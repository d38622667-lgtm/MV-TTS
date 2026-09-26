import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.io import imsave
from tqdm import tqdm

from ldm.models.diffusion.sync_dreamer import SyncDDIMSampler, SyncMultiviewDiffusion, repeat_to_batch
from ldm.util import instantiate_from_config, prepare_inputs


ROOT = Path(__file__).resolve().parent


def parse_view_list(text: str, view_num: int = 16) -> list[int]:
    if text is None or text == "":
        return []
    if text.lower() == "all":
        return list(range(view_num))
    views: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            views.extend(range(int(start), int(end) + 1))
        else:
            views.append(int(part))
    deduped = []
    for view in views:
        if view < 0 or view >= view_num:
            raise ValueError(f"View index out of range [0,{view_num - 1}]: {view}")
        if view not in deduped:
            deduped.append(view)
    return deduped


def parse_view_paths(items: Optional[list[str]], view_num: int = 16) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected VIEW=PATH, got: {item}")
        view_text, path_text = item.split("=", 1)
        view = int(view_text)
        if view < 0 or view >= view_num:
            raise ValueError(f"View index out of range [0,{view_num - 1}]: {view}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping[view] = path
    return mapping


def load_model(cfg: Path, ckpt: Path, device: str) -> SyncMultiviewDiffusion:
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f"loading model from {ckpt} ...", flush=True)
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    assert isinstance(model, SyncMultiviewDiffusion)
    return model


def load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGBA")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3] * alpha + 1.0 - alpha
    rgb = rgb * 2.0 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def load_uint8_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    rgb = arr[:, :, :3] * alpha + 1.0 - alpha
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def load_sample_view_paths(sample_dir: Path, view_num: int) -> dict[int, Path]:
    paths = {}
    for view in range(view_num):
        path = sample_dir / f"{view:03d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[view] = path
    return paths


@torch.no_grad()
def encode_view_images(
    model: SyncMultiviewDiffusion,
    view_paths: dict[int, Path],
    sample_num: int,
    image_size: int,
    device: str,
) -> dict[int, torch.Tensor]:
    latents = {}
    for view, path in sorted(view_paths.items()):
        image = load_rgb_tensor(path, image_size).unsqueeze(0).to(device)
        latent = model.encode_first_stage(image, sample=False)
        latents[view] = latent.repeat(sample_num, 1, 1, 1)
    return latents


def viewpoint_embedding_for_source(
    model: SyncMultiviewDiffusion,
    batch_size: int,
    elevation_ref: torch.Tensor,
    source_view: int,
) -> torch.Tensor:
    azimuth_input = model.azimuth[source_view].unsqueeze(0)
    azimuth_target = model.azimuth
    elevation_input = -elevation_ref
    elevation_target = -np.deg2rad(30)
    delta_elevation = elevation_target - elevation_input
    num_views = model.azimuth.shape[0]
    delta_elevation = delta_elevation.unsqueeze(1).repeat(1, num_views)
    delta_azimuth = azimuth_target - azimuth_input
    delta_azimuth = delta_azimuth.unsqueeze(0).repeat(batch_size, 1)
    delta_radius = torch.zeros_like(delta_azimuth)
    return torch.stack(
        [delta_elevation, torch.sin(delta_azimuth), torch.cos(delta_azimuth), delta_radius],
        -1,
    )


@torch.no_grad()
def predict_noise_from_source_view(
    sampler: SyncDDIMSampler,
    x_target_noisy: torch.Tensor,
    input_info: dict,
    clip_embed: torch.Tensor,
    time_steps: torch.Tensor,
    unconditional_scale: float,
    batch_view_num: int,
    source_view: int,
) -> torch.Tensor:
    x_input, elevation_input = input_info["x"], input_info["elevation"]
    batch, num_views, channels, height, width = x_target_noisy.shape
    model = sampler.model

    v_embed = viewpoint_embedding_for_source(model, batch, elevation_input, source_view)
    t_embed = model.embed_time(time_steps)
    spatial_volume = model.spatial_volume.construct_spatial_volume(
        x_target_noisy,
        t_embed,
        v_embed,
        model.poses,
        model.Ks,
    )

    noise_chunks = []
    target_indices = torch.arange(num_views, device=x_target_noisy.device)
    for start in range(0, num_views, batch_view_num):
        x_target_noisy_chunk = x_target_noisy[:, start : start + batch_view_num]
        chunk_views = x_target_noisy_chunk.shape[1]
        x_target_noisy_chunk = x_target_noisy_chunk.reshape(
            batch * chunk_views,
            channels,
            height,
            width,
        )
        time_steps_chunk = repeat_to_batch(time_steps, batch, chunk_views)
        target_indices_chunk = target_indices[start : start + batch_view_num].unsqueeze(0).repeat(batch, 1)
        clip_embed_chunk, volume_feats_chunk, x_concat_chunk = model.get_target_view_feats(
            x_input,
            spatial_volume,
            clip_embed,
            t_embed,
            v_embed,
            target_indices_chunk,
        )
        if unconditional_scale != 1.0:
            noise = model.model.predict_with_unconditional_scale(
                x_target_noisy_chunk,
                time_steps_chunk,
                clip_embed_chunk,
                volume_feats_chunk,
                x_concat_chunk,
                unconditional_scale,
            )
        else:
            noise = model.model(
                x_target_noisy_chunk,
                time_steps_chunk,
                clip_embed_chunk,
                volume_feats_chunk,
                x_concat_chunk,
                is_train=False,
            )
        noise_chunks.append(noise.view(batch, chunk_views, channels, height, width))
    return torch.cat(noise_chunks, 1)


def q_sample_from_alpha(x0: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise


def clamp_views_at_alpha(
    x: torch.Tensor,
    fixed_latents: dict[int, torch.Tensor],
    fixed_noise: dict[int, torch.Tensor],
    alpha: torch.Tensor,
) -> torch.Tensor:
    if not fixed_latents:
        return x
    alpha = alpha.to(x.device).float().view(1, 1, 1, 1)
    for view, latent in fixed_latents.items():
        x[:, view] = q_sample_from_alpha(latent, fixed_noise[view], alpha)
    return x


@torch.no_grad()
def inverse_resample(
    model: SyncMultiviewDiffusion,
    sampler: SyncDDIMSampler,
    input_info: dict,
    clip_embed: torch.Tensor,
    fixed_latents: dict[int, torch.Tensor],
    init_latents: Optional[torch.Tensor],
    source_view: int,
    sample_num: int,
    strength: float,
    cfg_scale: float,
    batch_view_num: int,
    seed: int,
) -> torch.Tensor:
    device = model.device
    num_views = model.view_num
    channels, height, width = 4, sampler.latent_size, sampler.latent_size
    total_steps = len(sampler.ddim_timesteps)

    strength = float(np.clip(strength, 0.0, 1.0))
    start_index = int(round((total_steps - 1) * strength))
    start_index = max(0, min(total_steps - 1, start_index))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if init_latents is None:
        if start_index != total_steps - 1:
            print(
                "[WARN] --strength < 1 without --init_sample_dir starts unknown views from random noise "
                "at an intermediate DDIM step.",
                flush=True,
            )
        x = torch.randn(
            sample_num,
            num_views,
            channels,
            height,
            width,
            device=device,
            generator=generator,
        )
    else:
        init_noise = torch.randn(
            init_latents.shape,
            device=device,
            generator=generator,
        )
        alpha_start = sampler.ddim_alphas[start_index].to(device).float().view(1, 1, 1, 1, 1)
        x = q_sample_from_alpha(init_latents, init_noise, alpha_start)

    fixed_noise = {
        view: torch.randn(latent.shape, device=device, generator=generator)
        for view, latent in fixed_latents.items()
    }
    alpha_start = sampler.ddim_alphas[start_index].to(device).float()
    x = clamp_views_at_alpha(x, fixed_latents, fixed_noise, alpha_start)

    ddim_indices = list(range(start_index, -1, -1))
    iterator = tqdm(ddim_indices, desc="Inverse DDIM")
    for index in iterator:
        step = int(sampler.ddim_timesteps[index])
        time_steps = torch.full((sample_num,), step, device=device, dtype=torch.long)
        alpha_t = sampler.ddim_alphas[index].to(device).float()
        x = clamp_views_at_alpha(x, fixed_latents, fixed_noise, alpha_t)
        noise_pred = predict_noise_from_source_view(
            sampler=sampler,
            x_target_noisy=x,
            input_info=input_info,
            clip_embed=clip_embed,
            time_steps=time_steps,
            unconditional_scale=cfg_scale,
            batch_view_num=batch_view_num,
            source_view=source_view,
        )
        x = sampler.denoise_apply_impl(x, index, noise_pred, is_step0=(index == 0))
        if index > 0:
            alpha_prev = sampler.ddim_alphas_prev[index].to(device).float()
            x = clamp_views_at_alpha(x, fixed_latents, fixed_noise, alpha_prev)

    for view, latent in fixed_latents.items():
        x[:, view] = latent
    return x


@torch.no_grad()
def decode_requested_views(
    model: SyncMultiviewDiffusion,
    latents: torch.Tensor,
    output_views: list[int],
) -> dict[int, np.ndarray]:
    decoded = {}
    for view in output_views:
        image = model.decode_first_stage(latents[:, view])
        image = (torch.clamp(image, min=-1.0, max=1.0) + 1.0) * 0.5
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()
        decoded[view] = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return decoded


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "View-level inverse-problem resampling for SyncDreamer. "
            "Input/output view ids are SyncDreamer's fixed 16 target views."
        )
    )
    parser.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "syncdreamer.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "ckpt" / "syncdreamer-pretrain.ckpt"))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--input_views",
        nargs="+",
        required=True,
        help="Observed views as VIEW=PATH, e.g. --input_views 0=000.png 4=004.png 9=009.png.",
    )
    parser.add_argument(
        "--output_views",
        type=str,
        default="all",
        help="Views to save, e.g. all, 0,1,2, 4-8, or 0,4-8,15.",
    )
    parser.add_argument(
        "--init_sample_dir",
        type=str,
        default=None,
        help="Optional complete SyncDreamer sample dir with 000.png ... 015.png for SDEdit-style repair.",
    )
    parser.add_argument(
        "--resample_views",
        type=str,
        default=None,
        help=(
            "Bad views to resample when --init_sample_dir is set. "
            "All other init views are fixed as additional conditions."
        ),
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        default=None,
        help="Optional image for SyncDreamer's original single-image condition. Defaults to --condition_view image.",
    )
    parser.add_argument(
        "--condition_view",
        type=int,
        default=None,
        help="View id of the condition image. Defaults to the smallest input view id.",
    )
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--crop_size", type=int, default=-1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--batch_view_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--decode_observed",
        action="store_true",
        help="Decode observed output views from VAE latents instead of copying the exact input pixels.",
    )
    args = parser.parse_args()

    cfg = resolve_path(args.cfg)
    ckpt = resolve_path(args.ckpt)
    output_dir = resolve_path(args.output)
    init_sample_dir = resolve_path(args.init_sample_dir) if args.init_sample_dir else None

    input_view_paths = parse_view_paths(args.input_views)
    output_views = parse_view_list(args.output_views)
    if not output_views:
        raise ValueError("No output views requested.")

    condition_view = args.condition_view
    if condition_view is None:
        condition_view = min(input_view_paths)
    if condition_view not in input_view_paths and args.condition_image is None:
        raise ValueError("--condition_view must be one of --input_views unless --condition_image is set.")

    condition_image = (
        resolve_path(args.condition_image)
        if args.condition_image
        else input_view_paths[condition_view]
    )
    if not condition_image.is_file():
        raise FileNotFoundError(condition_image)

    # SyncDreamer reads meta_info/camera-16.pkl and CLIP paths relative to its repo root.
    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        model = load_model(cfg, ckpt, args.device)
        sampler = SyncDDIMSampler(model, args.sample_steps, latent_size=model.image_size // 8)

        data = prepare_inputs(str(condition_image), args.elevation, args.crop_size, image_size=model.image_size)
        for key, value in data.items():
            data[key] = value.unsqueeze(0).to(args.device)
            data[key] = torch.repeat_interleave(data[key], args.sample_num, dim=0)

        _, clip_embed, input_info = model.prepare(data)

        fixed_view_paths = dict(input_view_paths)
        init_latents = None
        init_view_paths = {}
        resample_views = parse_view_list(args.resample_views) if args.resample_views else []
        if init_sample_dir is not None:
            init_view_paths = load_sample_view_paths(init_sample_dir, model.view_num)
            init_latent_dict = encode_view_images(
                model,
                init_view_paths,
                args.sample_num,
                model.image_size,
                args.device,
            )
            init_latents = torch.stack([init_latent_dict[view] for view in range(model.view_num)], 1)
            if resample_views:
                for view, path in init_view_paths.items():
                    if view not in resample_views and view not in fixed_view_paths:
                        fixed_view_paths[view] = path

        fixed_latents = encode_view_images(
            model,
            fixed_view_paths,
            args.sample_num,
            model.image_size,
            args.device,
        )

        latents = inverse_resample(
            model=model,
            sampler=sampler,
            input_info=input_info,
            clip_embed=clip_embed,
            fixed_latents=fixed_latents,
            init_latents=init_latents,
            source_view=condition_view,
            sample_num=args.sample_num,
            strength=args.strength,
            cfg_scale=args.cfg_scale,
            batch_view_num=args.batch_view_num,
            seed=args.seed,
        )
        decoded = decode_requested_views(model, latents, output_views)
    finally:
        os.chdir(original_cwd)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_pixels = {
        view: load_uint8_rgb(path, 256)
        for view, path in input_view_paths.items()
    }
    init_pixels = {
        view: load_uint8_rgb(path, 256)
        for view, path in init_view_paths.items()
    }

    manifest = {
        "input_views": {str(view): str(path) for view, path in sorted(input_view_paths.items())},
        "output_views": output_views,
        "condition_view": condition_view,
        "condition_image": str(condition_image),
        "init_sample_dir": str(init_sample_dir) if init_sample_dir else None,
        "resample_views": resample_views,
        "fixed_views": sorted(fixed_view_paths),
        "sample_num": args.sample_num,
        "sample_steps": args.sample_steps,
        "strength": args.strength,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for sample_idx in range(args.sample_num):
        sample_dir = output_dir / f"sample{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        row_images = []
        for view in output_views:
            if view in input_pixels and not args.decode_observed:
                image = input_pixels[view]
            elif view in init_pixels and init_sample_dir is not None and view not in resample_views and not args.decode_observed:
                image = init_pixels[view]
            else:
                image = decoded[view][sample_idx]
            imsave(sample_dir / f"{view:03d}.png", image)
            row_images.append(image)
        imsave(output_dir / f"{sample_idx:03d}_views_{'-'.join(f'{v:03d}' for v in output_views)}.png", np.concatenate(row_images, axis=1))

    print(f"Saved inverse-resampled views to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
