#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave
from tqdm import tqdm

from ldm.models.diffusion.sync_dreamer import SyncDDIMSampler, SyncMultiviewDiffusion
from ldm.util import prepare_inputs
from inverse_resample_syncdreamer_views import (
    ROOT,
    decode_requested_views,
    encode_view_images,
    load_model,
    load_sample_view_paths,
    load_uint8_rgb,
    parse_view_list,
    parse_view_paths,
    predict_noise_from_source_view,
    q_sample_from_alpha,
    resolve_path,
)


DEFAULT_GT_INPUT_ROOT = ROOT / "outputs" / "gt_original_sample_inputs_view270"


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None


def predict_x0_from_noise(x_t: torch.Tensor, noise_pred: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.to(x_t.device).float().view(1, 1, 1, 1, 1)
    return (x_t - (1.0 - alpha).sqrt() * noise_pred) / alpha.sqrt()


def noise_from_x0(x_t: torch.Tensor, x0_pred: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.to(x_t.device).float().view(1, 1, 1, 1, 1)
    sqrt_one_minus_alpha = torch.clamp(1.0 - alpha, min=1e-12).sqrt()
    return (x_t - alpha.sqrt() * x0_pred) / sqrt_one_minus_alpha


def posterior_ddim_use_measurement(
    sigma_y: float,
    alpha_bar_prev: torch.Tensor,
) -> bool:
    alpha_prev = alpha_bar_prev.sqrt()
    sigma_prev = torch.clamp(1.0 - alpha_bar_prev, min=0.0).sqrt()
    if sigma_y == 0.0:
        measurement_snr = torch.tensor(float("inf"), device=alpha_bar_prev.device)
    else:
        measurement_snr = torch.tensor(1.0 / max(sigma_y, 1e-8), device=alpha_bar_prev.device)
    diffusion_snr = alpha_prev / torch.clamp(sigma_prev, min=1e-8)
    return bool((measurement_snr > diffusion_snr).item())


def posterior_ddim_measurement_prev(
    y: torch.Tensor,
    sigma_y: float,
    alpha_bar_prev: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    alpha_bar_prev = alpha_bar_prev.to(device=y.device, dtype=y.dtype).view(1, 1, 1, 1, 1)
    alpha_prev = alpha_bar_prev.sqrt()
    sigma_prev_sq = torch.clamp(1.0 - alpha_bar_prev, min=0.0)
    std_sq = torch.clamp(sigma_prev_sq - alpha_bar_prev * (sigma_y ** 2), min=0.0)
    z = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
    return alpha_prev * y + std_sq.sqrt() * z


def infer_split_object(sample_dir: Path) -> tuple[str, str]:
    if sample_dir.name == "original_sample":
        return sample_dir.parent.parent.name, sample_dir.parent.name
    return sample_dir.parent.name, sample_dir.name


def resolve_gt_condition_image(gt_input_root: Path, sample_dir: Path, condition_view: int) -> Path:
    split, object_name = infer_split_object(sample_dir)
    candidates = [
        gt_input_root / split / object_name / "original_sample" / f"{condition_view:03d}.png",
        gt_input_root / split / object_name / f"{condition_view:03d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find GT condition image for "
        f"{split}/{object_name} view {condition_view:03d}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def make_grid_4x4(images: list[np.ndarray]) -> np.ndarray:
    if len(images) != 16:
        raise ValueError(f"4x4 grid requires exactly 16 images, got {len(images)}")
    rows = [
        np.concatenate(images[row * 4 : (row + 1) * 4], axis=1)
        for row in range(4)
    ]
    return np.concatenate(rows, axis=0)

@torch.no_grad()
def inverse_resample_view_ddnm(
    model: SyncMultiviewDiffusion,
    sampler: SyncDDIMSampler,
    input_info: dict,
    clip_embed: torch.Tensor,
    x0_ref: torch.Tensor,
    good_view_ids: list[int],
    bad_view_ids: list[int],
    source_view: int,
    sample_num: int,
    strength: float,
    cfg_scale: float,
    batch_view_num: int,
    seed: int,
    sigma_y: float = 0.0,
    always_measurement_view_ids: list[int] | None = None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    freeze_model(model)
    device = model.device
    total_steps = len(sampler.ddim_timesteps)
    good_view_ids = sorted(set(good_view_ids))
    bad_view_ids = sorted(set(bad_view_ids))
    always_measurement_view_ids = sorted(set(always_measurement_view_ids or []))
    if not good_view_ids:
        raise ValueError("View-Level Posterior-DDIM requires at least one good view.")
    if not bad_view_ids:
        raise ValueError("View-Level Posterior-DDIM requires at least one bad view.")
    overlap = sorted(set(good_view_ids) & set(bad_view_ids))
    if overlap:
        raise ValueError(f"good_view_ids and bad_view_ids overlap: {overlap}")
    missing_always_measurement = sorted(set(always_measurement_view_ids) - set(good_view_ids))
    if missing_always_measurement:
        raise ValueError(
            "always_measurement_view_ids must be a subset of good_view_ids, got "
            f"{missing_always_measurement}"
        )
    if sigma_y < 0.0:
        raise ValueError(f"sigma_y must be non-negative, got {sigma_y}")

    strength = float(np.clip(strength, 0.0, 1.0))
    start_index = int(round((total_steps - 1) * strength))
    start_index = max(0, min(total_steps - 1, start_index))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    alpha_start = sampler.ddim_alphas[start_index].to(device).float().view(1, 1, 1, 1, 1)
    noise = torch.randn(x0_ref.shape, device=device, generator=generator)
    x = q_sample_from_alpha(x0_ref, noise, alpha_start)
    measurement_snr = float("inf") if sigma_y == 0.0 else 1.0 / max(sigma_y, 1e-8)
    trace = {
        "snr_switch": {
            "sigma_y": sigma_y,
            "measurement_snr": measurement_snr,
            "start_index": start_index,
            "start_step": int(sampler.ddim_timesteps[start_index]),
            "first_measurement_index": None,
            "first_measurement_step": None,
            "first_prior_index": None,
            "first_prior_step": None,
            "always_measurement_view_ids": always_measurement_view_ids,
        }
    }

    for index in tqdm(list(range(start_index, -1, -1)), desc="View-Level Posterior-DDIM"):
        step = int(sampler.ddim_timesteps[index])
        time_steps = torch.full((sample_num,), step, device=device, dtype=torch.long)
        alpha_t = sampler.ddim_alphas[index].to(device).float()

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
        _x0_pred = predict_x0_from_noise(x, noise_pred, alpha_t)

        x_prev_prior = sampler.denoise_apply_impl(
            x,
            index,
            noise_pred,
            is_step0=(index == 0),
        )

        alpha_bar_prev = sampler.ddim_alphas_prev[index].to(device).float()
        use_measurement = posterior_ddim_use_measurement(sigma_y, alpha_bar_prev)
        if use_measurement and trace["snr_switch"]["first_measurement_index"] is None:
            trace["snr_switch"]["first_measurement_index"] = index
            trace["snr_switch"]["first_measurement_step"] = step
        if not use_measurement and trace["snr_switch"]["first_prior_index"] is None:
            trace["snr_switch"]["first_prior_index"] = index
            trace["snr_switch"]["first_prior_step"] = step

        measurement_view_ids = good_view_ids if use_measurement else always_measurement_view_ids
        x = x_prev_prior
        if measurement_view_ids:
            x_prev_measurement = posterior_ddim_measurement_prev(
                y=x0_ref[:, measurement_view_ids],
                sigma_y=sigma_y,
                alpha_bar_prev=alpha_bar_prev,
                generator=generator,
            )
            x[:, measurement_view_ids] = x_prev_measurement

    if return_trace:
        return x, trace
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View-Level Posterior-DDIM bad-view resampling for SyncDreamer."
    )
    parser.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "syncdreamer.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "ckpt" / "syncdreamer-pretrain.ckpt"))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--input_views",
        nargs="+",
        default=None,
        help=(
            "Optional observed views as VIEW=PATH. If omitted, condition_view is read "
            "from --init_sample_dir."
        ),
    )
    parser.add_argument("--output_views", type=str, default="all")
    parser.add_argument("--init_sample_dir", type=str, required=True, help="Complete 16-view x0_ref sample dir.")
    parser.add_argument("--bad_view_ids", "--resample_views", dest="bad_view_ids", type=str, required=True)
    parser.add_argument(
        "--good_view_ids",
        type=str,
        default=None,
        help="Known observation views. Defaults to all non-bad views.",
    )
    parser.add_argument("--condition_image", type=str, default=None)
    parser.add_argument(
        "--condition_source",
        choices=["gt", "sample"],
        default="gt",
        help=(
            "Default condition image source when --condition_image is omitted. "
            "gt reads from --gt_input_root; sample reads from --init_sample_dir."
        ),
    )
    parser.add_argument(
        "--gt_input_root",
        type=str,
        default=str(DEFAULT_GT_INPUT_ROOT),
        help="GT input root used when --condition_source gt.",
    )
    parser.add_argument("--condition_view", type=int, default=0)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--crop_size", type=int, default=-1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma_y", type=float, default=0.0, help="Measurement uncertainty for Posterior-DDIM SNR switching. 0 always uses the measurement branch for fixed views.")
    parser.add_argument("--batch_view_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--decode_observed",
        action="store_true",
        help="Decode good output views from VAE latents instead of copying exact x0_ref pixels.",
    )
    args = parser.parse_args()

    cfg = resolve_path(args.cfg)
    ckpt = resolve_path(args.ckpt)
    output_dir = resolve_path(args.output)
    init_sample_dir = resolve_path(args.init_sample_dir)
    gt_input_root = resolve_path(args.gt_input_root)
    if not init_sample_dir.is_dir():
        raise FileNotFoundError(init_sample_dir)
    if args.sigma_y < 0.0:
        raise ValueError(f"--sigma_y must be non-negative, got {args.sigma_y}")

    output_views = parse_view_list(args.output_views)
    bad_view_ids = parse_view_list(args.bad_view_ids)
    input_view_paths = parse_view_paths(args.input_views) if args.input_views else {}

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        model = load_model(cfg, ckpt, args.device)
        freeze_model(model)
        sampler = SyncDDIMSampler(model, args.sample_steps, latent_size=model.image_size // 8)

        ref_view_paths = load_sample_view_paths(init_sample_dir, model.view_num)
        condition_view = args.condition_view
        if condition_view < 0 or condition_view >= model.view_num:
            raise ValueError(f"--condition_view out of range: {condition_view}")
        if not input_view_paths and args.condition_source == "sample":
            input_view_paths = {condition_view: ref_view_paths[condition_view]}
        if (
            args.condition_source == "sample"
            and condition_view not in input_view_paths
            and args.condition_image is None
        ):
            raise ValueError("--condition_view must be one of --input_views unless --condition_image is set.")
        if args.condition_image:
            condition_image = resolve_path(args.condition_image)
        elif args.condition_source == "gt":
            condition_image = resolve_gt_condition_image(gt_input_root, init_sample_dir, condition_view)
            input_view_paths = {condition_view: condition_image}
        else:
            condition_image = input_view_paths[condition_view]
        if not condition_image.is_file():
            raise FileNotFoundError(condition_image)

        good_view_ids = (
            parse_view_list(args.good_view_ids, model.view_num)
            if args.good_view_ids
            else [view for view in range(model.view_num) if view not in bad_view_ids]
        )

        data = prepare_inputs(str(condition_image), args.elevation, args.crop_size, image_size=model.image_size)
        for key, value in data.items():
            data[key] = value.unsqueeze(0).to(args.device)
            data[key] = torch.repeat_interleave(data[key], args.sample_num, dim=0)
        _, clip_embed, input_info = model.prepare(data)

        ref_latent_dict = encode_view_images(
            model,
            ref_view_paths,
            args.sample_num,
            model.image_size,
            args.device,
        )
        x0_ref = torch.stack([ref_latent_dict[view] for view in range(model.view_num)], 1)

        latents = inverse_resample_view_ddnm(
            model=model,
            sampler=sampler,
            input_info=input_info,
            clip_embed=clip_embed,
            x0_ref=x0_ref,
            good_view_ids=good_view_ids,
            bad_view_ids=bad_view_ids,
            source_view=condition_view,
            sample_num=args.sample_num,
            strength=args.strength,
            cfg_scale=args.cfg_scale,
            batch_view_num=args.batch_view_num,
            seed=args.seed,
            sigma_y=args.sigma_y,
        )
        decoded = decode_requested_views(model, latents, output_views)
    finally:
        os.chdir(original_cwd)

    output_dir.mkdir(parents=True, exist_ok=True)
    ref_pixels = {view: load_uint8_rgb(path, 256) for view, path in ref_view_paths.items()}
    manifest = {
        "method": "view_level_posterior_ddim",
        "input_views": {str(view): str(path) for view, path in sorted(input_view_paths.items())},
        "output_views": output_views,
        "condition_view": condition_view,
        "condition_source": args.condition_source,
        "condition_image": str(condition_image),
        "gt_input_root": str(gt_input_root),
        "init_sample_dir": str(init_sample_dir),
        "good_view_ids": good_view_ids,
        "bad_view_ids": bad_view_ids,
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
            if view in good_view_ids and not args.decode_observed:
                image = ref_pixels[view]
            else:
                image = decoded[view][sample_idx]
            imsave(sample_dir / f"{view:03d}.png", image)
            row_images.append(image)
        imsave(
            output_dir / f"{sample_idx:03d}_views_{'-'.join(f'{view:03d}' for view in output_views)}.png",
            np.concatenate(row_images, axis=1),
        )
        if output_views == list(range(16)):
            grid = make_grid_4x4(row_images)
            imsave(output_dir / f"{sample_idx:03d}_grid_4x4.png", grid)
            imsave(sample_dir / "grid_4x4.png", grid)

    print(f"Saved View-Level Posterior-DDIM views to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
