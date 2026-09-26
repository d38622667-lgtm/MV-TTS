#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave
from tqdm import tqdm

from inverse_resample_syncdreamer_views import (
    ROOT,
    SyncDDIMSampler,
    decode_requested_views,
    encode_view_images,
    load_model,
    load_sample_view_paths,
    parse_view_list,
    prepare_inputs,
    resolve_path,
)
from inverse_resample_syncdreamer_views_view_ddnm import (
    DEFAULT_GT_INPUT_ROOT,
    freeze_model,
    inverse_resample_view_ddnm,
    make_grid_4x4,
    resolve_gt_condition_image,
)


DEFAULT_INPUT_ROOT = ROOT / "outputs" / "GSO_syncdreamer_view315_splits_1_new"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "GSO_syncdreamer_view315_splits_1_new_oracle_sample_nonbest_views_posterior_ddim_s4"
DEFAULT_PER_OBJECT = ROOT / "Angle_315" / "GSO_syncdreamer_view315_splits1_samples4_per_object.tsv"
DEFAULT_PER_VIEW = ROOT / "Angle_315" / "GSO_syncdreamer_view315_splits1_samples4_per_view_composed.tsv"
DEFAULT_CONDITION_ROOT = ROOT / "outputs" / "GSO_view315_rgba"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Posterior-DDIM for object-level oracle samples. The model is loaded once; "
            "for each object, views whose per-view oracle sample differs from the "
            "object-level best_sample are resampled."
        )
    )
    parser.add_argument("--cfg", type=Path, default=ROOT / "configs" / "syncdreamer.yaml")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "ckpt" / "syncdreamer-pretrain.ckpt")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per_object", type=Path, default=DEFAULT_PER_OBJECT)
    parser.add_argument("--per_view", type=Path, default=DEFAULT_PER_VIEW)
    parser.add_argument("--gt_input_root", type=Path, default=DEFAULT_GT_INPUT_ROOT)
    parser.add_argument("--condition_root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--splits", nargs="+", default=["all"])
    parser.add_argument(
        "--condition_source",
        choices=["sample", "gt", "condition_root"],
        default="sample",
        help=(
            "Use the first good sample view, matching GT condition view, or "
            "--condition_root/{split,all}/{object}.png as view0 condition."
        ),
    )
    parser.add_argument(
        "--force_view0_condition",
        action="store_true",
        help=(
            "Replace x0_ref view0 with --condition_root image and force view0 into "
            "good_view_ids so Posterior-DDIM treats it as a fixed measurement."
        ),
    )
    parser.add_argument("--output_views", type=str, default="all")
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma_y", type=float, default=0.2, help="Measurement uncertainty for Posterior-DDIM SNR switching. 0 always uses the measurement branch for fixed views.")
    parser.add_argument("--batch_view_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--crop_size", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--decode_observed",
        action="store_true",
        help="Deprecated compatibility flag; observed output views are always decoded now.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_best_samples(per_object_path: Path, input_root: Path, splits: set[str]) -> dict[tuple[str, str], str]:
    best_samples: dict[tuple[str, str], str] = {}
    with per_object_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row["split"]
            obj = row["object"]
            sample = row["best_sample"]
            if split not in splits:
                continue
            if sample == "ERROR":
                continue
            if not (input_root / split / obj / sample).is_dir():
                continue
            best_samples[(split, obj)] = sample
    return best_samples


def read_good_views(
    per_view_path: Path,
    best_samples: dict[tuple[str, str], str],
) -> dict[tuple[str, str], list[int]]:
    good_views: dict[tuple[str, str], list[int]] = {key: [] for key in best_samples}
    with per_view_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["split"], row["object"])
            if key not in best_samples:
                continue
            if row["selected_sample"] == best_samples[key]:
                good_views[key].append(int(row["view"]))
    return good_views


def resolve_condition_root_image(condition_root: Path, split: str, object_name: str) -> Path:
    candidates = [
        condition_root / split / f"{object_name}.png",
        condition_root / "all" / f"{object_name}.png",
        condition_root / f"{object_name}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find condition image for {split}/{object_name}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def build_jobs(args: argparse.Namespace) -> list[dict]:
    splits = set(args.splits)
    best_samples = read_best_samples(args.per_object, args.input_root, splits)
    good_views_by_key = read_good_views(args.per_view, best_samples)
    jobs = []
    for (split, obj), sample in sorted(best_samples.items()):
        good_views = sorted(set(good_views_by_key.get((split, obj), [])))
        if not good_views:
            good_views = [0]
        if args.force_view0_condition:
            good_views = sorted(set(good_views) | {0})
        bad_views = [view for view in range(16) if view not in good_views]
        if not bad_views:
            continue
        jobs.append(
            {
                "split": split,
                "object": obj,
                "best_sample": sample,
                "sample_dir": (args.input_root / split / obj / sample).resolve(),
                "good_view_ids": good_views,
                "bad_view_ids": bad_views,
            }
        )
        if args.limit and len(jobs) >= args.limit:
            break
    return jobs


def is_complete(output_dir: Path, output_views: list[int], sample_num: int) -> bool:
    if not (output_dir / "manifest.json").is_file():
        return False
    for sample_idx in range(sample_num):
        sample_dir = output_dir / f"sample{sample_idx:02d}"
        if not all((sample_dir / f"{view:03d}.png").is_file() for view in output_views):
            return False
    return True


def prepare_condition(model, condition_image: Path, sample_num: int, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    data = prepare_inputs(str(condition_image), args.elevation, args.crop_size, image_size=model.image_size)
    for key, value in data.items():
        data[key] = value.unsqueeze(0).to(args.device)
        data[key] = torch.repeat_interleave(data[key], sample_num, dim=0)
    _, clip_embed, input_info = model.prepare(data)
    return clip_embed, input_info


@torch.no_grad()
def run_job(
    model,
    sampler: SyncDDIMSampler,
    job: dict,
    output_dir: Path,
    output_views: list[int],
    args: argparse.Namespace,
) -> None:
    ref_view_paths = load_sample_view_paths(job["sample_dir"], model.view_num)
    view0_condition_image = None
    if args.force_view0_condition or args.condition_source == "condition_root":
        view0_condition_image = resolve_condition_root_image(
            args.condition_root,
            job["split"],
            job["object"],
        )
    if args.force_view0_condition:
        ref_view_paths[0] = view0_condition_image

    condition_view = 0 if args.condition_source == "condition_root" else job["good_view_ids"][0]
    if args.condition_source == "gt":
        condition_image = resolve_gt_condition_image(args.gt_input_root, job["sample_dir"], condition_view)
    elif args.condition_source == "condition_root":
        condition_image = view0_condition_image
    else:
        condition_image = ref_view_paths[condition_view]

    clip_embed, input_info = prepare_condition(model, condition_image, args.sample_num, args)
    ref_latent_dict = encode_view_images(
        model,
        ref_view_paths,
        args.sample_num,
        model.image_size,
        args.device,
    )
    x0_ref = torch.stack([ref_latent_dict[view] for view in range(model.view_num)], 1)

    latents, sampling_trace = inverse_resample_view_ddnm(
        model=model,
        sampler=sampler,
        input_info=input_info,
        clip_embed=clip_embed,
        x0_ref=x0_ref,
        good_view_ids=job["good_view_ids"],
        bad_view_ids=job["bad_view_ids"],
        source_view=condition_view,
        sample_num=args.sample_num,
        strength=args.strength,
        cfg_scale=args.cfg_scale,
        batch_view_num=args.batch_view_num,
        seed=args.seed,
        sigma_y=args.sigma_y,
        always_measurement_view_ids=[0] if 0 in job["good_view_ids"] else [],
        return_trace=True,
    )
    decoded = decode_requested_views(model, latents, output_views)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "oracle_sample_nonbest_view_posterior_ddim_batch",
        "split": job["split"],
        "object": job["object"],
        "best_sample": job["best_sample"],
        "init_sample_dir": str(job["sample_dir"]),
        "condition_source": args.condition_source,
        "condition_root": str(args.condition_root),
        "condition_view": condition_view,
        "condition_image": str(condition_image),
        "force_view0_condition": args.force_view0_condition,
        "view0_condition_image": str(view0_condition_image) if view0_condition_image is not None else None,
        "good_view_ids": job["good_view_ids"],
        "bad_view_ids": job["bad_view_ids"],
        "output_views": output_views,
        "sample_num": args.sample_num,
        "sample_steps": args.sample_steps,
        "strength": args.strength,
        "cfg_scale": args.cfg_scale,
        "sigma_y": args.sigma_y,
        "sampling_trace": sampling_trace,
        "batch_view_num": args.batch_view_num,
        "seed": args.seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for sample_idx in range(args.sample_num):
        sample_dir = output_dir / f"sample{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        row_images = []
        for view in output_views:
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

    del latents, decoded, x0_ref, ref_latent_dict, clip_embed, input_info
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_job_manifest(path: Path, jobs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("split\tobject\tbest_sample\tgood_views\tbad_views\tinit_sample_dir\n")
        for job in jobs:
            handle.write(
                f"{job['split']}\t{job['object']}\t{job['best_sample']}\t"
                f"{','.join(map(str, job['good_view_ids']))}\t"
                f"{','.join(map(str, job['bad_view_ids']))}\t{job['sample_dir']}\n"
            )


def main() -> None:
    args = parse_args()
    args.cfg = resolve_path(str(args.cfg))
    args.ckpt = resolve_path(str(args.ckpt))
    args.input_root = resolve_path(str(args.input_root))
    args.output_root = resolve_path(str(args.output_root))
    args.per_object = resolve_path(str(args.per_object))
    args.per_view = resolve_path(str(args.per_view))
    args.gt_input_root = resolve_path(str(args.gt_input_root))
    args.condition_root = resolve_path(str(args.condition_root))

    output_views = parse_view_list(args.output_views)
    if args.sigma_y < 0.0:
        raise ValueError(f"--sigma_y must be non-negative, got {args.sigma_y}")
    if args.force_view0_condition and args.condition_source != "condition_root":
        print(
            "[WARN] --force_view0_condition replaces view0 with --condition_root, "
            "but condition_source is not condition_root.",
            flush=True,
        )
    jobs = build_jobs(args)
    if not jobs:
        raise RuntimeError("No oracle Posterior-DDIM jobs found.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "oracle_posterior_ddim_jobs.tsv"
    write_job_manifest(manifest_path, jobs)

    print(f"[INFO] jobs={len(jobs)}", flush=True)
    print(f"[INFO] input_root={args.input_root}", flush=True)
    print(f"[INFO] condition_root={args.condition_root}", flush=True)
    print(f"[INFO] condition_source={args.condition_source}", flush=True)
    print(f"[INFO] force_view0_condition={args.force_view0_condition}", flush=True)
    print(f"[INFO] sigma_y={args.sigma_y}", flush=True)
    print(f"[INFO] output_root={args.output_root}", flush=True)
    print(f"[INFO] job_manifest={manifest_path}", flush=True)
    print("[INFO] loading model once...", flush=True)

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        model = load_model(args.cfg, args.ckpt, args.device)
        freeze_model(model)
        sampler = SyncDDIMSampler(model, args.sample_steps, latent_size=model.image_size // 8)

        for job in tqdm(jobs, desc="Oracle nonbest-view Posterior-DDIM jobs"):
            output_dir = args.output_root / job["split"] / job["object"] / job["best_sample"]
            if args.resume and not args.force and is_complete(output_dir, output_views, args.sample_num):
                print(f"[skip done] {output_dir}", flush=True)
                continue
            print(
                "[run] "
                f"{job['split']}/{job['object']}/{job['best_sample']} "
                f"good={','.join(map(str, job['good_view_ids']))} "
                f"bad={','.join(map(str, job['bad_view_ids']))}",
                flush=True,
            )
            run_job(model, sampler, job, output_dir, output_views, args)
    finally:
        os.chdir(original_cwd)

    print(f"[done] output_root={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
