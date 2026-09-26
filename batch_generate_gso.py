import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave

from generate import load_model
from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import prepare_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SyncDreamer inference. Loads the model once and runs all input PNGs."
    )
    parser.add_argument("--cfg", type=str, default="configs/syncdreamer.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpt/syncdreamer-pretrain.ckpt")
    parser.add_argument(
        "--input-root",
        type=str,
        default="../outputs/GSO_syncdreamer_inputs_from_gt16_rgba",
        help="Root containing hard/normal/easy PNG inputs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="../outputs/GSO_syncdreamer_from_gt16",
        help="Root where SyncDreamer outputs are saved.",
    )
    parser.add_argument("--splits", nargs="+", default=["hard", "normal", "easy"])
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=200)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--batch_view_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument(
        "--save-individual-views",
        action="store_true",
        help="Also save each generated view as output_dir/sampleXX/000.png ... 015.png.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def is_complete(output_dir: Path, sample_num: int) -> bool:
    return all((output_dir / f"{idx}.png").exists() for idx in range(sample_num))


def collect_tasks(input_root: Path, output_root: Path, splits, sample_num: int, overwrite: bool):
    tasks = []
    skipped = 0
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split directory: {split_dir}")
            continue
        for input_path in sorted(split_dir.glob("*.png")):
            output_dir = output_root / split / input_path.stem
            if is_complete(output_dir, sample_num) and not overwrite:
                skipped += 1
                continue
            tasks.append((split, input_path, output_dir))
    return tasks, skipped


def run_one(model, sampler, input_path: Path, output_dir: Path, args, task_index: int):
    torch.random.manual_seed(args.seed + task_index)
    np.random.seed(args.seed + task_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_inputs(str(input_path), args.elevation, args.crop_size)
    for key, value in data.items():
        data[key] = value.unsqueeze(0).cuda()
        data[key] = torch.repeat_interleave(data[key], args.sample_num, dim=0)

    with torch.no_grad():
        x_sample = model.sample(sampler, data, args.cfg_scale, args.batch_view_num)

    batch, view_num, _, _, _ = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    saved = []
    for batch_idx in range(batch):
        output_path = output_dir / f"{batch_idx}.png"
        imsave(output_path, np.concatenate([x_sample[batch_idx, view_idx] for view_idx in range(view_num)], 1))
        saved.append(str(output_path))

        if args.save_individual_views:
            view_dir = output_dir / f"sample{batch_idx:02d}"
            view_dir.mkdir(parents=True, exist_ok=True)
            for view_idx in range(view_num):
                view_path = view_dir / f"{view_idx:03d}.png"
                imsave(view_path, x_sample[batch_idx, view_idx])
                saved.append(str(view_path))
    return saved


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks, skipped = collect_tasks(input_root, output_root, args.splits, args.sample_num, args.overwrite)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"input root: {input_root}")
    print(f"output root: {output_root}")
    print(f"pending: {len(tasks)}, skipped complete: {skipped}")
    print(f"sample_num: {args.sample_num}, elevation: {args.elevation}, crop_size: {args.crop_size}")
    if not tasks:
        return 0

    model = load_model(args.cfg, args.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    sampler = SyncDDIMSampler(model, args.sample_steps)

    manifest_path = output_root / "syncdreamer_batch_manifest.csv"
    manifest_exists = manifest_path.exists() and not args.overwrite
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "status", "split", "input_path", "output_dir", "saved_paths", "seconds"],
        )
        if not manifest_exists:
            writer.writeheader()

        start = time.time()
        for task_index, (split, input_path, output_dir) in enumerate(tasks):
            item_start = time.time()
            try:
                saved = run_one(model, sampler, input_path, output_dir, args, task_index)
                status = "ok"
                saved_paths = ";".join(saved)
            except Exception as exc:
                status = f"error: {exc}"
                saved_paths = ""
                print(f"[{task_index + 1}/{len(tasks)}] error {split}/{input_path.name}: {exc}")
            else:
                elapsed = time.time() - start
                print(
                    f"[{task_index + 1}/{len(tasks)}] ok {split}/{input_path.stem} "
                    f"({elapsed / 60:.1f} min)"
                )

            writer.writerow(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": status,
                    "split": split,
                    "input_path": str(input_path),
                    "output_dir": str(output_dir),
                    "saved_paths": saved_paths,
                    "seconds": f"{time.time() - item_start:.2f}",
                }
            )
            handle.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
