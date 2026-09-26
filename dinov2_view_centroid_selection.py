from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_ROOT = REPO_ROOT / "SyncDreamer" / "outputs" / "GSO_syncdreamer_view270_splits_4"
DEFAULT_METRIC_SAMPLE_TSV = (
    REPO_ROOT
    / "SyncDreamer"
    / "eval_view270_splits4_true_bestof4_logs"
    / "GSO_syncdreamer_view270_splits4_true_bestof4_per_sample.tsv"
)
DEFAULT_METRIC_OBJECT_TSV = (
    REPO_ROOT
    / "SyncDreamer"
    / "eval_view270_splits4_true_bestof4_logs"
    / "GSO_syncdreamer_view270_splits4_true_bestof4_per_object.tsv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "SyncDreamer" / "eval_splits4_dinov2_view_centroid_logs"
DEFAULT_SAMPLES = [f"sample{i:02d}" for i in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one SyncDreamer sample per object by DINOv2 view centroids: "
            "for each view, average normalized DINO CLS features over samples, "
            "then choose the sample with the highest mean cosine to those view centroids."
        )
    )
    parser.add_argument("--pred_root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--splits", nargs="+", default=["hard"])
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--num_views", type=int, default=16)
    parser.add_argument(
        "--view_weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "edge000"],
        help=(
            "uniform averages all view similarities equally; edge000 gives higher "
            "weight to views near 000 on both sides of the circular view sequence."
        ),
    )
    parser.add_argument(
        "--edge_base_weight",
        type=float,
        default=1.0,
        help="Base weight for non-edge views when --view_weight_mode=edge000.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", type=str, default="syncdreamer_splits4_dinov2_view_centroid_hard")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=64,
        help="Number of view images to encode per DINOv2 forward pass.",
    )
    parser.add_argument("--metric_sample_tsv", type=Path, default=DEFAULT_METRIC_SAMPLE_TSV)
    parser.add_argument("--metric_object_tsv", type=Path, default=DEFAULT_METRIC_OBJECT_TSV)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load the DINOv2 model and processor from the local Hugging Face cache only.",
    )
    return parser.parse_args()


def build_view_weights(num_views: int, mode: str, edge_base_weight: float) -> np.ndarray:
    if mode == "uniform":
        return np.ones(num_views, dtype=np.float64)
    if mode != "edge000":
        raise ValueError(f"Unsupported view weight mode: {mode}")

    weights = np.full(num_views, edge_base_weight, dtype=np.float64)
    edge_weights = {
        0: 8.0,
        1: 8.0,
        2: 7.0,
        3: 6.0,
        num_views - 3: 6.0,
        num_views - 2: 7.0,
        num_views - 1: 8.0,
    }
    for view_idx, weight in edge_weights.items():
        if 0 <= view_idx < num_views:
            weights[view_idx] = max(weights[view_idx], weight)
    return weights


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def discover_objects(pred_root: Path, splits: list[str]) -> list[tuple[str, str, Path]]:
    objects = []
    for split in splits:
        split_dir = pred_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Missing split dir: {split_dir}", flush=True)
            continue
        for obj_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            objects.append((split, obj_dir.name, obj_dir))
    return objects


def build_metric_maps(sample_tsv: Path, object_tsv: Path) -> tuple[dict, dict]:
    sample_metrics = {}
    for row in read_tsv(sample_tsv):
        split = row.get("split", "")
        obj = row["object"]
        sample = row["sample"]
        sample_metrics[(split, obj, sample)] = {
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "lpips": float(row["lpips"]),
        }

    object_metrics = {}
    for row in read_tsv(object_tsv):
        split = row.get("split", "")
        obj = row["object"]
        object_metrics[(split, obj)] = {
            "true_best_sample": row["best_sample"],
            "true_best_rank_sum": float(row["best_rank_sum"]),
            "mean4_psnr": float(row["mean_psnr"]),
            "mean4_ssim": float(row["mean_ssim"]),
            "mean4_lpips": float(row["mean_lpips"]),
            "true_best_psnr": float(row["best_psnr"]),
            "true_best_ssim": float(row["best_ssim"]),
            "true_best_lpips": float(row["best_lpips"]),
            "view_best_psnr": float(row["view_best_psnr"]),
            "view_best_ssim": float(row["view_best_ssim"]),
            "view_best_lpips": float(row["view_best_lpips"]),
        }
    return sample_metrics, object_metrics


def add_true_rank_sums(rows: list[dict]) -> None:
    psnr_order = sorted(range(len(rows)), key=lambda idx: rows[idx]["selected_psnr"], reverse=True)
    ssim_order = sorted(range(len(rows)), key=lambda idx: rows[idx]["selected_ssim"], reverse=True)
    lpips_order = sorted(range(len(rows)), key=lambda idx: rows[idx]["selected_lpips"])
    rank_sum = np.zeros(len(rows), dtype=np.float32)
    for rank, idx in enumerate(psnr_order, start=1):
        rank_sum[idx] += rank
    for rank, idx in enumerate(ssim_order, start=1):
        rank_sum[idx] += rank
    for rank, idx in enumerate(lpips_order, start=1):
        rank_sum[idx] += rank
    for idx, row in enumerate(rows):
        row["selected_rank_sum"] = float(rank_sum[idx])


@torch.no_grad()
def encode_object(
    obj_dir: Path,
    samples: list[str],
    num_views: int,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: str,
    image_batch_size: int,
) -> torch.Tensor:
    images = []
    for sample in samples:
        sample_dir = obj_dir / sample
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Missing sample dir: {sample_dir}")
        for view_idx in range(num_views):
            images.append(load_image(sample_dir / f"{view_idx:03d}.png"))

    feature_chunks = []
    for start in range(0, len(images), image_batch_size):
        chunk = images[start:start + image_batch_size]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feature_chunks.append(outputs.pooler_output.detach().float().cpu())
    features = torch.cat(feature_chunks, dim=0).to(device)
    features = F.normalize(features, p=2, dim=1)
    return features.view(len(samples), num_views, -1)


def score_by_view_centroid(features: torch.Tensor, view_weights: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    centroids = F.normalize(features.mean(dim=0), p=2, dim=1)
    similarities = torch.sum(features * centroids.unsqueeze(0), dim=2)
    sample_scores = torch.sum(similarities * view_weights.unsqueeze(0), dim=1) / view_weights.sum()
    return (
        sample_scores.detach().cpu().numpy().astype(np.float64),
        similarities.detach().cpu().numpy().astype(np.float64),
    )


def mean_metric(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(values.mean()) if len(values) else float("nan")


def summarize(per_object_rows: list[dict]) -> dict:
    selected = {
        "psnr": mean_metric(per_object_rows, "selected_psnr"),
        "ssim": mean_metric(per_object_rows, "selected_ssim"),
        "lpips": mean_metric(per_object_rows, "selected_lpips"),
        "rank_sum": mean_metric(per_object_rows, "selected_rank_sum"),
    }
    return {
        "num_objects": len(per_object_rows),
        "hit_true_best": int(sum(row["hit_true_best"] for row in per_object_rows)),
        "hit_true_best_rate": float(np.mean([row["hit_true_best"] for row in per_object_rows]))
        if per_object_rows
        else float("nan"),
        "selected": selected,
        "mean4": {
            "psnr": mean_metric(per_object_rows, "mean4_psnr"),
            "ssim": mean_metric(per_object_rows, "mean4_ssim"),
            "lpips": mean_metric(per_object_rows, "mean4_lpips"),
        },
        "true_best_full_sample": {
            "psnr": mean_metric(per_object_rows, "true_best_psnr"),
            "ssim": mean_metric(per_object_rows, "true_best_ssim"),
            "lpips": mean_metric(per_object_rows, "true_best_lpips"),
        },
        "true_best_view_composed": {
            "psnr": mean_metric(per_object_rows, "view_best_psnr"),
            "ssim": mean_metric(per_object_rows, "view_best_ssim"),
            "lpips": mean_metric(per_object_rows, "view_best_lpips"),
        },
    }


def main() -> None:
    args = parse_args()
    args.pred_root = args.pred_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_sample_path = args.out_dir / f"{args.name}_all_samples.tsv"
    per_view_path = args.out_dir / f"{args.name}_per_view.tsv"
    per_object_path = args.out_dir / f"{args.name}_per_object.tsv"
    summary_json_path = args.out_dir / f"{args.name}_summary.json"
    summary_txt_path = args.out_dir / f"{args.name}_summary.txt"

    sample_metrics, object_metrics = build_metric_maps(args.metric_sample_tsv, args.metric_object_tsv)

    processor = AutoImageProcessor.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = model.to(args.device).eval()

    view_weights_np = build_view_weights(args.num_views, args.view_weight_mode, args.edge_base_weight)
    view_weights = torch.from_numpy(view_weights_np).to(args.device, dtype=torch.float32)

    sample_rows = []
    view_rows = []
    object_rows = []
    objects = discover_objects(args.pred_root, args.splits)

    for split, object_name, obj_dir in tqdm(objects, desc="DINOv2 view-centroid selection"):
        features = encode_object(
            obj_dir=obj_dir,
            samples=args.samples,
            num_views=args.num_views,
            processor=processor,
            model=model,
            device=args.device,
            image_batch_size=args.image_batch_size,
        )
        sample_scores, view_similarities = score_by_view_centroid(features, view_weights)
        best_idx = int(np.argmax(sample_scores))
        selected_sample = args.samples[best_idx]

        object_metric = object_metrics[(split, object_name)]
        selected_metric = sample_metrics[(split, object_name, selected_sample)]
        candidate_rows = []

        for sample_idx, sample in enumerate(args.samples):
            metric = sample_metrics[(split, object_name, sample)]
            row = {
                "split": split,
                "object": object_name,
                "sample": sample,
                "dino_view_centroid_weighted": float(sample_scores[sample_idx]),
                "dino_view_centroid_mean": float(view_similarities[sample_idx].mean()),
                "psnr": metric["psnr"],
                "ssim": metric["ssim"],
                "lpips": metric["lpips"],
                "is_selected": int(sample == selected_sample),
            }
            for view_idx in range(args.num_views):
                row[f"view{view_idx:03d}_sim"] = float(view_similarities[sample_idx, view_idx])
                view_rows.append(
                    {
                        "split": split,
                        "object": object_name,
                        "sample": sample,
                        "view": f"{view_idx:03d}",
                        "dino_centroid_similarity": float(view_similarities[sample_idx, view_idx]),
                        "is_selected_sample": int(sample == selected_sample),
                    }
                )
            sample_rows.append(row)
            candidate_rows.append(
                {
                    "selected_psnr": metric["psnr"],
                    "selected_ssim": metric["ssim"],
                    "selected_lpips": metric["lpips"],
                }
            )

        add_true_rank_sums(candidate_rows)
        selected_rank_sum = candidate_rows[best_idx]["selected_rank_sum"]

        object_rows.append(
            {
                "split": split,
                "object": object_name,
                "selected_sample": selected_sample,
                "selected_dino_view_centroid_weighted": float(sample_scores[best_idx]),
                "selected_dino_view_centroid_mean": float(view_similarities[best_idx].mean()),
                "true_best_sample": object_metric["true_best_sample"],
                "hit_true_best": int(selected_sample == object_metric["true_best_sample"]),
                "selected_rank_sum": float(selected_rank_sum),
                "selected_psnr": selected_metric["psnr"],
                "selected_ssim": selected_metric["ssim"],
                "selected_lpips": selected_metric["lpips"],
                **{key: value for key, value in object_metric.items() if key != "true_best_sample"},
            }
        )

    sample_fields = [
        "split",
        "object",
        "sample",
        "dino_view_centroid_weighted",
        "dino_view_centroid_mean",
        "psnr",
        "ssim",
        "lpips",
        "is_selected",
    ] + [f"view{idx:03d}_sim" for idx in range(args.num_views)]
    per_view_fields = [
        "split",
        "object",
        "sample",
        "view",
        "dino_centroid_similarity",
        "is_selected_sample",
    ]
    object_fields = [
        "split",
        "object",
        "selected_sample",
        "selected_dino_view_centroid_weighted",
        "selected_dino_view_centroid_mean",
        "true_best_sample",
        "hit_true_best",
        "selected_rank_sum",
        "selected_psnr",
        "selected_ssim",
        "selected_lpips",
        "mean4_psnr",
        "mean4_ssim",
        "mean4_lpips",
        "true_best_rank_sum",
        "true_best_psnr",
        "true_best_ssim",
        "true_best_lpips",
        "view_best_psnr",
        "view_best_ssim",
        "view_best_lpips",
    ]

    write_tsv(all_sample_path, sample_rows, sample_fields)
    write_tsv(per_view_path, view_rows, per_view_fields)
    write_tsv(per_object_path, object_rows, object_fields)

    summary = summarize(object_rows)
    summary["pred_root"] = str(args.pred_root)
    summary["model_name"] = args.model_name
    summary["splits"] = args.splits
    summary["samples"] = args.samples
    summary["image_batch_size"] = args.image_batch_size
    summary["view_weight_mode"] = args.view_weight_mode
    summary["edge_base_weight"] = args.edge_base_weight
    summary["view_weights"] = {f"{idx:03d}": float(weight) for idx, weight in enumerate(view_weights_np)}
    summary["outputs"] = {
        "all_samples": str(all_sample_path),
        "per_view": str(per_view_path),
        "per_object": str(per_object_path),
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        f"Name: {args.name}",
        f"Pred root: {args.pred_root}",
        f"Model: {args.model_name}",
        f"View weight mode: {args.view_weight_mode}",
        "View weights: " + ",".join(f"{idx:03d}:{weight:g}" for idx, weight in enumerate(view_weights_np)),
        f"Objects: {summary['num_objects']}",
        f"Hit true best: {summary['hit_true_best']}/{summary['num_objects']} "
        f"({summary['hit_true_best_rate'] * 100:.2f}%)",
        "",
        "Selected by DINO view centroid\t"
        f"PSNR={summary['selected']['psnr']:.5f}\t"
        f"SSIM={summary['selected']['ssim']:.5f}\t"
        f"LPIPS={summary['selected']['lpips']:.5f}\t"
        f"rank_sum={summary['selected']['rank_sum']:.3f}",
        "Mean-of-4\t"
        f"PSNR={summary['mean4']['psnr']:.5f}\t"
        f"SSIM={summary['mean4']['ssim']:.5f}\t"
        f"LPIPS={summary['mean4']['lpips']:.5f}",
        "True Best-of-4 full-sample\t"
        f"PSNR={summary['true_best_full_sample']['psnr']:.5f}\t"
        f"SSIM={summary['true_best_full_sample']['ssim']:.5f}\t"
        f"LPIPS={summary['true_best_full_sample']['lpips']:.5f}",
        "True View-composed Best-of-4\t"
        f"PSNR={summary['true_best_view_composed']['psnr']:.5f}\t"
        f"SSIM={summary['true_best_view_composed']['ssim']:.5f}\t"
        f"LPIPS={summary['true_best_view_composed']['lpips']:.5f}",
    ]
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
