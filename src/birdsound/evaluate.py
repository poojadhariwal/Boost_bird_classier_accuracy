from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report

from .config import TrainConfig
from .data import BirdSoundDataset, load_label_mapping, scan_dataset, set_seed, split_records
from .losses import FocalLoss
from .model import build_model
from .train import resolve_device, validate


def evaluate_run(run_dir: str | Path) -> dict[str, float]:
    run_dir = Path(run_dir)
    config = TrainConfig.from_json(run_dir / "config.json")
    set_seed(config.seed)
    label_to_index = load_label_mapping(run_dir / "label_to_index.json")

    all_records = scan_dataset(config.data_dir)
    _, val_records, _ = split_records(
        records=all_records,
        val_fraction=config.val_fraction,
        min_train_count_for_val=config.min_train_count_for_val,
        seed=config.seed,
    )

    dataset = BirdSoundDataset(
        records=val_records,
        label_to_index=label_to_index,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        clips_per_file=config.eval_clips_per_file,
        train=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(config.device)
    model = build_model(num_classes=len(label_to_index), freeze_backbone=config.freeze_backbone).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = FocalLoss(gamma=config.focal_gamma)

    _, metrics, predictions = validate(model, loader, device, criterion)
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)

    index_to_label = {index: label for label, index in label_to_index.items()}
    report = classification_report(
        predictions["target_index"],
        predictions["pred_index"],
        labels=sorted(index_to_label.keys()),
        target_names=[index_to_label[i] for i in sorted(index_to_label.keys())],
        zero_division=0,
        output_dict=True,
    )

    (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved run directory.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_run(args.run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
