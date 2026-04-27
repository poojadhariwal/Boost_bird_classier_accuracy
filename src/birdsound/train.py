from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import (
    BirdSoundDataset,
    build_weighted_sampler,
    compute_class_weights,
    make_label_mapping,
    save_label_mapping,
    scan_dataset,
    set_seed,
    split_records,
    summarize_records,
)
from .losses import FocalLoss
from .model import build_model


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(config: TrainConfig, label_to_index: dict[str, int]):
    records = scan_dataset(config.data_dir)
    train_records, val_records, ignored_for_val = split_records(
        records=records,
        val_fraction=config.val_fraction,
        min_train_count_for_val=config.min_train_count_for_val,
        seed=config.seed,
    )

    train_dataset = BirdSoundDataset(
        records=train_records,
        label_to_index=label_to_index,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        clips_per_file=config.train_clips_per_file,
        train=True,
    )
    val_dataset = BirdSoundDataset(
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=build_weighted_sampler(train_records, config.train_clips_per_file),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_records, val_records, ignored_for_val


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets, _ in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / max(len(loader.dataset), 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    model.eval()
    running_loss = 0.0
    grouped_logits: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_targets: dict[str, int] = {}

    with torch.no_grad():
        for inputs, targets, paths in tqdm(loader, desc="validate", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast(enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            target_values = targets.cpu().numpy()
            for path, prob, target in zip(paths, probs, target_values):
                grouped_logits[path].append(prob)
                grouped_targets[path] = int(target)

    rows = []
    y_true = []
    y_pred = []
    for path, logits_list in grouped_logits.items():
        mean_prob = np.mean(logits_list, axis=0)
        pred = int(np.argmax(mean_prob))
        target = grouped_targets[path]
        y_true.append(target)
        y_pred.append(pred)
        rows.append(
            {
                "path": path,
                "target_index": target,
                "pred_index": pred,
                "confidence": float(mean_prob[pred]),
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    predictions = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    loss_value = running_loss / max(len(loader.dataset), 1)
    return loss_value, metrics, predictions


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def run_training(config: TrainConfig) -> Path:
    set_seed(config.seed)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(run_dir / "config.json")

    all_records = scan_dataset(config.data_dir)
    label_to_index = make_label_mapping(all_records)
    save_label_mapping(run_dir / "label_to_index.json", label_to_index)

    train_loader, val_loader, train_records, val_records, ignored_for_val = create_dataloaders(config, label_to_index)
    device = resolve_device(config.device)

    model = build_model(num_classes=len(label_to_index), freeze_backbone=config.freeze_backbone).to(device)
    class_weights = compute_class_weights(train_records, label_to_index).to(device) if config.use_class_weights else None
    criterion = (
        FocalLoss(gamma=config.focal_gamma, weight=class_weights)
        if config.use_focal_loss
        else nn.CrossEntropyLoss(weight=class_weights)
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history_rows = []
    best_score = -1.0

    summary = {
        "all_records": summarize_records(all_records),
        "train_records": summarize_records(train_records),
        "val_records": summarize_records(val_records),
        "ignored_for_validation": ignored_for_val,
    }
    (run_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, metrics, predictions = validate(model, val_loader, device, criterion)
        scheduler.step()

        epoch_seconds = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "seconds": epoch_seconds,
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

        save_checkpoint(run_dir / "last_model.pt", model, optimizer, epoch)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch)
            predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(run_dir / "history.csv", index=False)

    build_confusion_matrix(run_dir, label_to_index)
    return run_dir


def build_confusion_matrix(run_dir: Path, label_to_index: dict[str, int]) -> None:
    predictions_path = run_dir / "validation_predictions.csv"
    if not predictions_path.exists():
        return

    df = pd.read_csv(predictions_path)
    index_to_label = {index: label for label, index in label_to_index.items()}
    labels = list(range(len(index_to_label)))
    matrix = confusion_matrix(df["target_index"], df["pred_index"], labels=labels)
    named = pd.DataFrame(
        matrix,
        index=[index_to_label[i] for i in labels],
        columns=[index_to_label[i] for i in labels],
    )
    named.to_csv(run_dir / "confusion_matrix.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bird audio classifier.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    run_dir = run_training(config)
    print(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
