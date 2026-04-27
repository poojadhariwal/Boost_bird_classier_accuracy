from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


# %% [markdown]
# # Bird Audio Classification
# This single Python file is organized like a Jupyter notebook.
# Each `# %%` block can be run as a separate cell in VS Code.


# %% [markdown]
# ## Section 1: Configuration

@dataclass
class Config:
    mode: str = "train"
    run_name: str = "single_file_run"
    data_dir: str = "iBC53"
    output_dir: str = "outputs/single_file_runs"
    audio_path: str = ""
    seed: int = 42
    sample_rate: int = 32000
    clip_seconds: float = 5.0
    train_clips_per_file: int = 2
    eval_clips_per_file: int = 3
    n_mels: int = 128
    fmin: int = 50
    fmax: int = 14000
    batch_size: int = 8
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    min_train_count_for_val: int = 2
    num_workers: int = 0
    device: str = "auto"
    freeze_backbone: bool = False
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


@dataclass(frozen=True)
class AudioRecord:
    path: str
    label: str


# %% [markdown]
# ## Section 2: Loss Function

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# %% [markdown]
# ## Section 3: Dataset Class

class BirdDataset(Dataset):
    def __init__(
        self,
        records: list[AudioRecord],
        label_to_index: dict[str, int],
        sample_rate: int,
        clip_seconds: float,
        n_mels: int,
        fmin: int,
        fmax: int,
        clips_per_file: int,
        train: bool,
    ) -> None:
        self.records = records
        self.label_to_index = label_to_index
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_seconds)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.clips_per_file = max(1, clips_per_file)
        self.train = train
        self.examples = [(record, clip_idx) for record in records for clip_idx in range(self.clips_per_file)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record, clip_index = self.examples[index]
        audio = load_audio(record.path, self.sample_rate)
        clip = extract_clip(audio, self.clip_samples, self.train, self.clips_per_file, clip_index)
        if self.train:
            clip = augment_waveform(clip, self.sample_rate)
        spec = audio_to_log_mel(clip, self.sample_rate, self.n_mels, self.fmin, self.fmax)
        if self.train:
            spec = spec_augment(spec)
        spec = normalize_spec(spec).repeat(3, 1, 1)
        return spec, self.label_to_index[record.label], record.path


# %% [markdown]
# ## Section 4: Utility Functions

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scan_dataset(data_dir: str | Path) -> list[AudioRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")
    records: list[AudioRecord] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav_path in sorted(class_dir.glob("*.wav")):
            records.append(AudioRecord(str(wav_path), class_dir.name))
    if not records:
        raise RuntimeError(f"No wav files found under {root}")
    return records


def split_records(
    records: list[AudioRecord],
    val_fraction: float,
    min_train_count_for_val: int,
    seed: int,
) -> tuple[list[AudioRecord], list[AudioRecord], dict[str, int]]:
    rng = random.Random(seed)
    grouped: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    train_records: list[AudioRecord] = []
    val_records: list[AudioRecord] = []
    kept_train_only: dict[str, int] = {}

    for label, items in sorted(grouped.items()):
        items = items[:]
        rng.shuffle(items)
        if len(items) < min_train_count_for_val:
            train_records.extend(items)
            kept_train_only[label] = len(items)
            continue
        val_count = min(max(1, round(len(items) * val_fraction)), len(items) - 1)
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])
    return train_records, val_records, kept_train_only


def make_label_mapping(records: list[AudioRecord]) -> dict[str, int]:
    labels = sorted({record.label for record in records})
    return {label: idx for idx, label in enumerate(labels)}


# %% [markdown]
# ## Section 5: Audio Preprocessing and Augmentation

def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if audio.size == 0:
        return np.zeros(sample_rate, dtype=np.float32)
    return audio.astype(np.float32)


def extract_clip(audio: np.ndarray, clip_samples: int, train: bool, clips_per_file: int, clip_index: int) -> np.ndarray:
    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    if len(audio) == clip_samples:
        return audio
    max_start = len(audio) - clip_samples
    if train:
        start = random.randint(0, max_start)
    elif clips_per_file == 1:
        start = max_start // 2
    else:
        position = clip_index / max(clips_per_file - 1, 1)
        start = int(round(position * max_start))
    return audio[start : start + clip_samples]


def augment_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    clip = audio.copy()
    if random.random() < 0.8:
        clip *= random.uniform(0.7, 1.3)
    if random.random() < 0.5:
        clip += np.random.normal(0.0, random.uniform(0.0005, 0.003), size=clip.shape).astype(np.float32)
    if random.random() < 0.5:
        clip = np.roll(clip, random.randint(-sample_rate // 4, sample_rate // 4))
    return np.clip(clip, -1.0, 1.0)


def audio_to_log_mel(audio: np.ndarray, sample_rate: int, n_mels: int, fmin: int, fmax: int) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)


def spec_augment(spec: torch.Tensor) -> torch.Tensor:
    augmented = spec.clone()
    _, freq_bins, time_bins = augmented.shape
    for _ in range(2):
        freq_width = random.randint(0, max(1, freq_bins // 12))
        if freq_width > 0:
            freq_start = random.randint(0, max(0, freq_bins - freq_width))
            augmented[:, freq_start : freq_start + freq_width, :] = augmented.mean()
        time_width = random.randint(0, max(1, time_bins // 12))
        if time_width > 0:
            time_start = random.randint(0, max(0, time_bins - time_width))
            augmented[:, :, time_start : time_start + time_width] = augmented.mean()
    return augmented


def normalize_spec(spec: torch.Tensor) -> torch.Tensor:
    std = spec.std().clamp_min(1e-6)
    return (spec - spec.mean()) / std


# %% [markdown]
# ## Section 6: Sampling, Weights, and Model

def build_sampler(records: list[AudioRecord], clips_per_file: int) -> WeightedRandomSampler:
    counts = Counter(record.label for record in records)
    weights: list[float] = []
    for record in records:
        weights.extend([1.0 / counts[record.label]] * max(1, clips_per_file))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_class_weights(records: list[AudioRecord], label_to_index: dict[str, int]) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    total = sum(counts.values())
    weights = np.zeros(len(label_to_index), dtype=np.float32)
    for label, idx in label_to_index.items():
        weights[idx] = total / max(counts[label], 1)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# %% [markdown]
# ## Section 7: Data Loader Preparation

def summarize_records(records: list[AudioRecord]) -> dict[str, float | int]:
    counts = [count for _, count in Counter(record.label for record in records).items()]
    return {
        "classes": len(counts),
        "files": len(records),
        "min_per_class": min(counts),
        "max_per_class": max(counts),
        "mean_per_class": float(np.mean(counts)),
        "median_per_class": float(np.median(counts)),
    }


def create_loaders(config: Config, label_to_index: dict[str, int]):
    all_records = scan_dataset(config.data_dir)
    train_records, val_records, kept_train_only = split_records(
        all_records,
        config.val_fraction,
        config.min_train_count_for_val,
        config.seed,
    )

    train_dataset = BirdDataset(
        train_records,
        label_to_index,
        config.sample_rate,
        config.clip_seconds,
        config.n_mels,
        config.fmin,
        config.fmax,
        config.train_clips_per_file,
        train=True,
    )
    val_dataset = BirdDataset(
        val_records,
        label_to_index,
        config.sample_rate,
        config.clip_seconds,
        config.n_mels,
        config.fmin,
        config.fmax,
        config.eval_clips_per_file,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=build_sampler(train_records, config.train_clips_per_file),
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

    summary = {
        "all_records": summarize_records(all_records),
        "train_records": summarize_records(train_records),
        "val_records": summarize_records(val_records) if val_records else {},
        "kept_train_only": kept_train_only,
    }
    return train_loader, val_loader, train_records, val_records, summary


# %% [markdown]
# ## Section 8: Training and Validation

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for inputs, targets, _ in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    return total_loss / max(len(loader.dataset), 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    grouped_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_targets: dict[str, int] = {}

    with torch.no_grad():
        for inputs, targets, paths in tqdm(loader, desc="validate", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            target_values = targets.cpu().numpy()
            for path, prob, target in zip(paths, probs, target_values):
                grouped_probs[path].append(prob)
                grouped_targets[path] = int(target)

    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for path, probs_list in grouped_probs.items():
        mean_prob = np.mean(probs_list, axis=0)
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
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
    }
    return total_loss / max(len(loader.dataset), 1), metrics, pd.DataFrame(rows)


# %% [markdown]
# ## Section 9: Saving Outputs and Reports

def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def train_pipeline(config: Config) -> None:
    set_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.run_dir / "config.json", asdict(config))

    all_records = scan_dataset(config.data_dir)
    label_to_index = make_label_mapping(all_records)
    save_json(config.run_dir / "label_to_index.json", label_to_index)

    train_loader, val_loader, train_records, _, summary = create_loaders(config, label_to_index)
    save_json(config.run_dir / "data_summary.json", summary)

    device = resolve_device(config.device)
    model = build_model(len(label_to_index), config.freeze_backbone).to(device)
    class_weights = compute_class_weights(train_records, label_to_index).to(device) if config.use_class_weights else None
    criterion = FocalLoss(config.focal_gamma, class_weights) if config.use_focal_loss else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    history_rows = []
    best_score = -1.0

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics, predictions = validate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "seconds": time.time() - start,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(config.run_dir / "history.csv", index=False)
        print(json.dumps(row, indent=2), flush=True)

        save_checkpoint(config.run_dir / "last_model.pt", model, optimizer, epoch)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            save_checkpoint(config.run_dir / "best_model.pt", model, optimizer, epoch)
            predictions.to_csv(config.run_dir / "validation_predictions.csv", index=False)
            save_json(config.run_dir / "metrics.json", metrics)

    build_reports(config.run_dir)
    print(f"Training complete. Outputs saved to: {config.run_dir}")


def build_reports(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    mapping = json.loads((run_dir / "label_to_index.json").read_text(encoding="utf-8"))
    index_to_label = {index: label for label, index in mapping.items()}
    pred_path = run_dir / "validation_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    labels = sorted(index_to_label.keys())
    matrix = confusion_matrix(df["target_index"], df["pred_index"], labels=labels)
    pd.DataFrame(
        matrix,
        index=[index_to_label[i] for i in labels],
        columns=[index_to_label[i] for i in labels],
    ).to_csv(run_dir / "confusion_matrix.csv")

    report = classification_report(
        df["target_index"],
        df["pred_index"],
        labels=labels,
        target_names=[index_to_label[i] for i in labels],
        zero_division=0,
        output_dict=True,
    )
    save_json(run_dir / "classification_report.json", report)


def plot_class_distribution(records: list[AudioRecord], title: str = "Class Distribution", top_n: int = 25) -> None:
    counts = Counter(record.label for record in records)
    frame = (
        pd.DataFrame([{"label": label, "count": count} for label, count in counts.items()])
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    plt.figure(figsize=(14, 6))
    plt.bar(frame["label"], frame["count"], color="steelblue")
    plt.title(title)
    plt.xlabel("Bird Class")
    plt.ylabel("Number of Audio Files")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.show()


def plot_train_val_distribution(train_records: list[AudioRecord], val_records: list[AudioRecord], top_n: int = 20) -> None:
    train_counts = Counter(record.label for record in train_records)
    val_counts = Counter(record.label for record in val_records)
    labels = sorted(set(train_counts) | set(val_counts))
    frame = pd.DataFrame(
        {
            "label": labels,
            "train_count": [train_counts.get(label, 0) for label in labels],
            "val_count": [val_counts.get(label, 0) for label in labels],
        }
    ).sort_values("train_count", ascending=False).head(top_n)

    x = np.arange(len(frame))
    width = 0.4
    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, frame["train_count"], width=width, label="Train", color="teal")
    plt.bar(x + width / 2, frame["val_count"], width=width, label="Validation", color="orange")
    plt.title("Train vs Validation Distribution")
    plt.xlabel("Bird Class")
    plt.ylabel("Number of Files")
    plt.xticks(x, frame["label"], rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(run_dir: str | Path) -> None:
    history_path = Path(run_dir) / "history.csv"
    if not history_path.exists():
        print(f"No history found at {history_path}")
        return

    history = pd.read_csv(history_path)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Val Loss")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["accuracy"], marker="o", label="Accuracy")
    plt.plot(history["epoch"], history["balanced_accuracy"], marker="o", label="Balanced Accuracy")
    plt.plot(history["epoch"], history["macro_f1"], marker="o", label="Macro F1")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_heatmap(run_dir: str | Path, max_classes: int = 25) -> None:
    matrix_path = Path(run_dir) / "confusion_matrix.csv"
    if not matrix_path.exists():
        print(f"No confusion matrix found at {matrix_path}")
        return

    matrix = pd.read_csv(matrix_path, index_col=0)
    if len(matrix) > max_classes:
        matrix = matrix.iloc[:max_classes, :max_classes]

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix.values, cmap="Blues", aspect="auto")
    plt.colorbar(label="Count")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(run_dir: str | Path) -> None:
    pred_path = Path(run_dir) / "validation_predictions.csv"
    if not pred_path.exists():
        print(f"No prediction file found at {pred_path}")
        return

    frame = pd.read_csv(pred_path)
    plt.figure(figsize=(10, 5))
    plt.hist(frame["confidence"], bins=20, color="slateblue", edgecolor="black")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Section 10: Main Pipelines

def evaluate_pipeline(config: Config) -> None:
    run_dir = config.run_dir
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(metrics_path.read_text(encoding="utf-8"))
    else:
        print(f"No metrics file found at {metrics_path}")


def predict_pipeline(config: Config) -> None:
    run_dir = config.run_dir
    if not config.audio_path:
        raise ValueError("Provide --audio-path for prediction mode.")

    saved_config = Config(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    mapping = json.loads((run_dir / "label_to_index.json").read_text(encoding="utf-8"))
    index_to_label = {index: label for label, index in mapping.items()}
    dummy_label = next(iter(mapping.keys()))

    dataset = BirdDataset(
        [AudioRecord(config.audio_path, dummy_label)],
        mapping,
        saved_config.sample_rate,
        saved_config.clip_seconds,
        saved_config.n_mels,
        saved_config.fmin,
        saved_config.fmax,
        saved_config.eval_clips_per_file,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=saved_config.eval_clips_per_file, shuffle=False)
    device = resolve_device(saved_config.device)
    model = build_model(len(mapping), saved_config.freeze_backbone).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for specs, _, _ in loader:
            specs = specs.to(device)
            logits = model(specs)
            mean_prob = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            break

    top_indices = np.argsort(mean_prob)[::-1][:5]
    results = [{"label": index_to_label[int(idx)], "probability": float(mean_prob[idx])} for idx in top_indices]
    print(json.dumps(results, indent=2))


# %% [markdown]
# ## Section 11: VS Code Interactive Cells
# Run these cells one by one in VS Code, similar to Google Colab.


# %%
# User settings cell
NOTEBOOK_CONFIG = Config(
    mode="train",
    run_name="vscode_notebook_run",
    data_dir="iBC53",
    output_dir="outputs/single_file_runs",
    audio_path="",
    seed=42,
    sample_rate=32000,
    clip_seconds=5.0,
    train_clips_per_file=2,
    eval_clips_per_file=3,
    n_mels=128,
    fmin=50,
    fmax=14000,
    batch_size=8,
    epochs=12,
    learning_rate=3e-4,
    weight_decay=1e-4,
    val_fraction=0.2,
    min_train_count_for_val=2,
    num_workers=0,
    device="auto",
    freeze_backbone=False,
    use_class_weights=True,
    use_focal_loss=True,
    focal_gamma=2.0,
)
print(NOTEBOOK_CONFIG)


# %%
# Dataset preview cell
set_seed(NOTEBOOK_CONFIG.seed)
all_records_preview = scan_dataset(NOTEBOOK_CONFIG.data_dir)
label_counts_preview = Counter(record.label for record in all_records_preview)
dataset_preview_df = (
    pd.DataFrame(
        [{"label": label, "count": count} for label, count in label_counts_preview.items()]
    )
    .sort_values(["count", "label"], ascending=[False, True])
    .reset_index(drop=True)
)
print("Total classes:", len(label_counts_preview))
print("Total audio files:", len(all_records_preview))
display(dataset_preview_df.head(15) if "display" in globals() else dataset_preview_df.head(15))


# %%
# Train/validation split preview cell
train_preview, val_preview, rare_preview = split_records(
    all_records_preview,
    NOTEBOOK_CONFIG.val_fraction,
    NOTEBOOK_CONFIG.min_train_count_for_val,
    NOTEBOOK_CONFIG.seed,
)
print("Training files:", len(train_preview))
print("Validation files:", len(val_preview))
print("Rare classes kept only in training:", rare_preview)


# %%
# Dataset graph cells
plot_class_distribution(all_records_preview, title="Overall Class Distribution", top_n=25)
plot_train_val_distribution(train_preview, val_preview, top_n=20)


# %%
# Training cell
# Run this cell when you want to start model training.
# train_pipeline(NOTEBOOK_CONFIG)


# %%
# Evaluation cell
# Run this after training is complete.
# evaluate_pipeline(NOTEBOOK_CONFIG)


# %%
# Analysis graph cells
# Run these after training is complete.
# plot_history(NOTEBOOK_CONFIG.run_dir)
# plot_confusion_heatmap(NOTEBOOK_CONFIG.run_dir, max_classes=25)
# plot_prediction_confidence(NOTEBOOK_CONFIG.run_dir)


# %%
# Prediction cell
# Set NOTEBOOK_CONFIG.audio_path first, then run this cell.
# NOTEBOOK_CONFIG.audio_path = r"iBC53\Acridotheres fuscus\1.wav"
# predict_pipeline(NOTEBOOK_CONFIG)


# %% [markdown]
# ## Section 12: Argument Parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file bird audio classification project.")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--run-name", default="single_file_run")
    parser.add_argument("--data-dir", default="iBC53")
    parser.add_argument("--output-dir", default="outputs/single_file_runs")
    parser.add_argument("--audio-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--train-clips-per-file", type=int, default=2)
    parser.add_argument("--eval-clips-per-file", type=int, default=3)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-count-for-val", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser


# %% [markdown]
# ## Section 13: Main Entry Point

def main() -> None:
    args = build_parser().parse_args()
    config = Config(
        mode=args.mode,
        run_name=args.run_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        audio_path=args.audio_path,
        seed=args.seed,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        train_clips_per_file=args.train_clips_per_file,
        eval_clips_per_file=args.eval_clips_per_file,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        min_train_count_for_val=args.min_train_count_for_val,
        num_workers=args.num_workers,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        use_class_weights=not args.no_class_weights,
        use_focal_loss=not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    if config.mode == "train":
        train_pipeline(config)
    elif config.mode == "evaluate":
        evaluate_pipeline(config)
    else:
        predict_pipeline(config)


if __name__ == "__main__":
    main()
