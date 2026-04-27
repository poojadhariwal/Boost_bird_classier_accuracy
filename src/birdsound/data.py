from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class AudioRecord:
    path: str
    label: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_dataset(data_dir: str | Path) -> list[AudioRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[AudioRecord] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for audio_path in sorted(class_dir.glob("*.wav")):
            records.append(AudioRecord(path=str(audio_path), label=class_dir.name))
    if not records:
        raise RuntimeError(f"No .wav files found under {root}")
    return records


def build_metadata_frame(records: list[AudioRecord]) -> pd.DataFrame:
    rows = [{"path": record.path, "label": record.label} for record in records]
    frame = pd.DataFrame(rows)
    frame["file_name"] = frame["path"].map(lambda x: Path(x).name)
    frame["class_count"] = frame.groupby("label")["label"].transform("count")
    return frame.sort_values(["label", "file_name"]).reset_index(drop=True)


def split_records(
    records: list[AudioRecord],
    val_fraction: float,
    min_train_count_for_val: int,
    seed: int,
) -> tuple[list[AudioRecord], list[AudioRecord], dict[str, int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    train_records: list[AudioRecord] = []
    val_records: list[AudioRecord] = []
    ignored_for_val: dict[str, int] = {}

    for label, items in sorted(by_label.items()):
        items = items[:]
        rng.shuffle(items)
        count = len(items)

        if count < min_train_count_for_val:
            train_records.extend(items)
            ignored_for_val[label] = count
            continue

        raw_val = max(1, int(round(count * val_fraction)))
        val_count = min(raw_val, count - 1)

        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    return train_records, val_records, ignored_for_val


def make_label_mapping(records: list[AudioRecord]) -> dict[str, int]:
    labels = sorted({record.label for record in records})
    return {label: index for index, label in enumerate(labels)}


def compute_class_weights(records: list[AudioRecord], label_to_index: dict[str, int]) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    weights = np.zeros(len(label_to_index), dtype=np.float32)
    total = sum(counts.values())
    for label, index in label_to_index.items():
        weights[index] = total / max(counts[label], 1)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(records: list[AudioRecord], clips_per_file: int) -> WeightedRandomSampler:
    counts = Counter(record.label for record in records)
    sample_weights = []
    for record in records:
        sample_weights.extend([1.0 / counts[record.label]] * max(1, clips_per_file))
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def save_label_mapping(path: str | Path, mapping: dict[str, int]) -> None:
    Path(path).write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def load_label_mapping(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BirdSoundDataset(Dataset):
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
        self.clips_per_file = clips_per_file
        self.train = train
        self.examples = [
            (record, clip_index)
            for record in records
            for clip_index in range(max(1, clips_per_file))
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record, clip_index = self.examples[index]
        audio = self._load_audio(record.path)
        clip = self._extract_clip(audio, clip_index)
        if self.train:
            clip = self._augment_waveform(clip)
        spec = self._to_log_mel(clip)
        if self.train:
            spec = self._spec_augment(spec)
        spec = self._normalize(spec)
        spec = spec.repeat(3, 1, 1)
        label = self.label_to_index[record.label]
        return spec, label, record.path

    def _load_audio(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if audio.size == 0:
            return np.zeros(self.clip_samples, dtype=np.float32)
        return audio.astype(np.float32)

    def _extract_clip(self, audio: np.ndarray, clip_index: int) -> np.ndarray:
        if len(audio) < self.clip_samples:
            pad_width = self.clip_samples - len(audio)
            audio = np.pad(audio, (0, pad_width))

        if len(audio) == self.clip_samples:
            return audio

        max_start = len(audio) - self.clip_samples
        if self.train:
            start = random.randint(0, max_start)
        else:
            if self.clips_per_file <= 1:
                start = max_start // 2
            else:
                position = clip_index / max(self.clips_per_file - 1, 1)
                start = int(round(position * max_start))
        return audio[start : start + self.clip_samples]

    def _augment_waveform(self, audio: np.ndarray) -> np.ndarray:
        clip = audio.copy()

        if random.random() < 0.8:
            gain = random.uniform(0.7, 1.3)
            clip *= gain

        if random.random() < 0.5:
            noise_scale = random.uniform(0.0005, 0.003)
            clip += np.random.normal(0.0, noise_scale, size=clip.shape).astype(np.float32)

        if random.random() < 0.5:
            shift = random.randint(-self.sample_rate // 4, self.sample_rate // 4)
            clip = np.roll(clip, shift)

        return np.clip(clip, -1.0, 1.0)

    def _to_log_mel(self, audio: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return tensor

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
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

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        mean = spec.mean()
        std = spec.std().clamp_min(1e-6)
        return (spec - mean) / std


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
