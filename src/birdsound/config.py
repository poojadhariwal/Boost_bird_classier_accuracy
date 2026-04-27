from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    run_name: str = "baseline_run"
    data_dir: str = "iBC53"
    output_dir: str = "outputs/runs"
    seed: int = 42
    sample_rate: int = 32000
    clip_seconds: float = 5.0
    train_clips_per_file: int = 3
    eval_clips_per_file: int = 5
    n_mels: int = 128
    fmin: int = 50
    fmax: int = 14000
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    min_train_count_for_val: int = 2
    num_workers: int = 0
    device: str = "auto"
    freeze_backbone: bool = True
    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 2.0

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
