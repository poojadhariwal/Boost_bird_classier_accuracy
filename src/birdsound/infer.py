from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .data import BirdSoundDataset, AudioRecord, load_label_mapping
from .model import build_model
from .train import resolve_device


def predict_file(run_dir: str | Path, audio_path: str | Path) -> list[dict[str, float | str]]:
    run_dir = Path(run_dir)
    audio_path = Path(audio_path)
    config = TrainConfig.from_json(run_dir / "config.json")
    label_to_index = load_label_mapping(run_dir / "label_to_index.json")
    index_to_label = {index: label for label, index in label_to_index.items()}

    dataset = BirdSoundDataset(
        records=[AudioRecord(path=str(audio_path), label=next(iter(label_to_index.keys())))],
        label_to_index=label_to_index,
        sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        clips_per_file=config.eval_clips_per_file,
        train=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.eval_clips_per_file, shuffle=False)

    device = resolve_device(config.device)
    model = build_model(num_classes=len(label_to_index), freeze_backbone=config.freeze_backbone).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        for specs, _, _ in loader:
            specs = specs.to(device)
            logits = model(specs)
            mean_prob = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            break

    ranked = np.argsort(mean_prob)[::-1][:5]
    return [
        {
            "label": index_to_label[int(index)],
            "probability": float(mean_prob[index]),
        }
        for index in ranked
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the bird label for one audio file.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--audio", required=True, help="Path to a wav file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = predict_file(args.run_dir, args.audio)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
