# BirdCLEF-Style Bird Audio Classification Research Project

This project turns the `iBC53` dataset into a reproducible bird-audio classification pipeline in Python. It is designed to push accuracy higher than a basic CNN baseline by combining transfer learning, mel-spectrogram modeling, class balancing, waveform augmentation, SpecAugment, focal loss, and test-time clip averaging.

## What this project is for

- Multi-class bird species classification from `.wav` recordings
- Research-paper style experimentation with reproducible train/validation splits
- Honest evaluation on imbalanced data
- A strong baseline that can be extended toward a 95-98% target if the data quality supports it

## Important note about the 95-98% target

The current dataset is highly imbalanced:

- 53 classes
- 1,368 recordings
- Some classes have only 1-3 recordings
- One class, `Mystery mystery`, has far more samples than the others

Because of that, **95-98% accuracy cannot be guaranteed from the raw dataset alone**. This project is built to maximize the chance of reaching that range while keeping the evaluation credible for a research paper.

## Project structure

```text
newdataset/
├── iBC53/
├── requirements.txt
├── README.md
├── configs/
│   └── baseline.json
├── docs/
│   └── methodology.md
├── scripts/
│   ├── build_metadata.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_audio.py
└── src/
    └── birdsound/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── losses.py
        ├── model.py
        ├── train.py
        ├── evaluate.py
        └── infer.py
```

## Core ideas used to improve accuracy

- Pretrained `EfficientNet-B0` on spectrogram images
- Log-mel spectrogram features
- Random audio cropping to fixed-duration clips
- Waveform augmentation: gain, noise, time shift
- SpecAugment masking on spectrograms
- Weighted sampling for class imbalance
- Focal loss with class weights
- Validation/test clip averaging from the same recording
- Balanced accuracy and macro-F1 reporting to avoid misleading conclusions

## Setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Default workflow

### 1. Build metadata

```powershell
python scripts/build_metadata.py --data-dir iBC53 --output outputs\metadata.csv
```

### 2. Train the model

```powershell
python scripts/train_model.py --config configs\baseline.json
```

For CPU-only machines, start with the lighter quick config:

```powershell
python scripts/train_model.py --config configs\cpu_quick.json
```

### 3. Evaluate the saved model

```powershell
python scripts/evaluate_model.py --run-dir outputs\runs\baseline_run
```

### 4. Predict a single audio file

```powershell
python scripts/predict_audio.py --run-dir outputs\runs\baseline_run --audio "iBC53\Acridotheres fuscus\1.wav"
```

## Outputs

Each training run creates a folder such as:

```text
outputs/runs/baseline_run/
```

Inside it, the code saves:

- `best_model.pt`
- `last_model.pt`
- `label_to_index.json`
- `config.json`
- `history.csv`
- `metrics.json`
- `confusion_matrix.csv`
- `validation_predictions.csv`

## Research paper guidance

Useful sections for your paper:

- Problem statement
- Dataset description and imbalance analysis
- Feature extraction using log-mel spectrograms
- Transfer-learning model architecture
- Data augmentation and regularization strategy
- Experimental setup and hyperparameters
- Accuracy, balanced accuracy, macro-F1, confusion matrix
- Error analysis on rare species

More wording support is included in [docs/methodology.md](docs/methodology.md).

## Recommended next upgrades if you want to push accuracy even higher

1. Remove or relabel the ambiguous `Mystery mystery` class if it is not a true species.
2. Add more recordings for classes with fewer than 10 files.
3. Use 5-fold group cross-validation by recording.
4. Try larger backbones such as `EfficientNet-B2` or `ConvNeXt-Tiny`.
5. Train an ensemble of 3-5 folds and average predictions.
6. Apply background-noise reduction and silence trimming before training.

## Suggested claim for the paper

Instead of promising 95-98% immediately, a safer research claim is:

> "We propose a transfer-learning based bird-call classification pipeline that significantly improves performance over a conventional baseline on the iBC53 dataset."

That wording is strong, honest, and publishable.
