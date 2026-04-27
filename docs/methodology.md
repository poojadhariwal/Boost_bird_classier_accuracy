# Methodology Notes

## Proposed title

Transfer Learning on Log-Mel Spectrograms for Bird Species Classification from Imbalanced Audio Recordings

## Abstract-style summary

This project performs automated bird species recognition from audio recordings using a deep-learning pipeline based on log-mel spectrogram representations and transfer learning. Raw `.wav` recordings are segmented into fixed-length clips, transformed into spectrogram images, and classified using a pretrained convolutional network. To address severe class imbalance and limited data, the system combines weighted sampling, focal loss, waveform augmentation, and spectrogram masking. Performance is measured using accuracy, balanced accuracy, and macro-F1 score to better reflect multi-class robustness under skewed class frequencies.

## Dataset description points

- Total classes: 53
- Data format: mono or stereo `.wav` audio recordings
- Class organization: one folder per species
- Dataset issue: some species contain very few recordings, which increases overfitting risk
- Evaluation issue: singleton classes cannot be split into both train and validation without leakage

## Preprocessing

- Resample audio to 32 kHz
- Convert to mono
- Segment each recording into 5-second clips
- Pad short clips and randomly crop long clips during training
- Convert clips to log-mel spectrograms with 128 mel bands

## Augmentation

- Random gain scaling
- Additive Gaussian noise
- Random temporal shift
- Frequency masking
- Time masking

## Model

- Backbone: pretrained EfficientNet-B0
- Input: 3-channel spectrogram image created by repeating a normalized single-channel log-mel spectrogram
- Output: fully connected layer with 53 logits

## Optimization

- Optimizer: AdamW
- Scheduler: cosine annealing
- Loss: focal loss with optional class weighting
- Regularization: transfer learning, augmentation, and early best-checkpoint selection

## Metrics to report

- Top-1 accuracy
- Balanced accuracy
- Macro-F1 score
- Per-class precision and recall
- Confusion matrix

## Honest limitations

- Extreme class imbalance can inflate plain accuracy while hiding rare-class failures
- Very small classes need more real recordings for reliable generalization
- Reported validation results should be accompanied by the exact split policy

## Strong experimental extension ideas

1. Group k-fold cross-validation
2. Ensembling across folds
3. Noise reduction and silence trimming
4. Pseudo-labeling on unlabeled field recordings
5. Attention pooling or transformer-based audio encoders
