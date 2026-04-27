from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
