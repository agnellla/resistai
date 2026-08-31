"""
src/model.py
------------
The classifier: pretrained EfficientNet-B0 with a fresh 2-class head.

    class 0 = real (authentic photo)
    class 1 = AI-generated

We use `timm` because it gives us EfficientNet-B0 with ImageNet weights in one
line and lets us swap the final layer with `num_classes=2`.
EfficientNet-B0 has ~5.3M parameters, well under the 2B competition limit.
"""

import timm
import torch.nn as nn


def build_model(backbone="efficientnet_b0", pretrained=True, num_classes=2, dropout=0.2):
    """
    Create the model.

    Args:
        backbone:    timm model name.
        pretrained:  if True, load ImageNet weights (recommended - we have little data).
        num_classes: 2 for this binary task.
        dropout:     dropout before the final linear layer, light regularisation.

    Returns:
        A torch.nn.Module. Output shape = (batch, 2) raw logits.
        Use CrossEntropyLoss during training; softmax at inference for probabilities.
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def count_parameters(model):
    """Return the number of trainable parameters - handy for the report / constraint check."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
