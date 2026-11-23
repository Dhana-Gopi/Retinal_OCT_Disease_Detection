"""Model definitions: custom CNN + ImageNet-pretrained transfer-learning models."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv

from src import config

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}


class CustomCNN(nn.Module):
    """A from-scratch CNN: 4 conv blocks -> global average pool -> 2 FC layers.

    Each block = Conv3x3 -> [BatchNorm] -> act -> Conv3x3 -> [BatchNorm] -> act
    -> MaxPool2x2, so a 224x224 input becomes a 256x14x14 feature map.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = 0.4,
                 batch_norm: bool = True, activation: str = "relu",
                 channels: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, in_ch = [], 3
        for out_ch in channels:
            for conv_in in (in_ch, out_ch):
                layers.append(nn.Conv2d(conv_in, out_ch, 3, padding=1, bias=not batch_norm))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(act(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, 128), act(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def _new_head(in_features: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, config.NUM_CLASSES))


def build_model(name: str, mode: str = "finetune", dropout: float = 0.3,
                pretrained: bool = True, **cnn_kwargs) -> nn.Module:
    """Create a model whose final layer outputs the 4 OCT classes.

    mode = "finetune"        -> all layers trainable
    mode = "feature_extract" -> pretrained backbone frozen, only new head trains
    """
    if name == "custom_cnn":
        return CustomCNN(dropout=cnn_kwargs.pop("dropout", dropout), **cnn_kwargs)

    weights = "DEFAULT" if pretrained else None
    if name == "resnet18":
        model = tv.resnet18(weights=weights)
        model.fc = _new_head(model.fc.in_features, dropout)
        head = model.fc
    elif name == "vgg16":
        model = tv.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, config.NUM_CLASSES)
        head = model.classifier
    elif name == "mobilenet_v2":
        model = tv.mobilenet_v2(weights=weights)
        model.classifier = _new_head(model.classifier[1].in_features, dropout)
        head = model.classifier
    elif name == "densenet121":
        model = tv.densenet121(weights=weights)
        model.classifier = _new_head(model.classifier.in_features, dropout)
        head = model.classifier
    else:
        raise ValueError(f"Unknown model: {name}")

    if mode == "feature_extract":
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True
    return model


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block of each architecture (used by Grad-CAM)."""
    return {
        "custom_cnn": lambda m: m.features[-2],   # last activation before final pool
        "resnet18": lambda m: m.layer4,
        "vgg16": lambda m: m.features[29],        # ReLU after last conv
        "mobilenet_v2": lambda m: m.features[-1],
        "densenet121": lambda m: m.features.norm5,
    }[name](model)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path, model: nn.Module, model_name: str, mode: str,
                    extra: dict | None = None) -> None:
    """Save weights plus everything needed to rebuild + preprocess identically."""
    payload = {
        "model_name": model_name, "mode": mode,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": config.CLASS_NAMES, "image_size": config.IMAGE_SIZE,
        "mean": config.IMAGENET_MEAN, "std": config.IMAGENET_STD,
        "resize_mode": config.RESIZE_MODE,
        "preprocessing": "grayscale -> resize to 224x224 (no padding) -> [0,1] "
                         "-> 3 channels -> ImageNet normalise",
    }
    payload.update(extra or {})
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the architecture from the checkpoint and load its weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_kwargs = ckpt.get("model_kwargs", {})
    model = build_model(ckpt["model_name"], mode=ckpt.get("mode", "finetune"),
                        pretrained=False, **model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
