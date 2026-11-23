"""Grad-CAM for every architecture in the project (custom CNN + 4 torchvision nets).

Public API (also used by the Streamlit app):

    heatmap, probs = compute_gradcam(model, model_name, input_tensor, target_class=None)
    rgb = overlay_heatmap(gray_uint8, heatmap)

How it works
------------
Instead of registering hooks (torchvision uses in-place ReLUs, and DenseNet
applies an in-place ReLU right after ``features.norm5``, both of which break
tensor hooks), each model is split explicitly into

    feature_fn(x) -> A        last convolutional feature map  [1, K, h, w]
    head_fn(A)    -> logits   everything after it              [1, 4]

``A`` is computed without gradients, detached, and marked ``requires_grad``;
the head is then run on it and the gradient of the target logit with respect
to ``A`` is taken with ``torch.autograd.grad``.  No parameter gradients are
created, nothing is registered on the model, and the model's train/eval mode
is restored afterwards, so the caller's model is left untouched.

Grad-CAM (Selvaraju et al., 2017):
    alpha_k = mean_{i,j} d y_c / d A_k[i,j]
    L_c     = ReLU( sum_k alpha_k * A_k )   -> upsampled to 224x224, scaled to [0, 1]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SUPPORTED_MODELS = ("custom_cnn", "resnet18", "vgg16", "mobilenet_v2", "densenet121")


def _split_model(model: nn.Module, model_name: str
                 ) -> tuple[Callable[[torch.Tensor], torch.Tensor],
                            Callable[[torch.Tensor], torch.Tensor]]:
    """Return (feature_fn, head_fn) so that head_fn(feature_fn(x)) == model(x)."""
    if model_name == "custom_cnn":
        # features = [conv blocks ..., ReLU, MaxPool]; stop before the last MaxPool
        # so the map is 28x28 (finer than the 14x14 after pooling).
        def feature_fn(x):
            return model.features[:-1](x)

        def head_fn(a):
            return model.classifier(model.pool(model.features[-1](a)))
    elif model_name == "resnet18":
        def feature_fn(x):
            m = model
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))

        def head_fn(a):
            return model.fc(torch.flatten(model.avgpool(a), 1))
    elif model_name == "vgg16":
        # features[29] is the ReLU after the last conv (14x14); [30] is the last MaxPool.
        def feature_fn(x):
            return model.features[:30](x)

        def head_fn(a):
            a = model.avgpool(model.features[30:](a))
            return model.classifier(torch.flatten(a, 1))
    elif model_name == "mobilenet_v2":
        def feature_fn(x):
            return model.features(x)   # ends with Conv-BN-ReLU6 -> 1280x7x7

        def head_fn(a):
            a = F.adaptive_avg_pool2d(a, (1, 1))
            return model.classifier(torch.flatten(a, 1))
    elif model_name == "densenet121":
        def feature_fn(x):
            # torchvision applies ReLU after norm5 in forward(); do it out-of-place here.
            return F.relu(model.features(x), inplace=False)

        def head_fn(a):
            a = F.adaptive_avg_pool2d(a, (1, 1))
            return model.classifier(torch.flatten(a, 1))
    else:
        raise ValueError(f"Grad-CAM not implemented for model '{model_name}'. "
                         f"Supported: {SUPPORTED_MODELS}")
    return feature_fn, head_fn


def compute_gradcam(model: nn.Module, model_name: str, input_tensor: torch.Tensor,
                    target_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM heatmap for one image.

    Args:
        model: a loaded model (e.g. from ``src.models.load_checkpoint``), any device.
        model_name: one of ``SUPPORTED_MODELS`` (``ckpt["model_name"]``).
        input_tensor: normalised image tensor ``[1, 3, 224, 224]`` (``to_model_tensor``).
        target_class: class index to explain; ``None`` = the predicted class.

    Returns:
        heatmap: float32 ``H x W`` (same H, W as the input) scaled to [0, 1].
        probs:   float32 ``(num_classes,)`` softmax probabilities of the model.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
        raise ValueError(f"expected input of shape [1, 3, H, W], got {tuple(input_tensor.shape)}")
    device = next(model.parameters()).device
    x = input_tensor.to(device=device, dtype=torch.float32)
    height, width = x.shape[-2:]
    feature_fn, head_fn = _split_model(model, model_name)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            feats = feature_fn(x)
        feats = feats.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = head_fn(feats)
            if target_class is None:
                target_class = int(logits.argmax(1).item())
            if not 0 <= int(target_class) < logits.shape[1]:
                raise ValueError(f"target_class {target_class} out of range")
            (grads,) = torch.autograd.grad(logits[0, int(target_class)], feats)
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), 1)[0].float().cpu().numpy()
            weights = grads.mean(dim=(2, 3), keepdim=True)          # alpha_k
            cam = F.relu((weights * feats.detach()).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(height, width), mode="bilinear",
                                align_corners=False)[0, 0]
            cam = cam - cam.min()
            peak = cam.max()
            cam = cam / peak if peak > 1e-12 else torch.zeros_like(cam)
            heatmap = cam.float().cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)
    return heatmap, probs.astype(np.float32)


def _colormap(name: str = "inferno") -> np.ndarray:
    """256x3 uint8 lookup table from matplotlib (imported lazily)."""
    from matplotlib import colormaps

    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def overlay_heatmap(gray_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55,
                    cmap: str = "inferno") -> np.ndarray:
    """Blend a [0, 1] heatmap onto a grayscale OCT scan.

    The colour is weighted by the heatmap itself, so low-attention regions keep
    the original scan visible and only the regions that drove the decision are
    tinted.  Returns an RGB uint8 image of the same H x W as ``gray_uint8``.
    """
    gray = np.asarray(gray_uint8)
    if gray.ndim == 3:   # accept RGB input too
        gray = gray[..., :3].mean(-1)
    gray = gray.astype(np.float32)
    if gray.max() <= 1.0:
        gray = gray * 255.0
    heat = np.clip(np.nan_to_num(np.asarray(heatmap, dtype=np.float32)), 0, 1)
    if heat.shape != gray.shape:
        heat_t = torch.from_numpy(heat)[None, None]
        heat = F.interpolate(heat_t, size=gray.shape, mode="bilinear",
                             align_corners=False)[0, 0].numpy().clip(0, 1)
    color = _colormap(cmap)[(heat * 255).astype(np.uint8)].astype(np.float32)
    weight = (alpha * heat)[..., None]
    rgb = gray[..., None] * (1 - weight) + color * weight
    return np.clip(rgb, 0, 255).astype(np.uint8)
