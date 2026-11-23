"""Neural-network fundamentals implemented from scratch with NumPy only.

This module covers the "Neural Network Fundamentals" part of the project:

1. A tiny worked example (2 inputs -> 2 sigmoid hidden units -> 1 sigmoid
   output) where every step of forward propagation, loss calculation,
   backpropagation (chain rule) and the gradient-descent update is printed.
   The analytic gradients are verified with numerical finite differences.
2. A configurable multi-layer perceptron (MLP) class with ReLU / Sigmoid /
   Tanh hidden activations, a softmax + cross-entropy output, optional L2
   weight decay and a gradient-check function.
3. Three hand-written optimizers: SGD, SGD with momentum, and Adam.
4. Experiments on real retinal OCT scans (downsampled to 32x32 pixels):
   (a) activation comparison and (b) optimizer comparison.

No deep-learning framework is used for the network itself -- every
gradient is derived by hand.  PyTorch is only imported indirectly by
``src.preprocessing`` (for its own helpers); nothing here runs on the GPU.

Run from the project root:
    python -m src.numpy_nn
"""
from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw figures to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import load_cache, patient_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
MET_DIR = ROOT / "outputs" / "metrics"
CLASS_NAMES = list(config.CLASS_NAMES)

# Colours for the 3 series in each comparison chart (fixed order).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_COLOR = "#52514e"


# ==========================================================================
# 1. Activation functions and their derivatives
# ==========================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid: squashes any number into (0, 1).  s(z) = 1 / (1 + e^-z)."""
    # np.clip avoids overflow warnings in exp() for very negative z.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: s(z) * (1 - s(z)).  Its maximum is only 0.25."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: keeps positive numbers, sets negative numbers to 0."""
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z > 0, otherwise 0."""
    return (z > 0).astype(z.dtype)


def tanh(z: np.ndarray) -> np.ndarray:
    """Tanh: squashes any number into (-1, 1); zero-centred, unlike sigmoid."""
    return np.tanh(z)


def tanh_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(z)^2.  Its maximum is 1."""
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn raw class scores (logits) into probabilities that sum to 1."""
    z = z - z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ==========================================================================
# 2. Worked example: 2 -> 2 (sigmoid) -> 1 (sigmoid), one training example
# ==========================================================================
def _tiny_forward(params: dict, x: np.ndarray, y: float) -> dict:
    """Forward pass of the tiny network; returns every intermediate value."""
    z1 = params["W1"] @ x + params["b1"]      # hidden weighted sums
    a1 = sigmoid(z1)                          # hidden activations
    z2 = params["W2"] @ a1 + params["b2"]     # output weighted sum
    a2 = sigmoid(z2)                          # prediction (probability)
    # Binary cross-entropy loss for one example.
    loss = float(-(y * np.log(a2) + (1 - y) * np.log(1 - a2))[0])
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "loss": loss}


def worked_example(lr: float = 0.5) -> str:
    """Run and print the step-by-step worked example; return the text."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        x = np.array([1.0, 0.5])
        y = 1.0
        params = {
            "W1": np.array([[0.10, 0.20],     # weights into hidden unit h1
                            [0.30, -0.10]]),  # weights into hidden unit h2
            "b1": np.array([0.0, 0.0]),
            "W2": np.array([[0.40, -0.20]]),  # weights from h1, h2 to output
            "b2": np.array([0.0]),
        }
        f = "{:.3f}".format

        print("=" * 68)
        print("WORKED EXAMPLE: 2 inputs -> 2 hidden (sigmoid) -> 1 output (sigmoid)")
        print("Loss: binary cross-entropy  L = -[y*log(p) + (1-y)*log(1-p)]")
        print("=" * 68)
        print(f"Input x = [x1, x2] = [{f(x[0])}, {f(x[1])}],   target y = {y:.0f}")
        print(f"W1 = [[w11, w12], [w21, w22]] = {params['W1'].tolist()}   b1 = {params['b1'].tolist()}")
        print(f"W2 = [v1, v2] = {params['W2'][0].tolist()}   b2 = {params['b2'].tolist()}")
        print(f"Learning rate = {lr}")

        # ---------------- forward propagation ----------------
        c = _tiny_forward(params, x, y)
        W1, W2 = params["W1"], params["W2"]
        print("\nSTEP 1 - FORWARD PROPAGATION")
        for i in range(2):
            print(f"  z_h{i+1} = {f(W1[i,0])}*{f(x[0])} + {f(W1[i,1])}*{f(x[1])} + {f(params['b1'][i])}"
                  f" = {f(c['z1'][i])}  ->  a_h{i+1} = sigmoid({f(c['z1'][i])}) = {f(c['a1'][i])}")
        print(f"  z_out = {f(W2[0,0])}*{f(c['a1'][0])} + {f(W2[0,1])}*{f(c['a1'][1])} + {f(params['b2'][0])}"
              f" = {f(c['z2'][0])}  ->  p = sigmoid({f(c['z2'][0])}) = {f(c['a2'][0])}")

        print("\nSTEP 2 - LOSS")
        print(f"  L = -[1*log({f(c['a2'][0])})] = {f(c['loss'])}")

        # ---------------- backpropagation ----------------
        print("\nSTEP 3 - BACKPROPAGATION (chain rule, from output back to input)")
        p = c["a2"][0]
        dz2 = p - y  # dL/dp * dp/dz_out simplifies to (p - y) for sigmoid + BCE
        print(f"  dL/dp      = -(y/p) + (1-y)/(1-p) = {f(-(y / p) + (1 - y) / (1 - p))}")
        print(f"  dp/dz_out  = p*(1-p) = {f(p * (1 - p))}")
        print(f"  dL/dz_out  = dL/dp * dp/dz_out = p - y = {f(dz2)}")
        dW2 = dz2 * c["a1"]
        db2 = dz2
        for j in range(2):
            print(f"  dL/dv{j+1}     = dL/dz_out * a_h{j+1} = {f(dz2)} * {f(c['a1'][j])} = {f(dW2[j])}")
        print(f"  dL/db2     = dL/dz_out = {f(db2)}")
        da1 = dz2 * W2[0]
        dz1 = da1 * c["a1"] * (1 - c["a1"])
        for j in range(2):
            print(f"  dL/da_h{j+1}  = dL/dz_out * v{j+1} = {f(dz2)} * {f(W2[0,j])} = {f(da1[j])}")
            print(f"  dL/dz_h{j+1}  = dL/da_h{j+1} * a_h{j+1}*(1-a_h{j+1}) = {f(da1[j])} * "
                  f"{f(c['a1'][j]*(1-c['a1'][j]))} = {dz1[j]:.4f}")
        dW1 = np.outer(dz1, x)
        db1 = dz1
        for i in range(2):
            for k in range(2):
                print(f"  dL/dw{i+1}{k+1}    = dL/dz_h{i+1} * x{k+1} = {dz1[i]:.4f} * {f(x[k])} = {dW1[i,k]:.4f}")
        print(f"  dL/db1     = dL/dz_h = [{db1[0]:.4f}, {db1[1]:.4f}]")
        analytic = {"W1": dW1, "b1": db1, "W2": dW2[None, :], "b2": np.array([db2])}

        # ---------------- gradient check ----------------
        print("\nSTEP 4 - GRADIENT CHECK (numerical finite differences)")
        print("  numerical dL/dw = [L(w + eps) - L(w - eps)] / (2*eps),  eps = 1e-5")
        eps = 1e-5
        max_rel = 0.0
        for name in ["W1", "b1", "W2", "b2"]:
            for idx in np.ndindex(params[name].shape):
                old = params[name][idx]
                params[name][idx] = old + eps
                lp = _tiny_forward(params, x, y)["loss"]
                params[name][idx] = old - eps
                lm = _tiny_forward(params, x, y)["loss"]
                params[name][idx] = old
                num = (lp - lm) / (2 * eps)
                ana = analytic[name][idx]
                rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
                max_rel = max(max_rel, rel)
                print(f"  {name}{list(idx)}: analytic = {ana:+.6f}   numerical = {num:+.6f}")
        verdict = "MATCH" if max_rel < 1e-6 else "MISMATCH"
        print(f"  Largest relative difference = {max_rel:.2e}  ->  {verdict}")

        # ---------------- weight update ----------------
        print(f"\nSTEP 5 - WEIGHT UPDATE (gradient descent: w_new = w - lr * dL/dw, lr = {lr})")
        new = {k: params[k] - lr * analytic[k] for k in params}
        for i in range(2):
            for k in range(2):
                print(f"  w{i+1}{k+1}: {f(W1[i,k])} - {lr}*({dW1[i,k]:.4f}) = {new['W1'][i,k]:.4f}")
        for j in range(2):
            print(f"  v{j+1}:  {f(W2[0,j])} - {lr}*({f(dW2[j])}) = {new['W2'][0,j]:.4f}")
        print(f"  b2:  {f(params['b2'][0])} - {lr}*({f(db2)}) = {new['b2'][0]:.4f}")
        after = _tiny_forward(new, x, y)
        print("\nSTEP 6 - CHECK: forward pass again with the new weights")
        print(f"  prediction {f(c['a2'][0])} -> {f(after['a2'][0])}   (target 1)")
        print(f"  loss       {f(c['loss'])} -> {f(after['loss'])}   (lower = better)")
    text = buf.getvalue()
    print(text)
    return text


# ==========================================================================
# 3. Optimizers (each updates the parameter arrays in place)
# ==========================================================================
class SGD:
    """Plain stochastic gradient descent: w = w - lr * grad."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply one update to every parameter."""
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    """SGD with momentum: keeps a running 'velocity' so updates build up speed
    in consistent directions and damp oscillations.

    v = beta * v + grad ;  w = w - lr * v
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr, self.beta = lr, beta
        self.velocity: list[np.ndarray] | None = None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply one update to every parameter."""
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    """Adam: momentum (1st moment m) + per-weight adaptive step size
    (2nd moment v), with bias correction for the first few steps.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Apply one update to every parameter."""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)  # bias correction
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==========================================================================
# 4. Multi-layer perceptron
# ==========================================================================
class MLP:
    """Fully-connected network: [Linear -> activation] * k -> Linear -> softmax.

    Args:
        layer_sizes: e.g. [1024, 128, 4] = 1024 inputs, one hidden layer of
            128 units, 4 output classes.
        activation: "relu", "sigmoid" or "tanh" for the hidden layers.
        l2: L2 weight-decay strength (0 = off).  Adds l2/2 * sum(W^2) to loss.
        seed: random seed for reproducible weight initialisation.
    """

    def __init__(self, layer_sizes: list[int], activation: str = "relu",
                 l2: float = 0.0, seed: int = 0):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        self.sizes = layer_sizes
        self.activation = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.l2 = l2
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He init suits ReLU; Xavier/Glorot init suits sigmoid/tanh.
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    @property
    def params(self) -> list[np.ndarray]:
        """All trainable arrays in a fixed order (W0, b0, W1, b1, ...)."""
        return [a for pair in zip(self.W, self.b) for a in pair]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """Forward propagation.  Returns class probabilities and a cache of
        the inputs/pre-activations of every layer (needed for backprop)."""
        cache = []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b                       # weighted sum
            cache.append((a, z))
            last = i == len(self.W) - 1
            a = softmax(z) if last else self.act(z)
        return a, cache

    def loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy (+ optional L2 penalty)."""
        n = len(y)
        ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        reg = 0.5 * self.l2 * sum(np.sum(W * W) for W in self.W)
        return float(ce + reg)

    def backward(self, probs: np.ndarray, y: np.ndarray, cache: list) -> list[np.ndarray]:
        """Backpropagation: gradients of the loss w.r.t. every parameter,
        returned in the same order as ``self.params``."""
        n = len(y)
        # Softmax + cross-entropy combine into a very simple gradient:
        # dL/dz = (probs - one_hot(y)) / n
        dz = probs.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_prev, _ = cache[i]
            grads_W[i] = a_prev.T @ dz + self.l2 * self.W[i]
            grads_b[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T                    # send error backwards
                dz = da * self.act_grad(cache[i - 1][1])  # through the activation
        return [g for pair in zip(grads_W, grads_b) for g in pair]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely class for each row of X."""
        return self.forward(X)[0].argmax(axis=1)


def gradient_check(model: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5,
                   n_checks: int = 20, seed: int = 0) -> float:
    """Compare backprop gradients with numerical finite differences on
    randomly chosen weights.  Returns the largest relative error
    (< 1e-6 means backprop is correct)."""
    rng = np.random.default_rng(seed)
    probs, cache = model.forward(X)
    analytic = model.backward(probs, y, cache)
    worst = 0.0
    for _ in range(n_checks):
        k = rng.integers(len(model.params))
        p = model.params[k]
        idx = tuple(rng.integers(s) for s in p.shape)
        old = p[idx]
        p[idx] = old + eps
        lp = model.loss(model.forward(X)[0], y)
        p[idx] = old - eps
        lm = model.loss(model.forward(X)[0], y)
        p[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = analytic[k][idx]
        worst = max(worst, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return worst


def train(model: MLP, optimizer, data: dict, epochs: int, batch_size: int = 64,
          seed: int = 0, label: str = "") -> dict:
    """Mini-batch training loop.  Records loss and accuracy every epoch."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            probs, cache = model.forward(Xtr[idx])
            optimizer.step(model.params, model.backward(probs, ytr[idx], cache))
        for split, X, y in [("train", Xtr, ytr), ("val", Xva, yva)]:
            probs, _ = model.forward(X)
            hist[f"{split}_loss"].append(model.loss(probs, y))
            hist[f"{split}_acc"].append(float((probs.argmax(1) == y).mean()))
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  [{label}] epoch {ep:2d}  train loss {hist['train_loss'][-1]:.3f} "
                  f"acc {hist['train_acc'][-1]:.3f} | val loss {hist['val_loss'][-1]:.3f} "
                  f"acc {hist['val_acc'][-1]:.3f}")
    hist["seconds"] = round(time.time() - t0, 1)
    return hist


# ==========================================================================
# 5. Real OCT data: balanced subset, 32x32, flattened, standardised
# ==========================================================================
def load_oct_subset(n_train: int = 1000, n_val: int = 250, size: int = 32,
                    seed: int = config.SEED) -> dict:
    """Sample a class-balanced subset from the patient-level split,
    downsample 224x224 -> size x size by block averaging, flatten and
    standardise each pixel with the training mean/std."""
    images, meta = load_cache("train")
    train_idx, val_idx = patient_split(meta)
    labels = meta["label"].to_numpy()
    rng = np.random.default_rng(seed)

    def pick(pool: np.ndarray, per_class: int) -> np.ndarray:
        chosen = [rng.choice(pool[labels[pool] == c], per_class, replace=False)
                  for c in range(len(CLASS_NAMES))]
        return np.sort(np.concatenate(chosen))  # sorted = faster memmap reads

    def to_features(idx: np.ndarray) -> np.ndarray:
        f = 224 // size  # 7 -> each output pixel = mean of a 7x7 block
        x = np.asarray(images[idx], dtype=np.float32) / 255.0
        x = x.reshape(len(idx), size, f, size, f).mean(axis=(2, 4))
        return x.reshape(len(idx), -1)

    tr, va = pick(train_idx, n_train), pick(val_idx, n_val)
    X_train, X_val = to_features(tr), to_features(va)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    data = {
        "X_train": ((X_train - mean) / std).astype(np.float64),
        "y_train": labels[tr], "X_val": ((X_val - mean) / std).astype(np.float64),
        "y_val": labels[va],
        "patients_overlap": len(set(meta.patient_id.iloc[tr]) & set(meta.patient_id.iloc[va])),
    }
    return data


# ==========================================================================
# 6. Figures
# ==========================================================================
def plot_activation_functions(path: Path) -> None:
    """Plot ReLU, sigmoid and tanh and their derivatives side by side."""
    z = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (name, (fn, grad)), color in zip(ACTIVATIONS.items(), SERIES_COLORS):
        axes[0].plot(z, fn(z), color=color, lw=2, label=name)
        axes[1].plot(z, grad(z), color=color, lw=2, label=name)
    axes[0].set_title("Activation functions f(z)")
    axes[1].set_title("Derivatives f'(z)  (what backprop multiplies by)")
    axes[1].annotate("sigmoid max slope = 0.25\n-> gradients shrink layer by layer",
                     xy=(0, 0.25), xytext=(1.3, 0.55), color=TEXT_COLOR, fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": TEXT_COLOR})
    for ax in axes:
        _style(ax)
        ax.set_xlabel("z (weighted sum)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style(ax) -> None:
    """Light, recessive grid and axes."""
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)


def plot_comparison(runs: dict, title: str, path: Path) -> None:
    """Loss and accuracy curves (train dashed, val solid) for 2-3 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, h), color in zip(runs.items(), SERIES_COLORS):
        ep = np.arange(1, len(h["val_loss"]) + 1)
        for ax, key in zip(axes, ["loss", "acc"]):
            ax.plot(ep, h[f"train_{key}"], color=color, lw=1.5, ls="--", alpha=0.8)
            ax.plot(ep, h[f"val_{key}"], color=color, lw=2,
                    label=f"{name} (final val acc {h['val_acc'][-1]:.3f})")
    axes[0].set_title("Cross-entropy loss (solid = val, dashed = train)")
    axes[1].set_title("Accuracy (solid = val, dashed = train)")
    axes[1].axhline(0.25, color=TEXT_COLOR, lw=0.8, ls=":")
    axes[1].text(1, 0.26, "chance (4 classes)", fontsize=8, color=TEXT_COLOR)
    for ax in axes:
        _style(ax)
        ax.set_xlabel("epoch")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==========================================================================
# 7. Main
# ==========================================================================
def main() -> None:
    """Run the worked example, gradient checks and both OCT experiments."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1) Worked example --------------------------------------------------
    text = worked_example()
    (MET_DIR / "numpy_worked_example.txt").write_text(text)

    # 2) Gradient check of the MLP class for every activation -----------
    print("Gradient check of the MLP class (random 10-8-6-3 net, L2 = 0.01):")
    rng = np.random.default_rng(1)
    Xg, yg = rng.normal(size=(16, 10)), rng.integers(0, 3, size=16)
    grad_checks = {}
    for act in ACTIVATIONS:
        err = gradient_check(MLP([10, 8, 6, 3], act, l2=0.01, seed=2), Xg, yg, n_checks=40)
        grad_checks[act] = err
        print(f"  {act:8s} max relative error = {err:.2e}  "
              f"{'PASS' if err < 1e-6 else 'FAIL'}")

    plot_activation_functions(FIG_DIR / "numpy_activation_functions.png")

    # 3) Real OCT data ------------------------------------------------------
    print("\nLoading balanced OCT subset (1,000/class train, 250/class val) ...")
    t0 = time.time()
    data = load_oct_subset()
    print(f"  X_train {data['X_train'].shape}, X_val {data['X_val'].shape}, "
          f"patients shared train/val = {data['patients_overlap']}, "
          f"load {time.time() - t0:.1f}s")

    epochs, hidden, batch = 30, 128, 64
    layers = [data["X_train"].shape[1], hidden, len(CLASS_NAMES)]
    lrs = {"sgd": 0.01, "momentum": 0.01, "adam": 0.001}
    make_opt = {"sgd": lambda: SGD(lrs["sgd"]),
                "momentum": lambda: Momentum(lrs["momentum"], 0.9),
                "adam": lambda: Adam(lrs["adam"])}

    print("\nExperiment A: activation comparison (Adam)")
    act_runs = {}
    for act in ["relu", "sigmoid", "tanh"]:
        model = MLP(layers, act, l2=1e-4, seed=0)
        act_runs[act] = train(model, make_opt["adam"](), data, epochs, batch, label=act)

    print("\nExperiment B: optimizer comparison (ReLU)")
    opt_runs = {"adam": act_runs["relu"]}  # identical setup -> reuse the run
    for opt in ["sgd", "momentum"]:
        model = MLP(layers, "relu", l2=1e-4, seed=0)
        opt_runs[opt] = train(model, make_opt[opt](), data, epochs, batch, label=opt)
    opt_runs = {k: opt_runs[k] for k in ["sgd", "momentum", "adam"]}

    plot_comparison(act_runs, "NumPy MLP on 32x32 OCT: activation comparison (Adam, lr 1e-3)",
                    FIG_DIR / "numpy_activation_comparison.png")
    plot_comparison({"SGD": opt_runs["sgd"], "SGD+momentum": opt_runs["momentum"],
                     "Adam": opt_runs["adam"]},
                    "NumPy MLP on 32x32 OCT: optimizer comparison (ReLU)",
                    FIG_DIR / "numpy_optimizer_comparison.png")

    def summary(h: dict) -> dict:
        best = int(np.argmax(h["val_acc"]))
        return {"final_train_acc": h["train_acc"][-1], "final_val_acc": h["val_acc"][-1],
                "final_val_loss": h["val_loss"][-1], "best_val_acc": h["val_acc"][best],
                "best_epoch": best + 1, "seconds": h["seconds"]}

    results = {
        "setup": {
            "data": "Kermany OCT train cache, patient-level split (src.preprocessing.patient_split)",
            "train_per_class": 1000, "val_per_class": 250, "class_names": CLASS_NAMES,
            "input": "224x224 grayscale -> 7x7 block mean -> 32x32 -> flatten (1024) -> "
                     "standardise with train mean/std",
            "architecture": layers, "epochs": epochs, "batch_size": batch, "l2": 1e-4,
            "learning_rates": lrs, "momentum_beta": 0.9, "adam_betas": [0.9, 0.999],
            "patients_shared_train_val": data["patients_overlap"],
            "seed": config.SEED, "framework": "NumPy only (CPU)",
        },
        "gradient_check_max_rel_error": grad_checks,
        "activation_comparison": {k: {"summary": summary(h), "curves": h}
                                  for k, h in act_runs.items()},
        "optimizer_comparison": {k: {"summary": summary(h), "curves": h}
                                 for k, h in opt_runs.items()},
        "total_runtime_seconds": round(time.time() - t_start, 1),
        "note": "A pixel MLP on 32x32 images ignores spatial structure; it is a teaching "
                "baseline and is expected to be far weaker than the CNNs.",
    }
    (MET_DIR / "numpy_results.json").write_text(json.dumps(results, indent=2))

    print("\nSUMMARY (final-epoch val accuracy, 1,000 val scans, chance = 0.25)")
    for group in ["activation_comparison", "optimizer_comparison"]:
        for k, v in results[group].items():
            s = v["summary"]
            print(f"  {group.split('_')[0]:10s} {k:9s} val acc {s['final_val_acc']:.3f} "
                  f"(best {s['best_val_acc']:.3f} @ ep {s['best_epoch']}), "
                  f"train acc {s['final_train_acc']:.3f}, {s['seconds']}s")
    print(f"Total runtime {results['total_runtime_seconds']}s")


if __name__ == "__main__":
    main()
