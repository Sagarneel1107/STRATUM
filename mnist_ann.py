"""
mnist_ann.py

A small fully-connected MNIST classifier used as the workload for the
memristive-crossbar decay analysis (see example_usage_nn_decay.py).

This file is deliberately self-contained:

    * It builds and trains a 784 -> HIDDEN -> 10 ReLU MLP.
    * Training uses PyTorch (CPU). Inference only needs NumPy, so the
      trained weights are exported to a plain .npz file and the crossbar
      analysis can run without touching PyTorch at all.
    * The activations fed into each layer are non-negative (pixels in
      [0, 1] and post-ReLU hidden units), which is exactly what a single
      memristor crossbar with positive read voltages can represent.

Run standalone to (re)train and cache the weights:

    python mnist_ann.py

The crossbar example calls `get_or_train()` which loads the cached weights
if present and trains them only the first time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Configuration  (all values come from the central simulation_config.py)

from simulation_config import (
    MNIST_HIDDEN_UNITS as HIDDEN_UNITS,
    MNIST_WEIGHTS_PATH as WEIGHTS_PATH,
    MNIST_EPOCHS as DEFAULT_EPOCHS,
    MNIST_BATCH_SIZE as DEFAULT_BATCH_SIZE,
    MNIST_LEARNING_RATE as DEFAULT_LEARNING_RATE,
    MNIST_SEED as DEFAULT_SEED,
)


# Data loading (no TensorFlow/Keras dependency)

def load_mnist(
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST as flattened float32 arrays.

    Returns (x_train, y_train, x_test, y_test) where each x is shape
    (N, 784) in [0, 1] (if normalize) and each y is shape (N,) int64.

    Resolution order:
        1. Cached Keras-format mnist.npz in ~/.keras/datasets (no TF needed)
        2. torchvision.datasets.MNIST (downloads to ./data if necessary)
        3. sklearn.datasets.fetch_openml('mnist_784')
    """
    arrays = _load_mnist_from_keras_cache()
    if arrays is None:
        arrays = _load_mnist_from_torchvision()
    if arrays is None:
        arrays = _load_mnist_from_openml()
    if arrays is None:
        raise RuntimeError(
            "Could not load MNIST from any source (keras cache, torchvision, openml)."
        )

    x_train, y_train, x_test, y_test = arrays

    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def _load_mnist_from_keras_cache():
    path = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def _load_mnist_from_torchvision():
    try:
        from torchvision import datasets  # type: ignore
    except Exception:
        return None
    try:
        root = os.path.join(os.getcwd(), "data")
        train = datasets.MNIST(root=root, train=True, download=True)
        test = datasets.MNIST(root=root, train=False, download=True)
        x_train = train.data.numpy()
        y_train = train.targets.numpy()
        x_test = test.data.numpy()
        y_test = test.targets.numpy()
        return x_train, y_train, x_test, y_test
    except Exception:
        return None


def _load_mnist_from_openml():
    try:
        from sklearn.datasets import fetch_openml  # type: ignore
    except Exception:
        return None
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        x = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.int64)
        x_train, x_test = x[:60000], x[60000:]
        y_train, y_test = y[:60000], y[60000:]
        return x_train, y_train, x_test, y_test
    except Exception:
        return None


# Weight container + NumPy inference (no PyTorch needed)

@dataclass
class MLPWeights:
    """
    Trained weights stored in crossbar-friendly orientation.

    W1: (784, HIDDEN)   so that  h_pre = x @ W1 + b1
    W2: (HIDDEN, 10)     so that  logits = relu(h_pre) @ W2 + b2
    """

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @property
    def hidden_units(self) -> int:
        return int(self.W1.shape[1])

    def layers(self):
        """Yield (name, W, b, has_relu) for each linear layer in order."""
        yield ("fc1", self.W1, self.b1, True)
        yield ("fc2", self.W2, self.b2, False)


def numpy_forward(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    """Software (float) forward pass. x: (N, 784) -> logits (N, 10)."""
    h = np.maximum(x @ weights.W1 + weights.b1, 0.0)   # ReLU
    return h @ weights.W2 + weights.b2


def predict(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    return np.argmax(numpy_forward(weights, x), axis=1)


def accuracy(weights: MLPWeights, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predict(weights, x) == y))


def save_weights(weights: MLPWeights, path: str | Path = WEIGHTS_PATH) -> None:
    np.savez(
        Path(path),
        W1=weights.W1, b1=weights.b1,
        W2=weights.W2, b2=weights.b2,
    )


def load_weights(path: str | Path = WEIGHTS_PATH) -> MLPWeights:
    with np.load(Path(path)) as d:
        return MLPWeights(W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"])


# Training (PyTorch)

def train_mnist(
    hidden_units: int = HIDDEN_UNITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> MLPWeights:
    """Train the MLP on MNIST and return crossbar-oriented NumPy weights."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, x_test, y_test = load_mnist(normalize=True)

    xtr = torch.from_numpy(x_train)
    ytr = torch.from_numpy(y_train)
    xte = torch.from_numpy(x_test)
    yte = torch.from_numpy(y_test)

    model = nn.Sequential(
        nn.Linear(784, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    n = xtr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            optimizer.step()
            running += float(loss) * len(idx)

        if verbose:
            model.eval()
            with torch.no_grad():
                test_acc = (model(xte).argmax(1) == yte).float().mean().item()
            print(f"  epoch {epoch + 1}/{epochs}  "
                  f"train_loss={running / n:.4f}  test_acc={test_acc:.4f}")

    # Export to crossbar orientation: torch Linear weight is (out, in).
    fc1, fc2 = model[0], model[2]
    weights = MLPWeights(
        W1=fc1.weight.detach().numpy().T.copy().astype(np.float64),
        b1=fc1.bias.detach().numpy().copy().astype(np.float64),
        W2=fc2.weight.detach().numpy().T.copy().astype(np.float64),
        b2=fc2.bias.detach().numpy().copy().astype(np.float64),
    )
    return weights


def get_or_train(
    path: str | Path = WEIGHTS_PATH,
    retrain: bool = False,
    **train_kwargs,
) -> MLPWeights:
    """Load cached weights if available, otherwise train and cache them."""
    path = Path(path)
    if path.exists() and not retrain:
        return load_weights(path)

    print("Training MNIST MLP (one-time; weights will be cached)...")
    weights = train_mnist(**train_kwargs)
    save_weights(weights, path)
    print(f"Saved trained weights -> {path}")
    return weights


# Standalone entry point

def main() -> None:
    weights = get_or_train(retrain=True)

    x_train, y_train, x_test, y_test = load_mnist(normalize=True)
    train_acc = accuracy(weights, x_train, y_train)
    test_acc = accuracy(weights, x_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINED MNIST MLP")
    print("=" * 60)
    print(f"Architecture:       784 -> {weights.hidden_units} -> 10  (ReLU)")
    print(f"Train accuracy:     {train_acc:.4f}")
    print(f"Test accuracy:      {test_acc:.4f}")
    print(f"W1 range (min/max): {weights.W1.min():+.4f} / {weights.W1.max():+.4f}")
    print(f"W2 range (min/max): {weights.W2.min():+.4f} / {weights.W2.max():+.4f}")
    print(f"Weights cached at:  {Path(WEIGHTS_PATH).resolve()}")


if __name__ == "__main__":
    main()
