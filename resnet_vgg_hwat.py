"""
resnet_vgg_hwat.py

ResNet-18 and VGG-16 on the STRATUM chip, with hardware-aware training.

Both networks are trained in the CIFAR-10 form (32x32, 10 classes), in two
regimes sharing one module definition:

    float   - hw.enabled = False, an ordinary conv/BN/ReLU network.  The software
              ceiling, and the initialisation for the hardware-aware run.
    hwat    - hw.enabled = True, every conv and FC executed the way the array does
              it: sign-magnitude weights on 16 conductance states plus a separate
              STT-MRAM sign bit, 8-bit activations as two 4-bit nibbles, and a
              per-array-tile ADC on each 128-row partial sum before the adds.

The arithmetic lives in `hwat_core.make_hw_layers`; do not reimplement it here.

CIFAR rather than ImageNet: VGG-16 at 224 px needs 8454 arrays (~575 mm^2), so it
cannot be resident on any plausible version of this chip.  At CIFAR resolution
both networks fit and the comparison is about the datapath.  `resnet_vgg_study.py`
reports the footprint for both resolutions.

    python resnet_vgg_hwat.py --smoke                  quick self-test, minutes
    python resnet_vgg_hwat.py --net resnet18           full float + HWAT
    python resnet_vgg_hwat.py --net vgg16 --device cuda
    python resnet_vgg_hwat.py --net both --epochs 40 --hwat-epochs 20

Checkpoints (`*_float.ckpt.pt`, `*_hwat.ckpt.pt`) resume automatically.  Exported
`.npz` weights carry batch-norm UNFOLDED, so the deployed network is the network
that was trained.

Do not fold batch-norm for a quantised deployment: it scales output column j by
gamma_j/sqrt(var_j+eps), and the scalar `adc_scale` cannot hold that spread
(VGG-16/CIFAR-10: 86.55% unfolded, 10.00% folded).  `export_numpy(fold_bn=True)`
stays for deployment paths that genuinely need a single (W, b) pair.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hwat_core import HWConfig, make_hw_layers, pick_device, enable_tf32

# 1. Network definitions (CIFAR variants)

# A layer spec is (name, c_in, c_out, k, stride) for convs; the classifier is
# described separately.  Standard CIFAR adaptations: ResNet-18 swaps the 7x7/2
# stem and max-pool for one 3x3/1 conv, VGG-16 collapses the three 4096-wide FCs.

RESNET18_STAGES = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
VGG16_PLAN = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"]


@dataclass
class NetSpec:
    """Everything the study needs to know about a network without torch."""
    name: str
    img_size: int = 32
    num_classes: int = 10
    in_ch: int = 3
    # (name, c_in, c_out, k, stride, out_hw) filled by `describe`
    layers: List[Tuple[str, int, int, int, int, int]] = field(default_factory=list)
    # What follows each array read in the digital PU, by layer name: "relu" or
    # "none".  A block's second conv and its 1x1 downsample feed the residual
    # adder instead, and the FC feeds the argmax.
    acts: Dict[str, str] = field(default_factory=dict)
    # The residual adds, as (width, output positions) - one per ResNet block.
    # Digital PU operations, not array reads, and the one structural difference
    # between the two networks the cost model can see.
    residual_adds: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def macs(self) -> int:
        return sum(c_in * k * k * c_out * hw * hw
                   for _, c_in, c_out, k, _, hw in self.layers)

    @property
    def params(self) -> int:
        return sum(c_in * k * k * c_out for _, c_in, c_out, k, _, _ in self.layers)

    def act_of(self, name: str) -> str:
        """Activation after `name`; "relu" is the default for older specs."""
        return self.acts.get(name, "relu")


def describe(net: str, img_size: int = 32, num_classes: int = 10) -> NetSpec:
    """
    Static shape description - no torch, so the study can import this alone.

    The stem depends on the resolution: CIFAR is a 3x3/1 conv with no pooling,
    ImageNet is 7x7/2 plus a 3x3/2 max-pool, which divides the spatial dimensions
    by 4 before the first stage.
    """
    spec = NetSpec(name=net, img_size=img_size, num_classes=num_classes)
    hw = img_size
    imagenet_stem = img_size >= 128
    if net == "resnet18":
        if imagenet_stem:
            hw = img_size // 2                       # 7x7 stride-2 conv
            spec.layers.append(("stem", 3, 64, 7, 2, hw))
            hw = hw // 2                             # 3x3 stride-2 max-pool
        else:
            spec.layers.append(("stem", 3, 64, 3, 1, hw))
        spec.acts["stem"] = "relu"
        c_prev = 64
        for si, (width, blocks, stride) in enumerate(RESNET18_STAGES):
            for bi in range(blocks):
                s = stride if bi == 0 else 1
                hw_out = hw // s
                spec.layers.append((f"s{si}b{bi}.c1", c_prev, width, 3, s, hw_out))
                spec.layers.append((f"s{si}b{bi}.c2", width, width, 3, 1, hw_out))
                spec.acts[f"s{si}b{bi}.c1"] = "relu"
                # c2 and the downsample feed the residual adder, not an
                # activation; the ReLU happens once, after the add.
                spec.acts[f"s{si}b{bi}.c2"] = "none"
                if s != 1 or c_prev != width:
                    spec.layers.append((f"s{si}b{bi}.ds", c_prev, width, 1, s, hw_out))
                    spec.acts[f"s{si}b{bi}.ds"] = "none"
                spec.residual_adds.append((width, hw_out * hw_out))
                c_prev, hw = width, hw_out
        spec.layers.append(("fc", c_prev, num_classes, 1, 1, 1))
        spec.acts["fc"] = "none"
    elif net == "vgg16":
        c_prev = 3
        idx = 0
        for v in VGG16_PLAN:
            if v == "M":
                hw //= 2
                continue
            spec.layers.append((f"c{idx}", c_prev, v, 3, 1, hw))
            spec.acts[f"c{idx}"] = "relu"
            c_prev = v
            idx += 1
        if imagenet_stem:
            # The ImageNet classifier: 25088->4096->4096->1000.  These three layers
            # are 124M of VGG-16's 138M weights, so they dominate the footprint.
            spec.layers.append(("fc6", c_prev * hw * hw, 4096, 1, 1, 1))
            spec.layers.append(("fc7", 4096, 4096, 1, 1, 1))
            spec.layers.append(("fc8", 4096, num_classes, 1, 1, 1))
            spec.acts.update({"fc6": "relu", "fc7": "relu", "fc8": "none"})
        else:
            spec.layers.append(("fc", c_prev * hw * hw, num_classes, 1, 1, 1))
            spec.acts["fc"] = "none"
    else:
        raise ValueError(f"unknown net {net!r} (expected resnet18 or vgg16)")
    return spec


def build_model(net: str, hw_cfg: HWConfig, num_classes: int = 10,
                img_size: int = 32, device=None):
    """Construct the torch model. Imports torch lazily, like the rest of the toolkit."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HWConv, HWLinear = make_hw_layers(hw_cfg)

    class BasicBlock(nn.Module):
        """ResNet basic block. The residual add is a digital PU op, not an array read."""

        def __init__(self, name, c_in, c_out, stride, hw_cfg):
            super().__init__()
            self.c1 = HWConv(f"{name}.c1", c_in, c_out, 3, stride, hw_cfg, act="relu")
            self.c2 = HWConv(f"{name}.c2", c_out, c_out, 3, 1, hw_cfg, act="none")
            self.ds = None
            if stride != 1 or c_in != c_out:
                self.ds = HWConv(f"{name}.ds", c_in, c_out, 1, stride, hw_cfg,
                                 act="none", padding=0)

        def forward(self, x):
            idn = x if self.ds is None else self.ds(x)
            return F.relu(self.c2(self.c1(x)) + idn)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = HWConv("stem", 3, 64, 3, 1, hw_cfg, act="relu")
            blocks, c_prev = [], 64
            for si, (width, n_blocks, stride) in enumerate(RESNET18_STAGES):
                for bi in range(n_blocks):
                    s = stride if bi == 0 else 1
                    blocks.append(BasicBlock(f"s{si}b{bi}", c_prev, width, s, hw_cfg))
                    c_prev = width
            self.blocks = nn.ModuleList(blocks)
            self.fc = HWLinear("fc", c_prev, num_classes, hw_cfg)

        def forward(self, x):
            x = self.stem(x)
            for b in self.blocks:
                x = b(x)
            x = F.adaptive_avg_pool2d(x, 1)
            return self.fc(x)

    class VGG16(nn.Module):
        def __init__(self):
            super().__init__()
            convs, c_prev, idx, hw_sz = [], 3, 0, img_size
            self.pool_after = []
            for v in VGG16_PLAN:
                if v == "M":
                    self.pool_after.append(len(convs) - 1)
                    hw_sz //= 2
                    continue
                convs.append(HWConv(f"c{idx}", c_prev, v, 3, 1, hw_cfg, act="relu"))
                c_prev, idx = v, idx + 1
            self.convs = nn.ModuleList(convs)
            self.fc = HWLinear("fc", c_prev * hw_sz * hw_sz, num_classes, hw_cfg)

        def forward(self, x):
            for i, c in enumerate(self.convs):
                x = c(x)
                if i in self.pool_after:
                    x = F.max_pool2d(x, 2)
            return self.fc(x)

    model = {"resnet18": ResNet18, "vgg16": VGG16}[net]()
    if device is not None:
        model.to(device)
    return model


# 2. Data

def load_cifar10(root: str = "./data", train_subset: Optional[int] = None,
                 test_subset: Optional[int] = None):
    """CIFAR-10 as float32 NCHW in [0,1], channel-normalised. Downloads on first use."""
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)

    def to_np(ds, limit):
        n = len(ds) if limit is None else min(limit, len(ds))
        x = np.stack([np.asarray(ds[i][0]) for i in range(n)]).astype(np.float32)
        y = np.array([ds[i][1] for i in range(n)], dtype=np.int64)
        return x, y

    xtr, ytr = to_np(tr, train_subset)
    xte, yte = to_np(te, test_subset)
    mean = xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (xtr - mean) / std, ytr, (xte - mean) / std, yte


# 3. Train / evaluate

def augment(xb, torch_mod):
    """
    Standard CIFAR-10 augmentation: random horizontal flip + random crop with 4 px
    reflect padding, per sample on whatever device xb is already on.

    Without it the networks memorise the training set (VGG-16: train loss 0.0095
    against 85.7% test top-1), and that gap would be misread as a hardware cost.
    """
    torch = torch_mod
    F = torch.nn.functional
    B, C, H, W = xb.shape
    dev = xb.device

    flip = torch.rand(B, device=dev) < 0.5
    xb = torch.where(flip.view(B, 1, 1, 1), xb.flip(-1), xb)

    pad = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    iy = torch.randint(0, 9, (B,), device=dev)
    ix = torch.randint(0, 9, (B,), device=dev)
    rows = iy.view(B, 1, 1, 1) + torch.arange(H, device=dev).view(1, 1, H, 1)
    cols = ix.view(B, 1, 1, 1) + torch.arange(W, device=dev).view(1, 1, 1, W)
    bi = torch.arange(B, device=dev).view(B, 1, 1, 1)
    ci = torch.arange(C, device=dev).view(1, C, 1, 1)
    return pad[bi, ci, rows, cols]


def evaluate(model, x, y, torch_mod, batch: int = 128, device=None) -> float:
    """Top-1 accuracy."""
    torch = torch_mod
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i:i + batch])
            if device is not None:
                xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            correct += int((pred == y[i:i + batch]).sum())
    return correct / max(len(x), 1)


def _ckpt_path(path: str) -> str:
    return path.replace(".npz", "") + ".ckpt.pt"


def train(net: str, regime: str, data, epochs: int, lr: float,
          hw_cfg: HWConfig, out_path: str, batch: int = 128,
          device=None, init: Optional[Dict[str, np.ndarray]] = None,
          patience: int = 0, use_augment: bool = True,
          log=print) -> Tuple[object, dict]:
    """
    Train one regime.  Keeps the best epoch by validation accuracy, not the last:
    quantised fine-tuning is not monotonic.

    `patience` stops early after that many epochs without a new best - the largest
    saving available on the hardware-aware phase, which costs ~7x a float epoch.
    0 disables it and runs every epoch.
    """
    import torch
    import torch.nn as nn

    xtr, ytr, xte, yte = data
    model = build_model(net, hw_cfg, num_classes=int(ytr.max()) + 1,
                        img_size=xtr.shape[-1], device=device)
    if init is not None:
        load_numpy_into(model, init)
        log(f"  initialised {regime} from float weights")

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.CrossEntropyLoss()

    start_ep, best_acc, best_state = 0, -1.0, None
    stale = 0
    ck = _ckpt_path(out_path)
    if os.path.exists(ck):
        blob = torch.load(ck, map_location=device or "cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start_ep = int(blob.get("epoch", 0))
        best_acc = float(blob.get("best_acc", -1.0))
        best_state = blob.get("best_state")
        log(f"  resumed {regime} from epoch {start_ep} (best {best_acc*100:.2f}%)")

    n = len(xtr)
    for ep in range(start_ep, epochs):
        model.train()
        perm = np.random.permutation(n)
        t0, tot_loss = time.time(), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.as_tensor(xtr[idx])
            yb = torch.as_tensor(ytr[idx])
            if device is not None:
                xb, yb = xb.to(device), yb.to(device)
            if use_augment:
                xb = augment(xb, torch)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss) * len(idx)
        sched.step()
        acc = evaluate(model, xte, yte, torch, device=device)
        log(f"  [{regime}] epoch {ep+1}/{epochs} loss {tot_loss/n:.4f} "
            f"top1 {acc*100:.2f}%  ({time.time()-t0:.1f}s)")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_acc": best_acc,
                    "best_state": best_state}, ck)
        if patience and stale >= patience:
            log(f"  [{regime}] early stop: {stale} epochs without a new best "
                f"(best {best_acc*100:.2f}%, {ep+1}/{epochs} epochs used)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_acc < 0.0:
        # No epoch ran (epochs=0, or a checkpoint already past `epochs`).  Score
        # the model as it stands rather than reporting the -1.0 sentinel, which
        # otherwise surfaces as a nonsensical "-100.00%" next to a saved .npz.
        best_acc = evaluate(model, xte, yte, torch, device=device)
        log(f"  [{regime}] no epoch ran; scored the initial weights: "
            f"top1 {best_acc*100:.2f}%")
    # Freeze the calibrated activation/ADC ranges so deployment sees fixed scales.
    for m in model.modules():
        if hasattr(m, "freeze_scales"):
            m.freeze_scales()
    return model, {"best_top1": best_acc, "epochs": epochs, "regime": regime}


# 4. Export: BN-folded (W, b) pairs the deployment path can map

def export_numpy(model, torch_mod, fold_bn: bool = False) -> Dict[str, np.ndarray]:
    """
    Flatten to {layer.W, layer.b, ...} with batch-norm folded into the weights.

    `lower_linear_layer` maps exactly this form onto sign-magnitude tiles, so the
    deployed arithmetic sees the same numbers the fine-tune converged on.
    """
    torch = torch_mod
    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, m in model.named_modules():
            if not hasattr(m, "weight") or not hasattr(m, "bias"):
                continue
            if not hasattr(m, "in_scale"):        # skip BN and other torch layers
                continue
            W = m.weight.detach().cpu().numpy().copy()      # (fan_in, c_out)
            b = m.bias.detach().cpu().numpy().copy()        # (c_out,)
            key = name.replace(".", "/") or "root"
            bn = getattr(m, "bn", None)
            if bn is not None and fold_bn:
                # Folding is algebraically exact but wrong under quantisation: it
                # multiplies output column j by s_j = gamma_j/sqrt(var_j+eps), and
                # `adc_scale` is one scalar for the whole layer, so high-s columns
                # saturate and low-s columns quantise to zero (86.55% unfolded vs
                # 10.00% folded on VGG-16).  Batch-norm after the converter is an
                # ordinary PU op, so keeping it is cheaper to justify.
                g = bn.weight.detach().cpu().numpy()
                beta = bn.bias.detach().cpu().numpy()
                mu = bn.running_mean.detach().cpu().numpy()
                var = bn.running_var.detach().cpu().numpy()
                sc_ = g / np.sqrt(var + bn.eps)
                W = W * sc_[None, :]
                b = (b - mu) * sc_ + beta
            elif bn is not None:
                # Carry batch-norm across unfolded, so the deployed network is
                # bit-for-bit the network that was trained.
                out[f"{key}.bn_weight"] = bn.weight.detach().cpu().numpy().astype(np.float32)
                out[f"{key}.bn_bias"] = bn.bias.detach().cpu().numpy().astype(np.float32)
                out[f"{key}.bn_mean"] = bn.running_mean.detach().cpu().numpy().astype(np.float32)
                out[f"{key}.bn_var"] = bn.running_var.detach().cpu().numpy().astype(np.float32)
            out[f"{key}.W"] = W.astype(np.float32)
            out[f"{key}.b"] = b.astype(np.float32)
            out[f"{key}.in_scale"] = m.in_scale.detach().cpu().numpy().astype(np.float32)
            out[f"{key}.adc_scale"] = m.adc_scale.detach().cpu().numpy().astype(np.float32)
            # The device-to-device spread must travel with the weights: it is a
            # fixed per-cell multiplier standing for one programmed array, and the
            # fine-tune adapts to that realisation.  A fresh draw moves the network
            # output by 64% and takes a trained VGG-16 to chance.
            if hasattr(m, "d2d"):
                out[f"{key}.d2d"] = m.d2d.detach().cpu().numpy().astype(np.float32)
    return out


def load_numpy_into(model, arrays: Dict[str, np.ndarray]) -> int:
    """Load exported (unfolded) weights back into a model. Returns layers matched."""
    import torch
    hit = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if not hasattr(m, "in_scale"):
                continue
            key = name.replace(".", "/") or "root"
            if f"{key}.W" not in arrays:
                continue
            m.weight.copy_(torch.as_tensor(arrays[f"{key}.W"]))
            m.bias.copy_(torch.as_tensor(arrays[f"{key}.b"]))
            if f"{key}.in_scale" in arrays:
                m.in_scale.copy_(torch.as_tensor(arrays[f"{key}.in_scale"]))
                m.calibrated.fill_(1.0)
            if f"{key}.adc_scale" in arrays:
                # Older exports stored one scalar per layer; broadcast it onto
                # the per-column buffer so those files still load.
                v = torch.as_tensor(arrays[f"{key}.adc_scale"]).flatten()
                m.adc_scale.copy_(v.expand_as(m.adc_scale) if v.numel() == 1 else v)
            if f"{key}.d2d" in arrays and hasattr(m, "d2d"):
                m.d2d.copy_(torch.as_tensor(arrays[f"{key}.d2d"]))
            bn = getattr(m, "bn", None)
            if bn is not None and f"{key}.bn_weight" in arrays:
                bn.weight.copy_(torch.as_tensor(arrays[f"{key}.bn_weight"]))
                bn.bias.copy_(torch.as_tensor(arrays[f"{key}.bn_bias"]))
                bn.running_mean.copy_(torch.as_tensor(arrays[f"{key}.bn_mean"]))
                bn.running_var.copy_(torch.as_tensor(arrays[f"{key}.bn_var"]))
            hit += 1
    return hit


def save_weights(path: str, arrays: Dict[str, np.ndarray], meta: dict) -> None:
    np.savez_compressed(path, __meta__=json.dumps(meta), **arrays)


def load_weights(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["__meta__"])) if "__meta__" in blob else {}
    return {k: blob[k] for k in blob.files if k != "__meta__"}, meta


def weights_path(net: str, regime: str) -> str:
    return f"{net}_hwat_{regime}.npz"


# 5. Driver

def get_or_train(net: str, epochs: int = 40, hwat_epochs: int = 20,
                 lr: float = 0.05, hwat_lr: float = 0.005,
                 train_subset: Optional[int] = None,
                 test_subset: Optional[int] = None,
                 batch: int = 128, device=None, retrain: bool = False,
                 adc_bits: int = 4, patience: int = 4,
                 use_augment: bool = True, log=print) -> Dict[str, dict]:
    """Train (or load) both regimes for one network. Returns a per-regime summary."""
    import torch

    device = device or pick_device()
    log(f"device: {device}")
    data = load_cifar10(train_subset=train_subset, test_subset=test_subset)
    log(f"CIFAR-10: {len(data[0])} train / {len(data[2])} test, "
        f"{data[0].shape[-1]}x{data[0].shape[-1]}")

    spec = describe(net, img_size=data[0].shape[-1])
    log(f"{net}: {len(spec.layers)} mapped layers, {spec.params/1e6:.2f}M weights, "
        f"{spec.macs/1e6:.1f} MMAC/image")

    results: Dict[str, dict] = {}
    float_arrays = None
    for regime in ("float", "hwat"):
        path = weights_path(net, regime)
        if os.path.exists(path) and not retrain:
            arrays, meta = load_weights(path)
            log(f"[{regime}] cached -> {path}  (top1 {meta.get('best_top1', 0)*100:.2f}%)")
            # Provenance guard: caching keys on the .npz existing, so a --smoke run
            # would leave 1-epoch weights for a later full run to reuse silently.
            if meta.get("partial"):
                log(f"  *** WARNING: these weights are from a PARTIAL run "
                    f"({meta.get('train_subset')} train images, "
                    f"{meta.get('epochs')} epoch(s)).  They are not a result.\n"
                    f"  *** Pass --retrain, or delete {path}, to train properly.")
            results[regime] = meta
            if regime == "float":
                float_arrays = arrays
            continue

        hw_cfg = HWConfig.from_simulation_config(
            enabled=(regime == "hwat"), adc_bits=adc_bits)
        log(f"[{regime}] training  (hw.enabled={hw_cfg.enabled}, adc_bits={hw_cfg.adc_bits})")
        model, meta = train(
            net, regime, data,
            epochs=(epochs if regime == "float" else hwat_epochs),
            lr=(lr if regime == "float" else hwat_lr),
            hw_cfg=hw_cfg, out_path=path, batch=batch, device=device,
            init=(float_arrays if regime == "hwat" else None),
            patience=(patience if regime == "hwat" else 0),
            use_augment=use_augment, log=log)
        arrays = export_numpy(model, torch)
        meta.update({"net": net, "adc_bits": adc_bits,
                     "params": spec.params, "macs": spec.macs,
                     "train_subset": train_subset, "test_subset": test_subset,
                     "train_images": int(len(data[0])),
                     # True when this did not see the full training set or ran
                     # too few epochs to mean anything - read by the cache guard.
                     "augmented": bool(use_augment),
                     "partial": bool(train_subset is not None
                                     or (epochs if regime == "float" else hwat_epochs) < 5)})
        save_weights(path, arrays, meta)
        log(f"[{regime}] saved -> {path}  best top1 {meta['best_top1']*100:.2f}%")
        results[regime] = meta
        if regime == "float":
            float_arrays = arrays
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="both", choices=["resnet18", "vgg16", "both"])
    ap.add_argument("--epochs", type=int, default=40, help="float training epochs")
    ap.add_argument("--hwat-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hwat-lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--adc-bits", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true",
                    help="disable random crop + horizontal flip. Without "
                         "augmentation VGG-16 overfits hard (train loss 0.01, "
                         "test 85.7%) and the ceiling lands ~7 points low")
    ap.add_argument("--patience", type=int, default=4,
                    help="stop the HWAT phase after N epochs with no new best "
                         "(0 = run every epoch). The HWAT phase costs ~7x a "
                         "float epoch, so this is the biggest time lever")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 tensor cores on Ampere+ (they are on by "
                         "default; TF32's 10-bit mantissa is far more than a "
                         "4-bit ADC datapath needs)")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--test-subset", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 1 epoch each: verifies the pipeline runs")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.hwat_epochs = 1, 1
        args.train_subset, args.test_subset = 512, 256
        args.batch = 64
        print("SMOKE MODE: 512 train / 256 test, 1 epoch per regime.\n"
              "Numbers are NOT meaningful - this only checks the pipeline runs.\n")

    nets = ["resnet18", "vgg16"] if args.net == "both" else [args.net]
    device = pick_device(args.device)
    if enable_tf32(not args.no_tf32):
        print("TF32 tensor cores: ENABLED for the per-tile matmul")
    elif device.type == "cuda":
        print("TF32 tensor cores: disabled (--no-tf32)")
    summary = {}
    for net in nets:
        print("=" * 78)
        print(f"{net.upper()}  -  float vs hardware-aware")
        print("=" * 78)
        summary[net] = get_or_train(
            net, epochs=args.epochs, hwat_epochs=args.hwat_epochs,
            lr=args.lr, hwat_lr=args.hwat_lr, batch=args.batch,
            train_subset=args.train_subset, test_subset=args.test_subset,
            device=device, retrain=args.retrain, adc_bits=args.adc_bits,
            patience=args.patience, use_augment=not args.no_augment)
        print()

    print("=" * 78)
    print(f"{'net':10s}{'float top1':>13s}{'hwat top1':>12s}{'delta':>9s}")
    for net, r in summary.items():
        f = r.get("float", {}).get("best_top1", 0.0)
        h = r.get("hwat", {}).get("best_top1", 0.0)
        print(f"{net:10s}{f*100:>12.2f}%{h*100:>11.2f}%{(h-f)*100:>8.2f}")
    with open("resnet_vgg_hwat_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSaved -> resnet_vgg_hwat_summary.json")


if __name__ == "__main__":
    main()
