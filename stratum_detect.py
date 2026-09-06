"""
stratum_detect.py

A detection task the chip can be scored on, plus hardware-aware training for it.

A deep detector cannot survive this datapath open-loop: 4-bit weights and a 4-bit
per-column ADC give roughly 8 dB of SNR per convolution, and dozens of chained
convolutions turn that into noise.  Two things fix that:

  1. A task with a real metric.  MNIST digits are composited into scenes, so the
     score is mAP@0.5, precision and recall rather than fidelity-vs-float.
  2. Training that knows about the hardware.  The chip's arithmetic sits inside
     the forward pass with straight-through gradients, so the weights converge on
     something that survives deployment.

What the hardware-aware forward models:

    weight quantisation   |w| snapped to the device's N conductance levels, sign
                          in the sign register (what `lower_linear_layer` does)
    input quantisation    activations to unsigned 8-bit codes, fed as two nibbles
    PER-TILE ADC          each 128-row array's partial sum digitised by its own
                          4-bit ADC BEFORE the partials are added
    device noise          persistent d2d spread on the programmed conductances,
                          plus transient read noise re-drawn every forward pass

Three details decide whether this works at all:

  * Model the datapath, not an approximation of it.  A single quantiser on the
    finished product trains against arithmetic the chip does not perform, and the
    weights do not transfer (14% mAP in its own quantiser, 0% on the chip).
  * Fine-tune, do not train from scratch.  Quantised training from a random
    initialisation has to discover the task and survive a 4-bit datapath at once.
  * Freeze the converter ranges.  They are observed from activations that depend
    on the previous layer's ranges, so leaving them adapting couples every scale
    to every other and the stack drifts.  Deployment calibrates once, too.

Training is PyTorch; the result is exported to numpy with batch-norm folded into
the convolutions, the form `stratum_detect_imc.py` deploys.

Run:
    python stratum_detect.py                (train float + HWAT, cache both)
    python stratum_detect.py --epochs 20
    python stratum_detect.py --retrain
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import simulation_config as _cfg


WEIGHTS_PATH_FLOAT = "stratum_detect_float.npz"
WEIGHTS_PATH_HWAT = "stratum_detect_hwat.npz"
DATASET_SEED = 1234

# --- task geometry -----------------------------------------------------------
IMG_SIZE = 64                 # square scene
STRIDE = 8                    # single detection level -> 8x8 grid
GRID = IMG_SIZE // STRIDE
NUM_CLASSES = 10              # the ten digits
MAX_OBJECTS = 3               # digits per scene
DIGIT_MIN_PX, DIGIT_MAX_PX = 16, 28


# 1. The task: MNIST digits composited into detection scenes

def _tight_bbox(patch: np.ndarray, thresh: float = 0.15):
    """Tight (x0, y0, x1, y1) around the inked pixels of a digit patch."""
    ys, xs = np.where(patch > thresh)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _resize_nearest(patch: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a square patch (no scipy dependency)."""
    idx = (np.arange(size) * (patch.shape[0] / size)).astype(int)
    return patch[np.ix_(idx, idx)]


def make_scene(rng: np.random.Generator, digits: np.ndarray, labels: np.ndarray,
               imgsz: int = IMG_SIZE, max_objects: int = MAX_OBJECTS
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    One detection scene.

    Returns (image (1, imgsz, imgsz) float32 in [0,1],
             targets (n, 5) = [class, cx, cy, w, h] in PIXELS).

    Digits are pasted with `maximum` compositing and rejected if their centre
    falls in a grid cell already taken - the assigner gives each cell one object.
    """
    canvas = np.zeros((imgsz, imgsz), dtype=np.float32)
    n = int(rng.integers(1, max_objects + 1))
    taken = set()
    targets = []
    for _ in range(n):
        i = int(rng.integers(0, digits.shape[0]))
        size = int(rng.integers(DIGIT_MIN_PX, DIGIT_MAX_PX + 1))
        patch = _resize_nearest(digits[i].reshape(28, 28), size)
        bb = _tight_bbox(patch)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        # place so the digit's tight box lies inside the canvas
        px = int(rng.integers(0, max(imgsz - (x1 - x0), 1)))
        py = int(rng.integers(0, max(imgsz - (y1 - y0), 1)))
        cx = px + (x1 - x0) / 2.0
        cy = py + (y1 - y0) / 2.0
        cell = (int(cx // STRIDE), int(cy // STRIDE))
        if cell in taken:
            continue
        taken.add(cell)
        # paste the cropped digit
        crop = patch[y0:y1, x0:x1]
        h, w = crop.shape
        canvas[py:py + h, px:px + w] = np.maximum(canvas[py:py + h, px:px + w], crop)
        targets.append([float(labels[i]), cx, cy, float(w), float(h)])
    if not targets:                      # degenerate draw -> retry
        return make_scene(rng, digits, labels, imgsz, max_objects)
    return canvas[None, :, :], np.asarray(targets, dtype=np.float32)


def build_dataset(n_train: int = 4000, n_val: int = 1000, seed: int = DATASET_SEED
                  ) -> Tuple[np.ndarray, list, np.ndarray, list]:
    """Synthesise the train / val scenes from the cached MNIST digits."""
    import mnist_ann
    x_train, y_train, x_test, y_test = mnist_ann.load_mnist(normalize=True)
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    for _ in range(n_train):
        img, tg = make_scene(rng, x_train, y_train)
        xs.append(img)
        ts.append(tg)
    rng_v = np.random.default_rng(seed + 1)
    xv, tv = [], []
    for _ in range(n_val):
        img, tg = make_scene(rng_v, x_test, y_test)
        xv.append(img)
        tv.append(tg)
    return (np.stack(xs), ts, np.stack(xv), tv)


# 2. Detection metric: mAP@0.5

def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[..., 0] = b[..., 0] - b[..., 2] / 2
    out[..., 1] = b[..., 1] - b[..., 3] / 2
    out[..., 2] = b[..., 0] + b[..., 2] / 2
    out[..., 3] = b[..., 1] + b[..., 3] / 2
    return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) -> (n,m) IoU."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float = 0.45) -> List[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i][None, :], boxes[order[1:]])[0]
        order = order[1:][ious < thresh]
    return keep


def decode(pred: np.ndarray, conf_thresh: float = 0.25,
           nms_thresh: float = 0.45) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw head output -> (boxes_xyxy, scores, classes) for ONE image.

    pred is (5 + NUM_CLASSES, GRID, GRID): objectness, tx, ty, tw, th, then class
    logits.  Anchor-free: the centre is the cell origin plus a sigmoid offset, the
    size an exponential of the raw prediction in stride units.
    """
    obj = 1.0 / (1.0 + np.exp(-pred[0]))
    tx, ty, tw, th = pred[1], pred[2], pred[3], pred[4]
    cls_logit = pred[5:]
    cls_prob = np.exp(cls_logit - cls_logit.max(axis=0, keepdims=True))
    cls_prob /= cls_prob.sum(axis=0, keepdims=True)
    cls_id = cls_prob.argmax(axis=0)
    cls_conf = cls_prob.max(axis=0)
    score = obj * cls_conf

    gy, gx = np.mgrid[0:pred.shape[1], 0:pred.shape[2]]
    cx = (gx + 1.0 / (1.0 + np.exp(-tx))) * STRIDE
    cy = (gy + 1.0 / (1.0 + np.exp(-ty))) * STRIDE
    w = np.exp(np.clip(tw, -4, 4)) * STRIDE
    h = np.exp(np.clip(th, -4, 4)) * STRIDE

    m = score >= conf_thresh
    if not np.any(m):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = _xywh_to_xyxy(np.stack([cx[m], cy[m], w[m], h[m]], axis=1))
    scores, classes = score[m], cls_id[m]
    keep = nms(boxes, scores, nms_thresh)
    return boxes[keep], scores[keep], classes[keep]


def mean_average_precision(all_preds: List[tuple], all_targets: List[np.ndarray],
                           iou_thresh: float = 0.5, n_classes: int = NUM_CLASSES
                           ) -> dict:
    """
    mAP@0.5 with the standard greedy match (highest-scoring detection first, each
    ground truth claimed once), plus micro precision / recall at the operating
    point the detections were decoded at.
    """
    aps, tp_all, fp_all, n_gt_all = [], 0, 0, 0
    for c in range(n_classes):
        recs = []          # (score, is_tp) over the whole val set for this class
        n_gt = 0
        for (boxes, scores, classes), tgt in zip(all_preds, all_targets):
            gt = tgt[tgt[:, 0] == c]
            gt_xyxy = _xywh_to_xyxy(gt[:, 1:5]) if gt.size else np.zeros((0, 4))
            n_gt += gt_xyxy.shape[0]
            sel = classes == c
            b, s = boxes[sel], scores[sel]
            if b.shape[0] == 0:
                continue
            order = np.argsort(-s)
            b, s = b[order], s[order]
            ious = iou_xyxy(b, gt_xyxy)
            claimed = set()
            for i in range(b.shape[0]):
                j = int(np.argmax(ious[i])) if gt_xyxy.shape[0] else -1
                hit = (j >= 0 and ious[i, j] >= iou_thresh and j not in claimed)
                if hit:
                    claimed.add(j)
                recs.append((float(s[i]), bool(hit)))
        n_gt_all += n_gt
        if n_gt == 0:
            continue
        if not recs:
            aps.append(0.0)
            continue
        recs.sort(key=lambda r: -r[0])
        tp = np.cumsum([1.0 if r[1] else 0.0 for r in recs])
        fp = np.cumsum([0.0 if r[1] else 1.0 for r in recs])
        tp_all += int(tp[-1])
        fp_all += int(fp[-1])
        recall = tp / max(n_gt, 1)
        precision = tp / np.maximum(tp + fp, 1e-9)
        # 101-point interpolated AP (COCO style)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = precision[recall >= t]
            ap += (p.max() if p.size else 0.0) / 101.0
        aps.append(float(ap))
    return {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "ap_per_class": aps,
        "precision": tp_all / max(tp_all + fp_all, 1),
        "recall": tp_all / max(n_gt_all, 1),
        "n_gt": n_gt_all,
    }


# 3. The detector (exported form: conv + bias + SiLU, BN already folded)

@dataclass
class LayerSpec:
    name: str
    c_in: int
    c_out: int
    k: int
    stride: int
    act: bool = True


# 64x64 -> 32 -> 16 -> 8, single detection level at stride 8.
# Deliberately sequential (no concat / residual): every layer is one reused
# crossbar VMM, which keeps the hardware deployment a straight chain.
ARCH: List[LayerSpec] = [
    LayerSpec("conv0", 1, 16, 3, 2),
    LayerSpec("conv1", 16, 32, 3, 2),
    LayerSpec("conv2", 32, 32, 3, 1),
    LayerSpec("conv3", 32, 64, 3, 2),
    LayerSpec("conv4", 64, 64, 3, 1),
    LayerSpec("head0", 64, 64, 3, 1),
    LayerSpec("head1", 64, 5 + NUM_CLASSES, 1, 1, act=False),
]


def arch_summary() -> str:
    macs = 0
    hw = IMG_SIZE
    for s in ARCH:
        hw = (hw + 2 * (s.k // 2) - s.k) // s.stride + 1
        macs += hw * hw * s.c_out * s.c_in * s.k * s.k
    params = sum(s.c_in * s.c_out * s.k * s.k + s.c_out for s in ARCH)
    return (f"{len(ARCH)} convs, {params/1e3:.1f} k params, {macs/1e6:.2f} MMAC "
            f"per {IMG_SIZE}x{IMG_SIZE} scene, {GRID}x{GRID} detection grid")


# 4. Hardware-aware training (PyTorch)

@dataclass
class HWConfig:
    """Which pieces of the chip's arithmetic sit inside the training forward."""
    enabled: bool = True
    weight_levels: int = 16          # device conductance states (4-bit)
    input_bits: int = 8              # digit-serial activation codes
    adc_bits: int = 4                # per-column bipolar ADC
    tile_rows: int = 128             # crossbar rows -> where partials get digitised
    # ---- MEMRISTOR non-idealities.  Every one of these perturbs the CELL
    # CONDUCTANCE, i.e. the stored weight - none of them is converter noise.
    # The ADC's own error is `adc_bits` and lives in `q_adc`, separately.
    d2d_sigma: float = 0.05          # persistent device-to-device spread
    read_sigma: float = 0.005        # transient read noise, re-drawn per forward
    # Write/programming scatter, redrawn at every programming event (unlike d2d,
    # fixed for the life of the die).  Programming error and cycle-to-cycle add in
    # quadrature, as in `single_memristor_nonidealities`.
    c2c_sigma: float = 0.0
    # Stuck-at defects: a cell that will not program, pinned to LRS (maximum
    # conductance -> full-scale magnitude) or HRS (g_min -> zero magnitude).
    stuck_at_prob: float = 0.0
    stuck_at_lrs_frac: float = 0.5
    # Retention drift.  The magnitude relaxes toward g_min as a Kohlrausch
    # stretched exponential; the sign is a separate STT-MRAM cell and never drifts.
    # The grid is linear from g_min, so the drift is a uniform gain on the weight
    # magnitude.  It bites because the ADC range is calibrated once at t=0 and then
    # frozen, so the shrinking signal under-ranges the converter.
    retention_time_s: float = 0.0
    retention_tau_s: float = 36000.0
    retention_beta: float = 0.5
    retention_model: str = "kohlrausch"
    # CROSSBAR-level non-idealities (the array, not the cell): wire IR drop,
    # parasitic + access-transistor series R, self-heating through the TCR, the
    # column current ceiling and the physical sense-noise floors.  Set this to a
    # `crossbar_nonidealities.CrossbarNonIdealityConfig` (e.g.
    # `simulation_config.build_workflow_crossbar_ni_config()`) and
    # `hwat_core._HWBase._crossbar_ni` applies them in training and at inference.
    # None keeps the ideal array.
    crossbar_ni: Optional[object] = None
    # Array operating point the current-domain mapping needs.  Defaults match
    # simulation_config; override together with `crossbar_ni` if the array changes.
    read_voltage_v: float = 0.4
    g_min_s: float = 5e-8
    g_max_s: float = 8e-7
    read_time_s: float = 7.2e-9

    # Apply the transient read noise at INFERENCE too, not only while training.
    # A deployed chip re-draws it on every read; leaving it False keeps the
    # historical behaviour, where a scored run is bit-for-bit repeatable.
    read_noise_at_inference: bool = False
    # Where the bias physically lives.  `_hw_matmul` folds it in digitally after
    # the ADC, but the chip rides the bias row, so with this True the bias decays
    # by the same retention factor as the magnitudes.  Only has an effect when
    # retention_time_s > 0.  Measured at t = 7200 s with the converter
    # recalibrated: 59.6% with a non-drifting digital bias against 81.6% with the
    # bias on the array.
    bias_on_array: bool = True
    scale_momentum: float = 0.95     # EMA for the calibrated ranges
    scale_margin: float = 1.20       # headroom over the observed peak
    # Range-tracking steps before the activation/ADC scales freeze themselves.
    # Tracking calls torch.quantile every step (20.3% of a training step) and is
    # waste once the EMA has settled.  0 = track for the whole run.
    calib_steps: int = 200
    # Batch-norm is folded into the weights before deployment, so a fine-tuning
    # run works directly on the folded (deployed) form and drops BN entirely.
    use_bn: bool = True

    @classmethod
    def from_simulation_config(cls, **over) -> "HWConfig":
        """Take the device/array numbers from simulation_config where they exist."""
        ni = _cfg.build_nn_decay_single_memristor_ni_config()
        base = dict(
            weight_levels=len(_cfg.CONDUCTANCE_LEVELS_S),
            input_bits=int(_cfg.INPUT_PRECISION_BITS),
            tile_rows=int(_cfg.XBAR_ROWS),
            d2d_sigma=float(ni.device_to_device_sigma_rel),
            read_sigma=float(ni.read_noise_sigma_rel),
            # Programming error and cycle-to-cycle add in quadrature, matching
            # `apply_programming_nonidealities`.  These used to stop at the NI
            # model and never reach the hardware-aware forward at all.
            c2c_sigma=float((float(ni.programming_error_sigma_rel) ** 2
                             + float(ni.cycle_to_cycle_sigma_rel) ** 2) ** 0.5),
            stuck_at_prob=float(ni.stuck_at_probability),
            stuck_at_lrs_frac=float(ni.stuck_at_lrs_fraction),
            retention_tau_s=float(ni.retention_tau_s),
            retention_beta=float(ni.retention_beta),
            # Which drift law, straight from the 'Non-ideal' tab's
            # SM_NI_RETENTION_MODEL - so 'none' there really does disable it.
            retention_model=str(ni.retention_model),
            # The ARRAY's non-idealities, not just the cell's.  Without this the
            # hardware-aware forward models a perfect crossbar wrapped around an
            # imperfect device, which is not the machine being built.
            crossbar_ni=_cfg.build_workflow_crossbar_ni_config(),
            read_voltage_v=float(_cfg.READ_VOLTAGE_V),
            g_min_s=float(min(_cfg.CONDUCTANCE_LEVELS_S)),
            g_max_s=float(max(_cfg.CONDUCTANCE_LEVELS_S)),
            read_time_s=float(_cfg.READ_TIME_S),
        )
        base.update(over)
        return cls(**base)


def build_torch_model(hw: "HWConfig", init: Optional[Dict[str, np.ndarray]] = None):
    """
    Construct the detector. Defined inside a function so this module imports
    (and the dataset / metric code runs) without PyTorch installed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def ste(quantised, original):
        """Straight-through: forward the quantised value, backward the identity."""
        return original + (quantised - original).detach()

    class HWConv(nn.Module):
        """
        One convolution executed the way the chip executes it.

        im2col -> unsigned 8-bit activation codes -> split the input across 128-row
        array tiles -> per-tile partial product against sign-magnitude quantised
        weights (+ device noise) -> per-tile 4-bit ADC -> sum the partials -> fold
        the bias -> BN -> SiLU.

        With `hw.enabled = False` every quantiser is bypassed, so the same module
        trains the float baseline and the hardware-aware one.
        """

        def __init__(self, spec: LayerSpec, hw: HWConfig):
            super().__init__()
            self.spec, self.hw = spec, hw
            fan_in = spec.c_in * spec.k * spec.k
            self.weight = nn.Parameter(
                torch.randn(fan_in, spec.c_out) * (2.0 / fan_in) ** 0.5)
            self.bias = nn.Parameter(torch.zeros(spec.c_out))
            self.bn = (nn.BatchNorm1d(spec.c_out)
                       if (spec.act and hw.use_bn) else None)
            # Calibrated ranges (EMA, frozen at eval) - the deployment does the
            # same: calibrate once, then keep the converter range fixed.
            self.register_buffer("in_scale", torch.ones(1))
            self.register_buffer("adc_scale", torch.ones(1))
            self.register_buffer("calibrated", torch.zeros(1))
            self.register_buffer("frozen", torch.zeros(1))
            # Device-to-device spread is drawn ONCE, like programming a physical
            # array: training then sees a fixed imperfect device, not fresh noise.
            self.register_buffer(
                "d2d", 1.0 + hw.d2d_sigma * torch.randn(fan_in, spec.c_out))

        # -- quantisers -------------------------------------------------

        def q_weight(self, w):
            """Sign-magnitude onto N conductance levels; the sign is a register."""
            hw = self.hw
            wmax = w.abs().amax().clamp_min(1e-12)
            step = wmax / (hw.weight_levels - 1)
            q = torch.round(w.abs() / step).clamp(0, hw.weight_levels - 1) * step
            q = q * torch.sign(w)
            if hw.d2d_sigma > 0:
                q = q * self.d2d                      # programmed-in, persistent
            if hw.read_sigma > 0 and self.training:
                q = q * (1.0 + hw.read_sigma * torch.randn_like(q))
            return ste(q, w)

        def to_nibbles(self, x):
            """
            Activations -> unsigned 8-bit codes -> radix-16 digit-serial nibbles.

            Returns the list of per-digit DAC drives, each in [0, 1] (the array
            is driven at nibble/digit_max of the read voltage), exactly what
            `DigitSerialInput.to_codes` / `.digits` hand the crossbar.
            """
            hw = self.hw
            s = self.in_scale.clamp_min(1e-9)
            n = 2 ** hw.input_bits - 1
            codes_f = (x / s).clamp(0, 1) * n
            codes = ste(torch.round(codes_f), codes_f)
            radix = 2 ** 4                       # 4-bit DAC digits
            dmax = radix - 1
            lo = torch.remainder(codes, radix)
            hi = torch.div(codes - lo, radix)
            return [lo / dmax, hi / dmax], s * dmax / n

        def q_adc(self, partial):
            """Bipolar ADC on ONE array's partial sum, before partials are added."""
            hw = self.hw
            s = self.adc_scale.clamp_min(1e-9)
            n = 2 ** (hw.adc_bits - 1)
            q = torch.round((partial / s).clamp(-1, 1) * n) / n * s
            return ste(q, partial)

        # -- forward ----------------------------------------------------

        def forward(self, x):
            sp, hw = self.spec, self.hw
            B = x.shape[0]
            p = sp.k // 2
            cols = F.unfold(x, sp.k, padding=p, stride=sp.stride)   # (B, Cin*k*k, L)
            cols = cols.transpose(1, 2)                             # (B, L, F)
            side = int(round(cols.shape[1] ** 0.5))

            if not hw.enabled:
                y = cols @ self.weight + self.bias
            else:
                # The DAC is unsigned: shift the activation non-negative and let
                # the bias carry the exact -q*colsum(W) correction (on the chip
                # that rides the bias row, which is sensed once and latched).
                shift = (-cols.amin()).clamp_min(0.0).detach()
                xs = cols + shift
                if self.training:
                    self._track_scales(xs)
                nibbles, conv = self.to_nibbles(xs)
                wq = self.q_weight(self.weight)
                # The real datapath does one analog read per nibble per array tile
                # and digitises each on its own 4-bit ADC before the PU shift-and-
                # add.  One quantiser on the finished product would train the
                # network against arithmetic the chip does not perform.
                T = hw.tile_rows
                raw = [nib[..., i:i + T] @ wq[i:i + T]
                       for nib in nibbles
                       for i in range(0, wq.shape[0], T)]
                if self.training:
                    self._track_adc(raw)
                acc = None
                k = 0
                for d, nib in enumerate(nibbles):
                    w_d = float(2 ** (4 * d))            # PU shift-left weight
                    for _ in range(0, wq.shape[0], T):
                        q = self.q_adc(raw[k]) * w_d
                        acc = q if acc is None else acc + q
                        k += 1
                y = acc * conv + (self.bias - shift * wq.sum(0))

            y = y.transpose(1, 2).reshape(B, sp.c_out, side, side)
            if self.bn is not None:
                y = self.bn(y.flatten(2)).reshape(B, sp.c_out, side, side)
            if sp.act:                     # SiLU follows every activated conv,
                y = F.silu(y)              # with or without a batch-norm before it
            return y

        # -- range calibration ------------------------------------------

        @torch.no_grad()
        def freeze_scales(self):
            """Stop adapting the converter ranges.

            They are observed from activations that depend on the previous layer's
            range, so leaving them adapting couples every layer's scale to every
            other one and the stack drifts.  Freezing matches deployment.
            """
            self.frozen.fill_(1.0)

        @torch.no_grad()
        def _track_scales(self, xs):
            """Track the DAC range: set on the first batch, then EMA until frozen."""
            if bool(self.frozen.item()):
                return
            flat = xs.flatten().float()
            obs = float(torch.quantile(
                flat[:: max(1, flat.numel() // 100000)], 0.999)) * self.hw.scale_margin
            obs = max(obs, 1e-9)
            if not bool(self.calibrated.item()):
                self.in_scale.fill_(obs)
                self.calibrated.fill_(1.0)
            else:
                m = self.hw.scale_momentum
                self.in_scale.mul_(m).add_((1 - m) * obs)

        @torch.no_grad()
        def _track_adc(self, raw_partials):
            """
            Track the per-array ADC full-scale from the partial sums observed,
            EMA'd across batches.

            It has to follow training: the weights move a long way, so a range
            calibrated on the random initialisation would saturate by the end.
            """
            if bool(self.frozen.item()):
                return
            peak = max(float(p.abs().amax()) for p in raw_partials)
            obs = max(peak * self.hw.scale_margin, 1e-9)
            if float(self.adc_scale.item()) <= 1.0 + 1e-12:      # still the init
                self.adc_scale.fill_(obs)
            else:
                m = self.hw.scale_momentum
                self.adc_scale.mul_(m).add_((1 - m) * obs)

    class Detector(nn.Module):
        """The 7-conv detector of ARCH, built from HWConv layers."""

        def __init__(self, hw: HWConfig):
            super().__init__()
            self.hw = hw
            self.layers = nn.ModuleList([HWConv(s, hw) for s in ARCH])

        def forward(self, x):
            for m in self.layers:
                x = m(x)
            return x

        @torch.no_grad()
        def load_folded(self, arrays: Dict[str, np.ndarray]) -> None:
            """Warm-start from an exported (BN-folded) weight set."""
            for m in self.layers:
                m.weight.copy_(torch.from_numpy(
                    arrays[f"{m.spec.name}.W"]).float())
                m.bias.copy_(torch.from_numpy(
                    arrays[f"{m.spec.name}.b"]).float())

        def freeze_scales(self) -> None:
            for m in self.layers:
                m.freeze_scales()

        @torch.no_grad()
        def export(self) -> Dict[str, np.ndarray]:
            """
            Fold BN into each convolution and emit plain numpy arrays.

            The chip has no batch-norm unit, and BN is a per-output-channel affine,
            so it folds into the weight columns and the bias.  What comes out is
            (W, b) per layer, the form `lower_linear_layer` maps onto tiles.
            """
            out: Dict[str, np.ndarray] = {}
            for m in self.layers:
                W = m.weight.detach().clone()
                b = m.bias.detach().clone()
                if m.bn is not None:
                    bn = m.bn
                    scale = bn.weight.detach() / torch.sqrt(
                        bn.running_var.detach() + bn.eps)
                    W = W * scale.unsqueeze(0)
                    b = (b - bn.running_mean.detach()) * scale + bn.bias.detach()
                out[f"{m.spec.name}.W"] = W.cpu().numpy().astype(np.float64)
                out[f"{m.spec.name}.b"] = b.cpu().numpy().astype(np.float64)
            return out

    model = Detector(hw)
    if init is not None:
        model.load_folded(init)
    return model


# 5. Assignment + loss

def build_targets(targets: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Anchor-free centre assignment: each ground-truth digit is owned by the grid
    cell its centre falls in, and scenes are generated so no two centres share one.

    Returns dense (B, GRID, GRID) arrays: obj mask, class id, and the regression
    targets in the network's own parameterisation (sigmoid offset, log size).
    """
    B = len(targets)
    obj = np.zeros((B, GRID, GRID), dtype=np.float32)
    cls = np.zeros((B, GRID, GRID), dtype=np.int64)
    box = np.zeros((B, 4, GRID, GRID), dtype=np.float32)
    for b, tg in enumerate(targets):
        for c, cx, cy, w, h in tg:
            gx, gy = int(cx // STRIDE), int(cy // STRIDE)
            if not (0 <= gx < GRID and 0 <= gy < GRID):
                continue
            obj[b, gy, gx] = 1.0
            cls[b, gy, gx] = int(c)
            ox = np.clip(cx / STRIDE - gx, 1e-4, 1 - 1e-4)
            oy = np.clip(cy / STRIDE - gy, 1e-4, 1 - 1e-4)
            box[b, 0, gy, gx] = np.log(ox / (1 - ox))      # inverse sigmoid
            box[b, 1, gy, gx] = np.log(oy / (1 - oy))
            box[b, 2, gy, gx] = np.log(max(w, 1.0) / STRIDE)
            box[b, 3, gy, gx] = np.log(max(h, 1.0) / STRIDE)
    return {"obj": obj, "cls": cls, "box": box}


def detection_loss(pred, tgt, torch_mod):
    """
    obj: BCE over every cell, with a positive weight - an 8x8 grid holding ~2
         digits is ~30:1 negative, and unweighted BCE just answers "nothing here"
    cls: cross-entropy on positives only
    box: L1 on the four regression channels of positives only

    Plain L1 rather than CIoU: the boxes are near-square digits on an 8x8 grid.
    """
    torch = torch_mod
    F = torch.nn.functional
    obj_t = torch.from_numpy(tgt["obj"])
    cls_t = torch.from_numpy(tgt["cls"])
    box_t = torch.from_numpy(tgt["box"])
    pos = obj_t > 0.5

    n_pos = float(obj_t.sum())
    pw = torch.tensor(
        max((obj_t.numel() - n_pos) / max(n_pos, 1.0), 1.0), dtype=pred.dtype)
    l_obj = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t, pos_weight=pw)
    if pos.any():
        cls_logits = pred[:, 5:].permute(0, 2, 3, 1)[pos]
        l_cls = F.cross_entropy(cls_logits, cls_t[pos])
        pb = pred[:, 1:5].permute(0, 2, 3, 1)[pos]
        tb = box_t.permute(0, 2, 3, 1)[pos]
        l_box = F.l1_loss(pb, tb)
    else:
        l_cls = l_box = torch.zeros((), dtype=pred.dtype)
    return l_obj + 0.5 * l_cls + 2.0 * l_box, (float(l_obj), float(l_cls),
                                               float(l_box))


# 6. Train / evaluate

# mAP needs the whole precision-recall curve, so detections are decoded at a low
# confidence; a high threshold would simply truncate the curve and under-report.
# Precision / recall are then quoted at a realistic operating point instead.
EVAL_CONF = 0.02
OPERATING_CONF = 0.30


def score_predictions(raw: List[np.ndarray], t_val: List[np.ndarray]) -> dict:
    """mAP@0.5 over the full curve, plus P/R at the operating threshold."""
    low = [decode(p, conf_thresh=EVAL_CONF) for p in raw]
    hi = [decode(p, conf_thresh=OPERATING_CONF) for p in raw]
    m = mean_average_precision(low, t_val)
    op = mean_average_precision(hi, t_val)
    m["precision"] = op["precision"]
    m["recall"] = op["recall"]
    m["operating_conf"] = OPERATING_CONF
    return m


def evaluate(model, x_val: np.ndarray, t_val: List[np.ndarray], torch_mod,
             batch: int = 64) -> dict:
    torch = torch_mod
    model.eval()
    raw = []
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch):
            xb = torch.from_numpy(x_val[i:i + batch].astype(np.float32))
            out = model(xb).cpu().numpy()
            raw.extend(out[j] for j in range(out.shape[0]))
    model.train()
    return score_predictions(raw, t_val)


def train_detector(hw: HWConfig, data, epochs: int = 12, batch: int = 64,
                   lr: float = 2e-3, seed: int = 0, init=None,
                   freeze_scales_after: int = 1, log=print) -> Tuple[object, dict]:
    """
    Train one detector (float or hardware-aware, per `hw.enabled`).

    `init` warm-starts from an exported weight set.  Hardware-aware training is a
    fine-tune of the float detector: quantised training from a random
    initialisation has to discover the task and survive a 4-bit datapath at once.

    `freeze_scales_after` stops the converter ranges adapting after that many
    epochs - see HWConv.freeze_scales.
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr, t_tr, x_va, t_va = data
    model = build_torch_model(hw, init=init)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = x_tr.shape[0]
    history = []
    best = {"map50": -1.0, "epoch": 0, "state": None}
    t0 = time.time()
    for ep in range(epochs):
        order = np.random.permutation(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = torch.from_numpy(x_tr[idx].astype(np.float32))
            tb = build_targets([t_tr[k] for k in idx])
            out = model(xb)
            loss, parts = detection_loss(out, tb, torch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        if hw.enabled and (ep + 1) >= freeze_scales_after:
            model.freeze_scales()
        m = evaluate(model, x_va, t_va, torch)
        history.append({"epoch": ep + 1, "loss": tot / max(nb, 1),
                        "map50": m["map50"], "precision": m["precision"],
                        "recall": m["recall"]})
        # Keep the BEST epoch, not the last.  Evaluating through the quantised
        # datapath re-draws the read noise every pass, so the score wanders by
        # several points between epochs and the final epoch is a lottery.
        if m["map50"] >= best["map50"]:
            best = {"map50": m["map50"], "epoch": ep + 1,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
        log(f"    epoch {ep+1:2d}/{epochs}  loss {tot/max(nb,1):6.4f}   "
            f"mAP@0.5 {m['map50']*100:5.2f} %   P {m['precision']*100:5.1f} %   "
            f"R {m['recall']*100:5.1f} %")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        log(f"    -> keeping epoch {best['epoch']} "
            f"(best mAP@0.5 {best['map50']*100:.2f} %)")
    final = evaluate(model, x_va, t_va, torch)
    return model, {"history": history, "final": final,
                   "best_epoch": best["epoch"], "seconds": time.time() - t0}


def save_weights(path: str, arrays: Dict[str, np.ndarray], meta: dict) -> None:
    np.savez(path, __meta__=np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8), **arrays)


def load_weights(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode("utf-8")) if "__meta__" in z else {}
    return ({k: z[k] for k in z.files if k != "__meta__"}, meta)


def get_or_train(retrain: bool = False, epochs: int = 12, n_train: int = 4000,
                 n_val: int = 1000, log=print) -> Dict[str, dict]:
    """
    Train (or load) BOTH detectors: the float baseline and the hardware-aware
    one.  Same architecture, same data, same schedule, same seed - the only
    difference is whether the chip's arithmetic was inside the forward pass.
    """
    todo = [k for k, p in (("float", WEIGHTS_PATH_FLOAT),
                           ("hwat", WEIGHTS_PATH_HWAT))
            if retrain or not os.path.exists(p)]
    out: Dict[str, dict] = {}
    for key, path in (("float", WEIGHTS_PATH_FLOAT), ("hwat", WEIGHTS_PATH_HWAT)):
        if key not in todo:
            w, m = load_weights(path)
            out[key] = {"weights": w, "meta": m}
            log(f"  loaded cached {key} detector ({path})")

    if not todo:
        return out

    log(f"  building the task: {n_train} train / {n_val} val scenes "
        f"({IMG_SIZE}x{IMG_SIZE}, up to {MAX_OBJECTS} digits each)")
    data = build_dataset(n_train, n_val)
    log(f"  detector: {arch_summary()}")

    if "float" in todo:
        log("\n  FLOAT baseline (no hardware in the loop)")
        m_f, r_f = train_detector(HWConfig(enabled=False), data, epochs=epochs,
                                  log=log)
        w_f = m_f.export()
        save_weights(WEIGHTS_PATH_FLOAT, w_f, r_f)
        out["float"] = {"weights": w_f, "meta": r_f}

    if "hwat" in todo:
        log("\n  HARDWARE-AWARE fine-tune of the float detector (4-bit "
            "sign-magnitude weights,\n  per-tile 4-bit ADC, 8-bit digit-serial "
            "nibbles, device spread + read noise)")
        # BN is already folded into the warm-start weights, so the fine-tuned
        # model runs in exactly the deployed form: conv -> bias -> SiLU.
        hw = HWConfig.from_simulation_config(enabled=True, use_bn=False)
        m_h, r_h = train_detector(hw, data, epochs=epochs, lr=5e-4,
                                  init=out["float"]["weights"], log=log)
        w_h = m_h.export()
        r_h["hw_config"] = {k: v for k, v in vars(hw).items()}
        save_weights(WEIGHTS_PATH_HWAT, w_h, r_h)
        out["hwat"] = {"weights": w_h, "meta": r_h}
    return out


def main() -> None:
    argv = sys.argv[1:]
    epochs = 12
    for i, a in enumerate(argv):
        if a == "--epochs" and i + 1 < len(argv):
            epochs = int(argv[i + 1])
    print("=" * 88)
    print("STRATUM detection task  -  training the float and hardware-aware "
          "detectors")
    print("=" * 88)
    res = get_or_train(retrain="--retrain" in argv, epochs=epochs)
    print("\n" + "-" * 88)
    print(f"  {'model':28s}{'mAP@0.5':>10s}{'precision':>12s}{'recall':>10s}"
          f"{'train (s)':>12s}")
    for key, label in (("float", "float-trained"),
                       ("hwat", "hardware-aware trained")):
        m = res[key]["meta"]
        f = m.get("final", {})
        print(f"  {label:28s}{f.get('map50',0)*100:>9.2f}%"
              f"{f.get('precision',0)*100:>11.1f}%{f.get('recall',0)*100:>9.1f}%"
              f"{m.get('seconds',0):>12.1f}")
    print("\n  (these are SOFTWARE scores; stratum_detect_imc.py deploys both "
          "on the chip)")


if __name__ == "__main__":
    main()
