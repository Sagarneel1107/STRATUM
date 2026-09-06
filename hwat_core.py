"""
hwat_core.py

Shared hardware-aware training layers: one differentiable definition of the
chip's arithmetic, used by every hardware-aware study in the toolkit.

`HWConv` encodes three corrections any new network must inherit:

  1. Weight quantisation is PER OUTPUT COLUMN, in training and deployment.
     Batch-norm folding leaves a layer's columns differing by orders of
     magnitude, so one global scale strands the quiet columns in the bottom few
     states.  Must match `lower_linear_layer(..., per_column_scale=True)`.
  2. The ADC straight-through estimator STOPS AT THE CLAMP.  Reaching through it
     hands identity gradient back for saturated partial sums, so only the
     rounding gets the straight-through treatment.
  3. The ADC is PER ARRAY TILE, applied to each 128-row tile's partial sum before
     the partials are added.

`resnet_vgg_hwat.py` uses these classes directly; new studies should import them
rather than growing an inline twin.  HWConfig comes from `stratum_detect`.
"""

from __future__ import annotations

import math
from typing import Optional

from stratum_detect import HWConfig     # re-exported: one config, one meaning

__all__ = ["HWConfig", "make_hw_layers", "pick_device"]


def pick_device(prefer: str = "auto"):
    """Resolve a torch device; CPU when CUDA is absent so smoke tests still run."""
    import torch
    if prefer in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def enable_tf32(enable: bool = True) -> bool:
    """
    Let the per-tile einsum use tensor cores on Ampere and newer.

    `allow_tf32` defaults to False in torch 2.x, so the batched per-tile matmul
    otherwise runs in strict FP32.  TF32's 10-bit mantissa is ample here: the
    partial sums go straight into a 4-bit ADC and the weights hold 16 states.

    Returns True when it actually applied (CUDA present).
    """
    import torch
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    return bool(enable)


def make_hw_layers(hw: HWConfig):
    """
    Build (HWConv, HWLinear) against the installed torch.

    Defined inside a function so this module imports on a machine without
    PyTorch, exactly as `stratum_detect` does.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def ste(q, x):
        """Straight-through: forward the quantised value, backward the identity."""
        return x + (q - x).detach()

    def lognormal_factor(cv: float, *shape):
        """
        Multiplicative conductance variation: positive, unit mean, CV = `cv`.

        Matches `single_memristor_nonidealities._lognormal_factor` exactly, so the
        training forward and the array it trains for share one device statistic.
        """
        cv = float(cv)
        if cv <= 0:
            return torch.ones(*shape)
        sigma_ln = math.sqrt(math.log(1.0 + cv ** 2))
        mu_ln = -0.5 * sigma_ln ** 2
        return torch.exp(mu_ln + sigma_ln * torch.randn(*shape))

    def _act(y, kind: str):
        if kind == "relu":
            return F.relu(y)
        if kind == "silu":
            return F.silu(y)
        return y

    class _HWBase(nn.Module):
        """Quantisers shared by the conv and linear forms."""

        def __init__(self, fan_in: int, c_out: int, hw: HWConfig):
            super().__init__()
            self.hw = hw
            self.fan_in, self.c_out = fan_in, c_out
            self.weight = nn.Parameter(
                torch.randn(fan_in, c_out) * (2.0 / fan_in) ** 0.5)
            self.bias = nn.Parameter(torch.zeros(c_out))
            self.register_buffer("in_scale", torch.ones(1))
            # PER-COLUMN ADC full-scale, not a scalar: the array has one converter
            # per output column, each with its own reference.  It matters most once
            # batch-norm is folded into the weights, which multiplies output column
            # j by gamma_j/sqrt(var_j+eps) - factors spanning orders of magnitude
            # that a single shared range cannot hold (86.55% unfolded vs 10.00%
            # folded on VGG-16/CIFAR-10).
            self.register_buffer("adc_scale", torch.ones(c_out))
            self.register_buffer("calibrated", torch.zeros(1))
            self.register_buffer("frozen", torch.zeros(1))
            # Steps of range tracking done so far.  Tracking calls `torch.quantile`
            # every step (20.3% of a training step) and is waste once the EMA has
            # settled, so the scales freeze after `calib_steps`.  Set 0 to track
            # forever.
            self.register_buffer("calib_steps_done", torch.zeros(1))
            # Device-to-device spread drawn ONCE, like programming a real array.
            # Lognormal, because a conductance is positive and its spread is
            # multiplicative - see `lognormal_factor`.
            self.register_buffer(
                "d2d", lognormal_factor(hw.d2d_sigma, fan_in, c_out))
            # Write/programming scatter, also frozen here.  A separate buffer from
            # d2d: it is redrawn on every reprogramming event, while d2d is a
            # property of the die.
            self.register_buffer(
                "c2c", lognormal_factor(getattr(hw, "c2c_sigma", 0.0), fan_in, c_out))
            # Stuck-at defects: which cells, and what they are pinned to.
            # `stuck_hi` marks the LRS ones (pinned to full-scale magnitude);
            # the rest are HRS (pinned to zero magnitude).
            p_stuck = float(getattr(hw, "stuck_at_prob", 0.0) or 0.0)
            self.register_buffer(
                "stuck_mask", (torch.rand(fan_in, c_out) < p_stuck))
            self.register_buffer(
                "stuck_hi",
                (torch.rand(fan_in, c_out) < float(getattr(hw, "stuck_at_lrs_frac", 0.5))))

        # -- retention --------------------------------------------------

        def retention_survival(self) -> float:
            """
            Fraction of the programmed conductance still there at `retention_time_s`.

            Kohlrausch stretched exponential toward the g_min floor.  The grid is
            linear from g_min and the floor is g_min, so this collapses to a plain
            gain S on the weight magnitude.  The sign is a separate STT-MRAM cell
            and never drifts, so a weight loses amplitude but cannot flip.
            """
            hw = self.hw
            t = float(getattr(hw, "retention_time_s", 0.0) or 0.0)
            if t <= 0:
                return 1.0
            tau = float(getattr(hw, "retention_tau_s", 36000.0))
            beta = float(getattr(hw, "retention_beta", 0.5))
            model = str(getattr(hw, "retention_model", "kohlrausch"))
            if model == "none":
                return 1.0
            if model == "exponential":
                return math.exp(-t / tau)
            if model == "power_law":
                return (1.0 + t / tau) ** (-beta)
            return math.exp(-((t / tau) ** beta))          # kohlrausch

        # -- quantisers -------------------------------------------------

        def q_weight(self, w):
            """Sign-magnitude onto N conductance levels, scaled PER OUTPUT COLUMN.

            FIX 1.  The magnitude lands on the device's `weight_levels` states;
            the SIGN is a separate physical element (the STT-MRAM sign cell), so
            a 16-state device carries 31 distinct signed values, not 16.
            """
            hw = self.hw
            wmax = w.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)   # (1, c_out)
            step = wmax / (hw.weight_levels - 1)
            # MAGNITUDE on the conductance grid; the sign is a separate cell.
            mag = torch.round(w.abs() / step).clamp(0, hw.weight_levels - 1) * step

            # Everything below perturbs the STORED CONDUCTANCE.  Retention relaxes
            # the magnitude toward g_min, which in the weight domain is a uniform
            # gain (see HWConfig.retention_time_s).
            surv = self.retention_survival()
            if surv < 1.0:
                mag = mag * surv
            # Persistent die spread, then the write scatter of this programming
            # event.  Both multiply the stored conductance.
            if hw.d2d_sigma > 0:
                mag = mag * self.d2d
            if getattr(hw, "c2c_sigma", 0.0) > 0:
                mag = mag * self.c2c
            # Stuck-at overrides whatever the programming achieved: LRS pins the
            # cell at full-scale magnitude, HRS at zero.  A gross error, not a
            # small perturbation - it does not average away across a tile.
            if float(getattr(hw, "stuck_at_prob", 0.0) or 0.0) > 0:
                pinned = torch.where(self.stuck_hi, wmax.expand_as(mag),
                                     torch.zeros_like(mag))
                mag = torch.where(self.stuck_mask, pinned, mag)
            # Transient read noise, redrawn per read.  Historically training
            # only; `read_noise_at_inference` extends it to a deployed read.
            # Lognormal, for the same reason as d2d/c2c.
            if hw.read_sigma > 0 and (self.training
                                      or getattr(hw, "read_noise_at_inference", False)):
                cv = float(hw.read_sigma)
                s_ln = math.sqrt(math.log(1.0 + cv ** 2))
                mag = mag * torch.exp(-0.5 * s_ln ** 2 + s_ln * torch.randn_like(mag))

            q = mag * torch.sign(w)
            return ste(q, w)

        def to_nibbles(self, x):
            """Activations -> unsigned 8-bit codes -> radix-16 digit-serial nibbles."""
            hw = self.hw
            s = self.in_scale.clamp_min(1e-9)
            n = 2 ** hw.input_bits - 1
            codes_f = (x / s).clamp(0, 1) * n
            codes = ste(torch.round(codes_f), codes_f)
            radix, dmax = 16, 15
            lo = torch.remainder(codes, radix)
            hi = torch.div(codes - lo, radix)
            return [lo / dmax, hi / dmax], s * dmax / n

        def q_adc(self, partial):
            """Bipolar ADC on ONE array's partial sum, before partials are added.

            FIX 2.  `r` is clamped BEFORE the straight-through round, so the
            gradient is zero outside the converter's range - which is the truth,
            a saturated ADC really is insensitive to its input.
            """
            hw = self.hw
            # (..., c_out) partial sums against a (c_out,) full-scale: the last
            # dimension broadcasts, giving every column its own converter range.
            s = self.adc_scale.clamp_min(1e-9)
            n = 2 ** (hw.adc_bits - 1)
            r = (partial / s).clamp(-1, 1)          # grad is 0 where saturated
            return ste(torch.round(r * n) / n, r) * s

        # -- range calibration ------------------------------------------

        @torch.no_grad()
        def freeze_scales(self):
            self.frozen.fill_(1.0)

        @torch.no_grad()
        def _track_scales(self, xs):
            if bool(self.frozen.item()):
                return
            flat = xs.detach().flatten().float()
            step = max(1, flat.numel() // 100000)
            obs = float(torch.quantile(flat[::step], 0.999)) * self.hw.scale_margin
            obs = max(obs, 1e-9)
            if not bool(self.calibrated.item()):
                self.in_scale.fill_(obs)
                self.calibrated.fill_(1.0)
            else:
                m = self.hw.scale_momentum
                self.in_scale.mul_(m).add_((1 - m) * obs)

        @torch.no_grad()
        def _track_adc(self, raw):
            if bool(self.frozen.item()):
                return
            # Peak per OUTPUT COLUMN (the last axis), not one peak for the layer.
            flat = raw.detach().abs().reshape(-1, raw.shape[-1])
            obs = (flat.amax(dim=0) * self.hw.scale_margin).clamp_min(1e-9)
            if bool((self.adc_scale <= 1.0 + 1e-12).all()):
                self.adc_scale.copy_(obs)
            else:
                m = self.hw.scale_momentum
                self.adc_scale.mul_(m).add_((1 - m) * obs)

        # -- the chip's GEMM --------------------------------------------

        def _crossbar_ni(self, raw):
            """
            CROSSBAR-level non-idealities on one tile's partial sums.

            `raw` is (D, B, L, n_tiles, c_out) in the normalised domain: each entry
            is `sum_i (nibble_i/15) * |w|_i` over one 128-row tile.  The array
            produces a current, so the terms are applied in the current domain and
            mapped back: with `v_i = (nib_i/15) * V_read` and
            `G_ij = g_min + |w_ij|/wmax * (g_max - g_min)`, a normalised partial `r`
            is `I = V_read * (g_max - g_min)/wmax * r` plus the g_min common mode.

            Modelled: wire IR drop (first-order, one pass), parasitic and
            access-transistor series R, self-heating through the TCR, the column
            current ceiling, and the physical sense noise floors.  Converter
            gain/offset ride on `adc_scale` in `q_adc`.

            Applied under `no_grad` and folded back with a straight-through
            estimator, so the forward value changes without a spurious gradient.
            """
            ni = getattr(self.hw, "crossbar_ni", None)
            if ni is None or not getattr(ni, "enabled", False):
                return raw

            v_read = float(getattr(self.hw, "read_voltage_v", 0.4))
            g_lo = float(getattr(self.hw, "g_min_s", 5e-8))
            g_hi = float(getattr(self.hw, "g_max_s", 8e-7))
            t_read = float(getattr(self.hw, "read_time_s", 7.2e-9))
            rows = int(self.hw.tile_rows)

            with torch.no_grad():
                # Normalised -> current.  `raw` already carries (nib/15)*|w|/wmax
                # implicitly, so one scale factor converts the whole tensor.
                k = v_read * (g_hi - g_lo)
                cur = raw * k
                out = cur

                # Series resistance (contact/via + access transistor) compresses the
                # large states most: G_eff = G/(1 + G*Rs).  At the column level the
                # first-order effect is a gain of 1/(1 + G_col*Rs).
                r_s = (float(ni.parasitic_series_resistance_ohm)
                       + float(ni.access_transistor_on_resistance_ohm))
                if r_s > 0:
                    g_col = rows * 0.5 * (g_lo + g_hi)
                    out = out / (1.0 + g_col * r_s)

                # Self-heating through the TCR: dT = R_th,cell*P_cell + R_th,arr*P_arr.
                tcr = float(ni.temperature_coefficient_per_k)
                if tcr != 0.0:
                    p_col = out.abs() * v_read          # ~ I*V dissipated per column
                    d_t = (float(ni.thermal_resistance_cell_k_per_w) * p_col / max(rows, 1)
                           + float(ni.thermal_resistance_array_k_per_w) * p_col)
                    if float(ni.max_temperature_rise_k) > 0:
                        d_t = d_t.clamp(max=float(ni.max_temperature_rise_k))
                    out = out * (1.0 + tcr * d_t)

                # Wire IR drop, first order: the line integral of the drop over the
                # column, i.e. r * rows^2/2 * I_avg, expressed as a column gain.
                r_w = float(ni.wire_resistance_ohm)
                if r_w > 0:
                    g_col = rows * 0.5 * (g_lo + g_hi)
                    out = out / (1.0 + 0.5 * rows * r_w * g_col)

                # Column current ceiling.
                sat = ni.current_saturation_a
                if sat is not None and sat > 0:
                    out = out.clamp(-float(sat), float(sat))

                # Sense noise.  Time-independent floors first, then the two physical
                # ones that fall as 1/sqrt(t_read) - which is what makes a faster
                # read cost effective bits instead of being free.
                var = torch.zeros_like(out)
                if float(ni.read_noise_current_a) > 0:
                    var = var + rows * float(ni.read_noise_current_a) ** 2
                if float(ni.read_noise_relative) > 0:
                    var = var + (float(ni.read_noise_relative) * out.abs()) ** 2
                if getattr(ni, "physical_sense_noise", False) and t_read > 0:
                    q, k_b = 1.602176634e-19, 1.380649e-23
                    g_col = rows * 0.5 * (g_lo + g_hi)
                    var = var + q * out.abs() / t_read                       # shot
                    var = var + 2.0 * k_b * float(ni.sense_temperature_k) * g_col / t_read
                if bool((var > 0).any()):
                    out = out + torch.sqrt(var) * torch.randn_like(out)

                delta = (out - cur) / k          # back to the normalised domain

            # Forward gets the perturbed value; the gradient flows through `raw`.
            return raw + delta

        def _hw_matmul(self, cols):
            """
            (B, L, fan_in) @ quantised weights, the way the array does it.

            FIX 3.  One analog read per (digit, array tile); each read is
            digitised on its OWN converter before the PU shift-and-add combines
            them.  The tiles are independent so they batch into one einsum.
            """
            hw = self.hw
            # The DAC is unsigned: shift the activation non-negative and let the
            # bias carry the exact -q*colsum(W) correction (on the chip that
            # rides the bias row, sensed once and latched).
            shift = (-cols.amin()).clamp_min(0.0).detach()
            xs = cols + shift
            tracking = self.training and not bool(self.frozen.item())
            if tracking:
                self._track_scales(xs)
                # Self-freeze once the ranges have had enough steps to settle.
                limit = int(getattr(hw, "calib_steps", 0) or 0)
                if limit:
                    self.calib_steps_done += 1
                    if float(self.calib_steps_done.item()) >= limit:
                        self.freeze_scales()
            nibbles, conv = self.to_nibbles(xs)
            wq = self.q_weight(self.weight)

            T = hw.tile_rows
            fan_in, c_out = wq.shape
            n_tiles = (fan_in + T - 1) // T
            pad = n_tiles * T - fan_in
            wq_p = F.pad(wq, (0, 0, 0, pad)) if pad else wq
            wq_t = wq_p.view(n_tiles, T, c_out)
            nib = torch.stack(nibbles, dim=0)              # (D, B, L, fan_in)
            if pad:
                nib = F.pad(nib, (0, pad))
            nib_t = nib.view(*nib.shape[:3], n_tiles, T)
            raw = torch.einsum("dblnt,nto->dblno", nib_t, wq_t)
            # The array between the row drivers and the sense amplifiers: IR drop,
            # series R, self-heating, the current ceiling and the sense-noise floors.
            raw = self._crossbar_ni(raw)
            if tracking:
                self._track_adc(raw)
            w_d = torch.tensor([16.0 ** d for d in range(len(nibbles))],
                               dtype=raw.dtype, device=raw.device)
            acc = (self.q_adc(raw) * w_d.view(-1, 1, 1, 1, 1)).sum(dim=(0, 3))
            # The bias rides the bias ROW of this array, so it is stored in the
            # same devices and relaxes with them.  Leaving it fixed while the
            # magnitudes fade lets a constant offset take over the layer.
            b = self.bias
            if getattr(hw, "bias_on_array", True):
                surv = self.retention_survival()
                if surv < 1.0:
                    b = b * surv
            return acc * conv + (b - shift * wq.sum(0))

    class HWConv(_HWBase):
        """One convolution as the chip performs it (im2col -> tiled VMM -> ADC)."""

        def __init__(self, name: str, c_in: int, c_out: int, k: int, stride: int,
                     hw: HWConfig, act: str = "relu", use_bn: bool = True,
                     padding: Optional[int] = None):
            super().__init__(c_in * k * k, c_out, hw)
            self.name, self.c_in = name, c_in
            self.k, self.stride, self.act_kind = k, stride, act
            self.padding = k // 2 if padding is None else int(padding)
            self.bn = nn.BatchNorm2d(c_out) if use_bn else None

        def forward(self, x):
            B, _, H, W = x.shape
            p = self.padding
            oh = (H + 2 * p - self.k) // self.stride + 1
            ow = (W + 2 * p - self.k) // self.stride + 1
            cols = F.unfold(x, self.k, padding=p, stride=self.stride).transpose(1, 2)
            y = (cols @ self.weight + self.bias) if not self.hw.enabled \
                else self._hw_matmul(cols)
            y = y.transpose(1, 2).reshape(B, self.c_out, oh, ow)
            if self.bn is not None:
                y = self.bn(y)
            return _act(y, self.act_kind)

    class HWLinear(_HWBase):
        """One fully-connected layer as the chip performs it."""

        def __init__(self, name: str, c_in: int, c_out: int, hw: HWConfig,
                     act: str = "none", use_bn: bool = False):
            super().__init__(c_in, c_out, hw)
            self.name, self.c_in = name, c_in
            self.act_kind = act
            self.bn = nn.BatchNorm1d(c_out) if use_bn else None

        def forward(self, x):
            x = x.flatten(1)
            if not self.hw.enabled:
                y = x @ self.weight + self.bias
            else:
                y = self._hw_matmul(x.unsqueeze(1)).squeeze(1)
            if self.bn is not None:
                y = self.bn(y)
            return _act(y, self.act_kind)

    return HWConv, HWLinear
