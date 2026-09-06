"""
imc_kernels.py

The model-agnostic primitives of the IMC datapath - they describe the CHIP, not
any particular network:

    im2col                      unroll a patch onto the crossbar rows
    silu                        the activation the datapath applies
    SCALE_MARGIN                converter headroom over the calibrated range
    ADC_CHOICE_BY_BITS          which hardware_peripherals ADC gives a resolution
    crossbar_factory
    conductance_levels          divide a measured range into programmable states
    precompute_layer_bias       sense a layer bias once and fold it into every
    run_layer_batched           following patch with a digital PUADD

Every study imports them from here, so they share one definition of the datapath
and their numbers stay commensurable.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from riscv_imc_workflow import (
    IMCProcessor, ExecutionCost, LinearLayerPlan, DIGITAL_CLOCK_MHZ,
)


# 1. Tensor primitives (numpy; mirrors what the chip's dataflow produces)

def im2col(x: np.ndarray, k: int, s: int, p: int) -> Tuple[np.ndarray, int, int]:
    """
    (c, h, w) -> ((n_patches, c*k*k), out_h, out_w).

    Row-major over output pixels; each row is one patch vector, exactly the
    vector the crossbar's rows are driven with.
    """
    c, h, w = x.shape
    if p:
        x = np.pad(x, ((0, 0), (p, p), (p, p)))
    oh = (h + 2 * p - k) // s + 1
    ow = (w + 2 * p - k) // s + 1
    # stride tricks: (oh, ow, c, k, k) view, then flatten to patch vectors
    sc, sh, sw = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(oh, ow, c, k, k), strides=(sh * s, sw * s, sc, sh, sw),
        writeable=False)
    return view.reshape(oh * ow, c * k * k), oh, ow


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# 2. Batched layer primitives

# The two layer primitives of the batched datapath: a layer bias is sensed once
# and latched, then folded into every following patch with a digital PUADD rather
# than a fresh row-mode analog read per patch.  Numerically identical to
# `IMCProcessor.run_layer`, since the bias is input-independent.

def precompute_layer_bias(proc: IMCProcessor, layer: LinearLayerPlan,
                          fixed_cost: Optional[ExecutionCost]
                          ) -> Dict[int, np.ndarray]:
    """Sense every bias tile of `layer` once; charge that one-time read to
    `fixed_cost` and return the latched bias vectors."""
    cached: Dict[int, np.ndarray] = {}
    for ti, tile in enumerate(layer.tiles):
        if tile.bias_weights is not None:
            cached[ti] = proc._row_mode_bias(tile, fixed_cost)
    return cached


def run_layer_batched(proc: IMCProcessor, layer: LinearLayerPlan, x: np.ndarray,
                      cached_bias: Dict[int, np.ndarray],
                      cost: Optional[ExecutionCost]) -> np.ndarray:
    """Run one layer with the bias added from the cached vector (one digital
    PUADD) instead of a fresh row-mode analog read."""
    y = np.zeros(layer.out_features, dtype=float)
    for ti, tile in enumerate(layer.tiles):
        x_tile = x[tile.row_slice]
        partial = proc._matrix_vmm(tile, x_tile, layer.in_scale, accumulate_cost=cost)
        y[tile.col_slice] += partial
        if ti in cached_bias:
            y[tile.col_slice] += cached_bias[ti]
            if cost is not None:               # fold cached bias in: one PUADD
                cost.add(proc._shift_add_cost(tile.out_count, shift=False, add=True))

    if cost is not None and layer.row_tiles > 1:
        cost.add(proc._reduce_add_cost(layer.out_features, layer.row_tiles))

    if layer.has_relu:
        y = np.maximum(y, 0.0)
        if cost is not None:
            cost.add(proc._elementwise_cost(layer.out_features))
    return y


# 3. Converter selection and device levels

SCALE_MARGIN = 1.25          # ADC/DAC headroom over the calibrated 99.9th pct


# hardware_peripherals ADC library: choice -> resolution (simulation_config:89).
ADC_CHOICE_BY_BITS = {4: 7, 8: 2, 10: 1}


def crossbar_factory(adc_choice: int) -> Callable[[], object]:
    """A crossbar template identical to the workflow's but with a chosen ADC."""
    def make():
        from simulation_config import build_crossbar_config, build_device as _bd
        from integrated_crossbar import IntegratedMemristorCrossbar
        from riscv_imc_workflow import (
            DAC_CHOICE_4BIT, DEFAULT_INPUT_PRECISION_BITS,
        )
        cfg = build_crossbar_config(
            adc_choice=adc_choice, dac_choice=DAC_CHOICE_4BIT,
            digital_frequency_mhz=DIGITAL_CLOCK_MHZ,
            input_precision_bits=DEFAULT_INPUT_PRECISION_BITS,
            name=f"acc_adc{adc_choice}")
        return IntegratedMemristorCrossbar(config=cfg, device=_bd())
    return make


def conductance_levels(n_states: int, base_levels) -> np.ndarray:
    """
    `n_states` programmable conductance levels spanning the device's measured
    range.  16 states is the 4-bit device the toolkit models; more states stand
    for a finer device (or several devices ganged per weight).
    """
    lo, hi = float(np.min(base_levels)), float(np.max(base_levels))
    return np.linspace(lo, hi, int(n_states))

