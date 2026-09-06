"""
resnet_vgg_study.py

Full-chip study for ResNet-18 and VGG-16: accuracy, array footprint, TOPS and
TOPS/W on the STRATUM chip, with and without hardware-aware training.

Accuracy comes from `resnet_vgg_hwat`, whose hardware-aware forward pass is the
chip datapath, so both regimes are scored through the same arithmetic.  Cost
comes from `riscv_imc_workflow.IMCProcessor._analog_read_cost`, the same
primitive every other study uses.  Every number is read-path: `fits` says when a
network has to be time-multiplexed through weight groups, but the write model
behind that is not calibrated, so its cost is not reported.

    python resnet_vgg_study.py                       both nets, CIFAR + ImageNet
    python resnet_vgg_study.py --net resnet18
    python resnet_vgg_study.py --no-accuracy         cost only, no torch needed
    python resnet_vgg_study.py --adc-bits 4 8 10

One command refreshes every artifact from a finished GPU run:

    python resnet_vgg_study.py --net both --img-sizes 32 224 \
        --weights-dir "out_log_22nd_Aug_ResNet;out_log_21st_Aug_VGG" \
        --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import simulation_config as sc
import riscv_imc_workflow as rw
from resnet_vgg_hwat import describe, weights_path, load_weights
from imc_kernels import ADC_CHOICE_BY_BITS, crossbar_factory

def chip_sizes():
    """
    The chip to cost: the one configured on the platform's 'Chip' tab.

    Read at call time, not at import, because the GUI writes these back into
    simulation_config just before the run.
    """
    t = int(sc.CHIP_NUM_TILES)
    p = int(sc.TILE_NUM_PUS)
    a = int(sc.PU_NUM_ARRAYS)
    return [(f"{t}x{p}x{a}", t, p, a)]

DEFAULT_ADC_BITS = [4, 8, 10]

# The resolution the trained weights are for; `resnet_vgg_hwat.build_model`
# always builds the CIFAR variant.  224 px is a cost projection only.
TRAINED_IMG_SIZE = 32


# 1. Mapping + cost

def map_and_cost(spec, adc_bits: int, sparse_modes: bool = True,
                 input_sparsity=0.5) -> dict:
    """
    Map every layer onto 128x128 arrays and cost one inference.

    `input_sparsity` is the fraction of activation rows that are zero after ReLU;
    a zero row drives no DAC and no cell, which is what the ISA's row decoder
    exploits.  Either a scalar (assumed, applied to every ReLU-fed layer) or a
    dict keyed by layer name (measured, from `measure_activation_sparsity`).
    """
    if adc_bits not in ADC_CHOICE_BY_BITS:
        raise ValueError(
            f"adc_bits={adc_bits} has no entry in ADC_CHOICE_BY_BITS "
            f"(available: {sorted(ADC_CHOICE_BY_BITS)}). The peripheral lookup "
            f"table holds real converters, so only those resolutions exist.")
    proc = rw.IMCProcessor(crossbar_factory=crossbar_factory(ADC_CHOICE_BY_BITS[adc_bits]),
                           sparse_modes=sparse_modes)
    cfg = proc.array.config
    R, C = cfg.rows, cfg.cols
    # Representative PROGRAMMED conductance.  `average_conductance_s` on this
    # template is the device's lowest state, which understates the read by 8.5x.
    V, G = cfg.read_voltage_v, proc.array.device_mean_conductance_s

    # Which layers see a DENSE input.  The stem reads the image, not a ReLU
    # output, so nothing is gated there; everything else is fed by an activation.
    dense_input = {"stem", "c0"}

    def sparsity_of(name: str) -> float:
        """Zero fraction at this layer's input: measured if given, else assumed."""
        if isinstance(input_sparsity, dict):
            # A layer with no measurement gets none assumed for it.
            return float(input_sparsity.get(name, 0.0))
        return 0.0 if name in dense_input else float(input_sparsity)

    cost = rw.ExecutionCost()
    arrays = 0
    cells = 0
    digital_ew = 0                  # PU elementwise ops (ReLU + residual adds)
    movement_energy_j = 0.0         # NoC + staging-buffer energy (see below)
    for (name, c_in, c_out, k, stride, out_hw) in spec.layers:
        K = c_in * k * k            # im2col fan-in
        M = c_out
        N = out_hw * out_hw         # output positions
        row_tiles = math.ceil(K / R)
        col_tiles = math.ceil(M / C)
        arrays += row_tiles * col_tiles
        cells += K * M
        sp = sparsity_of(name)
        # Cost one output position, then scale by the N positions in the feature
        # map: every position is its own VMM against the same programmed array.
        per_pos = rw.ExecutionCost()
        for rt in range(row_tiles):
            nr_full = min(R, K - rt * R)
            # Sparsity gates rows off; the stem sees a dense image, not ReLU output.
            nr = max(1, int(round(nr_full * (1.0 - sp))))
            for ct in range(col_tiles):
                nc = min(C, M - ct * C)
                one = proc._analog_read_cost(
                    np.full(nr, V), np.full((nr, nc), G), nc,
                    n_addressed_cells=nr_full * nc)
                # 8-bit activations -> two 4-bit nibble reads + one shift&add
                for _ in range(2):
                    per_pos.add(one)
                per_pos.add(proc._shift_add_cost(nc, shift=True, add=True))
        if row_tiles > 1:           # cross-tile accumulation
            per_pos.add(proc._reduce_add_cost(M, row_tiles))
        # Charge the activation this layer actually has.  A ResNet block's second
        # conv and 1x1 downsample feed the residual adder; the ReLU is after the add.
        if spec.act_of(name) != "none":
            per_pos.add(proc._elementwise_cost(M))
            digital_ew += M * N

        # Data movement for this layer, charged in energy as well as time:
        #   im2col   each output position needs its c_in*k*k fan-in on the row
        #            drivers; a line buffer keeps only k*c_in words fresh per step,
        #            plus a window restart on each output row.
        #   output   the M activations produced per position, staged for whichever
        #            arrays hold the next layer.
        words_in = N * k * c_in + out_hw * K
        hops = proc.noc_hops_for(row_tiles * col_tiles)
        move = proc.movement_cost(words_in * _ACT_BITS, hops=hops)
        move.add(proc.movement_cost(N * M * _ACT_BITS, hops=hops))
        movement_energy_j += move.energy_j
        cost.energy_j += move.energy_j
        cost.moved_bits += move.moved_bits

        cost.energy_j += per_pos.energy_j * N
        cost.latency_ns += per_pos.latency_ns * N
        cost.n_vmm += per_pos.n_vmm * N
        cost.n_shift_add += per_pos.n_shift_add * N
        cost.n_analog_reads += per_pos.n_analog_reads * N
        cost.cells_read += per_pos.cells_read * N
        cost.cells_skipped += per_pos.cells_skipped * N
        for kk, vv in per_pos.mode_counts.items():
            cost.mode_counts[kk] = cost.mode_counts.get(kk, 0) + vv * N

    # The residual adds: digital PU work with no array read behind them, and the
    # whole structural difference between ResNet-18 and VGG-16 on this chip.
    # Each is one add plus the post-add ReLU over the block's output positions.
    residual_energy_j = 0.0
    for width, positions in getattr(spec, "residual_adds", []):
        add = proc._elementwise_cost(width)         # x + f(x)
        relu = proc._elementwise_cost(width)        # ReLU after the add
        residual_energy_j += (add.energy_j + relu.energy_j) * positions
        cost.energy_j += (add.energy_j + relu.energy_j) * positions
        cost.latency_ns += (add.latency_ns + relu.latency_ns) * positions
        digital_ew += 2 * width * positions

    # Area of one array at THIS converter resolution, from the crossbar template.
    # ADCs are ~94% of an array (0.068 / 0.158 / 0.209 mm^2 at 4 / 8 / 10 bits), so
    # a fixed mm^2/array would hide the cost of a wider converter.  The 2T1R access
    # transistors and the sign cell are counted too (+8.5%/array).
    array_area_um2 = (proc.array.total_area_um2
                      + proc.array_signed_overhead_um2(R, C))
    ops = 2 * spec.macs
    return {
        "adc_bits": adc_bits,
        "sparse_modes": sparse_modes,
        "arrays": arrays,
        "cells": cells,
        "macs": spec.macs,
        "array_area_um2": array_area_um2,
        "area_mm2": arrays * array_area_um2 * 1e-6,
        "energy_j": cost.energy_j,
        "reads": cost.n_analog_reads,
        "read_latency_ns": proc._read_latency_ns,
        "read_time_ns": cfg.read_time_s * 1e9,
        "tops_per_watt": ops / (cost.energy_j * 1e12) if cost.energy_j > 0 else 0.0,
        "mode_counts": dict(cost.mode_counts),
        "sparsity_exploited": cost.sparsity_exploited,
        "sparsity_measured": isinstance(input_sparsity, dict),
        "sign_storage": proc.sign_storage_label,
        # Digital PU work carried alongside the analog reads, so a reader can
        # see how much of the budget is NOT the crossbar.
        "residual_adds": len(getattr(spec, "residual_adds", [])),
        "residual_energy_j": residual_energy_j,
        "residual_energy_frac": (residual_energy_j / cost.energy_j
                                 if cost.energy_j > 0 else 0.0),
        "digital_elementwise_ops": digital_ew,
        # Data movement, reported alongside the array term rather than only summed
        # into it, so a reader can see how much of TOPS/W is the crossbar.
        "movement_energy_j": movement_energy_j,
        "movement_energy_frac": (movement_energy_j / cost.energy_j
                                 if cost.energy_j > 0 else 0.0),
        "moved_bits": cost.moved_bits,
        "array_only_energy_j": cost.energy_j - movement_energy_j,
        "tops_per_watt_array_only": (
            ops / ((cost.energy_j - movement_energy_j) * 1e12)
            if cost.energy_j > movement_energy_j else 0.0),
    }


# NOTE: the reprogramming / weight-swap study was removed.  Its write-energy model
# integrated the pulse at the conductance the cell ends at, and there is no
# write-verify loop, pulse-count model or programming latency behind it, so
# nothing here constrains the number.  Restoring it needs a calibrated write model.

# Achievable throughput: what the datapath costs once the machine has to feed
# itself.  See `achievable_cost` for what each term is and where it comes from.

# Activation word width and the on-chip transport that carries them.  Taken from
# simulation_config so this model moves when the chip does.
_ACT_BITS = int(sc.INPUT_PRECISION_BITS)                  # 8-bit activations
_NOC_BITS = int(sc.NOC_DATA_WIDTH_BITS)                   # 128-bit links/ports
_NOC_CLK_NS = 1000.0 / float(sc.NOC_FREQUENCY_MHZ)        # 1 ns at 1 GHz
_DIG_CLK_NS = 1000.0 / float(sc.CHIP_DIGITAL_FREQUENCY_MHZ)
_WORDS_PER_PORT = max(1, _NOC_BITS // _ACT_BITS)          # 16 activations/cycle/port
_DIGITS = 2                                               # 8-bit input -> 2 nibbles


def achievable_cost(spec, mapping: dict, available: int, pus_on_chip: int,
                    arrays_per_pu: int) -> dict:
    """
    What the datapath reaches once it has to feed and balance itself.

    The reported `tops` is a roofline: `ceil(reads / arrays) * t_read`, which
    assumes two things it never states, both modelled explicitly here.

      * WEIGHT REPLICATION.  An array holds one copy of a layer's weights, so its
        H*W output positions stream past the same cells one at a time.  With one
        copy each the stages are wildly unbalanced; spare arrays go to extra copies
        of the position-heavy layers.
      * PIPELINING.  Layers are sequential for one image, so a single inference has
        one layer's arrays busy at a time.  With images in flight at every stage the
        rate is the slowest stage, not the sum.

    Charged on top, all computed from the mapping:

      * IM2COL DELIVERY, double-buffered against the reads, so only the part that
        exceeds the array time is exposed (`im2col_s`).
      * ACTIVATION MOVEMENT over the NoC to the next layer's arrays.
      * REDUCTION.  A layer split across `row_tiles` arrays needs an adder-tree pass;
        the tree pipelines against the reads, so it costs its fill, not a traversal
        per position.

    Returns four latencies spanning the two structural assumptions.
    """
    t_read = float(mapping["read_latency_ns"])
    R = C = int(sc.XBAR_ROWS)
    n_ports = max(1, int(pus_on_chip))

    # ---- geometry of every layer -------------------------------------------
    layers = []
    for (name, c_in, c_out, k, stride, out_hw) in spec.layers:
        K, M, P = c_in * k * k, c_out, out_hw * out_hw
        tiles_L = math.ceil(K / R) * math.ceil(M / C)
        layers.append(dict(name=name, c_in=c_in, k=k, K=K, M=M, P=P,
                           row_t=math.ceil(K / R), tiles=tiles_L, out_hw=out_hw))
    one_copy = sum(l["tiles"] for l in layers)

    # Replication: greedy water-filling.  Each step hands one more copy of a
    # layer's weights to whichever layer that shortens the most.
    reps = {l["name"]: 1 for l in layers}
    used = one_copy
    if available > one_copy:
        while True:
            best, gain = None, 0.0
            for l in layers:
                if used + l["tiles"] > available:
                    continue
                r = reps[l["name"]]
                g = (l["P"] / r) - (l["P"] / (r + 1))
                if g > gain:
                    best, gain = l, g
            if best is None or gain <= 0:
                break
            reps[best["name"]] += 1
            used += best["tiles"]

    def walk(replicate: bool):
        """Latency terms for one image; returns (stages, totals)."""
        stages, tot = [], dict(array=0.0, im2col=0.0, move=0.0, reduce=0.0)
        for l in layers:
            r = reps[l["name"]] if replicate else 1
            live = l["tiles"] * r
            # tiles of a layer (and its replicas) fire together
            conc = max(1, min(live, available))
            waves = math.ceil(live / conc)
            # positions split across the replicas
            pos = math.ceil(l["P"] / r)
            t_array = pos * waves * _DIGITS * t_read

            n_pu = max(1, math.ceil(conc / max(1, arrays_per_pu)))
            words = l["P"] * l["k"] * l["c_in"] + l["out_hw"] * l["K"]
            t_im2col = words / (n_pu * _WORDS_PER_PORT) * _DIG_CLK_NS

            t_move = (l["P"] * l["M"] * _ACT_BITS) / (_NOC_BITS * n_ports) * _NOC_CLK_NS
            # adder tree pipelines against the reads: fill only, once per layer
            t_reduce = (math.ceil(math.log2(l["row_t"])) * _DIG_CLK_NS
                        if l["row_t"] > 1 else 0.0)

            t_layer = max(t_array, t_im2col) + t_reduce
            tot["array"] += t_array
            tot["im2col"] += max(0.0, t_im2col - t_array)   # only what is EXPOSED
            tot["move"] += t_move
            tot["reduce"] += t_reduce
            stages.append(dict(name=l["name"], replicas=r, arrays=live,
                               positions=pos, layer_ns=t_layer, move_ns=t_move,
                               array_ns=t_array, im2col_ns=t_im2col,
                               input_bound=t_im2col > t_array))
        return stages, tot

    hops = max(1.0, (2.0 * math.ceil(math.sqrt(max(1, n_ports)))) / 3.0)
    t_fill = len(layers) * hops * _NOC_CLK_NS

    out = {}
    for tag, rep in (("plain", False), ("replicated", True)):
        stages, tot = walk(rep)
        single = sum(s["layer_ns"] for s in stages) + sum(
            s["move_ns"] for s in stages) + t_fill
        pipelined = max((s["layer_ns"] + s["move_ns"]) for s in stages) if stages else 0.0
        out[f"{tag}_single_s"] = single * 1e-9
        out[f"{tag}_pipelined_s"] = pipelined * 1e-9
        if rep:
            out.update({
                "array_s": tot["array"] * 1e-9,
                "im2col_s": tot["im2col"] * 1e-9,
                "activation_move_s": tot["move"] * 1e-9,
                "reduction_stall_s": tot["reduce"] * 1e-9,
                "pipeline_fill_s": t_fill * 1e-9,
                "input_bound_layers": sum(1 for s in stages if s["input_bound"]),
                "bottleneck": max(stages, key=lambda s: s["layer_ns"] + s["move_ns"])["name"]
                              if stages else "-",
                "per_layer": stages,
            })
    out.update({
        "latency_s": out["replicated_single_s"],
        "pipelined_latency_s": out["replicated_pipelined_s"],
        "arrays_one_copy": one_copy,
        "arrays_used": used,
        "replication_factor": used / one_copy if one_copy else 1.0,
        "max_replicas": max(reps.values()) if reps else 1,
        "n_layers": len(layers),
    })
    return out


def chip_point(spec, mapping: dict, tiles: int, pus: int, per_pu: int) -> dict:
    """
    Throughput of one chip size under a mapping.

      `tops`            - the roofline: every analog read spread evenly over every
                          array the network needs, no feeding, no waiting.  It
                          describes THIS chip only when `fits` is true.
      `tops_achievable` - the same inference with the datapath's own requirements
                          charged.  See `achievable_cost`.
    """
    available = tiles * pus * per_pu
    needed = mapping["arrays"]
    concurrent = min(available, needed) or 1
    lat_s = math.ceil(mapping["reads"] / concurrent) * mapping["read_latency_ns"] * 1e-9
    ops = 2 * mapping["macs"]

    ach = achievable_cost(spec, mapping, available, tiles * pus, per_pu)
    ach_lat = ach["latency_s"]
    fits = needed <= available
    # A network larger than the chip is time-multiplexed through weight groups.
    # `fits` reports whether that is necessary; its cost is not modelled.
    return {
        "chip": f"{tiles}x{pus}x{per_pu}",
        "arrays_available": available,
        "arrays_needed": needed,
        "fits": fits,
        "compute_latency_s": lat_s,
        "fps_compute_only": 1.0 / lat_s if lat_s > 0 else 0.0,
        "tops": ops / (lat_s * 1e12) if lat_s > 0 else 0.0,
        # The roofline only describes THIS chip when the network fits on it.
        # Carried as a flag so log, table and report can label it consistently.
        "tops_is_roofline": True,
        "tops_roofline_valid_for_chip": fits,
        "tops_per_watt": mapping["tops_per_watt"],
        # --- achievable, on the arrays this chip actually has ----------------
        "achievable_latency_s": ach_lat,
        "tops_achievable": ops / (ach_lat * 1e12) if ach_lat > 0 else 0.0,
        "fps_achievable": 1.0 / ach_lat if ach_lat > 0 else 0.0,
        "tops_pipelined": (ops / (ach["pipelined_latency_s"] * 1e12)
                           if ach["pipelined_latency_s"] > 0 else 0.0),
        "fps_pipelined": (1.0 / ach["pipelined_latency_s"]
                          if ach["pipelined_latency_s"] > 0 else 0.0),
        "roofline_optimism": (ach_lat / lat_s) if lat_s > 0 else 0.0,
        # The two structural assumptions the roofline makes, each switched on and
        # off, so the roofline can be read against the corner it belongs to.
        "tops_no_replication_single": (
            ops / (ach["plain_single_s"] * 1e12) if ach["plain_single_s"] > 0 else 0.0),
        "tops_no_replication_pipelined": (
            ops / (ach["plain_pipelined_s"] * 1e12) if ach["plain_pipelined_s"] > 0 else 0.0),
        "replication_factor": ach["replication_factor"],
        "max_replicas": ach["max_replicas"],
        "arrays_one_copy": ach["arrays_one_copy"],
        "arrays_used": ach["arrays_used"],
        "bottleneck_layer": ach.get("bottleneck", "-"),
        "overheads": {k: ach[k] for k in (
            "array_s", "im2col_s", "activation_move_s", "reduction_stall_s",
            "pipeline_fill_s", "input_bound_layers", "n_layers")},
    }


def find_weights(net: str, regime: str, weights_dir: str) -> Optional[str]:
    """
    Locate `<net>_hwat_<regime>.npz`, searching several directories.

    The two networks come out of separate GPU runs, so `weights_dir` accepts a
    list separated by ';' or ','; the first directory holding the file wins.
    """
    name = weights_path(net, regime)
    for part in str(weights_dir).replace(",", os.pathsep).replace(";", os.pathsep) \
            .split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        cand = os.path.join(part, name)
        if os.path.exists(cand):
            return cand
    return None


# 2. Measured activation sparsity

def measure_activation_sparsity(net: str, regime: str = "hwat",
                                weights_dir: str = ".", img_size: int = 32,
                                n_images: int = 256, batch: int = 64,
                                device=None, log=print) -> Dict[str, float]:
    """
    The measured fraction of array ROWS driven by a zero, per layer.

    One row of a 128-row tile is driven by one element of the im2col fan-in vector;
    if that element is zero the row's DAC stays off, which is what the ISA's row
    decoder exploits.  So the quantity that matters is the zero fraction of the
    UNFOLDED input, not of the feature map - they differ at the padded borders.

    Measured on the float arithmetic (`hw.enabled = False`): what is counted is the
    sign of each pre-activation, and running the quantised datapath to count zeros
    would cost ~5x for a third-decimal difference.

    Returns {layer_name: zero_fraction}, keyed to match `describe().layers`; empty
    when torch or the weights are missing, so the caller falls back to the scalar.
    """
    try:
        import torch
        import torch.nn.functional as F
        from hwat_core import HWConfig, pick_device
        from resnet_vgg_hwat import build_model, load_cifar10, load_numpy_into
    except Exception as exc:                      # noqa: BLE001
        log(f"  (sparsity measurement skipped: {exc})")
        return {}

    path = find_weights(net, regime, weights_dir)
    if path is None:
        log(f"  (sparsity measurement skipped: no "
            f"{weights_path(net, regime)} under {weights_dir})")
        return {}

    device = device or pick_device()
    arrays, _ = load_weights(path)
    cfg = HWConfig.from_simulation_config(enabled=False, adc_bits=4)
    model = build_model(net, cfg, num_classes=10, img_size=img_size, device=device)
    if load_numpy_into(model, arrays) == 0:
        log(f"  (sparsity measurement skipped: no layer matched in {path})")
        return {}
    model.eval()

    zeros: Dict[str, float] = {}
    total: Dict[str, float] = {}

    def hook(mod, inp):
        x = inp[0]
        if hasattr(mod, "k"):                     # HWConv: count on the im2col
            cols = F.unfold(x, mod.k, padding=mod.padding, stride=mod.stride)
        else:                                     # HWLinear
            cols = x.flatten(1)
        name = getattr(mod, "name", None)
        if name is None:
            return
        zeros[name] = zeros.get(name, 0.0) + float((cols == 0).sum())
        total[name] = total.get(name, 0.0) + float(cols.numel())

    handles = [m.register_forward_pre_hook(hook)
               for m in model.modules() if hasattr(m, "in_scale")]
    try:
        _, _, xte, _ = load_cifar10(test_subset=n_images)
        with torch.no_grad():
            for i in range(0, len(xte), batch):
                xb = torch.as_tensor(xte[i:i + batch])
                if device is not None:
                    xb = xb.to(device)
                model(xb)
    finally:
        for h in handles:
            h.remove()

    out = {k: (zeros[k] / total[k] if total.get(k) else 0.0) for k in zeros}
    if out:
        mean = sum(out.values()) / len(out)
        log(f"  measured activation sparsity on {min(n_images, len(xte))} images "
            f"({regime} weights): mean {mean*100:.1f}% of rows zero, "
            f"range {min(out.values())*100:.1f}-{max(out.values())*100:.1f}%")
    return out


# 3. Accuracy (optional - needs torch + trained weights)

def measure_accuracy(net: str, adc_bits: int, test_subset: Optional[int] = 2000,
                     device=None, calib_images: int = 512,
                     weights_dir: str = ".", seed: int = 20260822,
                     ni: Optional[dict] = None, recalibrate: bool = True,
                     redraw_d2d: bool = False, log=print) -> dict:
    """
    Score both regimes through the hardware forward - that is the chip datapath.

    Returns {} when torch or the trained weights are missing, so the cost half of
    the study still runs on a machine that has neither.

    `seed` matters: the ADC ranges are calibrated in train() mode, where `q_weight`
    injects transient read noise, so the full-scales are fitted against one noisy
    draw of the array.  Measured on one set of VGG-16 weights, only the draw
    varying: 87.60 / 87.60 / 86.60 / 87.00% (sd 0.42).  A difference under about a
    point is the calibration draw, not the architecture.
    """
    try:
        import torch
        from hwat_core import HWConfig, pick_device
        from resnet_vgg_hwat import build_model, load_cifar10, evaluate, load_numpy_into
    except Exception as exc:                      # noqa: BLE001
        log(f"  (accuracy skipped: {exc})")
        return {}

    out: dict = {}
    device = device or pick_device()
    _, _, xte, yte = load_cifar10(test_subset=test_subset)
    for regime in ("float", "hwat"):
        # Re-seed per regime, so float and hardware-aware are calibrated against
        # the same draw of read noise.
        torch.manual_seed(seed)
        path = find_weights(net, regime, weights_dir)
        if path is None:
            log(f"  (accuracy skipped for {regime}: no "
                f"{weights_path(net, regime)} under {weights_dir})")
            continue
        arrays, meta = load_weights(path)
        for hw_on in (False, True):
            # `ni` carries the device non-idealities held fixed for this run.
            # They only mean anything with the hardware datapath enabled.
            cfgm = HWConfig.from_simulation_config(enabled=hw_on, adc_bits=adc_bits,
                                                   **(ni or {}))
            model = build_model(net, cfgm, num_classes=int(yte.max()) + 1,
                                img_size=xte.shape[-1], device=device)
            n = load_numpy_into(model, arrays)
            if n == 0:
                continue

            # The trained .npz stores the d2d pattern the network was fitted
            # against, and `load_numpy_into` copies it over the freshly drawn
            # buffer.  Redrawing is an explicit switch rather than a silent choice.
            if redraw_d2d and hw_on:
                for mod in model.modules():
                    if hasattr(mod, "d2d"):
                        mod.d2d.copy_(1.0 + float(cfgm.d2d_sigma)
                                      * torch.randn_like(mod.d2d))

            # Always recalibrate for a hardware evaluation.  A float-trained
            # network never ran `_hw_matmul`, so its scales are still the 1.0
            # defaults and everything clamps.  A hardware-trained one carries
            # scales fitted on UNFOLDED weights, but `export_numpy` folds batch-norm
            # into them and `adc_scale` is a scalar, so the folded columns saturate
            # or lose resolution (86.55% from the state_dict vs 10.10% folded).
            # It also dominates retention: at t=7200 s VGG-16 scores 8.8% with the
            # ranges frozen against 81.6% refitted.
            needs_cal = hw_on and recalibrate
            if needs_cal:
                model.train()
                for m in model.modules():
                    if hasattr(m, "frozen"):
                        m.frozen.fill_(0.0)
                        m.calib_steps_done.fill_(0.0)
                        # Reset, do not merely unfreeze: `_track_scales` and
                        # `_track_adc` EMA towards the observed value at momentum
                        # 0.95 unless the range is still at its default, so stale
                        # ranges barely move (26.7% vs 84.4% with a real reset).
                        m.calibrated.fill_(0.0)
                        m.adc_scale.fill_(1.0)
                with torch.no_grad():
                    for i in range(0, min(calib_images, len(xte)), 64):
                        xb = torch.as_tensor(xte[i:i + 64])
                        if device is not None:
                            xb = xb.to(device)
                        model(xb)
                log(f"  ({regime}: calibrated ADC ranges on "
                    f"{min(calib_images, len(xte))} images before scoring)")

            for m in model.modules():
                if hasattr(m, "freeze_scales"):
                    m.freeze_scales()
            acc = evaluate(model, xte, yte, torch, device=device)
            out[f"{regime}/{'chip' if hw_on else 'software'}"] = acc
            log(f"  {regime:6s} -> {'chip' if hw_on else 'software':8s} "
                f"top1 {acc*100:6.2f}%")
    return out


# 4. Study

def _effective_ni(ni: Optional[dict]) -> dict:
    """
    The device the run actually saw: simulation_config's SM_NI_* values with the
    deployment overrides applied.  Falls back to the overrides alone if HWConfig
    is unavailable, so the cost-only half still runs on a machine without torch.
    """
    fields = ("d2d_sigma", "c2c_sigma", "stuck_at_prob", "read_sigma",
              "read_noise_at_inference", "retention_time_s", "retention_tau_s",
              "retention_beta", "retention_model", "bias_on_array")
    try:
        from hwat_core import HWConfig
        cfg = HWConfig.from_simulation_config(**(ni or {}))
        return {f: getattr(cfg, f) for f in fields if hasattr(cfg, f)}
    except Exception:                             # noqa: BLE001
        return dict(ni or {})


def run_study(nets: List[str], adc_bits_list: List[int], img_sizes: List[int],
              accuracy: bool = True, test_subset: Optional[int] = 2000,
              weights_dir: str = ".", input_sparsity: float = 0.5,
              sparse_modes: bool = True, measure_sparsity: bool = True,
              ni: Optional[dict] = None, recalibrate: bool = True,
              redraw_d2d: bool = False, log=print) -> dict:
    """
    `measure_sparsity` measures the real per-layer zero fraction from the trained
    weights and uses it instead of the `input_sparsity` scalar.  It falls back to
    the scalar automatically when torch or the weights are absent, and every
    mapping records which of the two it used (`sparsity_measured`).
    """
    summary: dict = {"config": {
        "read_time_ns": sc.READ_TIME_S * 1e9,
        "sign_storage": sc.SIGN_STORAGE_MODE,
        "wire_resistance_ohm": sc.WIRE_RESISTANCE_OHM,
        "wire_capacitance_f": sc.WIRE_CAPACITANCE_F,
        "conductance_states": len(sc.CONDUCTANCE_LEVELS_S),
        # The device non-idealities this run was scored under, recorded as the
        # EFFECTIVE values (SM_NI_* plus the deployment overrides).
        "non_idealities": _effective_ni(ni),
        "ni_overrides": dict(ni or {}),
        "adc_recalibrated": bool(recalibrate),
        "d2d_redrawn": bool(redraw_d2d),
    }, "nets": {}}

    log("=" * 96)
    log("RESNET / VGG FULL-CHIP STUDY")
    log("=" * 96)
    c = summary["config"]
    log(f"read time {c['read_time_ns']:.2f} ns (derived)   sign storage "
        f"{c['sign_storage']}   {c['conductance_states']} states + 1 sign bit")
    log(f"wire RC   {c['wire_resistance_ohm']} ohm / {c['wire_capacitance_f']*1e15:.2f} fF "
        f"per cell pitch")
    eff = summary["config"]["non_idealities"]
    if eff:
        log("device non-idealities held fixed for this run "
            "(SM_NI_* from the Non-ideal tab, plus deployment overrides):")
        _NI_LABEL = {
            "d2d_sigma": "device-to-device spread",
            "c2c_sigma": "write / cycle-to-cycle spread",
            "stuck_at_prob": "stuck-at defect probability",
            "read_sigma": "read noise",
            "read_noise_at_inference": "read noise applied at inference",
            "retention_time_s": "retention age (s since programming)",
            "bias_on_array": "bias rides the array row (drifts)",
            "retention_tau_s": "retention tau",
            "retention_beta": "retention beta",
            "retention_model": "retention model",
        }
        for k, v in eff.items():
            log(f"   {_NI_LABEL.get(k, k):42s} {v}")
        log(f"   {'ADC ranges refitted before scoring':42s} {bool(recalibrate)}")
    log("")

    for net in nets:
        summary["nets"][net] = {"resolutions": {}}

        # Measure per-layer activation sparsity once per network, at the resolution
        # the weights were trained at: sparsity is a property of the network.
        sparsity = {}
        if measure_sparsity:
            sparsity = measure_activation_sparsity(
                net, weights_dir=weights_dir, img_size=TRAINED_IMG_SIZE, log=log)
        sparsity_arg = sparsity if sparsity else input_sparsity
        summary["nets"][net]["activation_sparsity"] = sparsity
        if not sparsity and measure_sparsity:
            log(f"  falling back to the assumed {input_sparsity:.0%} activation "
                f"sparsity for {net}")

        for img in img_sizes:
            spec = describe(net, img_size=img)
            tag = f"{net}@{img}"
            log("-" * 96)
            log(f"{tag}:  {spec.params/1e6:.2f}M weights, {spec.macs/1e6:.1f} MMAC/image, "
                f"{len(spec.layers)} mapped layers"
                + (f", {len(spec.residual_adds)} residual adds"
                   if spec.residual_adds else ""))
            res = {"params": spec.params, "macs": spec.macs, "mappings": []}

            log(f"  {'ADC':>4}{'arrays':>8}{'mm^2':>8}{'reads':>10}{'uJ/inf':>10}"
                f"{'TOPS/W':>9}{'sparse%':>9}  modes")
            for bits in adc_bits_list:
                m = map_and_cost(spec, bits, sparse_modes=sparse_modes,
                                 input_sparsity=sparsity_arg)
                # The same mapping on a machine without the sub-mode opcodes: every
                # operation drives all 128 rows and digitises all 128 columns.
                if sparse_modes:
                    flat = map_and_cost(spec, bits, sparse_modes=False,
                                        input_sparsity=sparsity_arg)
                    m["all_matrix"] = {
                        "energy_j": flat["energy_j"],
                        "tops_per_watt": flat["tops_per_watt"],
                        "mode_counts": flat["mode_counts"],
                        "energy_ratio": (flat["energy_j"] / m["energy_j"]
                                         if m["energy_j"] else 0.0),
                    }
                area = m["area_mm2"]                # real, and ADC-resolution aware
                modes = ",".join(f"{k}:{v}" for k, v in sorted(m["mode_counts"].items()))
                log(f"  {bits:>3}b{m['arrays']:>8,}{area:>8.1f}{m['reads']:>10,}"
                    f"{m['energy_j']*1e6:>10.2f}{m['tops_per_watt']:>9.2f}"
                    f"{m['sparsity_exploited']*100:>7.1f}%{'m' if m['sparsity_measured'] else 'a'}"
                    f"  {modes}")
                if "all_matrix" in m:
                    am = m["all_matrix"]
                    # Same mapping without the sub-mode opcodes, for reference.
                    log(f"       sub-mode opcodes save "
                        f"{am['energy_ratio']:.2f}x over all-vmm "
                        f"({am['energy_j']*1e6:.2f} uJ/inf, "
                        f"{am['tops_per_watt']:.2f} TOPS/W)")
                m["chips"] = [chip_point(spec, m, t, p, a)
                              for _, t, p, a in chip_sizes()]
                res["mappings"].append(m)

            best = res["mappings"][0]
            # Peak TOPS only: spare arrays hold extra weight copies and
            # images are pipelined through every stage.
            log(f"\n  {'chip':>10}{'fits':>6}{'repl':>7}{'TOPS':>10}{'TOPS/W':>9}")
            log(f"  {'':>10}{'':>6}{'':>7}{'(peak)':>10}")
            for cp in best["chips"]:
                log(f"  {cp['chip']:>10}{('yes' if cp['fits'] else 'NO'):>6}"
                    f"{cp['replication_factor']:>6.1f}x"
                    f"{cp['tops_pipelined']:>10.0f}{cp['tops_per_watt']:>9.2f}")

            ref = next((c for c in best["chips"] if c["fits"]), best["chips"][0])
            log(f"\n  ==> {tag} on {ref['chip']}: {ref['tops_pipelined']:,.0f} TOPS peak, "
                f"{ref['tops_per_watt']:.2f} TOPS/W, "
                f"{ref['fps_achievable']:,.0f} FPS one image at a time.")

            # Accuracy is only defined at the trained resolution: `describe(img=224)`
            # models the ImageNet variant while `build_model` always builds CIFAR.
            if accuracy and img == TRAINED_IMG_SIZE:
                log("")
                res["accuracy"] = measure_accuracy(
                    net, adc_bits_list[0], test_subset=test_subset,
                    weights_dir=weights_dir, ni=ni, recalibrate=recalibrate,
                    redraw_d2d=redraw_d2d, log=log)
            elif accuracy:
                log(f"  (no accuracy at {img} px: the trained weights are the "
                    f"{TRAINED_IMG_SIZE} px CIFAR variant; this resolution is a "
                    f"cost projection only)")
            summary["nets"][net]["resolutions"][str(img)] = res
            log("")

    return summary


# 5. Platform (GUI) entry points

FIGURE_PATH = "resnet_vgg_results.png"
SUMMARY_PATH = "resnet_vgg_study_summary.json"

DEFAULT_RUN_SETTINGS: Dict[str, object] = {
    # Both backbones by default; the comparison is the study.
    "rv_nets": ["resnet18", "vgg16"],
    "rv_img_sizes": [32],                 # 32 = CIFAR (trained); 224 = ImageNet
    "rv_adc_bits": [4, 8, 10],            # converter sweep
    "rv_accuracy": True,                  # score the trained weights (needs torch)
    "rv_test_subset": 1000,               # test images for the accuracy score
    # Where the trained .npz live.  Several folders may be listed, separated by
    # ';' - the two networks come out of two separate GPU runs.
    "rv_weights_dir": "out_log_22nd_Aug_ResNet;out_log_21st_Aug_VGG",
    "rv_measure_sparsity": True,          # measure it rather than assume it
    "rv_input_sparsity": 0.5,             # the fallback when nothing is measured
    "rv_sparse_modes": True,              # exploit point/row/column ISA modes
    # Deployment conditions, each held fixed for the run.  The device
    # non-idealities (d2d, write/cycle-to-cycle, stuck-at, read noise) are NOT
    # repeated here: they live on the platform's 'Non-ideal' tab as SM_NI_* and
    # `HWConfig.from_simulation_config` reads them from there.
    "rv_retention_s": 0.0,         # age of the programmed array, in seconds
    "rv_read_at_inference": False, # apply read noise to a deployed read too
    "rv_bias_on_array": True,      # the bias row drifts with the weights
    "rv_recalibrate": True,        # refit the ADC ranges before scoring
    "rv_redraw_d2d": False,        # ignore the d2d pattern stored in the .npz
}


# Run-setting key -> HWConfig field.  Only the fields that have no SM_NI_*
# equivalent; everything else comes from simulation_config via the Non-ideal tab.
_NI_KEYS = {
    "rv_retention_s": "retention_time_s",
    "rv_read_at_inference": "read_noise_at_inference",
    "rv_bias_on_array": "bias_on_array",
}


def ni_from_run_settings(rs: dict) -> dict:
    """
    Deployment conditions from the run settings, as HWConfig overrides.

    Deliberately does NOT carry d2d / c2c / stuck-at / read noise: those come
    from simulation_config, which the platform's 'Non-ideal' tab already edits.
    """
    out = {}
    for key, field in _NI_KEYS.items():
        if key not in rs:
            continue
        v = rs[key]
        out[field] = bool(v) if field in ("read_noise_at_inference",
                                          "bias_on_array") else float(v)
    return out


def run_resnet_vgg_benchmark(rs: Optional[dict] = None, verbose: bool = True) -> dict:
    """Platform entry point: run the study under GUI run-settings."""
    rs = {**DEFAULT_RUN_SETTINGS, **(rs or {})}

    def log(*a):
        if verbose:
            print(*a)

    nets = [str(n) for n in rs["rv_nets"]] or ["resnet18"]
    summary = run_study(
        nets=nets,
        adc_bits_list=[int(b) for b in rs["rv_adc_bits"]],
        img_sizes=[int(s) for s in rs["rv_img_sizes"]],
        accuracy=bool(rs["rv_accuracy"]),
        test_subset=int(rs["rv_test_subset"]),
        weights_dir=str(rs["rv_weights_dir"]),
        # Wired through to `map_and_cost`, which otherwise used its own defaults.
        input_sparsity=float(rs["rv_input_sparsity"]),
        sparse_modes=bool(rs["rv_sparse_modes"]),
        measure_sparsity=bool(rs.get("rv_measure_sparsity", True)),
        ni=ni_from_run_settings(rs),
        recalibrate=bool(rs.get("rv_recalibrate", True)),
        redraw_d2d=bool(rs.get("rv_redraw_d2d", False)),
        log=log,
    )
    # Only the configured chip is costed, so there is nothing to filter: the table
    # reports the read-path numbers plus a `fits` flag.
    summary["resident_only"] = False
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


def _first_res(summary: dict, net: str) -> dict:
    res = summary.get("nets", {}).get(net, {}).get("resolutions", {})
    if not res:
        return {}
    return res[sorted(res, key=lambda k: int(k))[0]]


def build_results_figure(s: dict, fig) -> None:
    """Four panels: TOPS/W vs ADC, array footprint, accuracy, and on-chip throughput."""
    nets = list(s.get("nets", {}))
    axes = fig.subplots(2, 2)
    fig.suptitle("ResNet-18 / VGG-16 on the memristor PIM chip", fontsize=13)
    colours = {"resnet18": "#2b6cb0", "vgg16": "#c05621"}

    # (a) TOPS/W by converter resolution - the dominant efficiency lever.  Bars on
    # a pinned 10-35 axis; the two backbones land within 4% of each other.
    ax = axes[0][0]
    bit_list = sorted({m["adc_bits"] for net in nets
                       for m in _first_res(s, net).get("mappings", [])})
    if bit_list:
        series = []
        for net in nets:
            by_bits = {m["adc_bits"]: m["tops_per_watt"]
                       for m in _first_res(s, net).get("mappings", [])}
            series.append((net, [by_bits.get(b, 0.0) for b in bit_list]))
        plotted = [v for _net, ys in series for v in ys if v > 0]
        # The pinned window only holds while every bar is inside it; 8- and 10-bit
        # land at ~7.7 and ~2.2 TOPS/W, so the floor drops to zero for those.
        lo = 0.0 if plotted and min(plotted) < 10.0 else 10.0
        hi = max(35.0, max(plotted) * 1.15) if plotted else 35.0
        width = 0.8 / len(nets) if nets else 0.8
        for si, (net, ys) in enumerate(series):
            xs = [i + (si - (len(nets) - 1) / 2) * width
                  for i in range(len(bit_list))]
            ax.bar(xs, ys, width, label=net, color=colours.get(net))
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y + (hi - lo) * 0.015, f"{y:.1f}",
                            ha="center", fontsize=5.5)
        ax.set_xticks(range(len(bit_list)))
        ax.set_xticklabels([f"{int(b)}" for b in bit_list])
        ax.set_xlim(-0.5, len(bit_list) - 0.5)
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=8)
    else:
        ax.set_ylim(10, 35)
    ax.set_xlabel("ADC resolution (bits)")
    ax.set_ylabel("TOPS/W")
    ax.set_title("Efficiency is set by the converter")
    ax.grid(alpha=0.3, axis="y")

    # (b) array footprint per resolution, against the chip sizes.
    ax = axes[0][1]
    labels, vals, cols = [], [], []
    for net in nets:
        for r_key, res in sorted(s["nets"][net]["resolutions"].items(),
                                 key=lambda kv: int(kv[0])):
            ms = res.get("mappings", [])
            if not ms:
                continue
            labels.append(f"{net}\n@{r_key}")
            vals.append(ms[0]["arrays"])
            cols.append(colours.get(net, "#666"))
    ax.bar(range(len(vals)), vals, color=cols)
    for _, t, p, a in chip_sizes():
        ax.axhline(t * p * a, ls="--", lw=0.8, color="#888")
        ax.text(len(vals) - 0.4, t * p * a, f" {t}x{p}x{a}", fontsize=7,
                va="bottom", ha="right", color="#555")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("128x128 arrays needed")
    ax.set_yscale("log")
    ax.set_title("Does it fit on the chip?")
    ax.grid(alpha=0.3, axis="y")

    # (c) accuracy: software ceiling vs the same weights on the real datapath.
    # Index-safe when only one backbone has been trained - they come out of
    # separate GPU runs, so that is a normal intermediate state.
    ax = axes[1][0]
    series = [("float/software", "float, software", "#68d391"),
              ("float/chip", "float, on chip", "#fc8181"),
              ("hwat/chip", "HW-aware, on chip", "#4299e1")]
    scored = [n for n in nets if _first_res(s, n).get("accuracy")]
    if scored:
        width = 0.26
        for si, (key, label, colour) in enumerate(series):
            xs = [i + (si - 1) * width for i in range(len(scored))]
            ys = [_first_res(s, n).get("accuracy", {}).get(key, 0.0) * 100
                  for n in scored]
            ax.bar(xs, ys, width, label=label, color=colour)
            for x, y in zip(xs, ys):
                if y > 0:
                    ax.text(x, y + 1, f"{y:.1f}", ha="center", fontsize=5.5)
        ax.set_xticks(range(len(scored)))
        ax.set_xticklabels(scored, fontsize=7)
        # Pin the x range to the group slots, so one scored network does not
        # autoscale its three bars across the whole panel.
        ax.set_xlim(-0.5, len(scored) - 0.5)
        ax.set_ylim(0, 105)
        ax.set_ylabel("top-1 (%)")
        ax.legend(fontsize=6, loc="lower right")
        untrained = [n for n in nets if n not in scored]
        if untrained:
            ax.text(0.02, 0.96, f"no weights yet: {', '.join(untrained)}",
                    transform=ax.transAxes, fontsize=6, color="#a00", va="top")
    else:
        ax.text(0.5, 0.5, "no trained weights yet\n(run resnet_vgg_hwat.py)",
                ha="center", va="center", fontsize=9, color="#888")
        ax.set_xticks([])
    ax.set_title("Hardware-aware training on the real datapath")
    ax.grid(alpha=0.3, axis="y")

    # (d) throughput on the configured chip.  Green where the weights stay
    # resident; red where the chip is too small, in which case the bar is the
    # read-path rate only and is marked as an upper bound.
    ax = axes[1][1]
    xs, ys, cs, lbl = [], [], [], []
    for net in nets:
        for r_key, res in sorted(s["nets"][net]["resolutions"].items(),
                                 key=lambda kv: int(kv[0])):
            ms = res.get("mappings", [])
            if not ms:
                continue
            for cp in ms[0].get("chips", []):
                resident = cp["fits"]
                xs.append(len(xs))
                ys.append(cp["fps_compute_only"])
                cs.append(colours.get(net, "#666") if resident else "#e53e3e")
                lbl.append(f"{net[:3]}@{r_key}\n{cp['chip']}")
    if xs:
        ax.bar(xs, ys, color=cs)
        ax.set_xticks(xs)
        ax.set_xticklabels(lbl, fontsize=6, rotation=45, ha="right")
        # Zero-based linear axis so ResNet and VGG read against each other.
        ax.set_ylim(0, max(ys) * 1.25)
    else:
        ax.text(0.5, 0.5, "nothing mapped",
                ha="center", va="center", fontsize=9, color="#888")
        ax.set_xticks([])
    ax.set_ylabel("inferences / s")
    ax.set_title("Green = resident.  Red = needs weight swaps (read-path rate only).")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.96))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="both", choices=["resnet18", "vgg16", "both"])
    ap.add_argument("--adc-bits", type=int, nargs="+", default=DEFAULT_ADC_BITS)
    ap.add_argument("--img-sizes", type=int, nargs="+", default=[32, 224])
    ap.add_argument("--no-accuracy", action="store_true",
                    help="cost only; does not need torch or trained weights")
    ap.add_argument("--test-subset", type=int, default=2000)
    ap.add_argument("--weights-dir", default=".",
                    help="directory holding the trained *_hwat_*.npz files; "
                         "read in place, nothing is copied. Several may be "
                         "given, separated by ';' or ',' - the two networks "
                         "come out of two separate GPU runs")
    ap.add_argument("--input-sparsity", type=float, default=0.5,
                    help="assumed zero fraction of activation rows, used only "
                         "when the sparsity cannot be measured")
    ap.add_argument("--no-measure-sparsity", action="store_true",
                    help="do not measure the real per-layer activation sparsity "
                         "from the trained weights; use --input-sparsity for "
                         "every layer instead")
    ap.add_argument("--no-sparse-modes", action="store_true",
                    help="always issue a full matrix read instead of selecting "
                         "the point/row/column ISA modes")
    ap.add_argument("--report", action="store_true",
                    help="also write the dashboard figure and the standalone "
                         "HTML report, so one command refreshes every artifact "
                         "after a GPU run comes back")
    ap.add_argument("--out", default="resnet_vgg_study_summary.json")
    # Deployment conditions.  The device non-idealities come from
    # simulation_config's SM_NI_* values - one source, every study.
    ap.add_argument("--redraw-d2d", action="store_true",
                    help="draw a fresh device-to-device pattern at the "
                         "configured SM_NI_DEVICE_TO_DEVICE_SIGMA_REL instead of "
                         "reusing the one stored in the trained .npz")
    ap.add_argument("--read-at-inference", action="store_true",
                    help="apply read noise to deployed reads, not just calibration")
    ap.add_argument("--retention-s", type=float, default=None,
                    help="seconds since the array was programmed (default 0)")
    ap.add_argument("--no-bias-drift", action="store_true",
                    help="hold the bias fixed while the weights relax "
                         "(models a digital bias rather than a bias row)")
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="do NOT refit the ADC ranges before scoring - models a "
                         "chip calibrated once at t=0 and never re-ranged")
    args = ap.parse_args()

    ni = {}
    if args.retention_s is not None:
        ni["retention_time_s"] = float(args.retention_s)
    if args.read_at_inference:
        ni["read_noise_at_inference"] = True
    if args.no_bias_drift:
        ni["bias_on_array"] = False

    nets = ["resnet18", "vgg16"] if args.net == "both" else [args.net]
    summary = run_study(nets, args.adc_bits, args.img_sizes,
                        accuracy=not args.no_accuracy, test_subset=args.test_subset,
                        weights_dir=args.weights_dir,
                        input_sparsity=args.input_sparsity,
                        sparse_modes=not args.no_sparse_modes,
                        measure_sparsity=not args.no_measure_sparsity,
                        ni=ni, recalibrate=not args.no_recalibrate,
                        redraw_d2d=args.redraw_d2d)
    summary["resident_only"] = False
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"Saved summary -> {args.out}")

    if args.report:
        # Matplotlib and the platform are imported here so the cost half keeps
        # running on a machine with neither, which is why `--no-accuracy` exists.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 8))
        build_results_figure(summary, fig)
        fig.savefig(FIGURE_PATH, dpi=140)
        plt.close(fig)
        print(f"Saved figure  -> {FIGURE_PATH}")
        try:
            import stratum_platform as sp
            study = next(x for x in sp.build_default_studies()
                         if x.key == "resnet_vgg")
            path = sp.generate_html_report(
                summary, sp.Platform([study], initial="resnet_vgg"),
                png_path=FIGURE_PATH, out_path=study.report_path)
            print(f"Saved report  -> {path}")
        except Exception as exc:                  # noqa: BLE001
            print(f"(HTML report skipped: {exc})")


if __name__ == "__main__":
    main()
