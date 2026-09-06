"""
example_usage_riscv_workflow.py

Run:
    python example_usage_riscv_workflow.py            (tkinter GUI platform)
    python example_usage_riscv_workflow.py --batch    (classic one-shot run)

GUI mode launches STRATUM (stratum_platform.py), where every parameter the
workflow consumes - the simulation_config.py values plus the workflow-local
settings - is editable in grouped tabs before each run.  The simulation runs in
the background with its log streamed into the GUI; results come back as KPI
cards, a comparison table, the matplotlib dashboard and a self-contained HTML
report.  simulation_config.py on disk is never modified.  The STUDY selector also
reaches the ResNet-18 / VGG-16 backbones (resnet_vgg_study.py).

The simulation deploys the trained MNIST MLP (mnist_ann.py) onto the chip as a
RISC-V-like program, with every layer operation performed by the chip:

    * matrix multiply  -> matrix-mode VMM (vmm)            on the crossbar
    * bias add         -> row-mode read   (vmmr)           on the crossbar
    * digit combine    -> PU shift & add  (pusll / puadd)  in the PU
    * partial-sum acc. -> PU add          (puadd)          in the PU
    * ReLU             -> on-chip comparator (purelu)      in the PU

Chip design choices exercised and measured:

  1. Digit-serial 4-bit input: an 8-bit activation is fed as two 4-bit nibbles.
  2. Sign-magnitude weights: |w| on one array + a sign register (2T1R cell).
  3. 1 GHz digital clock, fixed, for maximum TOPS/W.

The run reports software-float accuracy, hardware accuracy (4-bit ADC),
energy/latency/TOPS/W at 1 GHz, the chip area breakdown, the sign-storage study
(CMOS register vs spintronic STT-MRAM), the compiled instruction program, and the
accuracy-retention-over-time analysis.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import mnist_ann
import simulation_config as _cfg
from simulation_config import (
    build_device,
    build_nn_decay_single_memristor_ni_config,
    build_sign_storage,
)
from riscv_imc_workflow import (
    DigitSerialInput, IMCProcessor, ExecutionCost, TwoT1RConfig,
    DIGITAL_CLOCK_MHZ,
    lower_linear_layer, calibrate_adc, _effective_forward,
    compile_network_to_instructions, efficiency_report,
    build_workflow_crossbar,
    freeze_programming, apply_time, mean_stored_conductance, weight_domain_logits,
)


# Workflow-local run settings; all editable in the platform UI, as are the
# simulation_config.py values (live in memory).  The t2_* values are the 2T1R
# signed-cell physics for the hardware read path.

DEFAULT_RUN_SETTINGS = {
    "test_subset_n": 300,        # test images through the full hardware path
    "calib_n": 128,              # images used to calibrate the ADC full-scale
    "digit_bits": 4,             # DAC digit width (4 => two nibbles / 8-bit input)
    # 2T1R cell: GUI-editable copies of the chip's cell, defined once in
    # simulation_config (section 2d) so every study models the same transistor.
    "t2_on_resistance_ohm": _cfg.TWOT1R_ON_RESISTANCE_OHM,
    "t2_off_resistance_ohm": _cfg.TWOT1R_OFF_RESISTANCE_OHM,
    "t2_rail_gain_mismatch": _cfg.TWOT1R_RAIL_GAIN_MISMATCH,
    "t2_rail_offset_a": _cfg.TWOT1R_RAIL_OFFSET_A,
    "t2_transistor_area_um2": _cfg.TWOT1R_TRANSISTOR_AREA_UM2,
    "t2_transistor_switch_energy_nj": _cfg.TWOT1R_TRANSISTOR_SWITCH_ENERGY_NJ,
    # MNIST:
    "force_retrain": False,      # retrain the MLP even if cached weights exist
}


# Commercial CPU / GPU reference points (peak INT8, datasheet-derived).  Fixed
# published figures, never run through the simulator - they exist so the simulated
# chip can be read against familiar hardware.  `tops` is vendor peak dense INT8
# and `power_w` the rated TDP, so the ratio is a datasheet efficiency.

COMMERCIAL_REFERENCES = [
    {"name": "NVIDIA H100 SXM",   "kind": "GPU", "tops": 1979.0, "power_w": 700.0},
    {"name": "NVIDIA A100 80GB",  "kind": "GPU", "tops": 624.0,  "power_w": 400.0},
    {"name": "NVIDIA RTX 4090",   "kind": "GPU", "tops": 660.6,  "power_w": 450.0},
    {"name": "Jetson AGX Orin",   "kind": "GPU", "tops": 275.0,  "power_w": 60.0},
    {"name": "Intel Xeon 8480+ (AMX)", "kind": "CPU", "tops": 118.0, "power_w": 350.0},
    {"name": "AMD EPYC 9654 (VNNI)",   "kind": "CPU", "tops": 39.0,  "power_w": 360.0},
]


def build_commercial_comparison(this_tops: float, this_tops_per_watt: float) -> dict:
    """
    Assemble this design's throughput/efficiency alongside the commercial
    reference parts, with a per-part efficiency ratio (how many times more
    energy-efficient this design is at peak INT8).
    """
    refs = []
    for r in COMMERCIAL_REFERENCES:
        tpw = r["tops"] / r["power_w"]
        refs.append({
            "name": r["name"], "kind": r["kind"],
            "tops": r["tops"], "power_w": r["power_w"], "tops_per_watt": tpw,
            "efficiency_ratio": (this_tops_per_watt / tpw) if tpw > 0 else 0.0,
        })
    return {
        "this_design": {"name": "This design (memristor PIM)",
                        "kind": "Memristor PIM",
                        "tops": this_tops, "tops_per_watt": this_tops_per_watt},
        "references": refs,
    }


def _place(num_pus: int):
    """Tile index -> (pu, array): fill arrays across the tile's PUs round-robin."""
    return lambda i: (i % num_pus, i // num_pus)


def _get_weights(rs) -> "mnist_ann.MLPWeights":
    """
    Load the cached MLP weights, retraining when the configuration demands it
    (hidden-width change or an explicit force_retrain).
    """
    path = _cfg.MNIST_WEIGHTS_PATH
    retrain = bool(rs.get("force_retrain", False))
    if os.path.exists(path) and not retrain:
        weights = mnist_ann.load_weights(path)
        if weights.hidden_units == _cfg.MNIST_HIDDEN_UNITS:
            return weights
        print(f"Cached weights have {weights.hidden_units} hidden units but the "
              f"config asks for {_cfg.MNIST_HIDDEN_UNITS} -> retraining...")
        retrain = True
    return mnist_ann.get_or_train(
        path=path, retrain=retrain,
        hidden_units=_cfg.MNIST_HIDDEN_UNITS,
        epochs=_cfg.MNIST_EPOCHS,
        batch_size=_cfg.MNIST_BATCH_SIZE,
        learning_rate=_cfg.MNIST_LEARNING_RATE,
        seed=_cfg.MNIST_SEED,
    )


def run_hardware(proc: IMCProcessor, layers, X, y):
    """Accuracy + mean cost over X for a given processor (digit/ADC config)."""
    correct = 0
    total = ExecutionCost()
    for i in range(X.shape[0]):
        c = ExecutionCost()
        h = X[i]
        for layer in layers:
            h = proc.run_layer(layer, h, c)
        if int(np.argmax(h)) == int(y[i]):
            correct += 1
        total.add(c)
    n = X.shape[0]
    mean = ExecutionCost(
        energy_j=total.energy_j / n, latency_ns=total.latency_ns / n,
        n_analog_reads=total.n_analog_reads // n, n_shift_add=total.n_shift_add // n,
        latency_parallel_ns=total.par_ns / n)
    return correct / n, mean


def sign_storage_costs(proc, layers, x_sub, y_sub, x_calib, n_arrays, rows, cols, macs):
    """
    Area / energy / TOPS of one sign-storage architecture on this deployment.

    The element changes only the sign area and the per-read sign energy - the
    analog currents, and so the accuracy, are identical.

    Read path only.  Programming energy is a write cost, paid once when the sign
    plane is loaded, so it is reported on its own line and never folded into
    `energy_per_inference_j` or TOPS/W.  Neither element has a characterised hold
    leakage, so that term is absent too.  The volatile register is cheaper per
    read; the MTJ's case is density and non-volatility, reported as `nonvolatile`.
    """
    calibrate_adc(proc, layers, x_calib)
    _, mean_cost = run_hardware(proc, layers, x_sub, y_sub)
    rep = efficiency_report(macs, mean_cost, proc.arrays_available)

    total_sign_bits = n_arrays * rows * cols
    cell = proc.sign_storage
    # Area: 2 access transistors (common to both schemes) + the sign-storage cell.
    tran_um2 = (rows * cols) * proc.cell_transistor_area_um2
    sign_um2 = total_sign_bits * cell.area_um2
    array_um2 = proc.array.total_area_um2
    total_area_um2 = n_arrays * (array_um2 + tran_um2) + sign_um2

    # Energy levers unique to the sign storage.  The one-time sign programming
    # energy is a write-path term and the write models here are uncalibrated, so
    # the comparison rests on per-read energy, area and non-volatility.
    # `standby_power_w` is 0 for both elements - neither has a characterised
    # hold-leakage figure.
    standby_power_w = proc.sign_standby_power_w(total_sign_bits)       # hold leakage
    latency_s = rep.latency_per_inference_s
    hold_energy_per_inf_j = standby_power_w * latency_s                # leaked while held

    return {
        "mode": proc.sign_storage_label,
        "nonvolatile": proc.sign_storage_nonvolatile,
        "area_per_bit_um2": cell.area_um2,
        "read_energy_per_bit_j": cell.energy_nj * 1e-9,
        "standby_power_per_bit_w": (standby_power_w / total_sign_bits
                                    if total_sign_bits else 0.0),
        "sign_area_total_um2": sign_um2,
        "total_area_um2": total_area_um2,
        "standby_power_total_w": standby_power_w,
        "hold_energy_per_inf_j": hold_energy_per_inf_j,
        "energy_per_inference_j": rep.energy_per_inference_j,
        # Fair "with hold-leakage" energy exposes the non-volatility advantage that
        # the dynamic-only pipeline number omits.
        "energy_incl_hold_j": rep.energy_per_inference_j + hold_energy_per_inf_j,
        "tops": rep.tops,
        "tops_per_watt": rep.tops_per_watt,
        "tops_per_watt_incl_hold": (
            rep.ops / ((rep.energy_per_inference_j + hold_energy_per_inf_j) * 1e12)
            if (rep.energy_per_inference_j + hold_energy_per_inf_j) > 0 else 0.0),
    }


def compare_sign_storage(layers, x_sub, y_sub, x_calib, twot1r, tech_node_nm,
                         n_arrays, rows, cols, macs, digit_io_kwargs):
    """Run both sign-storage techniques on the SAME deployment and tabulate them."""
    results = {}
    cb_ni = _cfg.build_workflow_crossbar_ni_config()
    for mode in ("register_2t1r", "spintronic_mram"):
        storage = build_sign_storage(tech_node_nm=tech_node_nm,
                                     frequency_mhz=DIGITAL_CLOCK_MHZ, mode=mode)
        proc = IMCProcessor(digit_io=DigitSerialInput(**digit_io_kwargs),
                            ideal_adc=False, twot1r=twot1r, sign_storage=storage,
                            crossbar_ni=cb_ni, rng=_cfg.CB_NI_RNG_SEED)
        results[mode] = sign_storage_costs(
            proc, layers, x_sub, y_sub, x_calib, n_arrays, rows, cols, macs)
    return results


def _time_labels(times_s):
    """Human-readable tick labels for the retention time points."""
    out = []
    for t in times_s:
        if t == 0:
            out.append("0")
        elif t < 60:
            out.append(f"{t:.0f}s")
        elif t < 3600:
            out.append(f"{t/60:.0f}m")
        else:
            out.append(f"{t/3600:.0f}h")
    return out


def build_results_figure(s, fig):
    """Draw the results dashboard for the summary dict `s` onto `fig`.

    Pure presentation - reads only the already-computed `summary` dict, so it
    changes none of the workflow logic. `fig` may be a pyplot figure (batch
    mode) or a plain matplotlib Figure (embedded in the tkinter GUI).
    """
    ret = s["retention_over_time"]
    t = np.array([r["time_s"] for r in ret])
    decay = np.array([r["conductance_decay_pct"] for r in ret])
    hw_acc_t = np.array([r["hardware_accuracy"] for r in ret])
    wd_acc_t = np.array([r["weight_domain_accuracy"] for r in ret])
    energy_t = np.array([r["energy_per_inference_j"] for r in ret])
    topsw_t = np.array([r["tops_per_watt"] for r in ret])

    x = np.arange(len(t))                # evenly spaced (t=0 breaks a log axis)
    xlabels = _time_labels(t)

    fig.suptitle(
        f"RISC-V IMC workflow - MNIST MLP ({s['architecture']}) deployed as an "
        f"instruction program\n"
        f"digit-serial {s['digit_serial_bits']}-bit input, sign-magnitude "
        f"weights ({s.get('sign_storage_mode', 'register_2t1r')} sign storage), 1 GHz",
        fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.52,
                          top=0.86, bottom=0.09, left=0.07, right=0.94)

    # ---- (a) accuracy bars --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    accs = [s["software_accuracy"], s["ideal_datapath_accuracy"],
            s["hardware_accuracy_4bit_adc"]]
    names = ["software\nfloat", "ideal\ndatapath", "hardware\n4-bit ADC"]
    bars = ax.bar(names, accs, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylim(min(0.80, max(0.0, min(accs) - 0.05)), 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_title("(a) Accuracy: float vs ideal vs hardware")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.003, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ---- (b) chip area breakdown --------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    n = s["n_arrays"]
    comps = ["memristor\narrays", "access\ntransistors", "sign\ncells"]
    vals_mm2 = [s["array_area_um2_per_array"] * n / 1e6,
                s["transistor_area_um2_per_array"] * n / 1e6,
                s["sign_area_um2_per_array"] * n / 1e6]
    bars = ax.bar(comps, vals_mm2, color=["#4c72b0", "#8ca7cc", "#dd8452"])
    ax.set_ylabel("area (mm$^2$)")
    total_mm2 = s["total_area_um2"] / 1e6
    ax.set_title(f"(b) Chip area breakdown  (total {total_mm2:.3f} mm$^2$, "
                 f"{n} arrays)")
    for b, a in zip(bars, vals_mm2):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ---- (c) energy efficiency vs commercial CPUs / GPUs -------------------
    ax = fig.add_subplot(gs[0, 2])
    comp = s.get("commercial_comparison")
    if comp:
        rows = [(comp["this_design"]["name"], comp["this_design"]["tops_per_watt"],
                 "this")]
        rows += [(r["name"], r["tops_per_watt"], r["kind"])
                 for r in comp["references"]]
        rows.sort(key=lambda d: d[1])                       # ascending -> top = best
        labels = [r[0].replace("This design (memristor PIM)", "This design (PIM)")
                  for r in rows]
        vals = [r[1] for r in rows]
        col_map = {"this": "#c44e52", "GPU": "#4c72b0", "CPU": "#dd8452",
                   "Memristor PIM": "#c44e52"}
        colors = [col_map.get(r[2], "#8ca7cc") for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("energy efficiency (TOPS/W, log)")
        ax.set_title("(c) Efficiency vs commercial CPUs/GPUs\n"
                     "(peak INT8, datasheet - context)")
        for yi, v in zip(y, vals):
            ax.text(v * 1.05, yi, f"{v:.2f}", va="center", fontsize=7)
        ax.set_xlim(right=max(vals) * 2.2)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#c44e52", label="this design"),
                           Patch(color="#4c72b0", label="GPU"),
                           Patch(color="#dd8452", label="CPU")],
                  fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    # ---- (d) retention: accuracy vs conductance decay ----------------------
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(x, hw_acc_t, "o-", color="#c44e52", label="hardware acc (4-bit ADC)")
    ax.plot(x, wd_acc_t, "s-", color="#55a868", label="weight-domain acc")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("elapsed time")
    ax.set_ylabel("test accuracy")
    acc_floor = min(float(np.min(hw_acc_t)), float(np.min(wd_acc_t)))
    ax.set_ylim(min(0.70, max(0.0, acc_floor - 0.05)), 1.0)
    ax.set_title(
        f"(d) Accuracy retention over time  "
        f"(Kohlrausch $\\tau$ = {s['retention_tau_s']:.0f} s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, decay, "^--", color="#7f7f7f", alpha=0.8,
             label="mean conductance decay")
    ax2.set_ylabel("mean conductance decay (%)", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")
    ax2.legend(loc="upper right", fontsize=8)

    # ---- (e) retention: efficiency vs time ---------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, energy_t * 1e12, "o-", color="#4c72b0", label="energy/inf (pJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("elapsed time")
    ax.set_ylabel("energy / inference (pJ)", color="#4c72b0")
    ax.tick_params(axis="y", labelcolor="#4c72b0")
    ax.set_title("(e) Efficiency over time")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, topsw_t, "s--", color="#dd8452", label="TOPS/W")
    ax2.set_ylabel("TOPS/W", color="#dd8452")
    ax2.tick_params(axis="y", labelcolor="#dd8452")
    return fig


def plot_results(s, out_path="riscv_workflow_results.png", show=True):
    """Batch-mode plotting: save the dashboard PNG and optionally show it."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    build_results_figure(s, fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_workflow(rs=None) -> dict:
    """
    Execute the full workflow with the given run settings (see
    DEFAULT_RUN_SETTINGS) and the CURRENT simulation_config values, print the
    engineering log, save the JSON summary, and return the summary dict.
    """
    rs = {**DEFAULT_RUN_SETTINGS, **(rs or {})}
    n_test = int(rs["test_subset_n"])
    n_calib = int(rs["calib_n"])
    digit_io_kwargs = dict(input_precision_bits=int(_cfg.INPUT_PRECISION_BITS),
                           digit_bits=int(rs["digit_bits"]))
    twot1r = TwoT1RConfig(
        on_resistance_ohm=float(rs["t2_on_resistance_ohm"]),
        off_resistance_ohm=float(rs["t2_off_resistance_ohm"]),
        rail_gain_mismatch=float(rs["t2_rail_gain_mismatch"]),
        rail_offset_a=float(rs["t2_rail_offset_a"]),
        transistor_area_um2=float(rs["t2_transistor_area_um2"]),
        transistor_switch_energy_nj=float(rs["t2_transistor_switch_energy_nj"]),
    )

    rng = np.random.default_rng(12345)

    # 1. Trained classifier + data
    weights = _get_weights(rs)
    x_train, y_train, x_test, y_test = mnist_ann.load_mnist(normalize=True)
    sw_acc = mnist_ann.accuracy(weights, x_test, y_test)

    print("=" * 74)
    print("RISC-V IMC WORKFLOW  -  MNIST MLP DEPLOYED AS AN INSTRUCTION PROGRAM")
    print("=" * 74)
    print(f"Architecture:             784 -> {weights.hidden_units} -> 10 (ReLU)")
    print(f"Software float test acc:  {sw_acc:.4f}  ({x_test.shape[0]} images)")

    device = build_device()
    levels = device.conductance_levels_s
    read_v = device.pulse.read_voltage_v
    array = build_workflow_crossbar()
    rows, cols = array.rows, array.cols

    print(f"Array:                    {rows} x {cols}, "
          f"{device.number_of_states}-state ({device.native_precision_bits}-bit) device")
    print(f"DAC / ADC precision:      {array.dac.precision_bits}-bit / "
          f"{array.adc.precision_bits}-bit     digital clock: 1 GHz")

    # Per-layer input scaling (activation value mapped to full DAC range).
    xcal = x_train[:n_calib]
    h1_cal = np.maximum(xcal @ weights.W1 + weights.b1, 0.0)
    in_scale_1 = float(max(np.percentile(xcal, 99.9), 1e-6))
    in_scale_2 = float(max(np.percentile(h1_cal, 99.9), 1e-6))

    # 2. Compile (lower) the network onto sign-magnitude array tiles
    place = _place(int(_cfg.TILE_NUM_PUS))
    layers = [
        lower_linear_layer("fc1", weights.W1, weights.b1, True, in_scale_1,
                           levels, rows, cols, read_v, pu_of_tile=place),
        lower_linear_layer("fc2", weights.W2, weights.b2, False, in_scale_2,
                           levels, rows, cols, read_v, pu_of_tile=place),
    ]
    n_arrays = sum(len(l.tiles) for l in layers)
    print("\n" + "-" * 74)
    print("MAPPING  (sign-magnitude: one array + one sign cell per tile)")
    print("-" * 74)
    for l in layers:
        print(f"  {l.name}: {l.in_features}x{l.out_features} -> "
              f"{len(l.tiles)} array(s) [{l.row_tiles} row-split x {l.col_tiles} col-split], "
              f"relu={l.has_relu}")
    print(f"  total arrays: {n_arrays}")

    # 3. Calibrate the bipolar ADC range, then run inference.  Sign storage is
    # chosen in simulation_config (SIGN_STORAGE_MODE): "register_2t1r" is a
    # volatile CMOS register, "spintronic_mram" a non-volatile MTJ cell.
    sign_storage = build_sign_storage(tech_node_nm=array.config.tech_node_nm,
                                      frequency_mhz=DIGITAL_CLOCK_MHZ)
    # CROSSBAR-level non-idealities on the analog path: wire IR drop, parasitic and
    # access-transistor series R, self-heating, current saturation, physical sense
    # noise and converter gain/offset.  Every accuracy figure below sees them.
    cb_ni = _cfg.build_workflow_crossbar_ni_config()
    proc = IMCProcessor(digit_io=DigitSerialInput(**digit_io_kwargs),
                        ideal_adc=False, twot1r=twot1r, sign_storage=sign_storage,
                        crossbar_ni=cb_ni, rng=_cfg.CB_NI_RNG_SEED)
    print(f"  Crossbar non-idealities:     wire={cb_ni.wire_resistance_ohm:.1f} ohm/pitch, "
          f"series={cb_ni.parasitic_series_resistance_ohm:.0f} ohm, "
          f"I_sat={cb_ni.current_saturation_a*1e3:.1f} mA, "
          f"sense noise={'on' if cb_ni.physical_sense_noise else 'off'}")
    print(f"  2T1R read path:              R_on={twot1r.on_resistance_ohm:.0f} ohm, "
          f"R_off={twot1r.off_resistance_ohm:.0e} ohm, "
          f"rail mismatch={twot1r.rail_gain_mismatch*100:.1f}%, "
          f"offset={twot1r.rail_offset_a*1e9:.1f} nA")
    print(f"  Sign storage:                {proc.sign_storage_label}"
          f"{'  (non-volatile MTJ)' if proc.sign_storage_nonvolatile else '  (volatile register)'}"
          f"   [SIGN_STORAGE_MODE={_cfg.SIGN_STORAGE_MODE!r}]")
    calibrate_adc(proc, layers, x_train[:n_calib])

    subset = rng.choice(x_test.shape[0], size=min(n_test, x_test.shape[0]), replace=False)
    x_sub, y_sub = x_test[subset], y_test[subset]
    m_cmp = min(64, x_sub.shape[0])      # subset for the sign-storage study

    # Exact-converter reference: the same chip with the ADC quantiser removed.  It
    # must carry the same 2T1R cell as `proc`, or the gap between the two rows also
    # absorbs R_on compression, wrong-rail leakage and sense mismatch.
    proc_ideal = IMCProcessor(digit_io=DigitSerialInput(**digit_io_kwargs),
                              ideal_adc=True, twot1r=twot1r,
                              sign_storage=sign_storage,
                              crossbar_ni=cb_ni, rng=_cfg.CB_NI_RNG_SEED)
    ideal_acc, _ = run_hardware(proc_ideal, layers, x_sub, y_sub)
    hw_acc, mean_cost = run_hardware(proc, layers, x_sub, y_sub)

    macs = sum(l.in_features * l.out_features for l in layers)
    rep = efficiency_report(macs, mean_cost, proc.arrays_available)

    print("\n" + "-" * 74)
    print(f"INFERENCE  ({x_sub.shape[0]} test images)")
    print("-" * 74)
    print(f"  Software float accuracy:        {sw_acc:.4f}")
    print(f"  Ideal-datapath accuracy:        {ideal_acc:.4f}  (4-bit weights, exact ADC)")
    print(f"  Hardware accuracy (4-bit ADC):  {hw_acc:.4f}")
    print(f"  Analog reads / inference:       {rep.n_analog_reads}")
    print(f"  Energy / inference:             {rep.energy_per_inference_j:.4e} J")
    print(f"  Arrays used / available:        {n_arrays} / {rep.arrays_concurrent}"
          f"   (concurrency limit: {min(n_arrays, rep.arrays_concurrent)} arrays)")
    print("  " + "-" * 62)
    print(f"  {'':30s}{'SERIAL':>14s}{'PARALLEL':>16s}")
    print(f"  {'Latency / inference':30s}"
          f"{rep.latency_per_inference_s*1e9:>11.1f} ns"
          f"{rep.latency_parallel_s*1e9:>13.1f} ns")
    print(f"  {'Power':30s}{rep.power_w*1e3:>11.3f} mW"
          f"{rep.power_parallel_w*1e3:>13.3f} mW")
    print(f"  {'Throughput':30s}{rep.tops:>9.4f} TOPS"
          f"{rep.tops_parallel:>11.4f} TOPS")
    print(f"  {'Energy efficiency':30s}{rep.tops_per_watt:>7.2f} TOPS/W"
          f"{rep.tops_per_watt:>9.2f} TOPS/W")
    print("  " + "-" * 62)
    # A layer's tiles are independent, so they issue together; energy is
    # identical either way, which is why TOPS/W does not move.
    print(f"  Parallel issue is {rep.parallel_speedup:.2f}x faster "
          f"(tiles read together; same joules per inference).")

    # 3b. Energy efficiency vs commercial CPUs / GPUs (context)
    comparison = build_commercial_comparison(rep.tops_parallel, rep.tops_per_watt)
    print("\n" + "-" * 74)
    print("ENERGY EFFICIENCY vs COMMERCIAL CPUs / GPUs  (peak INT8, datasheet)")
    print("-" * 74)
    print(f"  {'Device':32s}{'Class':>7s}{'TOPS/W':>10s}{'this vs it':>13s}")
    print(f"  {comparison['this_design']['name']:32s}{'PIM':>7s}"
          f"{comparison['this_design']['tops_per_watt']:>10.2f}{'1.0x':>13s}")
    for r in sorted(comparison["references"],
                    key=lambda d: d["tops_per_watt"], reverse=True):
        print(f"  {r['name']:32s}{r['kind']:>7s}{r['tops_per_watt']:>10.2f}"
              f"{r['efficiency_ratio']:>12.1f}x")
    print("  " + "-" * 62)
    # Datasheet peaks, for context only.
    print("  TOPS/W are peak-INT8 datasheet figures; absolute TOPS differ "
          "because this workload")
    print(f"  fills only a handful of arrays "
          f"({rep.tops_parallel:.2f} TOPS), so compare on TOPS/W.")

    # 4. Chip area breakdown (2T1R sign-magnitude arrays + sign cells)
    one_array_um2 = array.total_area_um2
    cells = rows * cols
    sign_reg_um2 = cells * proc.sign_storage.area_um2
    tran_um2 = cells * proc.cell_transistor_area_um2        # 2 access transistors / cell
    total_area_um2 = n_arrays * (one_array_um2 + sign_reg_um2 + tran_um2)
    print("\n" + "-" * 74)
    print(f"CHIP AREA  (2T1R sign-magnitude + {proc.sign_storage_label} sign cell)")
    print("-" * 74)
    print(f"  one array:                      {one_array_um2:.1f} um^2")
    print(f"  access transistors / array:     {tran_um2:.1f} um^2 "
          f"({proc.twot1r.transistors_per_cell} x {cells} cells)")
    print(f"  sign cell / array:              {sign_reg_um2:.1f} um^2 ({cells} bits, "
          f"{proc.sign_storage.area_um2:.4f} um^2/bit)")
    print(f"  total ({n_arrays} arrays):              {total_area_um2:.1f} um^2 "
          f"({total_area_um2/1e6:.4f} mm^2)")

    # 4b. Sign-storage architecture: register vs spintronic STT-MRAM
    #     (isolates the effect of the sign-storage choice on area/energy/TOPS)
    ss = compare_sign_storage(layers, x_sub[:m_cmp], y_sub[:m_cmp],
                              x_train[:n_calib], twot1r,
                              array.config.tech_node_nm,
                              n_arrays, rows, cols, macs, digit_io_kwargs)
    reg, mram = ss["register_2t1r"], ss["spintronic_mram"]
    print("\n" + "-" * 74)
    print("SIGN-STORAGE ARCHITECTURE  (2T1R CMOS register vs spintronic STT-MRAM)")
    print("-" * 74)
    print(f"  {'':30s}{'register_2t1r':>16s}{'spintronic_mram':>18s}")
    print(f"  {'non-volatile':30s}{str(reg['nonvolatile']):>16s}"
          f"{str(mram['nonvolatile']):>18s}")
    print(f"  {'sign area / bit (um^2)':30s}{reg['area_per_bit_um2']:>16.4f}"
          f"{mram['area_per_bit_um2']:>18.4f}")
    print(f"  {'sign area / all arrays (um^2)':30s}{reg['sign_area_total_um2']:>16.1f}"
          f"{mram['sign_area_total_um2']:>18.1f}")
    print(f"  {'total array area (mm^2)':30s}{reg['total_area_um2']/1e6:>16.4f}"
          f"{mram['total_area_um2']/1e6:>18.4f}")
    print(f"  {'read energy / bit (J)':30s}{reg['read_energy_per_bit_j']:>16.2e}"
          f"{mram['read_energy_per_bit_j']:>18.2e}")
    print("  " + "-" * 62)
    print("  READ / INFERENCE PATH  (the comparison; write costs excluded)")
    print(f"  {'energy/inf (J)':30s}{reg['energy_per_inference_j']:>16.3e}"
          f"{mram['energy_per_inference_j']:>18.3e}")
    print(f"  {'TOPS':30s}{reg['tops']:>16.4f}{mram['tops']:>18.4f}")
    print(f"  {'TOPS/W':30s}{reg['tops_per_watt']:>16.2f}"
          f"{mram['tops_per_watt']:>18.2f}")
    print("  " + "-" * 62)
    print("  OFF THE READ PATH  (reported, never charged per inference)")
    print(f"  {'sign hold power (W, all bits)':30s}{reg['standby_power_total_w']:>16.2e}"
          f"{mram['standby_power_total_w']:>18.2e}")
    print("    (hold power is 0 for both: neither element has a characterised")
    print("     leakage figure, and no other block on this chip carries one)")
    print("  " + "-" * 62)
    _area_saving = 100 * (1 - mram['total_area_um2'] / reg['total_area_um2'])
    _read_delta = 100 * (mram['energy_per_inference_j'] / reg['energy_per_inference_j'] - 1)
    print(f"  spintronic STT-MRAM vs register: {_area_saving:+.1f}% array area, "
          f"{_read_delta:+.2f}% read energy/inference,")
    print(f"    non-volatile (weights' signs survive power-down; >10 yr retention, "
          f"no sign reload), TOPS unchanged (sign read off the "
          f"{proc.array.config.read_time_s*1e9:.0f} ns analog-read critical path).")
    print(f"    On the read path alone the register is cheaper; the MTJ's case is "
          f"density and non-volatility.")
    print("  " + "-" * 62)
    print("  spintronic MTJ cell parameters (simulation_config.MRAM_*):")
    print(f"    cell area = {_cfg.MRAM_CELL_AREA_F2:.0f} F^2, "
          f"R_P = {_cfg.MRAM_R_PARALLEL_OHM:.0f} ohm, "
          f"TMR = {_cfg.MRAM_TMR_RATIO*100:.0f}% "
          f"(R_AP = {_cfg.MRAM_R_PARALLEL_OHM*(1+_cfg.MRAM_TMR_RATIO):.0f} ohm)")
    print(f"    read V = {_cfg.MRAM_READ_VOLTAGE_V:.2f} V, "
          f"STT write = {_cfg.MRAM_WRITE_CURRENT_A*1e6:.0f} uA @ "
          f"{_cfg.MRAM_WRITE_VOLTAGE_V:.2f} V for {_cfg.MRAM_WRITE_TIME_NS:.0f} ns")

    # 5. Show a slice of the compiled instruction program
    prog = compile_network_to_instructions(layers, proc.digit_io)
    print("\n" + "-" * 74)
    print(f"COMPILED PROGRAM  ({len(prog)} instructions; showing fc2 tail)")
    print("-" * 74)
    for ins in prog[-12:]:
        print("   " + str(ins))

    # 6. Accuracy retention over time: programming non-idealities (D2D, write error,
    # stuck-at) plus Kohlrausch drift, replayed at the elapsed times in
    # simulation_config, on this sign-magnitude / digit-serial datapath.
    ni_config = build_nn_decay_single_memristor_ni_config()
    dep = freeze_programming(layers, ni_config, levels)

    # Re-deploy at t=0 (programming NI, no drift) and recalibrate the ADC once;
    # the range then stays fixed, so as conductances relax the converter slowly
    # under-ranges - a real deployment effect (matches example_usage_nn_decay.py).
    apply_time(dep, layers, 0.0)
    calibrate_adc(proc, layers, x_train[:n_calib])
    base_g0 = mean_stored_conductance(layers)

    time_points = [float(t) for t in _cfg.NN_DECAY_TIME_POINTS_S]
    print("\n" + "=" * 74)
    print(f"ACCURACY RETENTION OVER TIME  (Kohlrausch tau = {ni_config.retention_tau_s:.0f} s, "
          f"sign in {proc.sign_storage_label} => drift loses amplitude, never flips sign)")
    print("=" * 74)
    print(f"  {'Time (s)':>9}  {'G decay %':>9}  {'Wt-dom acc':>10}  "
          f"{'HW acc':>7}  {'Energy/inf (J)':>14}  {'TOPS':>7}  {'TOPS/W':>7}")
    print(f"  {'':>9}  {'':>9}  {'':>10}  {'':>7}  {'':>14}  {'(par)':>7}")
    print("  " + "-" * 70)

    retention_rows = []
    for t in time_points:
        apply_time(dep, layers, t, read_noise=True)
        decay_pct = (1.0 - mean_stored_conductance(layers) / base_g0) * 100.0

        wd_logits = weight_domain_logits(layers, x_test)
        wd_acc = float(np.mean(np.argmax(wd_logits, axis=1) == y_test))

        t_acc, t_cost = run_hardware(proc, layers, x_sub, y_sub)
        t_rep = efficiency_report(macs, t_cost, proc.arrays_available)
        retention_rows.append({
            "time_s": t, "conductance_decay_pct": decay_pct,
            "weight_domain_accuracy": wd_acc, "hardware_accuracy": t_acc,
            "energy_per_inference_j": t_rep.energy_per_inference_j,
            "tops": t_rep.tops, "tops_parallel": t_rep.tops_parallel,
            "tops_per_watt": t_rep.tops_per_watt,
        })
        print(f"  {t:>9.0f}  {decay_pct:>9.3f}  {wd_acc:>10.4f}  {t_acc:>7.4f}  "
              f"{t_rep.energy_per_inference_j:>14.4e}  {t_rep.tops_parallel:>7.4f}  "
              f"{t_rep.tops_per_watt:>7.2f}")
    print("  " + "-" * 70)
    r0, rN = retention_rows[0], retention_rows[-1]
    print(f"  Hardware accuracy drop, t=0 -> t={rN['time_s']:.0f}s: "
          f"{100*(r0['hardware_accuracy'] - rN['hardware_accuracy']):.2f} pts "
          f"({rN['conductance_decay_pct']:.1f} % mean conductance decay)")

    # 7. Save summary
    summary = {
        "architecture": f"784-{weights.hidden_units}-10",
        "software_accuracy": sw_acc,
        "ideal_datapath_accuracy": ideal_acc,
        "hardware_accuracy_4bit_adc": hw_acc,
        "digit_serial_bits": int(rs["digit_bits"]),
        "test_subset_n": int(x_sub.shape[0]),
        "analog_reads_per_inference": rep.n_analog_reads,
        "energy_per_inference_j": rep.energy_per_inference_j,
        "latency_per_inference_ns": rep.latency_per_inference_s * 1e9,
        "power_w": rep.power_w,
        "tops": rep.tops,
        "tops_per_watt": rep.tops_per_watt,
        # Serial issue is the conservative bound; the chip actually reads a
        # layer's independent tiles together, which is what `*_parallel` costs.
        "latency_parallel_ns": rep.latency_parallel_s * 1e9,
        "power_parallel_w": rep.power_parallel_w,
        "tops_parallel": rep.tops_parallel,
        "parallel_speedup": rep.parallel_speedup,
        "arrays_concurrent_limit": rep.arrays_concurrent,
        "n_arrays": n_arrays,
        "array_area_um2_per_array": one_array_um2,
        "transistor_area_um2_per_array": tran_um2,
        "sign_area_um2_per_array": sign_reg_um2,
        "total_area_um2": total_area_um2,
        "instruction_count": len(prog),
        "program_tail": [str(ins) for ins in prog[-12:]],
        "retention_tau_s": ni_config.retention_tau_s,
        "retention_over_time": retention_rows,
        # Sign-storage architecture (SIGN_STORAGE_MODE) + register-vs-spintronic study.
        "sign_storage_mode": proc.sign_storage_label,
        "sign_storage_nonvolatile": proc.sign_storage_nonvolatile,
        "sign_storage_comparison": ss,
        "mram_parameters": {
            "cell_area_f2": float(_cfg.MRAM_CELL_AREA_F2),
            "r_parallel_ohm": float(_cfg.MRAM_R_PARALLEL_OHM),
            "tmr_ratio": float(_cfg.MRAM_TMR_RATIO),
            "read_voltage_v": float(_cfg.MRAM_READ_VOLTAGE_V),
            "write_current_a": float(_cfg.MRAM_WRITE_CURRENT_A),
            "write_voltage_v": float(_cfg.MRAM_WRITE_VOLTAGE_V),
            "write_time_ns": float(_cfg.MRAM_WRITE_TIME_NS),
        },
        # Commercial CPU/GPU efficiency context (peak INT8 datasheet figures).
        "commercial_comparison": comparison,
    }
    with open("riscv_workflow_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary -> riscv_workflow_summary.json")
    return summary


def main() -> None:
    argv = sys.argv[1:]
    if "--batch" in argv or "--classic" in argv:
        # Classic non-interactive behaviour: run once with the values currently
        # in simulation_config.py + the defaults above, pop up the figure.
        summary = run_workflow()
        plot_results(summary)
        return

    # Tkinter GUI platform (default): every study, opened on this one.
    from stratum_platform import Platform, build_default_studies
    Platform(build_default_studies(), initial="mlp").launch()


if __name__ == "__main__":
    main()
