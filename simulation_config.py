"""
simulation_config.py

Single control panel for the whole memristor PIM toolkit.  Every script reads
its parameters from here, so the scripts themselves never need editing.

  1.  MEMRISTOR DEVICE            - conductance states, pulse + read/write
  2.  CROSSBAR + PERIPHERALS      - array size, ADC/DAC choice, precisions, counts
  3.  LAYER MAPPING               - logical layer size for mapping estimates
  4.  SINGLE-MEMRISTOR NON-IDEAL. - D2D, write error, stuck-at, retention, read noise
  5.  CROSSBAR-LEVEL NON-IDEAL.   - IR drop, parasitics, ADC/DAC error, ...
  6.  CHIP ARCHITECTURE           - tiles / PUs / arrays / registers / interconnects
  7.  RUN / ANALYSIS SETTINGS     - elapsed times, RNG seeds, NN-decay + MNIST knobs
  8.  BUILDER FUNCTIONS           - turn the values above into objects

Units: conductance = Siemens, area = um^2, time = s, energy = J, voltage = V,
frequency = MHz, length = mm, resistance = Ohm, capacitance = F.

The builder functions also take keyword overrides for a one-off change, e.g.
build_single_memristor_ni_config(retention_tau_s=10_000.0).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from memristor_device import MemristorDevice
from integrated_crossbar import CrossbarConfig, IntegratedMemristorCrossbar
from single_memristor_nonidealities import SingleMemristorNonIdealityConfig
from crossbar_nonidealities import CrossbarNonIdealityConfig


# 1. MEMRISTOR DEVICE

# Measured/stored conductance states (Siemens). 16 states => 4-bit device.
# Linearly spaced 5 uS ... 80 uS in 5 uS steps.
CONDUCTANCE_LEVELS_S = [
    5e-8,  10e-8, 15e-8, 20e-8,
    25e-8, 30e-8, 35e-8, 40e-8,
    45e-8, 50e-8, 55e-8, 60e-8,
    65e-8, 70e-8, 75e-8, 80e-8,
]

DEVICE_AREA_UM2 = 0.01            # single-cell area

# E|w| / max|w| of a typical trained weight tensor: converts the conductance grid
# into the MEAN conductance a programmed array holds.
#     E[G] = g_min + (g_max - g_min) * TYPICAL_WEIGHT_ABS_MEAN_RATIO
# Measured over the 24 weight tensors shipped here (median 0.166).  Used by
# IntegratedMemristorCrossbar.representative_conductance_s when no weights are given.
TYPICAL_WEIGHT_ABS_MEAN_RATIO = 0.166

READ_VOLTAGE_V = 0.4             # read voltage

# 1b. PROGRAMMING (write) MODEL - internal, not an inference knob.
# Feeds only the write-energy and reprogramming-cost accounting; no accuracy,
# TOPS or TOPS/W figure reads them.  100 ns is an ordinary filamentary SET/RESET.
_WRITE_PULSE_WIDTH_S = 100e-9     # write pulse width
_WRITE_PULSE_INTERVAL_S = 100e-9  # gap between programming pulses
_WRITE_AMPLITUDE_V = 3.0          # write pulse amplitude

# Backwards-compatible aliases (external scripts / notebooks still import these).
PULSE_WIDTH_S = _WRITE_PULSE_WIDTH_S
PULSE_INTERVAL_S = _WRITE_PULSE_INTERVAL_S
WRITE_AMPLITUDE_V = _WRITE_AMPLITUDE_V

DEVICE_INITIAL_LEVEL_INDEX = 0    # starting conductance state index
DEVICE_NAME = "measured_multistate_memristor"

# Observed conductance-after-pulse trace used by the single-device demos
# (example_usage*.py) for write/read energy of one cell.
DEVICE_OBSERVED_TRACE_S = [6e-6, 9e-6, 14e-6, 22e-6, 35e-6]


# 2. CROSSBAR + INTEGRATED PERIPHERALS

XBAR_ROWS = 128
XBAR_COLS = 128

# If a load resistance is given the ADC sees a voltage; otherwise it sees current.
LOAD_RESISTANCE_OHM: Optional[float] = None

# Interconnect RC per cell pitch (NOT per whole line); the basis of
# `derive_read_time_s` below.  One cell pitch of a 65 nm-node crossbar at
# DEVICE_AREA_UM2 = 0.01 (~100 nm pitch).
WIRE_RESISTANCE_OHM = 2.5         # ohm per cell pitch of bit/word line
WIRE_CAPACITANCE_F = 0.2e-15      # farad per cell pitch of bit/word line

# Settling time constants before the converter samples.  None derives it from the
# ADC resolution (1 LSB, n_tau = bits * ln 2); a float pins it.
SETTLING_TAU_MULTIPLE: Optional[float] = None

# Extra margin on the derived read time (1.0 = settle exactly to 1 LSB).
READ_TIME_MARGIN = 1.2

# Peripheral selection (see hardware_peripherals.py tables for what each choice is):
#   ADC choice 7 -> 4-bit, 2 -> 8-bit, 1 -> 10-bit
#   DAC choice 6 -> 8-bit, 1 -> 1-bit
ADC_CHOICE = 7
DAC_CHOICE = 6

INPUT_PRECISION_BITS = 8
WEIGHT_PRECISION_BITS = 4
OUTPUT_PRECISION_BITS = 16

DIGITAL_FREQUENCY_MHZ = 100.0     # clock of the crossbar's digital periphery
TECH_NODE_NM = 65

# Physical peripheral counts. Keep None to use crossbar-based defaults:
#   ADCs = cols, DACs = rows, input regs = rows, output regs = cols,
#   shiftregs = rows, adders = cols.
NUM_ADCS: Optional[int] = None
NUM_DACS: Optional[int] = None
NUM_INPUT_REGS: Optional[int] = None
NUM_OUTPUT_REGS: Optional[int] = None
NUM_SHIFTREGS: Optional[int] = None
NUM_ADDERS: Optional[int] = None

CROSSBAR_NAME = "integrated_crossbar"


# 2c. DERIVED ANALOG READ TIME.  The larger of two constraints wins:
#   1. Converter aperture - the ADC's own conversion latency (peripheral table).
#   2. Column settling to 1 LSB - Elmore delay of the distributed RC column:
#          tau = (R_cells + R_wire_total/2) * C_line
#          t   = tau * n_tau,   n_tau = bits * ln 2
# The shot-noise limit is reported by `read_noise_sigma_a` rather than enforced
# here: it costs effective bits instead of gating the read.

def derive_read_time_s(
    adc_choice: Optional[int] = None,
    rows: Optional[int] = None,
    conductance_levels_s=None,
    wire_resistance_ohm: Optional[float] = None,
    wire_capacitance_f: Optional[float] = None,
    margin: Optional[float] = None,
    settling_tau_multiple: Optional[float] = None,
    return_terms: bool = False,
):
    """
    Physically-derived analog read duration in seconds.

    Every hardware number here is READ from the peripheral tables, never written,
    so this changes no device or peripheral specification.  Pass `return_terms`
    to get the individual constraints back for reporting.
    """
    from hardware_peripherals import ADC

    adc_choice = ADC_CHOICE if adc_choice is None else adc_choice
    rows = XBAR_ROWS if rows is None else rows
    levels = CONDUCTANCE_LEVELS_S if conductance_levels_s is None else conductance_levels_s
    r_wire = WIRE_RESISTANCE_OHM if wire_resistance_ohm is None else wire_resistance_ohm
    c_wire = WIRE_CAPACITANCE_F if wire_capacitance_f is None else wire_capacitance_f
    margin = READ_TIME_MARGIN if margin is None else margin
    if settling_tau_multiple is None:
        settling_tau_multiple = SETTLING_TAU_MULTIPLE

    adc = ADC(adc_choice)
    bits = int(adc.precision_bits or 8)

    # --- constraint 1: the converter's own aperture -------------------------
    t_adc_s = float(adc.latency_ns) * 1e-9

    # --- constraint 2: settle the column node to 1 LSB ----------------------
    g_mean = float(sum(levels)) / max(len(levels), 1)
    r_cells = 1.0 / max(rows * g_mean, 1e-18)      # all rows driving in parallel
    r_line = r_wire * rows                          # distributed bitline
    c_line = c_wire * rows
    tau_s = (r_cells + 0.5 * r_line) * c_line
    n_tau = settling_tau_multiple if settling_tau_multiple else bits * math.log(2.0)
    t_settle_s = tau_s * n_tau

    t_s = max(t_adc_s, t_settle_s) * float(margin)
    if not return_terms:
        return t_s
    return t_s, {
        "adc_bits": bits,
        "t_adc_s": t_adc_s,
        "tau_s": tau_s,
        "n_tau": n_tau,
        "t_settle_s": t_settle_s,
        "r_cells_ohm": r_cells,
        "r_line_ohm": r_line,
        "c_line_f": c_line,
        "margin": float(margin),
        "binding": "adc_aperture" if t_adc_s >= t_settle_s else "rc_settling",
    }


# The module-level value, derived at import from the configured ADC.  Editing
# ADC_CHOICE, XBAR_ROWS, the conductance levels or the wire RC now moves it.
READ_TIME_S = derive_read_time_s()


# 2b. SIGN-STORAGE ARCHITECTURE
# How the SIGN bit of each sign-magnitude / 2T1R synapse is stored.  Only the
# choice lives here; the hardware data is in hardware_peripherals.py (`Reg` and
# `STTMRAMSignCell`) and the wiring in riscv_imc_workflow.py.
#   "register_2t1r"   -> volatile 1-bit CMOS register driving the 2T1R gates.
#   "spintronic_mram" -> non-volatile 1T1MTJ cell biasing the same gates.
# Default is the MTJ: ~15% denser per sign bit and non-volatile, at slightly more
# read energy than a register access.  Each weight is a 4-bit magnitude on the
# memristor plus this 1-bit sign, i.e. 31 distinct signed levels.
SIGN_STORAGE_MODE = "spintronic_mram"

# STT-MRAM sign-cell physics, used only when SIGN_STORAGE_MODE ==
# "spintronic_mram".  Maps 1:1 onto hardware_peripherals.STTMRAMSignCell;
# defaults from embedded 28-65 nm perpendicular-MTJ macros.
MRAM_CELL_AREA_F2 = 36.0            # 1T1MTJ footprint in F^2 (F = feature size)
MRAM_R_PARALLEL_OHM = 6.0e3         # MTJ low-resistance (parallel) state
MRAM_TMR_RATIO = 1.2                # (R_AP - R_P)/R_P tunnel magnetoresistance (~120%)
MRAM_READ_VOLTAGE_V = 0.10          # MTJ sense voltage (kept << switching to avoid disturb)
MRAM_WRITE_CURRENT_A = 70e-6        # STT switching current
MRAM_WRITE_VOLTAGE_V = 0.70         # write-driver voltage
MRAM_WRITE_TIME_NS = 10.0           # STT switching pulse width


# 2d. 2T1R SIGNED SYNAPSE - the chip's access-transistor cell.
# One memristor holding |w| plus two access transistors whose gates are driven by
# the sign bit, so the read current is steered onto the + or - bitline; the column
# senses I_pos - I_neg.  Kept here so every study models the same cell.
# A bare `TwoT1RConfig()` is still all-ideal, for the self-tests and the exact-
# datapath references that ask for it explicitly.
TWOT1R_ON_RESISTANCE_OHM = 200.0            # access-transistor R_on, in series with the cell
TWOT1R_OFF_RESISTANCE_OHM = 1e9             # off-branch leak onto the wrong rail (Ion/Ioff ~1e6-1e9)
TWOT1R_RAIL_GAIN_MISMATCH = 0.01            # 1 % +/- sense-path gain mismatch
TWOT1R_RAIL_OFFSET_A = 1e-9                 # 1 nA differential sense offset
TWOT1R_TRANSISTOR_AREA_UM2 = 0.10           # area of ONE access transistor (two per cell)
TWOT1R_TRANSISTOR_SWITCH_ENERGY_NJ = 5e-7   # gate-drive energy per transistor per read
TWOT1R_SEPARATE_RAIL_ADC = False            # False: analog subtract + one bipolar ADC per column


def build_twot1r(**overrides):
    """The chip's 2T1R signed cell. Every study should read its cell from here."""
    from riscv_imc_workflow import TwoT1RConfig
    cfg = TwoT1RConfig(
        on_resistance_ohm=TWOT1R_ON_RESISTANCE_OHM,
        off_resistance_ohm=TWOT1R_OFF_RESISTANCE_OHM,
        rail_gain_mismatch=TWOT1R_RAIL_GAIN_MISMATCH,
        rail_offset_a=TWOT1R_RAIL_OFFSET_A,
        transistor_area_um2=TWOT1R_TRANSISTOR_AREA_UM2,
        transistor_switch_energy_nj=TWOT1R_TRANSISTOR_SWITCH_ENERGY_NJ,
        separate_rail_adc=TWOT1R_SEPARATE_RAIL_ADC,
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


# 3. LAYER MAPPING (for estimate_layer_mapping)

LAYER_ROWS = 1024
LAYER_COLS = 512


# 4. SINGLE-MEMRISTOR NON-IDEALITIES
#    (single_memristor_nonidealities.SingleMemristorNonIdealityConfig)

SM_NI_ENABLED = True

# Persistent cell-to-cell spread (coefficient of variation).
SM_NI_DEVICE_TO_DEVICE_SIGMA_REL = 0.05      # 5 %

# Random write error per programming event.
SM_NI_PROGRAMMING_ERROR_SIGMA_REL = 0.02     # 2 %
SM_NI_CYCLE_TO_CYCLE_SIGMA_REL = 0.01        # 1 %

# Transient read noise (re-drawn each read).
SM_NI_READ_NOISE_SIGMA_REL = 0.005           # 0.5 % relative
SM_NI_READ_NOISE_SIGMA_ABS_S = 0.0           # absolute floor (S)

# Stuck-at defects.
SM_NI_STUCK_AT_PROBABILITY = 0.001           # 0.1 % of cells
SM_NI_STUCK_AT_LRS_FRACTION = 0.5            # half stuck-LRS, half stuck-HRS

# Retention / drift: "none" | "exponential" | "power_law" | "kohlrausch".
SM_NI_RETENTION_MODEL = "kohlrausch"
SM_NI_RETENTION_TAU_S = 3600.0               # characteristic time (1 h)
SM_NI_RETENTION_BETA = 0.5                   # stretched-exponential exponent
SM_NI_RETENTION_ALPHA = 0.1                  # power-law exponent
SM_NI_RETENTION_FLOOR_S: Optional[float] = None

# Conductance clipping.
SM_NI_CLIP_TO_MEASURED_RANGE = True
SM_NI_MINIMUM_CONDUCTANCE_S = 1e-18

# Optional single-device pulse-update helper.
SM_NI_POTENTIATION_STEP_FRACTION = 0.10
SM_NI_DEPRESSION_STEP_FRACTION = 0.10


# 5. CROSSBAR-LEVEL NON-IDEALITIES
#    (crossbar_nonidealities.CrossbarNonIdealityConfig)

CB_NI_ENABLED = True

# Wire resistance of ONE cell pitch of bit/word line; same line as
# WIRE_RESISTANCE_OHM in section 2.
#   cell pitch 100 nm, line width 50 nm, aspect ratio 1.8 -> A = 4.5e-15 m^2
#   rho = 11 uOhm.cm (tungsten incl. TiN liner and surface scattering)
#   R = rho * L / A = 2.4 Ohm  ->  2.5 Ohm
# Tungsten rather than copper because Cu contaminates the switching oxide; a Cu
# line at the same geometry would be ~0.9 Ohm.
CB_NI_WIRE_RESISTANCE_OHM = 2.5
# Contact + via in series with EACH CELL, ohms per cell (2024 IITC: <50 ohm per
# kelvin via, two per cell).  Quoted per cell because that is what a process spec
# gives; `build_crossbar_ni_config` divides by XBAR_ROWS to get the column term,
# since 1/(1 + G_col*r_s) == 1/(1 + g_cell*R) when r_s = R / rows.
CB_NI_PARASITIC_SERIES_RESISTANCE_OHM = 100.0     # ohm PER CELL
CB_NI_PARASITIC_CAPACITANCE_F = 1e-15             # per wire segment

CB_NI_READ_NOISE_CURRENT_A = 1e-10               # absolute noise floor (scales sqrt(rows))
CB_NI_READ_NOISE_RELATIVE = 0.002                # fraction of column current

CB_NI_PROGRAMMING_VOLTAGE_DROP = 0.05            # max fractional drop at far corner [0,1]

# SET threshold and overdrive exponent.  Below V_th nothing switches; above it the
# conductance rises superlinearly with the overdrive.  V_th is a fraction of
# _WRITE_AMPLITUDE_V (3.0 V -> 1.65 V), an ordinary HfOx/TaOx SET voltage.
CB_NI_SET_THRESHOLD_FRACTION = 0.55
CB_NI_PROGRAMMING_OVERDRIVE_EXPONENT = 2.0

CB_NI_DAC_OFFSET_V = 1e-4                         # DAC output offset
CB_NI_DAC_GAIN_ERROR = 0.005                     # DAC scale error
CB_NI_ADC_OFFSET_A = 5e-11                        # ADC input offset
CB_NI_ADC_GAIN_ERROR = 0.005                     # ADC scale error

# Per-column current ceiling (A); None disables saturation.
CB_NI_CURRENT_SATURATION_A: Optional[float] = 5e-6

CB_NI_TEMPERATURE_COEFFICIENT_PER_K = 0.001       # TCR: fractional dG per K

# Self-heating is dissipation x thermal resistance, so the rise tracks the drive.
#   R_th,cell   1e7 K/W - a thermally confined ~10 nm filament; ~1.3 K at the
#               hottest cell of this array (0.4^2 V x 800 nS = 128 nW).
#   R_th,array  3e3 K/W - the whole 128x128 tile to ambient; ~3.3 K at full drive.
CB_NI_THERMAL_RESISTANCE_CELL_K_PER_W = 1.0e7
CB_NI_THERMAL_RESISTANCE_ARRAY_K_PER_W = 3.0e3
CB_NI_MAX_TEMPERATURE_RISE_K = 85.0              # CEILING (package limit), not a target

CB_NI_COMPARATOR_OFFSET_A = 1e-11                 # systematic sense-amp bias

# 1T1R/2T1R select-transistor R_on, ohms PER CELL: 500 from published sizing
# (W/L >= 5, 210 nm / 40 nm for >150 uA).  Divided by XBAR_ROWS in
# `build_crossbar_ni_config`.  Keep under ~10 kohm per cell: an under-driven word
# line pushes it past 1e5 ohm and the array stops working (32.4% top-1 measured).
CB_NI_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM = 500.0     # ohm PER CELL


# 6. CHIP ARCHITECTURE  (chip_architecture.py)

# Counts, as tiles x PUs/tile x arrays/PU -> 16 x 8 x 8 = 1024 arrays.
# Wide-and-shallow rather than the paper's 16 arrays/PU: input bandwidth to a
# layer scales with the number of PUs holding it.  1024 arrays holds one copy of
# ResNet-18 (698) and of VGG-16 (906) with room to replicate, so neither network
# pays a reprogramming cost.
CHIP_NUM_TILES = 16
TILE_NUM_PUS = 8
PU_NUM_ARRAYS = 8

# PU-local register file (Sec. 4.2): R0..R7, 128-bit each.
PU_NUM_REGISTERS = 8
PU_REGISTER_LENGTH_BITS = 128

# Tile-level register file (scalar/general + vector).
TILE_NUM_GENERAL_REGISTERS = 32
TILE_NUM_VECTOR_REGISTERS = 32
TILE_REGISTER_LENGTH_BITS = 128

# Tile instruction buffer + control-unit ALU.
INSTRUCTION_BUFFER_CAPACITY_BITS = 128 * 1024     # 16 kB
FUNCTIONAL_UNIT_BITWIDTH = 32

# Digital clock used by the chip's tile/PU logic (paper uses 1 GHz).
CHIP_DIGITAL_FREQUENCY_MHZ = 1000.0

# On-chip network (NoC) electrical parameters, shared by the PU / tile / chip
# interconnects. See chip_architecture.InterconnectConfig.
NOC_TOPOLOGY = "mesh"                              # "mesh" | "bus" | "point_to_point"
NOC_DATA_WIDTH_BITS = 128
NOC_FREQUENCY_MHZ = 1000.0
NOC_ROUTER_ENERGY_PJ_PER_BIT = 0.10               # per-hop router traversal
NOC_WIRE_ENERGY_PJ_PER_BIT_PER_MM = 0.20          # per-hop wire energy
NOC_LINK_LENGTH_MM = 1.0                          # physical length of one hop


# 7. RUN / ANALYSIS SETTINGS

# --- RNG seeds (reproducibility) ---
DATA_RNG_SEED = 0          # random state codes / input codes in the examples
SM_NI_RNG_SEED = 42        # single-memristor non-ideality model
CB_NI_RNG_SEED = 7         # crossbar non-ideality model

# --- Single-crossbar non-ideal examples (example_usage_nonideal / _full) ---
# Time between programming and the first read.
ELAPSED_TIME_S = 600.0     # 10 minutes
# example_usage_full time-series sweep step.
TIMESERIES_STEP_S = 25

# --- MNIST decay analysis (example_usage_nn_decay.py) ---
# Elapsed times after programming at which the network is measured.
NN_DECAY_TIME_POINTS_S = [0.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0]
# Test images pushed through the FULL non-ideal hardware path at each time point.
NN_DECAY_TEST_SUBSET_N = 300
# Images used only to calibrate per-layer input scaling (activation -> DAC range).
NN_DECAY_CALIB_N = 512
# Images used to calibrate each tile's ADC full-scale to the real current swing.
NN_DECAY_CALIB_ADC_N = 128
NN_DECAY_ADC_FULLSCALE_MARGIN = 1.2
# ADC choice for the crossbar read-out (choice 7 -> 4-bit).
NN_DECAY_ADC_CHOICE = 7
# Include the single-device transient read noise (re-drawn every read).
NN_DECAY_INCLUDE_DEVICE_READ_NOISE = True
# Gate off unused output columns (no current / no ADC energy on them).
NN_DECAY_ENABLE_COLUMN_GATING = True
NN_DECAY_BASE_SEED = 12345

# Non-ideality overrides specific to the NN-decay analysis: a longer retention
# study, a current ceiling above the differential-read common mode, and a 1T1R
# select-transistor on-resistance.
NN_DECAY_RETENTION_TAU_S = 36000.0
NN_DECAY_CURRENT_SATURATION_A = 2.0e-3
NN_DECAY_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM = 200.0

# --- MNIST training hyper-parameters (mnist_ann.py) ---
MNIST_HIDDEN_UNITS = 128          # maps cleanly onto a 128-col crossbar
MNIST_EPOCHS = 5
MNIST_BATCH_SIZE = 128
MNIST_LEARNING_RATE = 1e-3
MNIST_SEED = 0
MNIST_WEIGHTS_PATH = "mnist_mlp_weights.npz"


# 8. BUILDER FUNCTIONS  (turn the values above into objects)
#    You rarely need to edit below this line.

def build_device() -> MemristorDevice:
    """Construct the measured multi-state memristor device."""
    return MemristorDevice(
        conductance_levels_s=CONDUCTANCE_LEVELS_S,
        pulse_width_s=_WRITE_PULSE_WIDTH_S,
        pulse_interval_s=_WRITE_PULSE_INTERVAL_S,
        write_amplitude_v=_WRITE_AMPLITUDE_V,
        read_voltage_v=READ_VOLTAGE_V,
        read_time_s=READ_TIME_S,
        device_area_um2=DEVICE_AREA_UM2,
        initial_level_index=DEVICE_INITIAL_LEVEL_INDEX,
        name=DEVICE_NAME,
    )


def build_crossbar_config(**overrides) -> CrossbarConfig:
    """Construct the integrated-crossbar config (override any field by keyword)."""
    config = CrossbarConfig(
        rows=XBAR_ROWS,
        cols=XBAR_COLS,
        read_voltage_v=READ_VOLTAGE_V,
        read_time_s=READ_TIME_S,
        write_amplitude_v=_WRITE_AMPLITUDE_V,
        write_pulse_width_s=_WRITE_PULSE_WIDTH_S,
        load_resistance_ohm=LOAD_RESISTANCE_OHM,
        wire_resistance_ohm=WIRE_RESISTANCE_OHM,
        wire_capacitance_f=WIRE_CAPACITANCE_F,
        adc_choice=ADC_CHOICE,
        dac_choice=DAC_CHOICE,
        input_precision_bits=INPUT_PRECISION_BITS,
        weight_precision_bits=WEIGHT_PRECISION_BITS,
        output_precision_bits=OUTPUT_PRECISION_BITS,
        digital_frequency_mhz=DIGITAL_FREQUENCY_MHZ,
        tech_node_nm=TECH_NODE_NM,
        num_adcs=NUM_ADCS,
        num_dacs=NUM_DACS,
        num_input_regs=NUM_INPUT_REGS,
        num_output_regs=NUM_OUTPUT_REGS,
        num_shiftregs=NUM_SHIFTREGS,
        num_adders=NUM_ADDERS,
        name=CROSSBAR_NAME,
    )
    if overrides:
        config = dataclasses.replace(config, **overrides)
        # The read time is derived from the converter and the column RC, so an
        # override that moves either has to move it too.  An explicit
        # read_time_s in the overrides still wins.
        if "read_time_s" not in overrides and (
            {"adc_choice", "rows", "wire_resistance_ohm", "wire_capacitance_f"}
            & set(overrides)
        ):
            config = dataclasses.replace(
                config,
                read_time_s=derive_read_time_s(
                    adc_choice=config.adc_choice,
                    rows=config.rows,
                    wire_resistance_ohm=config.wire_resistance_ohm,
                    wire_capacitance_f=config.wire_capacitance_f,
                ),
            )
    return config


def build_crossbar(**overrides) -> IntegratedMemristorCrossbar:
    """Construct a fully integrated crossbar from the central config."""
    return IntegratedMemristorCrossbar(
        config=build_crossbar_config(**overrides),
        device=build_device(),
    )


def build_sign_storage(
    tech_node_nm: int = TECH_NODE_NM,
    frequency_mhz: float = CHIP_DIGITAL_FREQUENCY_MHZ,
    mode: Optional[str] = None,
):
    """
    Return the per-cell sign-storage element for the selected SIGN_STORAGE_MODE.

    A dispatcher only; the numeric hardware data lives in hardware_peripherals.py.
      "register_2t1r"   -> Reg(bitwidth=1)   (volatile CMOS)
      "spintronic_mram" -> STTMRAMSignCell   (non-volatile MTJ)
    """
    from hardware_peripherals import Reg, STTMRAMSignCell

    mode = SIGN_STORAGE_MODE if mode is None else mode
    if mode == "spintronic_mram":
        return STTMRAMSignCell(
            tech_node_nm=tech_node_nm, frequency_mhz=frequency_mhz,
            cell_area_f2=MRAM_CELL_AREA_F2,
            r_parallel_ohm=MRAM_R_PARALLEL_OHM,
            tmr_ratio=MRAM_TMR_RATIO,
            read_voltage_v=MRAM_READ_VOLTAGE_V,
            write_current_a=MRAM_WRITE_CURRENT_A,
            write_voltage_v=MRAM_WRITE_VOLTAGE_V,
            write_time_ns=MRAM_WRITE_TIME_NS,
        )
    if mode == "register_2t1r":
        return Reg(bitwidth=1, tech_node_nm=tech_node_nm, frequency_mhz=frequency_mhz)
    raise ValueError(
        f"unknown SIGN_STORAGE_MODE {mode!r}; "
        "use 'register_2t1r' or 'spintronic_mram'"
    )


def build_single_memristor_ni_config(**overrides) -> SingleMemristorNonIdealityConfig:
    """Single-memristor non-ideality config (override any field by keyword)."""
    config = SingleMemristorNonIdealityConfig(
        enabled=SM_NI_ENABLED,
        device_to_device_sigma_rel=SM_NI_DEVICE_TO_DEVICE_SIGMA_REL,
        programming_error_sigma_rel=SM_NI_PROGRAMMING_ERROR_SIGMA_REL,
        cycle_to_cycle_sigma_rel=SM_NI_CYCLE_TO_CYCLE_SIGMA_REL,
        read_noise_sigma_rel=SM_NI_READ_NOISE_SIGMA_REL,
        read_noise_sigma_abs_s=SM_NI_READ_NOISE_SIGMA_ABS_S,
        stuck_at_probability=SM_NI_STUCK_AT_PROBABILITY,
        stuck_at_lrs_fraction=SM_NI_STUCK_AT_LRS_FRACTION,
        retention_model=SM_NI_RETENTION_MODEL,
        retention_tau_s=SM_NI_RETENTION_TAU_S,
        retention_beta=SM_NI_RETENTION_BETA,
        retention_alpha=SM_NI_RETENTION_ALPHA,
        retention_floor_s=SM_NI_RETENTION_FLOOR_S,
        clip_to_measured_range=SM_NI_CLIP_TO_MEASURED_RANGE,
        minimum_conductance_s=SM_NI_MINIMUM_CONDUCTANCE_S,
        potentiation_step_fraction=SM_NI_POTENTIATION_STEP_FRACTION,
        depression_step_fraction=SM_NI_DEPRESSION_STEP_FRACTION,
    )
    return dataclasses.replace(config, **overrides) if overrides else config


def build_crossbar_ni_config(**overrides) -> CrossbarNonIdealityConfig:
    """Crossbar-level non-ideality config (override any field by keyword)."""
    config = CrossbarNonIdealityConfig(
        enabled=CB_NI_ENABLED,
        wire_resistance_ohm=CB_NI_WIRE_RESISTANCE_OHM,
        # PER-CELL -> COLUMN.  Both series terms are quoted per cell, while
        # `_crossbar_ni` applies them as a column gain 1/(1 + G_col*r_s).
        parasitic_series_resistance_ohm=(CB_NI_PARASITIC_SERIES_RESISTANCE_OHM
                                         / XBAR_ROWS),
        parasitic_capacitance_f=CB_NI_PARASITIC_CAPACITANCE_F,
        read_noise_current_a=CB_NI_READ_NOISE_CURRENT_A,
        read_noise_relative=CB_NI_READ_NOISE_RELATIVE,
        programming_voltage_drop=CB_NI_PROGRAMMING_VOLTAGE_DROP,
        set_threshold_fraction=CB_NI_SET_THRESHOLD_FRACTION,
        programming_overdrive_exponent=CB_NI_PROGRAMMING_OVERDRIVE_EXPONENT,
        dac_offset_v=CB_NI_DAC_OFFSET_V,
        dac_gain_error=CB_NI_DAC_GAIN_ERROR,
        adc_offset_a=CB_NI_ADC_OFFSET_A,
        adc_gain_error=CB_NI_ADC_GAIN_ERROR,
        current_saturation_a=CB_NI_CURRENT_SATURATION_A,
        temperature_coefficient_per_k=CB_NI_TEMPERATURE_COEFFICIENT_PER_K,
        thermal_resistance_cell_k_per_w=CB_NI_THERMAL_RESISTANCE_CELL_K_PER_W,
        thermal_resistance_array_k_per_w=CB_NI_THERMAL_RESISTANCE_ARRAY_K_PER_W,
        max_temperature_rise_k=CB_NI_MAX_TEMPERATURE_RISE_K,
        comparator_offset_a=CB_NI_COMPARATOR_OFFSET_A,
        access_transistor_on_resistance_ohm=(
            CB_NI_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM / XBAR_ROWS),
    )
    return dataclasses.replace(config, **overrides) if overrides else config


def build_nn_decay_single_memristor_ni_config() -> SingleMemristorNonIdealityConfig:
    """Single-memristor NI config for the NN-decay analysis (longer retention)."""
    return build_single_memristor_ni_config(retention_tau_s=NN_DECAY_RETENTION_TAU_S)


# Crossbar NI as the studies see it (MLP / ResNet / VGG / detector).  Two
# overrides on the single-crossbar defaults above:
#   * CURRENT CEILING must sit above the 2T1R differential pair's common mode,
#     or it clips both rails equally and erases their difference.
#   * ACCESS TRANSISTOR on-resistance is in series with every cell in a 1T1R/2T1R
#     array, so it matches TWOT1R_ON_RESISTANCE_OHM rather than 0.0.
WORKFLOW_CURRENT_SATURATION_A = 2.0e-3
# Ohms PER CELL, like the constant it shadows.  Decoupled from
# TWOT1R_ON_RESISTANCE_OHM: that is the 2T1R cell's own R_on, this is the select
# transistor as the array sees it.
WORKFLOW_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM = 500.0     # ohm PER CELL


def build_workflow_crossbar_ni_config(**overrides) -> CrossbarNonIdealityConfig:
    """
    The crossbar non-idealities the MLP / ResNet / VGG / detector studies run with.

    Pass this to `IMCProcessor(crossbar_ni=...)` (numpy datapath) or let
    `HWConfig.crossbar_ni` pick it up (torch datapath) so both halves of a
    hardware-aware experiment see the same array.
    """
    cfg = build_crossbar_ni_config(
        current_saturation_a=WORKFLOW_CURRENT_SATURATION_A,
        # Per-cell, so it takes the same /rows conversion the base builder does
        # to its own constants - an override bypasses that, being applied after.
        access_transistor_on_resistance_ohm=(
            WORKFLOW_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM / XBAR_ROWS),
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def build_nn_decay_crossbar_ni_config() -> CrossbarNonIdealityConfig:
    """Crossbar NI config for the NN-decay analysis (differential-read ceiling + 1T1R)."""
    return build_crossbar_ni_config(
        current_saturation_a=NN_DECAY_CURRENT_SATURATION_A,
        access_transistor_on_resistance_ohm=NN_DECAY_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM,
    )


def build_chip():
    """Build the chip (chip -> tile -> PU -> array) from the central config."""
    # Imported here to avoid import-order coupling with chip_architecture.
    from chip_architecture import (
        BufferConfig, IOPortConfig, FunctionalUnitConfig, InterconnectConfig,
        PUConfig, TileConfig, ChipConfig, Chip,
    )

    def _noc(name: str) -> InterconnectConfig:
        return InterconnectConfig(
            name=name,
            data_width_bits=NOC_DATA_WIDTH_BITS,
            frequency_mhz=NOC_FREQUENCY_MHZ,
            tech_node_nm=TECH_NODE_NM,
            topology=NOC_TOPOLOGY,
            router_energy_pj_per_bit=NOC_ROUTER_ENERGY_PJ_PER_BIT,
            wire_energy_pj_per_bit_per_mm=NOC_WIRE_ENERGY_PJ_PER_BIT_PER_MM,
            link_length_mm=NOC_LINK_LENGTH_MM,
        )

    pu_cfg = PUConfig(
        num_arrays=PU_NUM_ARRAYS,
        num_pu_registers=PU_NUM_REGISTERS,
        pu_register_length_bits=PU_REGISTER_LENGTH_BITS,
        input_precision_bits=INPUT_PRECISION_BITS,
        frequency_mhz=CHIP_DIGITAL_FREQUENCY_MHZ,
        tech_node_nm=TECH_NODE_NM,
        interconnect=_noc("pu_noc"),
    )
    tile_cfg = TileConfig(
        num_pus=TILE_NUM_PUS,
        num_general_registers=TILE_NUM_GENERAL_REGISTERS,
        num_vector_registers=TILE_NUM_VECTOR_REGISTERS,
        register_length_bits=TILE_REGISTER_LENGTH_BITS,
        frequency_mhz=CHIP_DIGITAL_FREQUENCY_MHZ,
        tech_node_nm=TECH_NODE_NM,
        pu=pu_cfg,
        buffer=BufferConfig(
            capacity_bits=INSTRUCTION_BUFFER_CAPACITY_BITS,
            data_width_bits=NOC_DATA_WIDTH_BITS,
            frequency_mhz=CHIP_DIGITAL_FREQUENCY_MHZ,
            tech_node_nm=TECH_NODE_NM,
        ),
        io=IOPortConfig(
            data_width_bits=NOC_DATA_WIDTH_BITS,
            frequency_mhz=CHIP_DIGITAL_FREQUENCY_MHZ,
            tech_node_nm=TECH_NODE_NM,
        ),
        functional_unit=FunctionalUnitConfig(
            bitwidth=FUNCTIONAL_UNIT_BITWIDTH,
            tech_node_nm=TECH_NODE_NM,
            frequency_mhz=CHIP_DIGITAL_FREQUENCY_MHZ,
        ),
        interconnect=_noc("tile_noc"),
    )
    chip_cfg = ChipConfig(num_tiles=CHIP_NUM_TILES, tile=tile_cfg, interconnect=_noc("chip_noc"))
    return Chip(chip_cfg, crossbar_factory=build_crossbar)
