"""
riscv_imc_workflow.py

RISC-V-like instruction workflow for the integrated memristor crossbar.

Adds the instruction-set / dataflow layer that `chip_architecture.py` leaves out,
following

    H. Zhou et al., "Heterogeneous Instruction Set Architecture for
    RRAM-enabled In-memory Computing", NANOARCH '23.

It turns the physical tile -> processing-unit -> array chip into a machine that
executes a small RISC-V-vector-style program.  A neural network is compiled into
that program, and every layer operation is carried out by the chip using the four
crossbar operation modes of the paper (Fig. 5c):

    point  (vmmp)  : one row  x one column   - scalar multiply / single cell
    column (vmmc)  : all rows x one column   - reduction (dot product down a col)
    row    (vmmr)  : one row  x all columns  - broadcast / per-output scaling
    matrix (vmm)   : all rows x all columns  - full VMM

The MUX decoder of Fig. 5b selects "one" vs "all" on each axis; the high bit of
the row and column address fields is the mode bit, as in the paper's 32-bit
operational address (Fig. 6).

Three deviations from the paper, specific to this chip:

1.  Digit-serial 4-bit input instead of bit-serial 1-bit.  With 16 conductance
    states the DAC is 4-bit, so an 8-bit input is two nibbles (2 VMM + 1
    shift&add) rather than eight 1-bit pulses.  See `DigitSerialInput` and
    `IMCProcessor._matrix_vmm`.

2.  Sign-magnitude weights in a 2T1R signed cell instead of differential pairs.
    Each weight stores |w| on one 4-bit memristor and its sign in a 1-bit cell
    whose bit drives two access transistors, steering the read current onto BL+
    or BL-.  A column develops

        I_pos = sum_{cells with sign=+} V_i * G_ij
        I_neg = sum_{cells with sign=-} V_i * G_ij

    and the differential sense amp forms I_pos - I_neg (a g_min dummy reference
    on each rail removes the common mode).  One memristor and one array per
    weight, so device count and read energy roughly halve.  The physics and
    non-idealities are in `TwoT1RConfig`; the sign STORAGE element is selected by
    `simulation_config.SIGN_STORAGE_MODE` ("register_2t1r" or "spintronic_mram").

3.  1 GHz digital clock, fixed.  All digital periphery runs at
    `DIGITAL_CLOCK_MHZ = 1000`.  Latency cancels in TOPS/W, so the clock buys
    throughput while the digit-serial single-array scheme buys the energy.

Accounting convention: area um^2, power W, latency ns, energy nJ (= W*ns);
energies are reported in J at the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, sqrt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hardware_peripherals import Reg, ShiftReg, Adder
from integrated_crossbar import IntegratedMemristorCrossbar, CrossbarConfig
from memristor_device import MemristorDevice


# Constants

_NJ_TO_J = 1e-9

# Digital clock for every peripheral in this workflow (paper: 1 GHz digital).
DIGITAL_CLOCK_MHZ = 1000.0

# Device is 16-state => 4-bit. The DAC is therefore 4-bit (radix-16 digit) and
# the ADC is 4-bit. (hardware_peripherals tables: DAC choice 4 -> 4-bit,
# ADC choice 7 -> 4-bit.)
DAC_CHOICE_4BIT = 4
ADC_CHOICE_4BIT = 7

# Input is 8-bit; a 4-bit DAC consumes it 4 bits (one nibble) at a time.
DEFAULT_INPUT_PRECISION_BITS = 8
DEFAULT_DIGIT_BITS = 4


# 1. Operation modes + 32-bit operational address (paper Fig. 5c, Fig. 6)

class OperationMode(Enum):
    """The four crossbar operation modes selected by the row/column MUX decoder."""
    POINT = "vmmp"      # one row,  one column
    COLUMN = "vmmc"     # all rows, one column   (reduction)
    ROW = "vmmr"        # one row,  all columns  (broadcast)
    MATRIX = "vmm"      # all rows, all columns  (full VMM)

    @property
    def row_single(self) -> bool:
        """True when the row MUX decoder activates exactly one row."""
        return self in (OperationMode.POINT, OperationMode.ROW)

    @property
    def col_single(self) -> bool:
        """True when the column MUX decoder activates exactly one column."""
        return self in (OperationMode.POINT, OperationMode.COLUMN)

    @staticmethod
    def from_flags(row_single: bool, col_single: bool) -> "OperationMode":
        if row_single and col_single:
            return OperationMode.POINT
        if row_single and not col_single:
            return OperationMode.ROW
        if (not row_single) and col_single:
            return OperationMode.COLUMN
        return OperationMode.MATRIX


@dataclass
class IMCAddress:
    """
    32-bit operational address (paper Fig. 6).

    Bit layout (low -> high):
        [ 0.. 8]  column index   (9 bits)
        [ 9    ]  column mode bit (1 = single column via decoder, 0 = all columns)
        [10..18]  row index      (9 bits)
        [19    ]  row mode bit    (1 = single row via decoder,    0 = all rows)
        [20..24]  array index in the PU   (5 bits, up to 32 arrays)
        [25..29]  PU index in the tile    (5 bits, up to 32 PUs)
        [30..31]  reserved (capacity expansion)

    The two mode bits are the MUX-decoder selectors of Fig. 5b and together pick
    one of the four `OperationMode`s.
    """
    mode: OperationMode
    row_index: int = 0
    col_index: int = 0
    array: int = 0
    pu: int = 0

    def encode(self) -> int:
        if not (0 <= self.row_index < 512 and 0 <= self.col_index < 512):
            raise ValueError("row/col index must fit in 9 bits (0..511)")
        if not (0 <= self.array < 32 and 0 <= self.pu < 32):
            raise ValueError("array/pu index must fit in 5 bits (0..31)")
        col_field = (self.col_index & 0x1FF) | ((1 << 9) if self.mode.col_single else 0)
        row_field = (self.row_index & 0x1FF) | ((1 << 9) if self.mode.row_single else 0)
        word = col_field
        word |= row_field << 10
        word |= (self.array & 0x1F) << 20
        word |= (self.pu & 0x1F) << 25
        return word & 0xFFFFFFFF

    @staticmethod
    def decode(word: int) -> "IMCAddress":
        col_index = word & 0x1FF
        col_single = bool((word >> 9) & 0x1)
        row_index = (word >> 10) & 0x1FF
        row_single = bool((word >> 19) & 0x1)
        array = (word >> 20) & 0x1F
        pu = (word >> 25) & 0x1F
        return IMCAddress(
            mode=OperationMode.from_flags(row_single, col_single),
            row_index=row_index, col_index=col_index, array=array, pu=pu,
        )

    def __str__(self) -> str:
        return (f"0x{self.encode():08x}"
                f"[{self.mode.value} pu{self.pu} arr{self.array}"
                f" r{self.row_index} c{self.col_index}]")


# 2. Instruction set (RISC-V-vector style, 32-bit), paper Sec. 4

class Opcode(Enum):
    VMM = "vmm"        # analog read in one of the four modes -> PU ADC registers
    PROG = "prog"      # write cells (progp single cell / progc whole column)
    PUSLL = "pusll"    # PU logical-shift-left immediate (digit/precision weighting)
    PUADD = "puadd"    # PU add (accumulate shifted/partial sums)
    PUSUB = "pusub"    # PU subtract (sign handling / residuals)
    PURELU = "purelu"  # on-chip ReLU via the sign register + sense-amp comparator
    PUMAX = "pumax"    # on-chip max-reduce (pooling), uses column-mode style reduce
    MV = "mv"          # tile <-> PU / inter-tile data move


# RISC-V opcode bytes used purely so the encoded words look RVV-like when printed.
_OPCODE_BITS = {
    Opcode.VMM: 0x0B, Opcode.PROG: 0x2B, Opcode.PUSLL: 0x13, Opcode.PUADD: 0x33,
    Opcode.PUSUB: 0x33, Opcode.PURELU: 0x5B, Opcode.PUMAX: 0x5B, Opcode.MV: 0x03,
}
_MODE_FUNCT3 = {
    OperationMode.POINT: 0b000, OperationMode.COLUMN: 0b001,
    OperationMode.ROW: 0b010, OperationMode.MATRIX: 0b011,
}


@dataclass
class Instruction:
    """
    One decoded instruction.

    Not every field is used by every opcode; unused ones stay None.  `vd`, `vs1`,
    `vs2` are register *names* (PU registers "R0".."R7" or tile vector regs
    "v0".."v31").  `addr` is the operational address for analog ops.  `imm` is the
    immediate (shift amount, or signed op_V voltage code for writes).
    """
    op: Opcode
    addr: Optional[IMCAddress] = None
    vd: Optional[str] = None
    vs1: Optional[str] = None
    vs2: Optional[str] = None
    imm: Optional[int] = None
    comment: str = ""

    def encode(self) -> int:
        """Pack into a 32-bit word (RVV-flavoured; exact bits are illustrative)."""
        word = _OPCODE_BITS[self.op] & 0x7F
        if self.addr is not None:
            word |= _MODE_FUNCT3[self.addr.mode] << 12
        # Source/dest register numbers, where present (5-bit RVV fields).
        word |= (_reg_num(self.vd) & 0x1F) << 7
        word |= (_reg_num(self.vs1) & 0x1F) << 15
        word |= (_reg_num(self.vs2) & 0x1F) << 20
        if self.imm is not None:
            word |= (self.imm & 0x7F) << 25
        return word & 0xFFFFFFFF

    def __str__(self) -> str:
        if self.op == Opcode.VMM:
            mode = self.addr.mode.value
            src = self.vs1 or "vs"
            asm = f"{mode:6s} -, {self.addr}, {src}"
        elif self.op == Opcode.PROG:
            asm = f"prog   {self.addr}, {self.vs1}, #opV={self.imm}"
        elif self.op == Opcode.PUSLL:
            asm = f"pusll  {self.vd}, {self.vs1}, #{self.imm}"
        elif self.op in (Opcode.PUADD, Opcode.PUSUB):
            asm = f"{self.op.value:6s} {self.vd}, {self.vs1}, {self.vs2}"
        elif self.op == Opcode.PURELU:
            asm = f"purelu {self.vd}, {self.vs1}"
        elif self.op == Opcode.PUMAX:
            asm = f"pumax  {self.vd}, {self.vs1}"
        else:
            asm = f"{self.op.value:6s} {self.vd}, {self.vs1}"
        tail = f"   ; {self.comment}" if self.comment else ""
        return f"{self.encode():08x}  {asm}{tail}"


def _reg_num(name: Optional[str]) -> int:
    if not name:
        return 0
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


# 3. Sign-magnitude weight mapping (sign register, single array)  - deviation #2

@dataclass
class SignMagnitudeWeights:
    """
    A signed weight matrix stored as |w| on one 4-bit array + a sign register.

    magnitude_codes : int (rows, cols) in 0..n_states-1  (4-bit state code)
    conductance_s   : float (rows, cols) on-grid magnitude conductance (levels[code])
    sign_bits       : uint8 (rows, cols), 0 = positive, 1 = negative  (the register)
    scale_s_per_w   : Siemens of (G - g_min) per unit weight
    g_min_s         : the baseline (off) conductance subtracted as the reference rail
    """
    magnitude_codes: np.ndarray
    conductance_s: np.ndarray
    sign_bits: np.ndarray
    scale_s_per_w: float          # scalar, or a (cols,) vector for per-column scaling
    g_min_s: float

    @property
    def signs(self) -> np.ndarray:
        """+1 / -1 float matrix from the stored sign bits."""
        return np.where(self.sign_bits == 0, 1.0, -1.0)

    def scale_for(self, out_count: int):
        """
        The S-per-unit-weight factor for the first `out_count` columns.

        A scalar layer scale applies to every column; a per-column scale is a
        vector and has to be sliced to the columns this tile actually drives.
        """
        s = self.scale_s_per_w
        return s[:out_count] if isinstance(s, np.ndarray) else s

    @property
    def effective_weight(self) -> np.ndarray:
        """
        The signed weight the array actually realises:

            w_eff = sign * (G_on_grid - g_min) / scale

        The (G - g_min) subtraction is the dummy-column common-mode removal that
        lets a sign-magnitude array represent a zero weight.
        """
        return self.signs * (self.conductance_s - self.g_min_s) / self.scale_s_per_w

    @staticmethod
    def from_weights(
        W: np.ndarray, levels_s: Sequence[float],
        g_min: Optional[float] = None, g_max: Optional[float] = None,
    ) -> "SignMagnitudeWeights":
        levels = np.asarray(levels_s, dtype=float)
        g_min = float(levels.min()) if g_min is None else g_min
        g_max = float(levels.max()) if g_max is None else g_max

        w_absmax = float(np.max(np.abs(W)))
        if w_absmax <= 0:
            w_absmax = 1.0
        scale = (g_max - g_min) / w_absmax                  # S per unit |w|

        # Magnitude -> target conductance in [g_min, g_max] -> nearest 4-bit state.
        target = g_min + np.abs(W) * scale
        codes = np.argmin(np.abs(target[..., None] - levels), axis=-1).astype(int)
        conductance = levels[codes]

        sign_bits = (W < 0).astype(np.uint8)                # 0=+, 1=-
        return SignMagnitudeWeights(
            magnitude_codes=codes, conductance_s=conductance,
            sign_bits=sign_bits, scale_s_per_w=scale, g_min_s=g_min,
        )


# 3b. 2T1R signed-cell model: two access transistors steer |w| onto +/- rails

@dataclass
class TwoT1RConfig:
    """
    Physical model + non-idealities of the 2T1R signed synapse.

    One memristor holding |w| plus two access transistors.  The 1-bit sign turns
    exactly one transistor on, routing the read current onto BL+ or BL-; the
    column output is I_pos - I_neg.

    on_resistance_ohm
        In series with the memristor: G_branch = G / (1 + G*R_on).
    off_resistance_ohm
        Leakage of the unselected transistor onto the wrong rail (sign crosstalk).
        inf disables it.
    rail_gain_mismatch
        Relative gain difference of the + and - sense paths: 1 +/- m/2.
    rail_offset_a
        Systematic differential offset current of the sense amp.
    separate_rail_adc
        False (default): analog subtraction, one bipolar ADC per column.  True:
        one unipolar ADC per rail, subtracted digitally.
    transistor_area_um2
        Area of one access transistor (two per cell).
    transistor_switch_energy_nj
        Gate-drive energy charged once per access transistor per read.

    An all-ideal config (R_on=0, R_off=inf, no mismatch/offset) reproduces the
    ideal sign-magnitude result exactly.
    """
    enabled: bool = True
    on_resistance_ohm: float = 0.0
    off_resistance_ohm: float = float("inf")
    rail_gain_mismatch: float = 0.0
    rail_offset_a: float = 0.0
    separate_rail_adc: bool = False
    transistor_area_um2: float = 0.10
    transistor_switch_energy_nj: float = 0.0

    transistors_per_cell: int = 2

    def branch_on_conductance(self, G):
        """Conductance of the selected (ON) branch: memristor in series with R_on."""
        G = np.asarray(G, dtype=float)
        if self.on_resistance_ohm <= 0.0:
            return G
        return G / (1.0 + G * self.on_resistance_ohm)

    def branch_off_conductance(self, G):
        """Leakage conductance of the un-selected (OFF) branch onto the wrong rail."""
        G = np.asarray(G, dtype=float)
        if not np.isfinite(self.off_resistance_ohm):
            return np.zeros_like(G)
        return G / (1.0 + G * self.off_resistance_ohm)

    @property
    def gain_pos(self) -> float:
        return 1.0 + 0.5 * self.rail_gain_mismatch

    @property
    def gain_neg(self) -> float:
        return 1.0 - 0.5 * self.rail_gain_mismatch


# 4. Digit-serial input encoder (8-bit -> 4-bit nibbles)  - deviation #1

@dataclass
class DigitSerialInput:
    """
    Split an 8-bit input vector into radix-2^digit_bits digits.

    For 8-bit input and a 4-bit DAC this yields TWO nibbles (low, high) instead of
    eight 1-bit slices.  Each digit `d` is weighted by `2^(digit_bits*d)` and that
    weighting is applied by the PU shift&adder AFTER digitisation (`<< 4` for the
    high nibble).
    """
    input_precision_bits: int = DEFAULT_INPUT_PRECISION_BITS
    digit_bits: int = DEFAULT_DIGIT_BITS

    @property
    def num_digits(self) -> int:
        return ceil(self.input_precision_bits / self.digit_bits)

    @property
    def digit_max_code(self) -> int:
        return (1 << self.digit_bits) - 1

    @property
    def input_max_code(self) -> int:
        return (1 << self.input_precision_bits) - 1

    def to_codes(self, x: np.ndarray, in_scale: float) -> np.ndarray:
        """Quantise activations to 8-bit integer codes (non-negative)."""
        codes = np.round((np.asarray(x, dtype=float) / in_scale) * self.input_max_code)
        return np.clip(codes, 0, self.input_max_code).astype(int)

    def digits(self, codes: np.ndarray) -> List[np.ndarray]:
        """Return [digit0 (LSBs), digit1, ...] each in 0..digit_max_code."""
        mask = self.digit_max_code
        out = []
        for d in range(self.num_digits):
            out.append((codes >> (self.digit_bits * d)) & mask)
        return out

    def digit_weight(self, d: int) -> int:
        """Binary weight applied to digit `d` (the PU shift-left amount is its log2)."""
        return 1 << (self.digit_bits * d)


# 5. Per-array deployment record + workflow crossbar factory

def build_workflow_crossbar() -> IntegratedMemristorCrossbar:
    """
    A crossbar matched to this workflow: 4-bit DAC + 4-bit ADC, 1 GHz digital.

    Size, device, read voltage etc. come from the user's `simulation_config`; only
    the converter choices and the digital clock are overridden here so the array
    reflects the digit-serial datapath and the 1 GHz periphery.
    """
    from simulation_config import build_crossbar_config, build_device
    cfg = build_crossbar_config(
        adc_choice=ADC_CHOICE_4BIT,
        dac_choice=DAC_CHOICE_4BIT,
        digital_frequency_mhz=DIGITAL_CLOCK_MHZ,
        input_precision_bits=DEFAULT_INPUT_PRECISION_BITS,
        name="riscv_imc_array",
    )
    return IntegratedMemristorCrossbar(config=cfg, device=build_device())


@dataclass
class ArrayTile:
    """
    One physical array holding a (row-block x col-block) slice of a layer.

    Mapped to (pu, array) inside the chip; carries the sign-magnitude weights, the
    bias-row weights (if this tile owns the layer bias), and the calibrated bipolar
    ADC full-scale used to digitise the signed nibble current.
    """
    pu: int
    array: int
    row_slice: slice
    col_slice: slice
    in_count: int
    out_count: int
    weights: SignMagnitudeWeights
    bias_weights: Optional[SignMagnitudeWeights] = None   # 1 x out_count augmented row
    adc_fullscale_a: float = 0.0          # bipolar full-scale of I_pos - I_neg
    bias_fullscale_a: float = 0.0
    rail_fullscale_a: float = 0.0         # unipolar full-scale of a single rail
    bias_rail_fullscale_a: float = 0.0
    # PER-DIGIT converter references (index = digit, LSB first).  The sense
    # amplifier reference is already re-programmed per read phase, and the digits
    # have different current statistics, so a pooled reference is set by whichever
    # digit swings widest.  Empty falls back to the shared `adc_fullscale_a`.
    adc_fullscale_per_digit: List[float] = field(default_factory=list)
    rail_fullscale_per_digit: List[float] = field(default_factory=list)

    def digit_fullscale_a(self, d: int) -> float:
        """Bipolar converter reference for digit `d`, falling back to the shared one."""
        fs = self.adc_fullscale_per_digit
        return fs[d] if d < len(fs) and fs[d] > 0 else self.adc_fullscale_a

    def digit_rail_fullscale_a(self, d: int) -> float:
        """Unipolar per-rail reference for digit `d`, falling back to the shared one."""
        fs = self.rail_fullscale_per_digit
        return fs[d] if d < len(fs) and fs[d] > 0 else self.rail_fullscale_a


@dataclass
class LinearLayerPlan:
    name: str
    in_features: int
    out_features: int
    has_relu: bool
    in_scale: float
    read_voltage_v: float
    rows: int
    cols: int
    row_tiles: int
    col_tiles: int
    tiles: List[ArrayTile] = field(default_factory=list)


# 6. The processor: executes instructions on the chip, tracking cost + value

@dataclass
class ExecutionCost:
    energy_j: float = 0.0
    latency_ns: float = 0.0
    n_vmm: int = 0
    n_shift_add: int = 0
    n_analog_reads: int = 0
    # Per-ISA-mode read counts (vmm / vmmr / vmmc / vmmp) plus the cells that
    # sparsity kept switched off.  Lets a study report how often each operation
    # mode actually fired instead of assuming everything is a full matrix read.
    mode_counts: Dict[str, int] = field(default_factory=dict)
    cells_skipped: int = 0
    cells_read: int = 0
    # Activation bits pushed across the NoC / staging buffers.  Reported so the
    # movement term can be read against the array term rather than only summed
    # into it - see `IMCProcessor.movement_cost`.
    moved_bits: float = 0.0
    # CRITICAL-PATH latency when independent array reads issue concurrently.
    # `latency_ns` is the serial instruction stream; a layer's tiles hold different
    # weights in different arrays and share no state, so they read at once and the
    # layer costs one tile.  None means no concurrency was modelled at this level,
    # i.e. identical to `latency_ns`; `par_ns` resolves that.
    latency_parallel_ns: Optional[float] = None

    @property
    def par_ns(self) -> float:
        """Critical-path latency, falling back to the serial one when unset."""
        return (self.latency_ns if self.latency_parallel_ns is None
                else self.latency_parallel_ns)

    def add(self, other: "ExecutionCost") -> None:
        # Sequential composition: both latencies accumulate, because `add` is only
        # used for things that really do follow one another.  Concurrency is
        # introduced by `run_layer` setting `latency_parallel_ns` explicitly.
        self.latency_parallel_ns = self.par_ns + other.par_ns
        self.energy_j += other.energy_j
        self.latency_ns += other.latency_ns
        self.n_vmm += other.n_vmm
        self.n_shift_add += other.n_shift_add
        self.n_analog_reads += other.n_analog_reads
        self.cells_skipped += other.cells_skipped
        self.cells_read += other.cells_read
        self.moved_bits += other.moved_bits
        for k, v in other.mode_counts.items():
            self.mode_counts[k] = self.mode_counts.get(k, 0) + v

    @property
    def sparsity_exploited(self) -> float:
        """Fraction of addressed cells that sparsity kept off the rails."""
        total = self.cells_read + self.cells_skipped
        return (self.cells_skipped / total) if total else 0.0


class IMCProcessor:
    """
    Executes the compiled NN program on a real `Chip` instance.

    Produces both the numerical layer output and the area/energy/latency cost,
    reusing the existing crossbar peripheral models for every cost term.
    """

    def __init__(
        self,
        digit_io: Optional[DigitSerialInput] = None,
        crossbar_factory: Callable[[], IntegratedMemristorCrossbar] = build_workflow_crossbar,
        ideal_adc: bool = False,
        twot1r: Optional[TwoT1RConfig] = None,
        sign_storage: Optional[object] = None,
        sparse_modes: bool = True,
        arrays_available: Optional[int] = None,
        charge_skipped_macs: bool = False,
        crossbar_ni: Optional[object] = None,
        rng: Optional[object] = None,
    ):
        # CROSSBAR-LEVEL non-idealities on the analog read path (wire IR drop,
        # parasitic + access-transistor series R, self-heating, current saturation,
        # physical sense noise, converter offset and gain).  `_form_rails` routes
        # the two bit-line currents through the model when one is supplied.
        # None keeps the ideal analog path, which the self-tests want; pass
        # `simulation_config.build_workflow_crossbar_ni_config()` for the real array.
        self._rng = (rng if isinstance(rng, np.random.Generator)
                     else np.random.default_rng(rng))
        # Accept either a built model or a bare config (wrapped here, sharing this
        # processor's RNG so a seeded run is reproducible end to end).
        if crossbar_ni is not None and not hasattr(crossbar_ni, "config"):
            from crossbar_nonidealities import CrossbarNonIdealityModel
            crossbar_ni = CrossbarNonIdealityModel(crossbar_ni, rng=self._rng)
        self.crossbar_ni = crossbar_ni
        # Whether a row whose activation digit is zero still costs peripheral energy.
        # False (default): the ISA's row decoder gates that row off - its DAC is not
        # clocked, its sign cell is not accessed, its access-transistor gates are not
        # driven.  True: a fixed-schedule machine with no per-row gating.  These are
        # two different machines, so only one of this flag and `sparse_modes` can
        # answer for a given study.
        self.charge_skipped_macs = bool(charge_skipped_macs)
        # Exploit the ISA's point/row/column decoders: when the active set collapses
        # to a single row and/or column, address it with the narrow mode.  False pins
        # every read to MATRIX mode.
        self.sparse_modes = bool(sparse_modes)
        # How many arrays the chip can read simultaneously.  A layer's tiles are
        # independent, so they issue together and the layer costs one tile's read.
        # Defaults to the whole chip (tiles x PUs x arrays/PU); pass 1 for strictly
        # serial accounting, or less to cost a partition of the chip.
        if arrays_available is None:
            from simulation_config import (CHIP_NUM_TILES, TILE_NUM_PUS,
                                           PU_NUM_ARRAYS)
            arrays_available = int(CHIP_NUM_TILES) * int(TILE_NUM_PUS) * int(PU_NUM_ARRAYS)
        self.arrays_available = max(1, int(arrays_available))
        self.digit_io = digit_io or DigitSerialInput()
        self.array = crossbar_factory()        # one template array (peripherals shared)
        self.ideal_adc = ideal_adc
        # 2T1R signed-cell physics + non-idealities.  Defaults to the chip's own
        # cell from simulation_config.build_twot1r(); pass `TwoT1RConfig()`
        # explicitly for an all-ideal reference datapath.
        if twot1r is None:
            from simulation_config import build_twot1r
            twot1r = build_twot1r()
        self.twot1r = twot1r

        # Per-cell sign-storage element: volatile 1-bit CMOS register
        # ("register_2t1r") or non-volatile STT-MRAM MTJ ("spintronic_mram").
        # The choice comes from simulation_config; default follows the config.
        if sign_storage is None:
            from simulation_config import build_sign_storage
            sign_storage = build_sign_storage(
                tech_node_nm=self.array.config.tech_node_nm,
                frequency_mhz=DIGITAL_CLOCK_MHZ,
            )
        self.sign_storage = sign_storage
        self.sign_reg = sign_storage           # backward-compatible alias
        self.shiftreg = ShiftReg(bitwidth=self.array.config.output_precision_bits,
                                 tech_node_nm=self.array.config.tech_node_nm,
                                 frequency_mhz=DIGITAL_CLOCK_MHZ)
        self.adder = Adder(bitwidth=self.array.config.output_precision_bits,
                           tech_node_nm=self.array.config.tech_node_nm,
                           frequency_mhz=DIGITAL_CLOCK_MHZ)
        self._clk_ns = 1000.0 / DIGITAL_CLOCK_MHZ

        # Last issued instruction stream (for inspection / printing).
        self.last_program: List[Instruction] = []

    # 2T1R area accounting (two access transistors + a sign bit per cell)

    @property
    def cell_transistor_area_um2(self) -> float:
        """Silicon area of the two access transistors of one 2T1R signed cell."""
        return self.twot1r.transistors_per_cell * self.twot1r.transistor_area_um2

    @property
    def signed_cell_overhead_um2(self) -> float:
        """Per-cell overhead of the sign-magnitude 2T1R scheme (2 transistors + sign bit)."""
        return self.cell_transistor_area_um2 + self.sign_storage.area_um2

    def array_signed_overhead_um2(self, rows: int, cols: int) -> float:
        """Total 2T1R + sign-storage overhead added on top of one memristor array."""
        return rows * cols * self.signed_cell_overhead_um2

    # Sign-storage architecture accounting (register vs spintronic STT-MRAM)

    @property
    def sign_storage_nonvolatile(self) -> bool:
        """True for the STT-MRAM sign cell (state kept at power-down)."""
        return bool(getattr(self.sign_storage, "nonvolatile", False))

    @property
    def sign_storage_label(self) -> str:
        return ("spintronic_mram" if self.sign_storage_nonvolatile
                else "register_2t1r")

    def sign_program_energy_j(self, n_cells: int) -> float:
        """
        One-time energy to WRITE all `n_cells` sign bits.

        The STT-MRAM cell exposes a per-bit switching energy; the volatile CMOS
        register has no dedicated programming energy (it is simply latched), so
        this is ~0 for the register scheme.
        """
        w_nj = getattr(self.sign_storage, "write_energy_nj", 0.0)
        return n_cells * w_nj * _NJ_TO_J

    def sign_standby_power_w(self, n_cells: int) -> float:
        """
        Continuous sign-HOLD (standby-leakage) power for `n_cells` sign bits.

        Only a characterised leakage figure counts here, and neither element in
        `hardware_peripherals` has one, so both report 0 and the comparison rests
        on per-read energy and area.  Nothing else on the chip carries a leakage
        term either, so a sign-plane figure would be the only one and would
        dominate by default.

        The register's real disadvantage is volatility - the sign plane must be
        reloaded after power-down - which is reported as `nonvolatile`.
        """
        return n_cells * float(getattr(self.sign_storage, "standby_power_w", 0.0) or 0.0)

    # Cost primitives (reuse existing peripheral unit energies)

    @property
    def _read_latency_ns(self) -> float:
        a = self.array
        return max(a.config.read_time_s * 1e9, a.dac.latency_ns, a.adc.latency_ns,
                   a.output_reg.latency_ns)

    def select_mode(
        self, n_active_rows: int, n_active_cols: int,
        total_rows: Optional[int] = None, total_cols: Optional[int] = None,
    ) -> OperationMode:
        """
        Pick the ISA operation mode that matches the ACTUAL active set.

        Read energy scales with the lines actually driven and digitised, so the
        cheapest legal mode is the one whose decoder matches the activity:

            1 row  + 1 col   -> POINT   (vmmp)
            1 row  + N cols  -> ROW     (vmmr)   e.g. a bias strip
            N rows + 1 col   -> COLUMN  (vmmc)   e.g. a reduction / pool
            N rows + N cols  -> MATRIX  (vmm)

        This never beats MATRIX per-MAC - only MATRIX amortises the converter over
        a full array.  The sub-modes win by not computing results that would be
        discarded: a total-energy win on sparse and irregular ops.

        The label is derived from the actual active set unconditionally;
        `sparse_modes` selects the machine (see `_analog_read_cost`).
        """
        return OperationMode.from_flags(
            row_single=(n_active_rows <= 1), col_single=(n_active_cols <= 1),
        )

    def _analog_read_cost(
        self, voltages_active: np.ndarray, G_active: np.ndarray, n_active_cols: int,
        n_addressed_cells: Optional[int] = None,
    ) -> ExecutionCost:
        """
        Cost of ONE analog read of one nibble on a single 2T1R array.

        The ISA operation mode is chosen from the active row/column counts (see
        `select_mode`); energy follows the lines actually driven and digitised.

        voltages_active : (n_active_rows,)   nibble row voltages (only nonzero rows)
        G_active        : (n_active_rows, n_active_cols)  magnitude conductances
        n_addressed_cells : cells the dense form would have touched, for the
                            sparsity accounting (defaults to the active count).
        """
        a = self.array
        t2 = self.twot1r
        n_rows = int(voltages_active.shape[0])
        cols = int(n_active_cols)
        addressed = (n_rows * cols if n_addressed_cells is None
                     else int(n_addressed_cells))

        # `sparse_modes` selects WHICH MACHINE is being costed.
        #   True  - this chip: the operational address carries the mode bits, so the
        #           MUX decoder drives exactly the lines the operation needs.
        #   False - a machine with no sub-mode opcodes: every operation drives all
        #           rows and digitises all columns, discarding what it did not want.
        if not self.sparse_modes:
            n_rows, cols = a.config.rows, a.config.cols
            voltages_active = np.full(n_rows, float(a.config.read_voltage_v))
            G_active = np.full((n_rows, cols), a.device_mean_conductance_s)
            addressed = n_rows * cols

        # Rows the periphery is actually clocked for.  With `charge_skipped_macs`
        # that is every ADDRESSED row, including the zero-digit ones.  The CELL
        # current of a 0 V row is still zero, and `e_xbar` follows the voltages.
        addressed_rows = int(round(addressed / cols)) if cols else n_rows
        n_rows_billed = max(addressed_rows, n_rows) if self.charge_skipped_macs else n_rows

        n_cells = n_rows_billed * cols
        mode = self.select_mode(n_rows_billed, cols)

        # Row drivers (DAC) for the billed rows + sign reads for their cells.
        # e_sign uses the selected sign-storage element's per-read energy: a CMOS
        # register access, or the STT-MRAM MTJ sense in the spintronic scheme.
        e_dac = n_rows_billed * a.dac.energy_nj
        e_sign = n_cells * self.sign_storage.energy_nj
        # 2T1R steering: charge the two access-transistor gates of each cell.
        e_tran = t2.transistors_per_cell * n_cells * t2.transistor_switch_energy_nj

        # Crossbar analog energy: t * sum_i V_i^2 * sum_j G_ij  (single array).
        # Driven rows only - a row held at 0 V dissipates nothing in the cells,
        # whatever the periphery around it costs.
        if n_rows > 0 and cols > 0:
            row_power_w = (voltages_active ** 2) @ G_active[:, :cols]
            e_xbar_j = a.config.read_time_s * float(np.sum(row_power_w))
        else:
            e_xbar_j = 0.0

        # Column side: the signed sense chain digitises the difference of the two
        # rails.  Analog-subtraction keeps ONE bipolar ADC per column; digitising
        # each rail separately needs TWO unipolar ADCs per column.
        n_adc_per_col = 2 if t2.separate_rail_adc else 1
        e_adc = n_adc_per_col * cols * a.adc.energy_nj
        e_oreg = cols * a.output_reg.energy_nj

        energy_j = (e_dac + e_sign + e_tran + e_adc + e_oreg) * _NJ_TO_J + e_xbar_j
        return ExecutionCost(
            energy_j=energy_j, latency_ns=self._read_latency_ns,
            n_vmm=1, n_analog_reads=1,
            mode_counts={mode.value: 1},
            cells_read=n_cells,
            cells_skipped=max(addressed - n_cells, 0),
        )

    # Data movement: the term the read-path budget used to leave out

    def movement_cost(self, num_bits: float, hops: Optional[float] = None,
                      buffer_accesses: int = 2) -> ExecutionCost:
        """
        Energy and latency of moving `num_bits` of activation across the chip.

        Three terms, all from `simulation_config`'s NoC/buffer parameters:

            router   NOC_ROUTER_ENERGY_PJ_PER_BIT        per bit per hop
            wire     NOC_WIRE_ENERGY_PJ_PER_BIT_PER_MM   per bit per hop per mm
            buffer   Buffer.ACCESS_ENERGY_PJ_PER_BIT     per bit per access

        `buffer_accesses` defaults to 2 - one write into the staging buffer and one
        read out - which is what a produce-then-consume hop costs.  Movement is the
        dominant energy term on any real accelerator, so leaving it free would bias
        TOPS/W in the flattering direction.
        """
        if num_bits <= 0:
            return ExecutionCost()
        from simulation_config import (
            NOC_ROUTER_ENERGY_PJ_PER_BIT, NOC_WIRE_ENERGY_PJ_PER_BIT_PER_MM,
            NOC_LINK_LENGTH_MM, NOC_FREQUENCY_MHZ, TECH_NODE_NM,
        )
        from chip_architecture import Buffer

        if hops is None:
            hops = self.average_noc_hops
        pj_per_bit_hop = (NOC_ROUTER_ENERGY_PJ_PER_BIT
                          + NOC_WIRE_ENERGY_PJ_PER_BIT_PER_MM * NOC_LINK_LENGTH_MM)
        buf_pj_per_bit = (Buffer.ACCESS_ENERGY_PJ_PER_BIT_65NM
                          * (TECH_NODE_NM / 65.0) ** 2)
        energy_pj = num_bits * (hops * pj_per_bit_hop
                                + buffer_accesses * buf_pj_per_bit)
        return ExecutionCost(
            energy_j=energy_pj * 1e-12,
            latency_ns=hops * (1000.0 / float(NOC_FREQUENCY_MHZ)),
            moved_bits=float(num_bits),
        )

    def noc_hops_for(self, n_arrays: Optional[int] = None) -> float:
        """
        Mean Manhattan hop count for traffic among `n_arrays` arrays (~2/3 of a
        mesh side).

        Scoped to the arrays the LAYER occupies, not the whole chip: a mapper places
        a layer's tiles together, so an 8-array MLP layer does not pay the hop count
        of an 8,230-array VGG layer.  None means chip-wide.
        """
        from simulation_config import (CHIP_NUM_TILES, TILE_NUM_PUS, PU_NUM_ARRAYS,
                                       NOC_TOPOLOGY)
        if NOC_TOPOLOGY in ("bus", "point_to_point"):
            return 1.0
        chip_nodes = max(1, int(CHIP_NUM_TILES) * int(TILE_NUM_PUS))
        if n_arrays is None:
            nodes = chip_nodes
        else:
            # PUs the layer spans, capped at the chip.
            nodes = max(1, min(chip_nodes,
                               ceil(max(1, int(n_arrays)) / max(1, int(PU_NUM_ARRAYS)))))
        return max(1.0, (2.0 * ceil(sqrt(nodes))) / 3.0)

    @property
    def average_noc_hops(self) -> float:
        """Chip-wide mean hop count (the whole mesh)."""
        return self.noc_hops_for(None)

    def _shift_add_cost(self, n_cols: int, shift: bool, add: bool) -> ExecutionCost:
        e = 0.0
        lat = 0.0
        n = 0
        if shift:
            e += n_cols * self.shiftreg.operation_metrics(1).energy_nj
            lat += self._clk_ns
            n += n_cols
        if add:
            e += n_cols * self.adder.accumulation_metrics(2).energy_nj
            lat += self._clk_ns
            n += n_cols
        return ExecutionCost(energy_j=e * _NJ_TO_J, latency_ns=lat, n_shift_add=n)

    def _reduce_add_cost(self, n_cols: int, n_terms: int) -> ExecutionCost:
        """Cost of summing `n_terms` partial columns (cross-tile accumulation)."""
        if n_terms <= 1:
            return ExecutionCost()
        m = self.adder.accumulation_metrics(n_terms, use_tree=True)
        return ExecutionCost(energy_j=n_cols * m.energy_nj * _NJ_TO_J,
                             latency_ns=m.latency_ns, n_shift_add=n_cols)

    def _elementwise_cost(self, n_cols: int) -> ExecutionCost:
        """One on-chip comparator/ALU op per element (ReLU, max, ...)."""
        m = self.adder.accumulation_metrics(2)
        return ExecutionCost(energy_j=n_cols * m.energy_nj * _NJ_TO_J,
                             latency_ns=self._clk_ns)

    # 2T1R signed read: form the +/- rails, then digitise their difference

    def _form_rails(
        self, v_rows: np.ndarray, signs: np.ndarray, G: np.ndarray, g_min: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the two physical bitline currents of a 2T1R signed column block.

        The sign bit routes each cell's ON branch onto its own rail and leaks a
        little through the OFF branch onto the other.  A g_min dummy cell is
        subtracted on each rail to remove the sign-magnitude common mode.

            I_pos = sum_i V_i * [ (sign=+ ? g_on - g_on0 : g_off - g_off0) ]_ij
            I_neg = sum_i V_i * [ (sign=- ? g_on - g_on0 : g_off - g_off0) ]_ij
        """
        t2 = self.twot1r
        pos = signs > 0                                     # (rows, cols) boolean
        g_on = t2.branch_on_conductance(G)                 # selected branch
        g_off = t2.branch_off_conductance(G)               # leaking branch
        g_on0 = float(t2.branch_on_conductance(g_min))     # dummy reference (ON)
        g_off0 = float(t2.branch_off_conductance(g_min))   # dummy reference (OFF)

        # Physical branch conductances actually hanging on each bit line.  The dummy
        # reference is subtracted AFTER sensing: the wire carries the full cell
        # current, so IR drop, saturation and self-heating must see it un-subtracted.
        pos_rail_g = np.where(pos, g_on, g_off)
        neg_rail_g = np.where(pos, g_off, g_on)
        pos_ref_g = np.where(pos, g_on0, g_off0)
        neg_ref_g = np.where(pos, g_off0, g_on0)

        i_pos = self._sense_rail(v_rows, pos_rail_g) - v_rows @ pos_ref_g
        i_neg = self._sense_rail(v_rows, neg_rail_g) - v_rows @ neg_ref_g
        return i_pos, i_neg

    def _sense_rail(self, v_rows: np.ndarray, G_rail: np.ndarray) -> np.ndarray:
        """
        Current sensed on one bit line, through the crossbar non-ideality model.

        Each rail is its own wire with its own sense amplifier, so the two get
        INDEPENDENT noise draws - which is why the differential read is noisier than
        either rail.  Common-mode terms cancel in the difference; the residual
        differential offset is `TwoT1RConfig.rail_offset_a`, applied in
        `_readout_signed`.  The dummy reference is an ideal bias, not a second noisy
        read, since a reference column is heavily averaged in any real design.
        """
        ni = self.crossbar_ni
        if ni is None or not getattr(ni.config, "enabled", False) or v_rows.size == 0:
            return v_rows @ G_rail
        return ni._compute_column_currents(
            np.asarray(v_rows, dtype=float),
            np.asarray(G_rail, dtype=float),
            read_voltage_v=self.array.config.read_voltage_v,
            rows=int(v_rows.shape[0]),
            read_time_s=float(getattr(self.array.config, "read_time_s", 0.0) or 0.0),
        )

    def _rail_read(
        self, tile: ArrayTile, nibble_rows: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Physical 2T1R read of ONE input nibble on a signed array.

        Returns (i_pos, i_neg, active_voltages, active_G): the two rail currents
        (reference-subtracted) plus the active-cell info for energy accounting.
        """
        w = tile.weights
        vread = self.array.config.read_voltage_v
        dmax = self.digit_io.digit_max_code
        ic, oc = tile.in_count, tile.out_count

        # 4-bit DAC: nibble value -> row voltage in [0, Vread], with the driver's
        # gain and offset error.  A row commanded to 0 V does not sit at exactly
        # 0 V, which is one of the ways a "skipped" row still perturbs the column.
        v_rows = (nibble_rows[:ic].astype(float) / dmax) * vread    # (in_count,)
        ni = self.crossbar_ni
        if ni is not None and getattr(ni.config, "enabled", False):
            c = ni.config
            v_rows = np.maximum(v_rows * (1.0 + c.dac_gain_error) + c.dac_offset_v, 0.0)
        signs = w.signs[:ic, :oc]
        G = w.conductance_s[:ic, :oc]
        i_pos, i_neg = self._form_rails(v_rows, signs, G, w.g_min_s)

        active_mask = nibble_rows[:ic] > 0
        v_active = v_rows[active_mask]
        G_active = G[active_mask]
        return i_pos, i_neg, v_active, G_active

    def _signed_nibble_current(
        self, tile: ArrayTile, nibble_rows: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Back-compat helper: the ideal differential rail current I_pos - I_neg."""
        i_pos, i_neg, v_active, G_active = self._rail_read(tile, nibble_rows)
        return i_pos - i_neg, v_active, G_active

    def _readout_signed(
        self, i_pos: np.ndarray, i_neg: np.ndarray,
        diff_fullscale_a: float, rail_fullscale_a: float,
    ) -> np.ndarray:
        """
        Convert the two rail currents into one signed, digitised value.

        Default (separate_rail_adc=False): a differential sense amp forms
        gain_pos*I_pos - gain_neg*I_neg (+ offset) in the analog domain and ONE
        bipolar ADC digitises it.  Otherwise each rail gets its own unipolar ADC
        and the difference is taken digitally.  Rail gain/offset mismatch is the
        2T1R sense non-ideality.
        """
        t2 = self.twot1r
        if t2.separate_rail_adc:
            ip = self._adc_rail(self._adc_front_end(i_pos), rail_fullscale_a)
            ineg = self._adc_rail(self._adc_front_end(i_neg), rail_fullscale_a)
            return t2.gain_pos * ip - t2.gain_neg * ineg + t2.rail_offset_a
        diff = t2.gain_pos * i_pos - t2.gain_neg * i_neg + t2.rail_offset_a
        return self._adc_signed(self._adc_front_end(diff), diff_fullscale_a)

    def _adc_front_end(self, current: np.ndarray) -> np.ndarray:
        """
        Converter gain and offset error, applied to whatever the ADC digitises.

        A real converter's transfer function is `k*(1+e_gain)*I + I_offset`, not the
        identity.  On the differential path these act on the DIFFERENCE (one bipolar
        ADC), which is why they are applied here rather than per rail - a per-rail
        offset is common-mode and already cancelled in `_form_rails`.
        """
        ni = self.crossbar_ni
        if ni is None or not getattr(ni.config, "enabled", False):
            return current
        c = ni.config
        return current * (1.0 + c.adc_gain_error) + c.adc_offset_a

    def _row_mode_bias(
        self, tile: ArrayTile, cost: Optional[ExecutionCost],
    ) -> np.ndarray:
        """
        Bias add as a ROW-mode operation (vmmr): one bias row, all columns.

        The augmented bias row is a 2T1R signed strip driven at the constant full
        read voltage; its two rails are sensed and subtracted on the bias ADC
        range, so the bias carries its own current->value conversion.  This
        realises the layer bias entirely on the crossbar - no host-side vector add.
        """
        bw = tile.bias_weights
        if bw is None:
            return np.zeros(tile.out_count, dtype=float)
        oc = tile.out_count
        vread = self.array.config.read_voltage_v
        v_rows = np.array([vread])
        i_pos, i_neg = self._form_rails(
            v_rows, bw.signs[:1, :oc], bw.conductance_s[:1, :oc], bw.g_min_s)
        i_bias_q = self._readout_signed(
            i_pos, i_neg, tile.bias_fullscale_a, tile.bias_rail_fullscale_a)
        y_bias = i_bias_q / (vread * bw.scale_for(oc))

        if cost is not None:
            # One row-mode analog read (1 active row, oc active columns) + a PUADD
            # to fold the bias into the column accumulator.
            G_active = bw.conductance_s[:1, :oc]
            cost.add(self._analog_read_cost(v_rows, G_active, oc))
            cost.add(self._shift_add_cost(oc, shift=False, add=True))
        return y_bias

    def _adc_signed(self, current: np.ndarray, fullscale_a: float) -> np.ndarray:
        """Digitise the signed rail difference with the bipolar 4-bit ADC."""
        if self.ideal_adc or fullscale_a <= 0:
            return current
        adc = self.array.adc
        levels = 2 ** adc.precision_bits
        adc.configure_intervals(-fullscale_a, fullscale_a)
        codes = adc.quantize(current)
        step = (2.0 * fullscale_a) / levels
        return (codes.astype(float) + 0.5) * step - fullscale_a

    def _adc_rail(self, current: np.ndarray, fullscale_a: float) -> np.ndarray:
        """Digitise one (non-negative) rail current with a unipolar 4-bit ADC."""
        if self.ideal_adc or fullscale_a <= 0:
            return current
        adc = self.array.adc
        levels = 2 ** adc.precision_bits
        adc.configure_intervals(0.0, fullscale_a)
        codes = adc.quantize(np.maximum(current, 0.0))
        step = fullscale_a / levels
        return (codes.astype(float) + 0.5) * step

    def _matrix_vmm(
        self, tile: ArrayTile, x_tile: np.ndarray, in_scale: float,
        accumulate_cost: Optional[ExecutionCost],
    ) -> np.ndarray:
        """
        Full matrix-mode VMM of one 2T1R array, digit-serial over the input nibbles.

        Implements deviation #1: TWO nibble reads + ONE shift&add (not 8 + 7).
        Each nibble read forms the +/- rails (deviation #2) and senses I_pos-I_neg.
        Returns the tile's partial output in the weight domain (length out_count).
        """
        oc = tile.out_count
        vread = self.array.config.read_voltage_v
        scale = tile.weights.scale_for(oc)

        codes = self.digit_io.to_codes(x_tile, in_scale)          # 8-bit codes
        nibbles = self.digit_io.digits(codes)                     # [low, high]

        i_comb = np.zeros(oc, dtype=float)
        for d, nib in enumerate(nibbles):
            i_pos, i_neg, v_active, G_active = self._rail_read(tile, nib)
            # Each digit is digitised against its own converter reference.  See
            # ArrayTile.adc_fullscale_per_digit.
            cur_q = self._readout_signed(
                i_pos, i_neg, tile.digit_fullscale_a(d), tile.digit_rail_fullscale_a(d))
            i_comb = i_comb + self.digit_io.digit_weight(d) * cur_q
            if accumulate_cost is not None:
                accumulate_cost.add(self._analog_read_cost(
                    v_active, G_active, oc,
                    n_addressed_cells=tile.in_count * oc))
                # Digit weighting in the PU shift&adder: shift the high nibble, add.
                accumulate_cost.add(self._shift_add_cost(oc, shift=(d > 0), add=(d > 0)))

        # Convert signed combined current back to the weight domain.
        # y = i_comb * in_scale * digit_max / (Vread * input_max * scale)
        conv = in_scale * self.digit_io.digit_max_code / (
            vread * self.digit_io.input_max_code * scale)
        return i_comb * conv

    # Layer execution (matrix VMM + bias row + cross-tile add + ReLU)

    def run_layer(
        self, layer: LinearLayerPlan, x: np.ndarray, cost: Optional[ExecutionCost] = None,
    ) -> np.ndarray:
        """
        Execute one linear layer for a single activation vector `x`.

        The layer's tiles are costed both ways: energy and serial latency accumulate
        as before, and alongside them the critical path is built on the tiles being
        independent, so `arrays_available` of them issue together and the layer costs
        `ceil(n_tiles / concurrent)` waves of one tile.

        Sequential, and charged after the waves: the two nibble reads inside a tile,
        the cross-tile reduction, the ReLU, and the next layer.
        """
        y = np.zeros(layer.out_features, dtype=float)

        tiles_cost = ExecutionCost() if cost is not None else None
        tile_latencies: List[float] = []
        for tile in layer.tiles:
            x_tile = x[tile.row_slice]
            # Cost each tile into its OWN accumulator so the critical path can
            # see one tile's latency; the totals are merged straight after.
            tc = ExecutionCost() if cost is not None else None
            partial = self._matrix_vmm(tile, x_tile, layer.in_scale, accumulate_cost=tc)
            y[tile.col_slice] += partial
            # Bias rides a ROW-mode read on the first row-tile that owns each
            # output column block (deviation: non-VMM op done on the crossbar).
            if tile.bias_weights is not None:
                y[tile.col_slice] += self._row_mode_bias(tile, tc)
            if tc is not None:
                tile_latencies.append(tc.latency_ns)
                tiles_cost.add(tc)

        if cost is not None:
            # The waves.  A tile carrying the bias strip runs one extra row-mode
            # read, so a wave costs the slowest tile in that wave rather than
            # `waves * global_max`.  At `arrays_available = 1` this collapses to
            # sum(tile_latencies), the serial number.
            n_tiles = len(layer.tiles)
            concurrent = max(1, min(n_tiles, self.arrays_available))
            tiles_cost.latency_parallel_ns = sum(
                max(tile_latencies[i:i + concurrent])
                for i in range(0, n_tiles, concurrent)) if tile_latencies else 0.0
            cost.add(tiles_cost)

        # Cross-tile (row-split) accumulation cost: one reduce per output column.
        if cost is not None and layer.row_tiles > 1:
            cost.add(self._reduce_add_cost(layer.out_features, layer.row_tiles))

        # DATA MOVEMENT.  The activation vector has to reach this layer's row
        # drivers and the result has to be staged for the next layer; both cross the
        # NoC and touch a staging buffer.  The input is broadcast to every column
        # tile, the output is written back once.
        if cost is not None:
            act_bits = int(self.array.config.input_precision_bits)
            out_bits = int(self.array.config.output_precision_bits)
            col_tiles = max(1, layer.col_tiles)
            hops = self.noc_hops_for(len(layer.tiles))
            cost.add(self.movement_cost(
                layer.in_features * act_bits * col_tiles, hops=hops))
            cost.add(self.movement_cost(layer.out_features * out_bits, hops=hops))

        if layer.has_relu:
            y = np.maximum(y, 0.0)                 # on-chip ReLU (sign bit + comparator)
            if cost is not None:
                cost.add(self._elementwise_cost(layer.out_features))
        return y

    def run_network(
        self, layers: List[LinearLayerPlan], x: np.ndarray,
    ) -> Tuple[np.ndarray, ExecutionCost]:
        cost = ExecutionCost()
        h = x
        for layer in layers:
            h = self.run_layer(layer, h, cost)
        return h, cost

    # Column-mode reduction primitive (a non-VMM op done on the crossbar)

    def reduce_sum(self, x: np.ndarray, cost: Optional[ExecutionCost] = None) -> float:
        """
        Sum a vector using COLUMN mode (all rows -> one column of 1-weights).

        This is the "every NN op on the chip" path for reductions such as global
        sum / average pooling: a column of unit-weight cells driven by the whole
        input vector returns sum_i x_i in a single analog column read.
        """
        if cost is not None:
            # all-rows / one-column read: in_count active rows, 1 active column.
            n = int(np.count_nonzero(x))
            v = np.full(n, self.array.config.read_voltage_v)
            # Representative PROGRAMMED conductance, not the unprogrammed
            # template's minimum state (see device_mean_conductance_s).
            G = np.full((n, 1), self.array.device_mean_conductance_s)
            cost.add(self._analog_read_cost(v, G, 1, n_addressed_cells=x.size))
        return float(np.sum(x))


# 7. NN compiler: lower a network onto sign-magnitude tiles + an instruction stream

def lower_linear_layer(
    name: str, W: np.ndarray, b: np.ndarray, has_relu: bool, in_scale: float,
    levels_s: Sequence[float], rows: int, cols: int, read_voltage_v: float,
    pu_of_tile: Callable[[int], Tuple[int, int]] = lambda i: (0, i),
    per_column_scale: bool = False,
) -> LinearLayerPlan:
    """
    Map a software weight matrix W (in x out) onto sign-magnitude array tiles.

    Large layers are tiled across `rows x cols` arrays; each tile stores |w| on a
    single array + a sign register.  The first tile of every output column block
    also carries the bias as an augmented row.  `pu_of_tile(i) -> (pu, array)`
    decides physical placement.

    per_column_scale
        False (default): one S-per-unit-weight scale for the whole layer, so the
        largest weight sets the grid for every column and quiet columns lose
        resolution.  True: each output column gets its own scale and uses the full
        state range, at no physical cost - each column already has its own ADC and
        the factor folds into the next layer.  Row-tiles of the same column still
        share a scale.
    """
    in_f, out_f = W.shape
    levels = np.asarray(levels_s, dtype=float)
    g_min, g_max = float(levels.min()), float(levels.max())

    n_row_tiles = ceil(in_f / rows)
    n_col_tiles = ceil(out_f / cols)

    plan = LinearLayerPlan(
        name=name, in_features=in_f, out_features=out_f, has_relu=has_relu,
        in_scale=in_scale, read_voltage_v=read_voltage_v, rows=rows, cols=cols,
        row_tiles=n_row_tiles, col_tiles=n_col_tiles,
    )

    # Row-tiles of a column always share a scale, so partial sums stay
    # commensurate; only whether COLUMNS share one is optional.
    if per_column_scale:
        w_absmax = np.maximum(np.max(np.abs(W), axis=0), 1e-12)     # (out_f,)
        b_absmax = np.maximum(np.abs(b), 1e-12)                     # (out_f,)
    else:
        w_absmax = float(np.max(np.abs(W))) or 1.0
        b_absmax = float(np.max(np.abs(b))) or 1.0
    scale = (g_max - g_min) / w_absmax
    bias_scale = (g_max - g_min) / b_absmax

    idx = 0
    for rt in range(n_row_tiles):
        r0, r1 = rt * rows, min((rt + 1) * rows, in_f)
        for ct in range(n_col_tiles):
            c0, c1 = ct * cols, min((ct + 1) * cols, out_f)
            in_count, out_count = r1 - r0, c1 - c0

            # Pad to the full array with zero-weight (g_min) spare cells.
            Wpad = np.zeros((rows, cols), dtype=float)
            Wpad[:in_count, :out_count] = W[r0:r1, c0:c1]
            w_sc = _tile_scale(scale, c0, c1, cols)
            sm = _sign_mag_with_scale(Wpad, levels, g_min, w_sc)

            bias_sm = None
            if rt == 0:                              # first row-tile owns the bias row
                Bpad = np.zeros((1, cols), dtype=float)
                Bpad[0, :out_count] = b[c0:c1]
                # Bias row uses the same array; map with its own magnitude range
                # so the bias keeps the full device grid.
                bias_sm = _sign_mag_with_scale(
                    Bpad, levels, g_min, _tile_scale(bias_scale, c0, c1, cols))

            pu, array = pu_of_tile(idx)
            plan.tiles.append(ArrayTile(
                pu=pu, array=array,
                row_slice=slice(r0, r1), col_slice=slice(c0, c1),
                in_count=in_count, out_count=out_count,
                weights=sm, bias_weights=bias_sm,
            ))
            idx += 1
    return plan


def _tile_scale(scale, c0: int, c1: int, cols: int):
    """
    The scale a tile stores: a scalar passes through; a per-column vector is
    sliced to this tile's columns and padded out to the physical array width
    (spare columns hold a harmless copy of the last real scale).
    """
    if not isinstance(scale, np.ndarray):
        return scale
    out = np.full(cols, scale[c1 - 1], dtype=float)
    out[:c1 - c0] = scale[c0:c1]
    return out


def _sign_mag_with_scale(
    W: np.ndarray, levels: np.ndarray, g_min: float, scale,
) -> SignMagnitudeWeights:
    """Sign-magnitude mapping with a *fixed* S/weight scale (for tile commensurability)."""
    target = g_min + np.abs(W) * scale
    target = np.clip(target, levels.min(), levels.max())
    codes = np.argmin(np.abs(target[..., None] - levels), axis=-1).astype(int)
    return SignMagnitudeWeights(
        magnitude_codes=codes, conductance_s=levels[codes],
        sign_bits=(W < 0).astype(np.uint8), scale_s_per_w=scale, g_min_s=g_min,
    )


def calibrate_adc(
    proc: IMCProcessor, layers: List[LinearLayerPlan], X_calib: np.ndarray,
    percentile: float = 99.9, margin: float = 1.2,
) -> None:
    """
    Set each tile's bipolar ADC full-scale from the observed per-nibble signed
    current swing (run once at deployment).

    Calibrated per digit: the digits do not share a current distribution, and the
    high digit is the one multiplied by 16 in the combine, so its quantisation error
    is the one that survives.  The pooled values are still recorded on
    `adc_fullscale_a` / `rail_fullscale_a` as the fallback.
    """
    h = X_calib
    vread = proc.array.config.read_voltage_v
    n_digits = proc.digit_io.num_digits
    for layer in layers:
        for tile in layer.tiles:
            diff_peaks: List[List[float]] = [[0.0] for _ in range(n_digits)]
            rail_peaks: List[List[float]] = [[0.0] for _ in range(n_digits)]
            for i in range(h.shape[0]):
                codes = proc.digit_io.to_codes(h[i, tile.row_slice], layer.in_scale)
                for d, nib in enumerate(proc.digit_io.digits(codes)):
                    i_pos, i_neg, _, _ = proc._rail_read(tile, nib)
                    if i_pos.size:
                        diff_peaks[d].append(float(np.max(np.abs(i_pos - i_neg))))
                        rail_peaks[d].append(float(max(np.max(i_pos), np.max(i_neg))))
            # Bipolar ADC sees I_pos - I_neg; each unipolar rail ADC sees one rail.
            tile.adc_fullscale_per_digit = [
                max(float(np.percentile(p, percentile)) * margin, 1e-15)
                for p in diff_peaks]
            tile.rail_fullscale_per_digit = [
                max(float(np.percentile(p, percentile)) * margin, 1e-15)
                for p in rail_peaks]
            # Pooled fallback, for any consumer that has not been told about digits.
            tile.adc_fullscale_a = max(tile.adc_fullscale_per_digit)
            tile.rail_fullscale_a = max(tile.rail_fullscale_per_digit)
            # Bias row-mode ADC range: the worst-case single-row rail currents.
            if tile.bias_weights is not None:
                bw = tile.bias_weights
                oc = tile.out_count
                i_pos, i_neg = proc._form_rails(
                    np.array([vread]), bw.signs[:1, :oc], bw.conductance_s[:1, :oc],
                    bw.g_min_s)
                tile.bias_fullscale_a = max(
                    float(np.max(np.abs(i_pos - i_neg))) * margin, 1e-15)
                tile.bias_rail_fullscale_a = max(
                    float(max(np.max(i_pos), np.max(i_neg))) * margin, 1e-15)
        # Propagate activations to the next layer with the (accurate) effective weights.
        h = np.stack([_effective_forward(layer, h[i]) for i in range(h.shape[0])])


def _effective_forward(layer: LinearLayerPlan, x: np.ndarray) -> np.ndarray:
    """Accurate forward using each tile's effective signed weights (for calibration)."""
    y = np.zeros(layer.out_features, dtype=float)
    bias_done = set()
    for tile in layer.tiles:
        W_eff = tile.weights.effective_weight[:tile.in_count, :tile.out_count]
        y[tile.col_slice] += x[tile.row_slice] @ W_eff
        key = (tile.col_slice.start, tile.col_slice.stop)
        if tile.bias_weights is not None and key not in bias_done:
            y[tile.col_slice] += tile.bias_weights.effective_weight[0, :tile.out_count]
            bias_done.add(key)
    return np.maximum(y, 0.0) if layer.has_relu else y


# Instruction-stream generation (the readable RISC-V-like program)

def compile_layer_to_instructions(
    layer: LinearLayerPlan, digit_io: DigitSerialInput,
    input_reg: str = "v1", acc_reg: str = "R4",
) -> List[Instruction]:
    """
    Emit the RISC-V-vector-style instruction stream for one inference of a layer.

    Per array tile (matrix mode), digit-serial input -> two VMMs + one shift&add;
    partial sums across row-tiles accumulate with PUADD; bias rides the augmented
    row inside the matrix VMM; ReLU is an on-chip PURELU.
    """
    prog: List[Instruction] = []
    n_digits = digit_io.num_digits
    seen_col: set = set()

    for ti, tile in enumerate(layer.tiles):
        addr = IMCAddress(mode=OperationMode.MATRIX,
                          row_index=0, col_index=0, array=tile.array, pu=tile.pu)
        adc_reg = f"R{ti % 4}"          # ADC-group register (paper Sec. 4.2)
        for d in range(n_digits):
            prog.append(Instruction(
                Opcode.VMM, addr=addr, vd=adc_reg, vs1=input_reg,
                comment=f"{layer.name} tile{ti} nibble{d} (matrix mode)"))
            if d > 0:
                shift = digit_io.digit_bits * d
                prog.append(Instruction(
                    Opcode.PUSLL, vd=adc_reg, vs1=adc_reg, imm=shift,
                    comment=f"weight high nibble << {shift}"))
                prog.append(Instruction(
                    Opcode.PUADD, vd=acc_reg, vs1=acc_reg, vs2=adc_reg,
                    comment="combine digit-serial partials"))
        key = (tile.col_slice.start, tile.col_slice.stop)
        if key in seen_col:
            prog.append(Instruction(
                Opcode.PUADD, vd=acc_reg, vs1=acc_reg, vs2=adc_reg,
                comment="accumulate row-split partial (cross-tile)"))
        seen_col.add(key)

        # Bias as a ROW-mode read (vmmr) of a dedicated 1-row bias strip, then PUADD.
        # The strip is its own small array (1 x cols) so a fully packed weight array
        # need not give up an input row.
        if tile.bias_weights is not None:
            bias_addr = IMCAddress(mode=OperationMode.ROW,
                                   row_index=0, col_index=0,
                                   array=tile.array, pu=tile.pu)
            prog.append(Instruction(
                Opcode.VMM, addr=bias_addr, vd="R3", vs1="x1",
                comment=f"{layer.name} tile{ti} bias strip (row mode, const input)"))
            prog.append(Instruction(
                Opcode.PUADD, vd=acc_reg, vs1=acc_reg, vs2="R3",
                comment="add bias to column accumulator"))

    if layer.has_relu:
        prog.append(Instruction(
            Opcode.PURELU, vd=acc_reg, vs1=acc_reg,
            comment=f"{layer.name} on-chip ReLU (sign reg + comparator)"))
    return prog


def compile_network_to_instructions(
    layers: List[LinearLayerPlan], digit_io: DigitSerialInput,
) -> List[Instruction]:
    prog: List[Instruction] = []
    for li, layer in enumerate(layers):
        prog.append(Instruction(Opcode.MV, vd="v1", vs1=f"v{li}",
                                comment=f"load {layer.name} input vector to PU bus"))
        prog.extend(compile_layer_to_instructions(layer, digit_io))
    return prog


# 8. Efficiency reporting

@dataclass
class EfficiencyReport:
    macs: int
    energy_per_inference_j: float
    latency_per_inference_s: float
    n_analog_reads: int
    n_shift_add: int
    # Critical-path latency with a layer's tiles reading CONCURRENTLY, and how
    # many arrays that concurrency was allowed to use.  `latency_per_inference_s`
    # stays the serial instruction stream so nothing that depended on it moves.
    latency_parallel_s: float = 0.0
    arrays_concurrent: int = 1

    @property
    def ops(self) -> int:
        return 2 * self.macs            # 1 MAC = multiply + accumulate

    @property
    def tops(self) -> float:
        """Serial issue: one array at a time.  The conservative bound."""
        return self.ops / (self.latency_per_inference_s * 1e12) if self.latency_per_inference_s > 0 else 0.0

    @property
    def tops_parallel(self) -> float:
        """Tiles of a layer read together, layers still in sequence."""
        return (self.ops / (self.latency_parallel_s * 1e12)
                if self.latency_parallel_s > 0 else 0.0)

    @property
    def parallel_speedup(self) -> float:
        return (self.latency_per_inference_s / self.latency_parallel_s
                if self.latency_parallel_s > 0 else 1.0)

    @property
    def power_w(self) -> float:
        return self.energy_per_inference_j / self.latency_per_inference_s if self.latency_per_inference_s > 0 else 0.0

    @property
    def power_parallel_w(self) -> float:
        """Same energy in less time - the parallel machine draws proportionally more."""
        return (self.energy_per_inference_j / self.latency_parallel_s
                if self.latency_parallel_s > 0 else 0.0)

    @property
    def tops_per_watt(self) -> float:
        """
        Unchanged by concurrency, and that is the point: energy per operation is
        a property of the array and its converter, not of how many arrays happen
        to be switched on at the same instant.  Firing 8 arrays together cuts the
        latency 8x and raises the power 8x, and TOPS/W does not move.
        """
        return self.ops / (self.energy_per_inference_j * 1e12) if self.energy_per_inference_j > 0 else 0.0


def efficiency_report(macs: int, cost: ExecutionCost,
                      arrays_concurrent: int = 1) -> EfficiencyReport:
    return EfficiencyReport(
        macs=macs,
        energy_per_inference_j=cost.energy_j,
        latency_per_inference_s=cost.latency_ns * 1e-9,
        n_analog_reads=cost.n_analog_reads,
        n_shift_add=cost.n_shift_add,
        latency_parallel_s=cost.par_ns * 1e-9,
        arrays_concurrent=int(arrays_concurrent),
    )


# 9. Retention / accuracy-over-time (single-device non-idealities + drift)
# Layers the time-dependent device physics of single_memristor_nonidealities.py
# onto the sign-magnitude deployment, reproducing the accuracy-retention sweep of
# example_usage_nn_decay.py.  Retention drifts only the magnitude conductance
# (toward g_min); the sign lives in the register, so a weight loses amplitude but
# never flips sign, unlike a differential pair with unequal G+/G- drift.

@dataclass
class _TileProgramming:
    """Frozen post-programming magnitude conductances + the per-tile NI model."""
    ni: object                                  # SingleMemristorNonIdealityModel
    programmed_weight_s: np.ndarray
    programmed_bias_s: Optional[np.ndarray]


@dataclass
class RetentionDeployment:
    """Per-tile programming state used to replay retention at any elapsed time."""
    per_tile: List[List[_TileProgramming]]      # indexed [layer][tile]


def freeze_programming(
    layers: List[LinearLayerPlan], ni_config, levels_s: Sequence[float],
    seed: int = 20240517,
) -> RetentionDeployment:
    """
    Apply programming non-idealities (D2D spread, write error, stuck-at) ONCE to
    each tile's magnitude conductances and freeze them.  Retention drift is then a
    pure function of elapsed time applied on top (see `apply_time`).
    """
    from single_memristor_nonidealities import SingleMemristorNonIdealityModel
    per_tile: List[List[_TileProgramming]] = []
    idx = 0
    for layer in layers:
        row: List[_TileProgramming] = []
        for tile in layer.tiles:
            ni = SingleMemristorNonIdealityModel(levels_s, ni_config, rng=seed + idx)
            idx += 1
            pw = ni.apply_programming_nonidealities(tile.weights.conductance_s)
            pb = (ni.apply_programming_nonidealities(tile.bias_weights.conductance_s)
                  if tile.bias_weights is not None else None)
            row.append(_TileProgramming(ni=ni, programmed_weight_s=pw, programmed_bias_s=pb))
        per_tile.append(row)
    return RetentionDeployment(per_tile=per_tile)


def apply_time(
    dep: RetentionDeployment, layers: List[LinearLayerPlan], elapsed_time_s: float,
    read_noise: bool = False,
) -> None:
    """
    Drift every tile's stored magnitude conductances to `elapsed_time_s` after
    programming, writing the retained (optionally read-noisy) values back into the
    sign-magnitude weights.  Sign bits are untouched (they live in registers).
    """
    for li, layer in enumerate(layers):
        for ti, tile in enumerate(layer.tiles):
            tp = dep.per_tile[li][ti]
            w = tp.ni.apply_retention(tp.programmed_weight_s, elapsed_time_s=elapsed_time_s)
            if read_noise:
                w = tp.ni.apply_read_noise(w)
            tile.weights.conductance_s = w
            if tp.programmed_bias_s is not None and tile.bias_weights is not None:
                b = tp.ni.apply_retention(tp.programmed_bias_s, elapsed_time_s=elapsed_time_s)
                if read_noise:
                    b = tp.ni.apply_read_noise(b)
                tile.bias_weights.conductance_s = b


def mean_stored_conductance(layers: List[LinearLayerPlan]) -> float:
    """Mean magnitude conductance across all tiles (for the decay-% metric)."""
    vals = []
    for layer in layers:
        for tile in layer.tiles:
            vals.append(tile.weights.conductance_s[:tile.in_count, :tile.out_count].ravel())
            if tile.bias_weights is not None:
                vals.append(tile.bias_weights.conductance_s[0, :tile.out_count].ravel())
    return float(np.mean(np.concatenate(vals)))


def layer_effective_matrix(layer: LinearLayerPlan) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the (in x out) effective signed weight matrix + bias of a layer."""
    W = np.zeros((layer.in_features, layer.out_features), dtype=float)
    b = np.zeros(layer.out_features, dtype=float)
    bias_done = set()
    for tile in layer.tiles:
        W[tile.row_slice, tile.col_slice] = (
            tile.weights.effective_weight[:tile.in_count, :tile.out_count])
        key = (tile.col_slice.start, tile.col_slice.stop)
        if tile.bias_weights is not None and key not in bias_done:
            b[tile.col_slice] = tile.bias_weights.effective_weight[0, :tile.out_count]
            bias_done.add(key)
    return W, b


def weight_domain_logits(layers: List[LinearLayerPlan], X: np.ndarray) -> np.ndarray:
    """Batched ideal-read forward using the current (drifted) effective weights."""
    h = X
    for layer in layers:
        W, b = layer_effective_matrix(layer)
        z = h @ W + b
        h = np.maximum(z, 0.0) if layer.has_relu else z
    return h


# 10. Self-test demo (synthetic, always runnable)

def _self_test() -> None:
    rng = np.random.default_rng(0)
    from simulation_config import build_device
    device = build_device()
    levels = device.conductance_levels_s

    # A tiny 12 -> 8 -> 3 network with non-negative inputs (post-ReLU style).
    W1 = rng.standard_normal((12, 8)) * 0.5
    b1 = rng.standard_normal(8) * 0.1
    W2 = rng.standard_normal((8, 3)) * 0.5
    b2 = rng.standard_normal(3) * 0.1
    X = np.abs(rng.standard_normal((64, 12)))

    rows = cols = 128
    read_v = device.pulse.read_voltage_v
    layers = [
        lower_linear_layer("fc1", W1, b1, True, float(np.percentile(X, 99.9)),
                           levels, rows, cols, read_v),
        lower_linear_layer("fc2", W2, b2, False, 1.0, levels, rows, cols, read_v),
    ]
    # in_scale for fc2 depends on fc1 activations; estimate from the effective fwd.
    h1 = np.stack([_effective_forward(layers[0], X[i]) for i in range(X.shape[0])])
    layers[1].in_scale = float(max(np.percentile(h1, 99.9), 1e-6))

    print("=" * 70)
    print("RISC-V IMC WORKFLOW - SELF TEST  (synthetic 12->8->3 net)")
    print("=" * 70)

    # 1) Datapath consistency: with grid-aligned inputs and an ideal ADC, the
    #    digit-serial + sign-magnitude + row-mode-bias datapath must reproduce the
    #    effective-weight forward to numerical precision.  The assert needs
    #    R_on=0 / R_off=inf, so this asks for a bare TwoT1RConfig().
    proc_ideal = IMCProcessor(ideal_adc=True, twot1r=TwoT1RConfig())
    io = proc_ideal.digit_io
    layer0 = layers[0]
    # Inputs that are exact multiples of the 8-bit quantization step.
    codes = rng.integers(0, io.input_max_code + 1, size=(32, layer0.in_features))
    Xg = codes.astype(float) * layer0.in_scale / io.input_max_code
    max_err = 0.0
    for i in range(Xg.shape[0]):
        y_ref = _effective_forward(layer0, Xg[i])
        y_hw = proc_ideal.run_layer(layer0, Xg[i])
        max_err = max(max_err, float(np.max(np.abs(y_hw - y_ref))))
    print(f"Datapath vs effective-weight forward (grid inputs): max abs err = {max_err:.2e}")
    assert max_err < 1e-12, "digit-serial / sign-magnitude / bias datapath is inconsistent!"

    # 1b) 2T1R rails: the ideal two-rail read (R_on=0, R_off=inf) must reproduce the
    #     numeric sign-magnitude result exactly; a non-ideal 2T1R cell then perturbs
    #     it via series R_on, off-rail leakage, and +/- sense mismatch.
    tile0 = layer0.tiles[0]
    ic0, oc0 = tile0.in_count, tile0.out_count
    nib0 = io.digits(io.to_codes(Xg[0][tile0.row_slice], layer0.in_scale))[0]
    ip, ineg, _, _ = proc_ideal._rail_read(tile0, nib0)
    n_pos = int(np.count_nonzero(tile0.weights.signs[:ic0, :oc0] > 0))
    print(f"2T1R rails (ideal): I_pos[:3]={np.round(ip[:3]*1e9,3)} nA, "
          f"I_neg[:3]={np.round(ineg[:3]*1e9,3)} nA  "
          f"(+cells={n_pos}/{ic0*oc0})")

    ni_2t1r = TwoT1RConfig(on_resistance_ohm=2000.0, off_resistance_ohm=1e9,
                           rail_gain_mismatch=0.01, rail_offset_a=1e-9,
                           transistor_switch_energy_nj=5e-7)
    proc_2t = IMCProcessor(ideal_adc=True, twot1r=ni_2t1r)
    errs = []
    for i in range(Xg.shape[0]):
        y_ref = _effective_forward(layer0, Xg[i])
        y_2t = proc_2t.run_layer(layer0, Xg[i])
        errs.append(float(np.mean(np.abs(y_2t - y_ref))))
    print(f"Non-ideal 2T1R (R_on=2 kohm, R_off=1 Gohm, 1% rail mismatch, "
          f"1 nA offset): mean abs err vs ideal = {np.mean(errs):.2e}")

    # 2) Real 4-bit ADC path + efficiency at 1 GHz (chip's real 2T1R cell).
    proc = IMCProcessor(ideal_adc=False)
    calibrate_adc(proc, layers, X[:32])
    cost = ExecutionCost()
    for i in range(X.shape[0]):
        proc.run_layer(layers[0], X[i], cost)
    cost_total = ExecutionCost()
    for i in range(X.shape[0]):
        _, c = proc.run_network(layers, X[i])
        cost_total.add(c)
    macs = sum(l.in_features * l.out_features for l in layers)
    per = ExecutionCost(energy_j=cost_total.energy_j / X.shape[0],
                        latency_ns=cost_total.latency_ns / X.shape[0],
                        n_analog_reads=cost_total.n_analog_reads // X.shape[0],
                        n_shift_add=cost_total.n_shift_add // X.shape[0])
    rep = efficiency_report(macs, per)
    print(f"Digit-serial cycles/array: {proc.digit_io.num_digits} "
          f"(vs 8 for bit-serial); analog reads/inf: {rep.n_analog_reads}")
    print(f"Energy/inference: {rep.energy_per_inference_j:.3e} J   "
          f"Latency: {rep.latency_per_inference_s*1e9:.1f} ns")
    print(f"Throughput: {rep.tops:.4f} TOPS   Efficiency: {rep.tops_per_watt:.2f} TOPS/W")

    # 3) Show the instruction stream.
    prog = compile_network_to_instructions(layers, proc.digit_io)
    print(f"\nCompiled instruction stream ({len(prog)} instructions):")
    for ins in prog:
        print("   " + str(ins))
    print("\nSelf test passed.")


if __name__ == "__main__":
    _self_test()
