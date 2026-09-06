"""
chip_architecture.py

Hierarchical chip-architecture model for the memristor-enabled heterogeneous
in-memory-computing system described in:

    H. Zhou, Z. Zhou, S. Ren, J. Chen, Y. Li, X. Miao,
    "Heterogeneous Instruction Set Architecture for RRAM-enabled
     In-memory Computing", NANOARCH '23.

This is the physical three-level organisation of Fig. 3 and Sec. 3 (chip -> tile
-> processing unit -> array) plus the on-chip interconnects; the ISA layer lives
in riscv_imc_workflow.py.  The crossbar array and all of its peripherals come
verbatim from `crossbar_user_config.build_crossbar`, so no crossbar physics is
re-implemented here - the chip model only aggregates it.

Accounting, as in hardware_peripherals.py: area um^2, power W, latency ns,
energy nJ (= W * ns).

    Chip
     |-- chip-level interconnect  (NoC mesh between tiles + host IO)
     `-- Tiles[ num_tiles ]
           |-- IO port             (host <-> tile)
           |-- instruction buffer  (CU fetches instructions from here)
           |-- tile register file  (scalar/general + vector registers)
           |-- functional unit     (digital ALU inside the control unit)
           |-- tile interconnect   (CU <-> PUs)
           `-- Processing Units[ num_pus ]
                 |-- PU register file (local R0..R7: ADC + shift/add groups)
                 |-- PU shift & add   (intra-PU post-processing)
                 |-- PU interconnect  (array group <-> registers)
                 `-- Array group[ num_arrays ]   (only ONE active at a time)
                       `-- IntegratedMemristorCrossbar
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import ceil, sqrt
from pathlib import Path
from typing import Callable, Optional
import json

import numpy as np

# --- existing, unmodified building blocks -------------------------------------
from hardware_peripherals import Reg, ShiftReg, Adder, HardwareMetrics
from integrated_crossbar import IntegratedMemristorCrossbar


# Helpers

# Unit conversions used throughout this module.
_PJ_TO_NJ = 1e-3      # 1 pJ = 1e-3 nJ
_NJ_TO_J = 1e-9       # 1 nJ = 1e-9 J


def _default_crossbar_factory() -> IntegratedMemristorCrossbar:
    """
    Build one crossbar (the array) straight from the user's existing config.

    Imported lazily so this module never forces a particular crossbar config at
    import time; the chip simply reuses whatever `crossbar_user_config` defines.
    """
    from crossbar_user_config import build_crossbar
    return build_crossbar()


# New digital building blocks (not present in hardware_peripherals.py)

@dataclass
class InterconnectConfig:
    """
    On-chip-network (NoC) parameters.

    The same model is reused at every level: array<->register inside a PU, CU<->PU
    inside a tile, and tile<->tile across the chip.  Energy splits into a per-hop
    router term and a per-hop wire term.  All numbers are 65 nm estimates.
    """
    name: str = "noc"
    data_width_bits: int = 128
    frequency_mhz: float = 1000.0
    tech_node_nm: int = 65
    topology: str = "mesh"                       # "mesh" | "bus" | "point_to_point"

    router_area_um2: Optional[float] = None      # per router/switch node
    router_static_power_w: Optional[float] = None
    router_energy_pj_per_bit: float = 0.10       # dynamic traversal energy / bit / hop
    wire_energy_pj_per_bit_per_mm: float = 0.20  # link wire energy / bit / mm
    link_length_mm: float = 1.0                  # physical length of one hop


class Interconnect:
    """
    Parameterised NoC connecting `num_nodes` end-points.

    area_um2      : routers + a coarse link-area allowance
    transfer_*    : dynamic energy/latency for moving a payload across the network
    """

    def __init__(self, config: InterconnectConfig, num_nodes: int):
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        self.config = config
        self.num_nodes = num_nodes
        self._fill_defaults()

    def _tech_scale(self) -> float:
        return (self.config.tech_node_nm / 65.0) ** 2

    def _fill_defaults(self) -> None:
        c = self.config
        # A 5-port router area scales with the flit width.  ~80 um^2 per bit at
        # 65 nm is a representative slice for a light NoC router.
        if c.router_area_um2 is None:
            c.router_area_um2 = 80.0 * c.data_width_bits * self._tech_scale()
        # Static (leakage) power per router, ~5 nW/bit at 65 nm.
        if c.router_static_power_w is None:
            c.router_static_power_w = 5e-9 * c.data_width_bits * self._tech_scale()

    @property
    def num_routers(self) -> int:
        if self.config.topology == "bus":
            return 1
        return self.num_nodes

    @property
    def area_um2(self) -> float:
        # Routers plus a coarse link allowance (one link of router-sized wiring
        # per node) so the wiring is not free.
        link_allowance = 0.10 * self.config.router_area_um2 * self.num_nodes
        return self.num_routers * self.config.router_area_um2 + link_allowance

    @property
    def static_power_w(self) -> float:
        return self.num_routers * self.config.router_static_power_w

    def average_hops(self) -> float:
        """Average number of hops a payload traverses for this topology."""
        if self.config.topology == "bus":
            return 1.0
        if self.config.topology == "point_to_point":
            return 1.0
        # Mesh: average Manhattan distance on a ~square grid ~ (2/3) * side.
        side = ceil(sqrt(self.num_nodes))
        return max(1.0, (2.0 * side) / 3.0)

    def transfer_metrics(
        self,
        num_bits: int,
        num_hops: Optional[float] = None,
    ) -> HardwareMetrics:
        """
        Cost of moving `num_bits` of payload across `num_hops` (default: the
        topology's average hop count).
        """
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        hops = self.average_hops() if num_hops is None else float(num_hops)
        c = self.config

        energy_pj = num_bits * hops * (
            c.router_energy_pj_per_bit
            + c.wire_energy_pj_per_bit_per_mm * c.link_length_mm
        )
        energy_nj = energy_pj * _PJ_TO_NJ
        latency_ns = hops * (1000.0 / c.frequency_mhz)
        return HardwareMetrics(
            area_um2=self.area_um2,
            power_w=self.static_power_w,
            latency_ns=latency_ns,
            energy_nj=energy_nj,
        )

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.static_power_w, 0.0, 0.0)


@dataclass
class BufferConfig:
    """Instruction/data buffer (on-chip SRAM) held in the tile."""
    name: str = "instruction_buffer"
    capacity_bits: int = 128 * 1024        # 16 kB default
    data_width_bits: int = 128
    frequency_mhz: float = 1000.0
    tech_node_nm: int = 65


class Buffer:
    """
    SRAM-style buffer.

    area    : capacity * per-bit cell area (6T SRAM ~1.0 um^2/bit at 65 nm)
    power   : leakage proportional to capacity
    access  : dynamic energy ~ 0.05 pJ per accessed bit at 65 nm
    """

    AREA_PER_BIT_65NM_UM2 = 1.0
    LEAKAGE_PER_BIT_65NM_W = 1e-11
    ACCESS_ENERGY_PJ_PER_BIT_65NM = 0.05

    def __init__(self, config: BufferConfig):
        if config.capacity_bits <= 0:
            raise ValueError("capacity_bits must be positive")
        self.config = config
        scale = (config.tech_node_nm / 65.0) ** 2
        self.area_um2 = self.AREA_PER_BIT_65NM_UM2 * config.capacity_bits * scale
        self.static_power_w = self.LEAKAGE_PER_BIT_65NM_W * config.capacity_bits * scale
        self._access_energy_pj_per_bit = self.ACCESS_ENERGY_PJ_PER_BIT_65NM * scale
        self.latency_ns = 1000.0 / config.frequency_mhz

    def access_metrics(self, num_accesses: int = 1, bits_per_access: Optional[int] = None) -> HardwareMetrics:
        if num_accesses < 0:
            raise ValueError("num_accesses must be non-negative")
        bits = self.config.data_width_bits if bits_per_access is None else bits_per_access
        energy_nj = num_accesses * bits * self._access_energy_pj_per_bit * _PJ_TO_NJ
        latency_ns = num_accesses * self.latency_ns
        return HardwareMetrics(self.area_um2, self.static_power_w, latency_ns, energy_nj)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.static_power_w, self.latency_ns, 0.0)


@dataclass
class IOPortConfig:
    """Tile/host IO interface."""
    name: str = "io_port"
    data_width_bits: int = 128
    frequency_mhz: float = 1000.0
    tech_node_nm: int = 65
    area_um2: Optional[float] = None
    energy_pj_per_bit: float = 1.0          # crossing a chip/host boundary is costlier than on-chip wire


class IOPort:
    """Digital IO port that moves control/data between the host and a tile."""

    def __init__(self, config: IOPortConfig):
        self.config = config
        scale = (config.tech_node_nm / 65.0) ** 2
        if config.area_um2 is None:
            # ~150 um^2 per bit of pad + driver logic at 65 nm.
            config.area_um2 = 150.0 * config.data_width_bits * scale
        self.area_um2 = config.area_um2
        self.latency_ns = 1000.0 / config.frequency_mhz

    def transfer_metrics(self, num_bits: int) -> HardwareMetrics:
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        energy_nj = num_bits * self.config.energy_pj_per_bit * _PJ_TO_NJ
        return HardwareMetrics(self.area_um2, 0.0, self.latency_ns, energy_nj)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, 0.0, self.latency_ns, 0.0)


@dataclass
class FunctionalUnitConfig:
    """Digital functional unit (ALU) inside the tile control unit."""
    name: str = "functional_unit"
    bitwidth: int = 32
    tech_node_nm: int = 65
    frequency_mhz: float = 1000.0
    complexity_factor: float = 4.0          # ALU (add/sub/logic/shift/compare) vs. a plain adder


class FunctionalUnit:
    """
    Digital ALU used by the control unit for scalar/vector arithmetic.

    Area/power are anchored to the existing `Adder` model (same 65 nm numbers)
    and multiplied by `complexity_factor` to reflect the full ALU.  No ISA /
    decode logic is modelled here - only the datapath cost.
    """

    def __init__(self, config: FunctionalUnitConfig):
        self.config = config
        self._adder = Adder(
            bitwidth=config.bitwidth,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.frequency_mhz,
        )
        self.area_um2 = self._adder.area_um2 * config.complexity_factor
        self.power_w = self._adder.power_w * config.complexity_factor
        self.latency_ns = self._adder.latency_ns

    def operation_metrics(self, num_ops: int = 1) -> HardwareMetrics:
        if num_ops < 0:
            raise ValueError("num_ops must be non-negative")
        latency_ns = num_ops * self.latency_ns
        energy_nj = self.power_w * latency_ns
        return HardwareMetrics(self.area_um2, self.power_w, latency_ns, energy_nj)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, 0.0)


# Processing Unit (Sec. 3.1, Fig. 3)

@dataclass
class PUConfig:
    """
    Processing-unit parameters.

    Defaults follow Table 1 of the paper: 16 arrays per PU, 8 local registers.
    The array group is large for capacity, but only ONE array is activated per
    operation (on-chip-network constraint), so dynamic energy uses a single
    crossbar while area counts all `num_arrays`.
    """
    name: str = "pu"
    num_arrays: int = 16
    num_pu_registers: int = 8
    pu_register_length_bits: int = 128
    input_precision_bits: int = 8
    frequency_mhz: float = 1000.0
    tech_node_nm: int = 65
    interconnect: InterconnectConfig = field(
        default_factory=lambda: InterconnectConfig(name="pu_noc", topology="mesh")
    )


@dataclass
class PUReadResult:
    activated_array_index: int
    crossbar_energy_j: float
    register_energy_j: float
    shift_add_energy_j: float
    interconnect_energy_j: float
    total_energy_j: float
    total_latency_ns: float


class ProcessingUnit:
    """
    One processing unit: an array group + local registers + shift&add + NoC.

    The array is built once as a template because all `num_arrays` are identical
    and only one is ever active.  PU-level registers and shift&add are additional
    structures on top of the per-array peripherals inside the crossbar.
    """

    def __init__(
        self,
        config: PUConfig,
        crossbar_factory: Callable[[], IntegratedMemristorCrossbar] = _default_crossbar_factory,
    ):
        if config.num_arrays <= 0:
            raise ValueError("num_arrays must be positive")
        self.config = config

        # One representative array; reused for the (single) activated array.
        self.array_template: IntegratedMemristorCrossbar = crossbar_factory()
        self.active_array_index = 0

        # PU-local register file (Sec. 4.2: group-0 ADC outputs, group-1 shift/add).
        self.register = Reg(
            bitwidth=config.pu_register_length_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.frequency_mhz,
        )

        # PU-level shift & add module (intra-PU post-processing, Sec. 4.5).
        self.shiftreg = ShiftReg(
            bitwidth=config.pu_register_length_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.frequency_mhz,
        )
        self.adder = Adder(
            bitwidth=config.pu_register_length_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.frequency_mhz,
        )

        # Array-group <-> register interconnect.
        self.interconnect = Interconnect(
            config.interconnect, num_nodes=config.num_arrays + 1
        )

    # ----- structure --------------------------------------------------------

    @property
    def active_array(self) -> IntegratedMemristorCrossbar:
        return self.array_template

    @property
    def array_group_area_um2(self) -> float:
        # All arrays counted, each with its full crossbar+peripheral overhead.
        return self.array_template.total_area_um2 * self.config.num_arrays

    @property
    def register_area_um2(self) -> float:
        return self.register.area_um2 * self.config.num_pu_registers

    @property
    def shift_add_area_um2(self) -> float:
        return self.shiftreg.area_um2 + self.adder.area_um2

    @property
    def interconnect_area_um2(self) -> float:
        return self.interconnect.area_um2

    @property
    def total_area_um2(self) -> float:
        return (
            self.array_group_area_um2
            + self.register_area_um2
            + self.shift_add_area_um2
            + self.interconnect_area_um2
        )

    def area_breakdown_um2(self) -> dict[str, float]:
        return {
            "array_group": self.array_group_area_um2,
            "one_array": self.array_template.total_area_um2,
            "pu_registers": self.register_area_um2,
            "pu_shift_add": self.shift_add_area_um2,
            "pu_interconnect": self.interconnect_area_um2,
            "total": self.total_area_um2,
        }

    # ----- a single activated VMM read -------------------------------------

    def read(
        self,
        input_codes: Optional[np.ndarray] = None,
        active_partial_sums: int = 1,
        accumulate_energy: bool = True,
    ) -> PUReadResult:
        """
        Execute one VMM on the (single) activated array and add PU-level
        register, shift&add, and interconnect costs on top.
        """
        xbar = self.active_array

        if input_codes is None:
            max_code = 2 ** xbar.dac.precision_bits - 1
            input_codes = np.full(xbar.rows, max_code, dtype=int)

        res = xbar.read_from_digital_input(
            input_codes,
            accumulate_energy=accumulate_energy,
            active_partial_sums=active_partial_sums,
        )

        # ADC outputs are latched into the PU register file (group-0) and then
        # combined across input-precision bits by the PU shift&add (group-1).
        n_adc = xbar.num_adcs
        reg_energy_j = n_adc * self.register.energy_nj * _NJ_TO_J

        shift_cycles = max(1, self.config.input_precision_bits - 1)
        shift_metrics = self.shiftreg.operation_metrics(shift_cycles)
        add_metrics = self.adder.accumulation_metrics(self.config.input_precision_bits, use_tree=True)
        shift_add_energy_j = (shift_metrics.energy_nj + add_metrics.energy_nj) * _NJ_TO_J

        # Move ADC results (n_adc * adc precision bits) from array to registers.
        ic = self.interconnect.transfer_metrics(n_adc * xbar.adc.precision_bits)
        interconnect_energy_j = ic.energy_nj * _NJ_TO_J

        total_energy_j = (
            res.total_energy_j
            + reg_energy_j
            + shift_add_energy_j
            + interconnect_energy_j
        )
        total_latency_ns = (
            res.total_latency_ns
            + self.register.latency_ns
            + shift_metrics.latency_ns
            + add_metrics.latency_ns
            + ic.latency_ns
        )

        return PUReadResult(
            activated_array_index=self.active_array_index,
            crossbar_energy_j=res.total_energy_j,
            register_energy_j=reg_energy_j,
            shift_add_energy_j=shift_add_energy_j,
            interconnect_energy_j=interconnect_energy_j,
            total_energy_j=total_energy_j,
            total_latency_ns=total_latency_ns,
        )

    def full_summary(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "num_arrays": self.config.num_arrays,
            "num_pu_registers": self.config.num_pu_registers,
            "pu_register_length_bits": self.config.pu_register_length_bits,
            "area_breakdown_um2": self.area_breakdown_um2(),
            "array_summary": self.array_template.full_summary(),
        }


# Tile (Sec. 3.1, Fig. 3)

@dataclass
class TileConfig:
    name: str = "tile"
    num_pus: int = 4

    # Tile-level register file (scalar/general + vector registers, Sec. 3.1).
    num_general_registers: int = 32
    num_vector_registers: int = 32
    register_length_bits: int = 128

    frequency_mhz: float = 1000.0
    tech_node_nm: int = 65

    pu: PUConfig = field(default_factory=PUConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    io: IOPortConfig = field(default_factory=IOPortConfig)
    functional_unit: FunctionalUnitConfig = field(default_factory=FunctionalUnitConfig)
    interconnect: InterconnectConfig = field(
        default_factory=lambda: InterconnectConfig(name="tile_noc", topology="mesh")
    )


@dataclass
class TileReadResult:
    pu_index: int
    pu_energy_j: float
    register_energy_j: float
    functional_unit_energy_j: float
    buffer_energy_j: float
    interconnect_energy_j: float
    total_energy_j: float
    total_latency_ns: float


class Tile:
    """
    One tile: IO + instruction buffer + register file + functional unit + a set
    of PUs, all stitched together by the tile-level interconnect.
    """

    def __init__(
        self,
        config: TileConfig,
        crossbar_factory: Callable[[], IntegratedMemristorCrossbar] = _default_crossbar_factory,
    ):
        if config.num_pus <= 0:
            raise ValueError("num_pus must be positive")
        self.config = config

        self.io = IOPort(config.io)
        self.buffer = Buffer(config.buffer)
        self.register = Reg(
            bitwidth=config.register_length_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.frequency_mhz,
        )
        self.functional_unit = FunctionalUnit(config.functional_unit)
        self.pus = [
            ProcessingUnit(config.pu, crossbar_factory=crossbar_factory)
            for _ in range(config.num_pus)
        ]
        self.interconnect = Interconnect(config.interconnect, num_nodes=config.num_pus + 1)

    # ----- structure --------------------------------------------------------

    @property
    def num_registers(self) -> int:
        return self.config.num_general_registers + self.config.num_vector_registers

    @property
    def register_area_um2(self) -> float:
        return self.register.area_um2 * self.num_registers

    @property
    def pu_array_area_um2(self) -> float:
        return sum(pu.total_area_um2 for pu in self.pus)

    @property
    def total_area_um2(self) -> float:
        return (
            self.io.area_um2
            + self.buffer.area_um2
            + self.register_area_um2
            + self.functional_unit.area_um2
            + self.pu_array_area_um2
            + self.interconnect.area_um2
        )

    def area_breakdown_um2(self) -> dict[str, float]:
        per_pu = self.pus[0].area_breakdown_um2() if self.pus else {}
        return {
            "io": self.io.area_um2,
            "buffer": self.buffer.area_um2,
            "registers": self.register_area_um2,
            "functional_unit": self.functional_unit.area_um2,
            "pus_total": self.pu_array_area_um2,
            "one_pu": per_pu.get("total", 0.0),
            "interconnect": self.interconnect.area_um2,
            "total": self.total_area_um2,
        }

    # ----- a single activated VMM through one PU ---------------------------

    def execute_vmm(
        self,
        pu_index: int = 0,
        input_codes: Optional[np.ndarray] = None,
        active_partial_sums: int = 1,
        accumulate_energy: bool = True,
    ) -> TileReadResult:
        """
        Run one VMM on `pu_index` and add the tile-level digital overheads:
        instruction fetch (buffer), operand read (register file), control/ALU
        work (functional unit), and CU<->PU transport (interconnect).
        """
        if not 0 <= pu_index < len(self.pus):
            raise IndexError("pu_index out of range")

        pu_res = self.pus[pu_index].read(
            input_codes=input_codes,
            active_partial_sums=active_partial_sums,
            accumulate_energy=accumulate_energy,
        )

        # Instruction fetch from the buffer (one instruction word).
        buf = self.buffer.access_metrics(num_accesses=1)
        buffer_energy_j = buf.energy_nj * _NJ_TO_J

        # Read the two source operand registers (addr + rs/vs, Sec. 4.3).
        reg_energy_j = 2 * self.register.energy_nj * _NJ_TO_J

        # Functional unit does the address/control arithmetic for the issue.
        fu = self.functional_unit.operation_metrics(num_ops=1)
        functional_unit_energy_j = fu.energy_nj * _NJ_TO_J

        # Transport the input vector (register_length bits) CU -> PU.
        ic = self.interconnect.transfer_metrics(self.config.register_length_bits)
        interconnect_energy_j = ic.energy_nj * _NJ_TO_J

        total_energy_j = (
            pu_res.total_energy_j
            + buffer_energy_j
            + reg_energy_j
            + functional_unit_energy_j
            + interconnect_energy_j
        )
        total_latency_ns = (
            pu_res.total_latency_ns
            + buf.latency_ns
            + self.register.latency_ns
            + fu.latency_ns
            + ic.latency_ns
        )

        return TileReadResult(
            pu_index=pu_index,
            pu_energy_j=pu_res.total_energy_j,
            register_energy_j=reg_energy_j,
            functional_unit_energy_j=functional_unit_energy_j,
            buffer_energy_j=buffer_energy_j,
            interconnect_energy_j=interconnect_energy_j,
            total_energy_j=total_energy_j,
            total_latency_ns=total_latency_ns,
        )

    def full_summary(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "num_pus": self.config.num_pus,
            "num_registers": self.num_registers,
            "area_breakdown_um2": self.area_breakdown_um2(),
            "pu_summary": self.pus[0].full_summary() if self.pus else {},
        }


# Chip (top level, Fig. 1 & Fig. 3)

@dataclass
class ChipConfig:
    name: str = "memristor_imc_chip"
    num_tiles: int = 4
    tile: TileConfig = field(default_factory=TileConfig)
    interconnect: InterconnectConfig = field(
        default_factory=lambda: InterconnectConfig(name="chip_noc", topology="mesh")
    )


@dataclass
class ChipReadResult:
    tile_index: int
    pu_index: int
    tile_energy_j: float
    interconnect_energy_j: float
    total_energy_j: float
    total_latency_ns: float


class Chip:
    """
    Top-level chip: a set of tiles joined by the chip-level NoC.

    Mirrors Fig. 3 (tiles, each with IO/buffer/registers/PUs) and Fig. 1 (tiles
    interconnected on a single die).  Only the physical architecture and its
    area/energy/latency are modelled; the ISA is left out.
    """

    def __init__(
        self,
        config: ChipConfig,
        crossbar_factory: Callable[[], IntegratedMemristorCrossbar] = _default_crossbar_factory,
    ):
        if config.num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.config = config
        self.tiles = [
            Tile(config.tile, crossbar_factory=crossbar_factory)
            for _ in range(config.num_tiles)
        ]
        self.interconnect = Interconnect(config.interconnect, num_nodes=config.num_tiles)

    # ----- structure --------------------------------------------------------

    @property
    def tiles_area_um2(self) -> float:
        return sum(t.total_area_um2 for t in self.tiles)

    @property
    def total_area_um2(self) -> float:
        return self.tiles_area_um2 + self.interconnect.area_um2

    @property
    def total_area_mm2(self) -> float:
        return self.total_area_um2 * 1e-6

    def area_breakdown_um2(self) -> dict[str, float]:
        one_tile = self.tiles[0].area_breakdown_um2() if self.tiles else {}
        return {
            "tiles_total": self.tiles_area_um2,
            "one_tile": one_tile.get("total", 0.0),
            "chip_interconnect": self.interconnect.area_um2,
            "total": self.total_area_um2,
        }

    # ----- a single activated VMM through one tile/PU ----------------------

    def execute_vmm(
        self,
        tile_index: int = 0,
        pu_index: int = 0,
        input_codes: Optional[np.ndarray] = None,
        active_partial_sums: int = 1,
        accumulate_energy: bool = True,
    ) -> ChipReadResult:
        if not 0 <= tile_index < len(self.tiles):
            raise IndexError("tile_index out of range")

        tile_res = self.tiles[tile_index].execute_vmm(
            pu_index=pu_index,
            input_codes=input_codes,
            active_partial_sums=active_partial_sums,
            accumulate_energy=accumulate_energy,
        )

        # Chip-level transport: deliver the input vector to the target tile.
        ic = self.interconnect.transfer_metrics(self.config.tile.register_length_bits)
        interconnect_energy_j = ic.energy_nj * _NJ_TO_J

        total_energy_j = tile_res.total_energy_j + interconnect_energy_j
        total_latency_ns = tile_res.total_latency_ns + ic.latency_ns

        return ChipReadResult(
            tile_index=tile_index,
            pu_index=pu_index,
            tile_energy_j=tile_res.total_energy_j,
            interconnect_energy_j=interconnect_energy_j,
            total_energy_j=total_energy_j,
            total_latency_ns=total_latency_ns,
        )

    # ----- reporting --------------------------------------------------------

    def full_summary(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "num_tiles": self.config.num_tiles,
            "total_area_um2": self.total_area_um2,
            "total_area_mm2": self.total_area_mm2,
            "area_breakdown_um2": self.area_breakdown_um2(),
            "tile_summary": self.tiles[0].full_summary() if self.tiles else {},
        }

    def save_summary_json(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.full_summary(), f, indent=2)

    def __repr__(self) -> str:
        return (
            f"Chip(name={self.config.name!r}, tiles={self.config.num_tiles}, "
            f"pus/tile={self.config.tile.num_pus}, "
            f"arrays/pu={self.config.tile.pu.num_arrays}, "
            f"total_area={self.total_area_mm2:.4f} mm^2)"
        )


# Convenience builder + demo

def build_chip(
    num_tiles: int = 4,
    num_pus_per_tile: int = 4,
    num_arrays_per_pu: int = 16,
    crossbar_factory: Callable[[], IntegratedMemristorCrossbar] = _default_crossbar_factory,
) -> Chip:
    """
    Build a chip whose arrays come straight from the existing crossbar config.

    Counts default to Table 1 of the paper (16 arrays/PU); tiles and PUs/tile
    are free parameters (the 32-bit address format supports up to 32 of each).
    """
    pu_cfg = PUConfig(num_arrays=num_arrays_per_pu)
    tile_cfg = TileConfig(num_pus=num_pus_per_tile, pu=pu_cfg)
    chip_cfg = ChipConfig(num_tiles=num_tiles, tile=tile_cfg)
    return Chip(chip_cfg, crossbar_factory=crossbar_factory)


if __name__ == "__main__":
    # Build from the central control panel so the demo reflects whatever is set
    # in simulation_config.py (counts, interconnect, clock, crossbar/device).
    from simulation_config import build_chip as build_chip_from_config
    chip = build_chip_from_config()

    print(chip)
    print("\n" + "=" * 60)
    print("CHIP AREA BREAKDOWN")
    print("=" * 60)
    for key, value in chip.area_breakdown_um2().items():
        print(f"  {key:18s}: {value:.4e} um^2  ({value * 1e-6:.4f} mm^2)")

    print("\n" + "=" * 60)
    print("ONE TILE AREA BREAKDOWN")
    print("=" * 60)
    for key, value in chip.tiles[0].area_breakdown_um2().items():
        print(f"  {key:18s}: {value:.4e} um^2")

    print("\n" + "=" * 60)
    print("ONE PU AREA BREAKDOWN")
    print("=" * 60)
    for key, value in chip.tiles[0].pus[0].area_breakdown_um2().items():
        print(f"  {key:18s}: {value:.4e} um^2")

    print("\n" + "=" * 60)
    print("SINGLE ACTIVATED VMM (tile 0, PU 0)")
    print("=" * 60)
    result = chip.execute_vmm(tile_index=0, pu_index=0, active_partial_sums=4)
    print(f"  PU-array path energy : {result.tile_energy_j:.4e} J")
    print(f"  chip interconnect    : {result.interconnect_energy_j:.4e} J")
    print(f"  total VMM energy     : {result.total_energy_j:.4e} J")
    print(f"  total VMM latency    : {result.total_latency_ns:.4e} ns")

    chip.save_summary_json("chip_architecture_summary.json")
    print("\nSummary written -> chip_architecture_summary.json")
