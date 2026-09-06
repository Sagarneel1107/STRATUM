"""
integrated_crossbar.py

Fully integrated memristor crossbar model.

This file integrates:
    - memristor device
    - crossbar conductance matrix
    - ADC
    - DAC
    - input/output registers
    - shift register
    - adder

The peripherals are not optional hooks. They are instantiated as part of every
IntegratedMemristorCrossbar object and their area is always counted. Their
operation energy/latency is also included in read and mapping estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from pathlib import Path
from typing import Optional, Sequence
import json
import numpy as np

from memristor_device import MemristorDevice
from hardware_peripherals import ADC, DAC, Reg, ShiftReg, Adder, HardwareMetrics


# Width of one digit-serial input digit. An 8-bit activation is fed to the array
# as radix-16 nibbles, matching riscv_imc_workflow.DigitIO (digit_max_code = 15)
# and resnet_vgg_study (_DIGITS = 2). One analog read per digit.
DIGIT_BITS = 4


def digits_per_input(input_precision_bits: int) -> int:
    """How many analog reads one input word costs on the digit-serial datapath."""
    return max(1, ceil(int(input_precision_bits) / DIGIT_BITS))


@dataclass
class CrossbarConfig:
    rows: int
    cols: int

    # Crossbar read/write conditions
    read_voltage_v: float = 0.1
    read_time_s: float = 50e-9
    write_amplitude_v: float = 1.0
    write_pulse_width_s: float = 100e-9

    # Output conversion
    load_resistance_ohm: Optional[float] = None

    # Wire placeholders for later refinement
    wire_resistance_ohm: float = 0.0
    wire_capacitance_f: float = 0.0

    # Peripheral architecture
    adc_choice: int = 2
    dac_choice: int = 1
    input_precision_bits: int = 8
    weight_precision_bits: int = 4
    output_precision_bits: int = 16
    digital_frequency_mhz: float = 100.0
    tech_node_nm: int = 65

    # Counts. If None, sensible crossbar-based defaults are used.
    num_adcs: Optional[int] = None
    num_dacs: Optional[int] = None
    num_input_regs: Optional[int] = None
    num_output_regs: Optional[int] = None
    num_shiftregs: Optional[int] = None
    num_adders: Optional[int] = None

    name: str = "integrated_memristor_crossbar"

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.read_time_s < 0 or self.write_pulse_width_s < 0:
            raise ValueError("times must be non-negative")
        if self.digital_frequency_mhz <= 0:
            raise ValueError("digital_frequency_mhz must be positive")


@dataclass
class IntegratedReadResult:
    input_codes: np.ndarray
    input_voltages_v: np.ndarray
    column_currents_a: np.ndarray
    column_analog_values: np.ndarray
    adc_output_codes: np.ndarray

    crossbar_energy_j: float
    dac_energy_j: float
    adc_energy_j: float
    input_reg_energy_j: float
    output_reg_energy_j: float
    shiftreg_energy_j: float
    adder_energy_j: float

    total_energy_j: float
    total_latency_ns: float
    # Energy efficiency: 2 ops per MAC; latency cancels in the ratio.
    tops_per_watt: float


@dataclass
class MappingEstimate:
    layer_rows: int
    layer_cols: int
    xbar_rows: int
    xbar_cols: int
    row_splits: int
    col_splits: int
    total_crossbars: int

    one_crossbar_area_um2: float
    total_crossbar_area_um2: float
    total_peripheral_area_um2: float
    total_area_um2: float

    accumulation_terms: int
    adder_latency_ns: float
    adder_energy_j: float


class IntegratedMemristorCrossbar:
    """
    Crossbar with peripherals integrated into the object.

    The physical computation is:
        digital input codes
        -> input registers
        -> shift registers
        -> DAC row voltages
        -> conductance matrix-vector multiplication
        -> ADC digitization
        -> output registers
        -> adder accumulation model

    This makes peripheral accounting part of the crossbar, not a later hook.
    """

    def __init__(
        self,
        config: CrossbarConfig,
        device: MemristorDevice,
        initial_conductance_s: Optional[float] = None,
    ):
        config.validate()

        self.config = config
        self.device = device

        if initial_conductance_s is None:
            if device.conductance_levels_s:
                initial_conductance_s = device.conductance_levels_s[0]
            elif device.current_conductance_s > 0:
                initial_conductance_s = device.current_conductance_s
            else:
                raise ValueError("initial_conductance_s is required")

        if initial_conductance_s <= 0:
            raise ValueError("initial_conductance_s must be positive")

        self.conductance_matrix_s = np.full(
            (config.rows, config.cols),
            float(initial_conductance_s),
            dtype=float,
        )

        # Fully integrated peripheral instances.  The converters are told the period
        # they are actually issued at (the analog read window), not their datasheet
        # sample rate: both are bias-dominated, so one idling between reads still
        # burns its static term.
        conversion_period_s = float(config.read_time_s) or None
        self.adc = ADC(choice=config.adc_choice,
                       conversion_period_s=conversion_period_s)
        self.dac = DAC(choice=config.dac_choice,
                       conversion_period_s=conversion_period_s)
        self.input_reg = Reg(
            bitwidth=config.input_precision_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.digital_frequency_mhz,
        )
        self.output_reg = Reg(
            bitwidth=config.output_precision_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.digital_frequency_mhz,
        )
        self.shiftreg = ShiftReg(
            bitwidth=config.input_precision_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.digital_frequency_mhz,
        )
        self.adder = Adder(
            bitwidth=config.output_precision_bits,
            tech_node_nm=config.tech_node_nm,
            frequency_mhz=config.digital_frequency_mhz,
        )

        # Count how many of each peripheral physically exist around this crossbar.
        self.num_adcs = config.num_adcs if config.num_adcs is not None else config.cols
        self.num_dacs = config.num_dacs if config.num_dacs is not None else config.rows
        self.num_input_regs = config.num_input_regs if config.num_input_regs is not None else config.rows
        self.num_output_regs = config.num_output_regs if config.num_output_regs is not None else config.cols
        self.num_shiftregs = config.num_shiftregs if config.num_shiftregs is not None else config.rows
        self.num_adders = config.num_adders if config.num_adders is not None else config.cols

        self.total_read_energy_j = 0.0
        self.total_peripheral_energy_j = 0.0
        self.last_read_result: Optional[IntegratedReadResult] = None

        self._configure_adc_from_crossbar_range()

    # Areas

    @property
    def rows(self) -> int:
        return self.config.rows

    @property
    def cols(self) -> int:
        return self.config.cols

    @property
    def crossbar_cell_area_um2(self) -> float:
        return self.rows * self.cols * self.device.device_area_um2

    @property
    def peripheral_area_um2(self) -> float:
        return (
            self.num_adcs * self.adc.area_um2
            + self.num_dacs * self.dac.area_um2
            + self.num_input_regs * self.input_reg.area_um2
            + self.num_output_regs * self.output_reg.area_um2
            + self.num_shiftregs * self.shiftreg.area_um2
            + self.num_adders * self.adder.area_um2
        )

    @property
    def total_area_um2(self) -> float:
        return self.crossbar_cell_area_um2 + self.peripheral_area_um2

    @property
    def average_conductance_s(self) -> float:
        """Mean of the conductances CURRENTLY stored in this array."""
        return float(np.mean(self.conductance_matrix_s))

    @property
    def device_mean_conductance_s(self) -> float:
        """
        Representative conductance of an array holding real weights, using the
        toolkit's typical weight statistic.  See `representative_conductance_s`.
        """
        return self.representative_conductance_s()

    @staticmethod
    def weight_abs_mean_ratio(W) -> float:
        """
        `E|W| / max|W|` for a weight tensor - the only statistic of the weights that
        the representative conductance depends on.  Feed the result to
        `representative_conductance_s` to cost an array holding THESE weights.
        """
        W = np.abs(np.asarray(W, dtype=float))
        peak = float(W.max()) if W.size else 0.0
        return float(W.mean() / peak) if peak > 0 else 0.0

    def representative_conductance_s(
        self, weight_abs_mean_ratio: Optional[float] = None
    ) -> float:
        """
        Mean conductance of an array holding real weights.

        The sign-magnitude mapping is `G = g_min + |w| * (g_max - g_min)/max|w|`, so

            E[G] = g_min + (g_max - g_min) * E|w| / max|w|

        This is not the mean of the conductance LEVEL TABLE.  Trained tensors are
        sharply peaked at zero, so `E|w|/max|w|` is ~0.166 (median over the 24
        weight tensors shipped here), not the 0.5 a level-table mean assumes - about
        175 nS for a programmed array, against 50 nS for an unprogrammed template
        and 425 nS for the level-table mean.

        Pass the measured ratio (see `weight_abs_mean_ratio`) whenever the weights
        are in hand; omit it and the toolkit-wide typical value is used.
        """
        levels = self.device.conductance_levels_s
        if not levels:
            return self.average_conductance_s
        g_min, g_max = float(np.min(levels)), float(np.max(levels))
        if weight_abs_mean_ratio is None:
            from simulation_config import TYPICAL_WEIGHT_ABS_MEAN_RATIO
            weight_abs_mean_ratio = TYPICAL_WEIGHT_ABS_MEAN_RATIO
        r = float(np.clip(float(weight_abs_mean_ratio), 0.0, 1.0))
        return g_min + (g_max - g_min) * r

    @property
    def total_energy_j(self) -> float:
        return self.total_read_energy_j + self.total_peripheral_energy_j

    def area_breakdown_um2(self) -> dict[str, float]:
        return {
            "crossbar_cells": self.crossbar_cell_area_um2,
            "adc_total": self.num_adcs * self.adc.area_um2,
            "dac_total": self.num_dacs * self.dac.area_um2,
            "input_reg_total": self.num_input_regs * self.input_reg.area_um2,
            "output_reg_total": self.num_output_regs * self.output_reg.area_um2,
            "shiftreg_total": self.num_shiftregs * self.shiftreg.area_um2,
            "adder_total": self.num_adders * self.adder.area_um2,
            "peripheral_total": self.peripheral_area_um2,
            "total": self.total_area_um2,
        }

    # Conductance storage/programming

    def set_conductance_matrix(self, conductance_matrix_s: Sequence[Sequence[float]]) -> None:
        matrix = np.asarray(conductance_matrix_s, dtype=float)

        if matrix.shape != (self.rows, self.cols):
            raise ValueError(f"conductance matrix must have shape {(self.rows, self.cols)}")

        if np.any(matrix <= 0):
            raise ValueError("all conductance values must be positive")

        self.conductance_matrix_s = matrix
        self._configure_adc_from_crossbar_range()

    def set_conductance_from_state_codes(self, state_code_matrix: Sequence[Sequence[int]]) -> None:
        codes = np.asarray(state_code_matrix, dtype=int)

        if codes.shape != (self.rows, self.cols):
            raise ValueError(f"state code matrix must have shape {(self.rows, self.cols)}")

        n_states = len(self.device.conductance_levels_s)
        if n_states == 0:
            raise RuntimeError("Device has no conductance levels stored")
        if np.any((codes < 0) | (codes >= n_states)):
            raise IndexError("conductance code is out of range")

        # Vectorised numpy indexing rather than an O(rows*cols) Python loop.
        levels = np.asarray(self.device.conductance_levels_s, dtype=float)
        self.set_conductance_matrix(levels[codes])

    def program_matrix(
        self,
        target_conductance_matrix_s: Sequence[Sequence[float]],
        pulse_counts: int | Sequence[Sequence[int]] = 1,
        accumulate_energy: bool = True,
    ) -> float:
        """
        Program all cells to a target conductance matrix.  Returns 0.0.

        Write energy is not reported: the model integrated the pulse at the
        conductance the cell ends at, and there is no write-verify loop,
        pulse-count model or programming latency behind the flat matrix write, so
        nothing here constrains the number.  `pulse_counts` is still validated so a
        broken schedule is not silently accepted.
        """
        target = np.asarray(target_conductance_matrix_s, dtype=float)

        if target.shape != (self.rows, self.cols):
            raise ValueError(f"target matrix must have shape {(self.rows, self.cols)}")
        if np.any(target <= 0):
            raise ValueError("target conductances must be positive")

        if isinstance(pulse_counts, int):
            pulses = np.full((self.rows, self.cols), pulse_counts, dtype=float)
        else:
            pulses = np.asarray(pulse_counts, dtype=float)
            if pulses.shape != (self.rows, self.cols):
                raise ValueError(f"pulse count matrix must have shape {(self.rows, self.cols)}")

        if np.any(pulses <= 0):
            raise ValueError("pulse counts must be positive")

        self.conductance_matrix_s = target
        self._configure_adc_from_crossbar_range()
        return 0.0

    def program_cell_from_trace(
        self,
        row: int,
        col: int,
        conductance_trace_s: Sequence[float],
        pulse_amplitudes_v: Optional[Sequence[float]] = None,
        pulse_widths_s: Optional[Sequence[float]] = None,
        accumulate_energy: bool = True,
    ) -> float:
        self._check_cell(row, col)

        if len(conductance_trace_s) == 0:
            raise ValueError("conductance_trace_s cannot be empty")

        # Write energy is not reported - see `program_matrix`.
        e_j = 0.0

        final_g = float(conductance_trace_s[-1])
        if final_g <= 0:
            raise ValueError("final conductance must be positive")

        self.conductance_matrix_s[row, col] = final_g
        self._configure_adc_from_crossbar_range()
        return e_j

    # Integrated read path

    def read_from_digital_input(
        self,
        input_codes: Sequence[int],
        accumulate_energy: bool = True,
        active_partial_sums: int = 1,
    ) -> IntegratedReadResult:
        """
        Full integrated read path.

        1. input_codes are latched once by the input registers
        2. the word is split into radix-16 digits (DIGIT_BITS wide)
        3. for each digit: DAC drives the rows, the array performs the analog MVM,
           every column is digitised, and the output registers latch it
        4. shift registers weight the higher digit and the adders fold it in
        5. adders estimate cross-tile partial-sum accumulation

        The returned currents and ADC codes are the full-precision analog result;
        the energy and latency describe the digit-serial datapath.
        """
        codes = np.asarray(input_codes, dtype=int)

        if codes.shape != (self.rows,):
            raise ValueError(f"input_codes must have shape ({self.rows},)")

        v_min = 0.0
        v_max = self.config.read_voltage_v
        vin = self.dac.codes_to_voltages(codes, v_min=v_min, v_max=v_max)

        return self._integrated_read_from_voltages(
            input_codes=codes,
            input_voltages_v=vin,
            accumulate_energy=accumulate_energy,
            active_partial_sums=active_partial_sums,
        )

    def read_from_analog_voltages(
        self,
        input_voltages_v: Sequence[float],
        accumulate_energy: bool = True,
        active_partial_sums: int = 1,
    ) -> IntegratedReadResult:
        """
        Full integrated read path when row voltages are already known.

        DAC energy is still included because the integrated architecture has DAC
        row drivers. The corresponding digital code is approximated by inverse
        DAC mapping.
        """
        vin = np.asarray(input_voltages_v, dtype=float)

        if vin.shape != (self.rows,):
            raise ValueError(f"input_voltages_v must have shape ({self.rows},)")

        max_code = 2 ** self.dac.precision_bits - 1
        if self.config.read_voltage_v == 0:
            codes = np.zeros(self.rows, dtype=int)
        else:
            codes = np.rint(vin / self.config.read_voltage_v * max_code).astype(int)
            codes = np.clip(codes, 0, max_code)

        return self._integrated_read_from_voltages(
            input_codes=codes,
            input_voltages_v=vin,
            accumulate_energy=accumulate_energy,
            active_partial_sums=active_partial_sums,
        )

    def _integrated_read_from_voltages(
        self,
        input_codes: np.ndarray,
        input_voltages_v: np.ndarray,
        accumulate_energy: bool,
        active_partial_sums: int,
    ) -> IntegratedReadResult:
        # Crossbar MVM current:
        column_currents_a = input_voltages_v @ self.conductance_matrix_s

        # If load resistance exists, ADC sees voltage. Otherwise ADC sees current.
        if self.config.load_resistance_ohm is None:
            analog_values = column_currents_a
        else:
            analog_values = column_currents_a * self.config.load_resistance_ohm

        adc_codes = self.adc.quantize(analog_values)

        # Digit-serial accounting, matching riscv_imc_workflow._matrix_vmm and
        # resnet_vgg_study.  An 8-bit activation reaches the array as `n_digits`
        # radix-16 nibbles, and each digit is a separate physical event: rows
        # re-driven at that digit's DAC code, array re-read, every column
        # re-digitised, and only then the PU shift-and-add.
        n_digits = digits_per_input(self.config.input_precision_bits)

        # Row-side peripherals only burn energy for rows that actually drive a
        # non-zero voltage; column-side peripherals are active on every column.
        active_rows = int(np.count_nonzero(input_voltages_v))

        # Crossbar read energy, per digit:
        # E = t * sum_rows sum_cols V_row^2 * G_ij
        row_power_w = (input_voltages_v ** 2) @ self.conductance_matrix_s
        crossbar_energy_j = float(
            n_digits * self.config.read_time_s * np.sum(row_power_w))

        # One DAC drive, one conversion and one output latch PER DIGIT.
        dac_energy_j = n_digits * active_rows * self.dac.energy_nj * 1e-9
        adc_energy_j = n_digits * self.num_adcs * self.adc.energy_nj * 1e-9
        output_reg_energy_j = (
            n_digits * self.num_output_regs * self.output_reg.energy_nj * 1e-9)

        # The activation word is latched ONCE and then decomposed into digits.
        input_reg_energy_j = active_rows * self.input_reg.energy_nj * 1e-9

        # Combining the digits: shift the higher digit and add it to the running
        # column accumulator, so (n_digits - 1) shift+add pairs on the OUTPUT side.
        # (`_shift_add_cost(n_cols, shift=d>0, add=d>0)` in riscv_imc_workflow.)
        digit_combines = max(n_digits - 1, 0)
        shift_metrics = self.shiftreg.operation_metrics(digit_combines)
        shiftreg_energy_j = self.num_output_regs * shift_metrics.energy_nj * 1e-9
        digit_add_metrics = self.adder.accumulation_metrics(2, use_tree=True)
        digit_add_energy_j = (
            digit_combines * self.num_adders * digit_add_metrics.energy_nj * 1e-9)

        # Cross-tile partial-sum accumulation, on top of the digit combine.
        # If active_partial_sums=1, energy is zero but area remains counted.
        adder_metrics = self.adder.accumulation_metrics(active_partial_sums, use_tree=True)
        adder_energy_j = (
            self.num_adders * adder_metrics.energy_nj * 1e-9 + digit_add_energy_j)

        total_energy_j = (
            crossbar_energy_j
            + dac_energy_j
            + adc_energy_j
            + input_reg_energy_j
            + output_reg_energy_j
            + shiftreg_energy_j
            + adder_energy_j
        )

        # Per-digit read latency keeps the max() form (the stages overlap in a
        # pipelined read); the digits themselves are sequential, because the second
        # nibble needs the same rows re-driven at a different DAC code.
        per_digit_latency_ns = max(
            self.config.read_time_s * 1e9,
            self.dac.latency_ns,
            self.adc.latency_ns,
            self.input_reg.latency_ns,
            self.output_reg.latency_ns,
        )
        clk_ns = 1000.0 / self.config.digital_frequency_mhz
        total_latency_ns = (
            n_digits * per_digit_latency_ns
            + digit_combines * 2.0 * clk_ns          # one shift + one add per combine
            + adder_metrics.latency_ns
        )

        # TOPS/W: 1 MAC = 2 ops (multiply + accumulate).
        # TOPS   = 2 * active_rows * cols / latency_s / 1e12
        # Power  = total_energy_j / latency_s
        # TOPS/W = TOPS / Power  =>  latency cancels  =>  2*rows*cols / (E_J * 1e12)
        tops_per_watt = (
            (2.0 * active_rows * self.cols) / (total_energy_j * 1e12)
            if total_energy_j > 0 and active_rows > 0
            else 0.0
        )

        result = IntegratedReadResult(
            input_codes=input_codes,
            input_voltages_v=input_voltages_v,
            column_currents_a=column_currents_a,
            column_analog_values=analog_values,
            adc_output_codes=adc_codes,
            crossbar_energy_j=crossbar_energy_j,
            dac_energy_j=dac_energy_j,
            adc_energy_j=adc_energy_j,
            input_reg_energy_j=input_reg_energy_j,
            output_reg_energy_j=output_reg_energy_j,
            shiftreg_energy_j=shiftreg_energy_j,
            adder_energy_j=adder_energy_j,
            total_energy_j=total_energy_j,
            total_latency_ns=total_latency_ns,
            tops_per_watt=tops_per_watt,
        )

        self.last_read_result = result

        if accumulate_energy:
            self.total_read_energy_j += crossbar_energy_j
            self.total_peripheral_energy_j += total_energy_j - crossbar_energy_j

        return result

    # Layer mapping with integrated area and adder estimate

    def estimate_layer_mapping(self, layer_rows: int, layer_cols: int) -> MappingEstimate:
        """
        Estimate the number of crossbars and the peripheral cost for a layer.

        For a layer_rows x layer_cols weight matrix: `row_splits` splits the input
        dimension across crossbar rows, `col_splits` the output dimension across
        columns, and total_crossbars = row_splits * col_splits.  With row_splits > 1
        the partial sums from several crossbars must be added.
        """
        if layer_rows <= 0 or layer_cols <= 0:
            raise ValueError("layer dimensions must be positive")

        row_splits = ceil(layer_rows / self.rows)
        col_splits = ceil(layer_cols / self.cols)
        total_crossbars = row_splits * col_splits

        accumulation_terms = row_splits
        adder_metrics = self.adder.accumulation_metrics(accumulation_terms, use_tree=True)
        # One adder tree per output column block. Approximate count: layer_cols.
        adder_energy_j = layer_cols * adder_metrics.energy_nj * 1e-9

        one_area = self.total_area_um2
        total_crossbar_area = self.crossbar_cell_area_um2 * total_crossbars
        total_peripheral_area = self.peripheral_area_um2 * total_crossbars

        return MappingEstimate(
            layer_rows=layer_rows,
            layer_cols=layer_cols,
            xbar_rows=self.rows,
            xbar_cols=self.cols,
            row_splits=row_splits,
            col_splits=col_splits,
            total_crossbars=total_crossbars,
            one_crossbar_area_um2=one_area,
            total_crossbar_area_um2=total_crossbar_area,
            total_peripheral_area_um2=total_peripheral_area,
            total_area_um2=total_crossbar_area + total_peripheral_area,
            accumulation_terms=accumulation_terms,
            adder_latency_ns=adder_metrics.latency_ns,
            adder_energy_j=adder_energy_j,
        )

    # Energy/metric reports

    def energy_breakdown_j(self) -> dict[str, float]:
        if self.last_read_result is None:
            last = {
                "last_crossbar_read_energy_j": 0.0,
                "last_dac_energy_j": 0.0,
                "last_adc_energy_j": 0.0,
                "last_input_reg_energy_j": 0.0,
                "last_output_reg_energy_j": 0.0,
                "last_shiftreg_energy_j": 0.0,
                "last_adder_energy_j": 0.0,
                "last_total_read_path_energy_j": 0.0,
            }
        else:
            r = self.last_read_result
            last = {
                "last_crossbar_read_energy_j": r.crossbar_energy_j,
                "last_dac_energy_j": r.dac_energy_j,
                "last_adc_energy_j": r.adc_energy_j,
                "last_input_reg_energy_j": r.input_reg_energy_j,
                "last_output_reg_energy_j": r.output_reg_energy_j,
                "last_shiftreg_energy_j": r.shiftreg_energy_j,
                "last_adder_energy_j": r.adder_energy_j,
                "last_total_read_path_energy_j": r.total_energy_j,
            }

        totals = {
            "accumulated_crossbar_read_energy_j": self.total_read_energy_j,
            "accumulated_peripheral_energy_j": self.total_peripheral_energy_j,
            "accumulated_total_energy_j": self.total_energy_j,
        }

        return {**last, **totals}

    def full_summary(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "rows": self.rows,
            "cols": self.cols,
            "device_states": self.device.number_of_states,
            "device_native_precision_bits": self.device.native_precision_bits,
            "average_conductance_s": self.average_conductance_s,
            "area_breakdown_um2": self.area_breakdown_um2(),
            "energy_breakdown_j": self.energy_breakdown_j(),
            "peripheral_counts": {
                "num_adcs": self.num_adcs,
                "num_dacs": self.num_dacs,
                "num_input_regs": self.num_input_regs,
                "num_output_regs": self.num_output_regs,
                "num_shiftregs": self.num_shiftregs,
                "num_adders": self.num_adders,
            },
            "peripheral_unit_metrics": {
                "adc": asdict(self.adc.metrics()),
                "dac": asdict(self.dac.metrics()),
                "input_reg": asdict(self.input_reg.metrics()),
                "output_reg": asdict(self.output_reg.metrics()),
                "shiftreg": asdict(self.shiftreg.metrics()),
                "adder": asdict(self.adder.metrics()),
            },
        }

    def reset_energy_counters(self) -> None:
        self.total_read_energy_j = 0.0
        self.total_peripheral_energy_j = 0.0
        self.last_read_result = None

    # IO

    def save_conductance_matrix_csv(self, path: str | Path) -> None:
        np.savetxt(Path(path), self.conductance_matrix_s, delimiter=",")

    def load_conductance_matrix_csv(self, path: str | Path) -> None:
        matrix = np.loadtxt(Path(path), delimiter=",")
        self.set_conductance_matrix(matrix)

    def save_summary_json(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.full_summary(), f, indent=2)

    # Internal helpers

    def _configure_adc_from_crossbar_range(self) -> None:
        """
        Configure ADC thresholds based on the present conductance matrix and
        read-voltage range.

        If load_resistance is given, thresholds are in volts.
        Otherwise thresholds are in amperes.
        """
        g_min = float(np.min(self.conductance_matrix_s))
        g_max = float(np.max(self.conductance_matrix_s))

        # Minimum output assumes all rows at 0 V.
        out_min = 0.0

        # Maximum output assumes every row receives max read voltage and every
        # device is at max conductance.
        out_max = self.rows * self.config.read_voltage_v * g_max

        if self.config.load_resistance_ohm is not None:
            out_max *= self.config.load_resistance_ohm

        if out_max <= out_min:
            out_max = out_min + 1e-15

        self.adc.configure_intervals(out_min, out_max)

    def _check_cell(self, row: int, col: int) -> None:
        if not 0 <= row < self.rows:
            raise IndexError("row index out of range")
        if not 0 <= col < self.cols:
            raise IndexError("col index out of range")

    def __repr__(self) -> str:
        return (
            f"IntegratedMemristorCrossbar(rows={self.rows}, cols={self.cols}, "
            f"avg_G={self.average_conductance_s:.4e} S, "
            f"total_area={self.total_area_um2:.4e} um^2)"
        )
