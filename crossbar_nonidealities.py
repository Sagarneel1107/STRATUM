"""
crossbar_nonidealities.py

Crossbar-level non-ideality model for the integrated memristor PIM simulator.

1.  IR drop / wire resistance       : voltage loss along row and column wires
3.  Parasitic series resistance     : contact and via resistance per cell
4.  Parasitic capacitance           : wire capacitance, extra energy and RC latency
5.  Read noise accumulation         : thermal/shot noise summed across each column
6.  Programming voltage drop        : reduced write amplitude at far cells
7.  ADC/DAC quantization errors     : systematic offset and gain errors
8.  Current saturation              : per-column current ceiling
9.  Temperature gradient            : self-heating shifts conductance via TCR
10. Peripheral comparator offset    : systematic sense-amplifier bias
11. Access transistor + col gating  : 1T1R select-transistor on-resistance; unused
                                      output columns are disconnected from their
                                      sense amplifiers

Sneak paths are not modelled: this array is 1T1R, and the select transistor is
the device that eliminates them.  A 0T1R study would need the term back, and
would then have to drop the access transistor.

Use after single_memristor_nonidealities.py has written to
xbar.conductance_matrix_s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from integrated_crossbar import IntegratedReadResult, digits_per_input


@dataclass
class CrossbarNonIdealityConfig:
    """Parameters for all crossbar-level non-idealities."""

    enabled: bool = True

    # 1. Wire resistance - one segment between adjacent row/col nodes (ohm)
    wire_resistance_ohm: float = 0.0

    # 1b. Fixed-point passes used to solve the IR-drop network. The drop sets the
    # currents and the currents set the drop, so one pass is not a solution; four
    # passes settle a 128x128 array to well below a converter LSB.
    ir_drop_iterations: int = 4

    # 3. Parasitic series resistance - contact + via per cell (ohm)
    parasitic_series_resistance_ohm: float = 0.0

    # 4. Parasitic capacitance - per wire segment (F)
    parasitic_capacitance_f: float = 0.0

    # 5. Read noise accumulation
    read_noise_current_a: float = 0.0   # absolute noise floor (A RMS); scales as sqrt(rows)
    read_noise_relative: float = 0.0    # fraction of column current

    # 5b. PHYSICAL (time-dependent) sense noise.  The two terms above are
    # integration-time independent; these add the floors that scale as
    # 1/sqrt(t_read), so a faster read costs effective bits:
    #     shot     sigma = sqrt(q * |I_col| / t)
    #     thermal  sigma = sqrt(2 * kB * T * G_col / t)
    # False recovers the time-blind behaviour.
    physical_sense_noise: bool = True
    sense_temperature_k: float = 300.0

    # 6. Programming voltage drop - max fractional drop at the farthest cell corner [0, 1]
    programming_voltage_drop: float = 0.0

    # 6b. SET threshold and overdrive exponent for the programming model.  Below
    # V_th nothing switches; above it the conductance grows superlinearly with the
    # overdrive, and a cell below V_th stays at g_min rather than programming
    # partially.
    #   overdrive = clip((V_cell - V_th) / (V_write - V_th), 0, 1)
    #   G_written = g_min + (G_target - g_min) * overdrive ** exponent
    set_threshold_fraction: float = 0.55     # V_th as a fraction of the write amplitude
    programming_overdrive_exponent: float = 2.0

    # 7. ADC/DAC errors
    dac_offset_v: float = 0.0       # systematic DAC output offset (V)
    dac_gain_error: float = 0.0     # relative DAC scale error (dimensionless)
    adc_offset_a: float = 0.0       # systematic ADC input offset (A)
    adc_gain_error: float = 0.0     # relative ADC scale error (dimensionless)

    # 8. Current saturation - per-column ceiling (A); None disables
    current_saturation_a: Optional[float] = None

    # 9. Self-heating
    temperature_coefficient_per_k: float = 0.0   # TCR: fractional dG per K
    # Temperature rise is dissipation times a thermal resistance, over two paths:
    #   dT_cell = R_th_cell  * P_cell        own filament, thermally confined
    #   dT_bulk = R_th_array * P_array       die-level background, shared by all
    # `max_temperature_rise_k` is a ceiling (package / runaway limit) applied after
    # the physics; 0 or negative disables the clamp.
    thermal_resistance_cell_k_per_w: float = 1.0e7    # K/W, one confined filament
    thermal_resistance_array_k_per_w: float = 3.0e3   # K/W, whole array to ambient
    max_temperature_rise_k: float = 0.0                # ceiling on the rise (K); 0 = none

    # 10. Peripheral comparator offset - systematic bias on every column output (A)
    comparator_offset_a: float = 0.0

    # 11. Access/select transistor (1T1R-style) on-resistance when ON (ohm), in
    #     series with every active cell.  The transistors also gate unused output
    #     columns off (see read_from_digital_input's active_output_cols): a
    #     disconnected column draws no current and burns no ADC / register energy.
    access_transistor_on_resistance_ohm: float = 0.0

    def validate(self) -> None:
        nonneg = [
            "wire_resistance_ohm",
            "parasitic_series_resistance_ohm",
            "parasitic_capacitance_f",
            "read_noise_current_a",
            "read_noise_relative",
            "max_temperature_rise_k",
            "access_transistor_on_resistance_ohm",
            "thermal_resistance_cell_k_per_w",
            "thermal_resistance_array_k_per_w",
        ]
        for field in nonneg:
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")
        if not 0.0 <= self.programming_voltage_drop <= 1.0:
            raise ValueError("programming_voltage_drop must be in [0, 1]")
        if not 0.0 <= self.set_threshold_fraction < 1.0:
            raise ValueError("set_threshold_fraction must be in [0, 1)")
        if self.programming_overdrive_exponent <= 0:
            raise ValueError("programming_overdrive_exponent must be positive")
        if self.current_saturation_a is not None and self.current_saturation_a <= 0:
            raise ValueError("current_saturation_a must be positive when set")


class CrossbarNonIdealityModel:
    """
    Applies crossbar-level non-idealities during programming and read.

        cb_model = CrossbarNonIdealityModel(config, rng=0)
        cb_model.apply_to_crossbar(xbar)               # programming voltage drop
        result = cb_model.read_from_digital_input(xbar, input_codes)
        stats = cb_model.nonideal_stats(xbar, v_applied)
    """

    #: Relative change in the COLUMN CURRENT at which the IR-drop fixed point is
    #: called converged.  Three orders below a 4-bit converter LSB.
    IR_CONVERGENCE_RTOL = 1e-4

    def __init__(
        self,
        config: Optional[CrossbarNonIdealityConfig] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.config = config if config is not None else CrossbarNonIdealityConfig()
        self.config.validate()
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # Programming path

    def apply_to_crossbar(self, xbar, g_min_s: Optional[float] = None) -> None:
        """
        Apply programming-phase crossbar non-idealities in place.

        Cells far from the drivers receive a reduced pulse amplitude, and one that
        does not clear its SET threshold does not program at all.  Call after the
        single-device non-idealities have been programmed into xbar.

        Idempotent: the pristine target is stashed on the crossbar the first time,
        and any later `set_conductance_matrix` clears and re-arms it.
        """
        if not self.config.enabled or self.config.programming_voltage_drop == 0.0:
            return

        stash = getattr(xbar, "_pristine_program_target_s", None)
        target = xbar.conductance_matrix_s
        if stash is not None and stash.shape == target.shape and np.array_equal(
                getattr(xbar, "_program_drop_result_s", None), target):
            # Already degraded by this model and untouched since: re-derive from the
            # pristine target instead of degrading the degraded matrix again.
            target = stash
        else:
            stash = np.array(target, dtype=float, copy=True)

        if g_min_s is None:
            levels = getattr(getattr(xbar, "device", None), "conductance_levels_s", None)
            g_min_s = float(min(levels)) if levels else float(np.min(target))

        G_eff = self._apply_programming_voltage_drop(
            target, xbar.rows, xbar.cols, float(g_min_s)
        )
        xbar.set_conductance_matrix(G_eff)
        xbar._pristine_program_target_s = stash
        xbar._program_drop_result_s = np.array(
            xbar.conductance_matrix_s, dtype=float, copy=True)

    def _apply_programming_voltage_drop(
        self, G: np.ndarray, rows: int, cols: int, g_min_s: float = 0.0
    ) -> np.ndarray:
        """
        Conductance actually written, given the amplitude that reaches each cell.

        Both the row wire (driven from column 0) and the column wire (driven from
        row 0) drop voltage, so the delivered amplitude falls with distance;
        `norm_pos` averages the two directions.  The response is not proportional
        to amplitude - a filamentary cell has a SET threshold.  With `V_write`
        normalised to 1:

            v_cell    = 1 - drop * norm_pos
            overdrive = clip((v_cell - v_th) / (1 - v_th), 0, 1)
            G_written = g_min + (G_target - g_min) * overdrive ** exponent
        """
        c = self.config
        i = np.arange(rows, dtype=float)[:, None]
        j = np.arange(cols, dtype=float)[None, :]
        norm_pos = (i / max(rows - 1, 1) + j / max(cols - 1, 1)) / 2.0

        v_cell = 1.0 - c.programming_voltage_drop * norm_pos
        v_th = float(c.set_threshold_fraction)
        overdrive = np.clip((v_cell - v_th) / max(1.0 - v_th, 1e-12), 0.0, 1.0)
        written = g_min_s + (G - g_min_s) * overdrive ** c.programming_overdrive_exponent
        return np.maximum(written, 1e-18)

    # Read path

    def read_from_digital_input(
        self,
        xbar,
        input_codes,
        active_partial_sums: int = 1,
        accumulate_energy: bool = True,
        active_output_cols: Optional[int] = None,
    ) -> IntegratedReadResult:
        """
        Full non-ideal integrated read path, replacing
        xbar.read_from_digital_input.  Injects crossbar-level non-idealities
        between DAC outputs and ADC inputs, in this order:

        DAC offset/gain -> parasitic + access-transistor R -> temperature ->
        IR drop -> current saturation -> read noise -> comparator offset ->
        ADC offset/gain -> ADC quantization

        `active_output_cols` is the number of leading output columns whose access
        transistors are ON; the rest are gated off and carry no current and no ADC
        / register / adder energy.  None leaves every column active.
        """
        codes = np.asarray(input_codes, dtype=int)

        # DAC: convert codes to voltages, then apply offset and gain errors.
        v_min, v_max = 0.0, xbar.config.read_voltage_v
        v_nominal = xbar.dac.codes_to_voltages(codes, v_min, v_max)

        if self.config.enabled:
            v_applied = np.maximum(
                v_nominal * (1.0 + self.config.dac_gain_error) + self.config.dac_offset_v,
                0.0,
            )
        else:
            v_applied = v_nominal

        # Column access-transistor gating: output columns at/after n_active_cols are
        # disconnected from their sense amplifiers (no current, no column energy).
        if active_output_cols is None:
            n_active_cols = xbar.cols
        else:
            n_active_cols = int(max(0, min(active_output_cols, xbar.cols)))

        G_used = xbar.conductance_matrix_s
        if n_active_cols < xbar.cols:
            G_used = G_used.copy()
            G_used[:, n_active_cols:] = 0.0

        # Non-ideal MVM.
        if self.config.enabled:
            I_col = self._compute_column_currents(
                v_applied, G_used, xbar.config.read_voltage_v, xbar.rows,
                read_time_s=float(getattr(xbar.config, "read_time_s", 0.0) or 0.0),
            )
        else:
            I_col = v_applied @ G_used

        if n_active_cols < xbar.cols:
            I_col[n_active_cols:] = 0.0

        # ADC: apply gain/offset errors, then quantize.
        if self.config.enabled:
            I_sensed = I_col * (1.0 + self.config.adc_gain_error) + self.config.adc_offset_a
        else:
            I_sensed = I_col

        analog_values = (
            I_sensed * xbar.config.load_resistance_ohm
            if xbar.config.load_resistance_ohm is not None
            else I_sensed
        )
        adc_codes = xbar.adc.quantize(analog_values)

        # Energy (mirrors integrated_crossbar._integrated_read_from_voltages).
        # Digit-serial: each of the `n_digits` radix-16 nibbles is its own physical
        # event - rows re-driven, array re-read, active columns re-digitised.
        n_digits = digits_per_input(xbar.config.input_precision_bits)
        digit_combines = max(n_digits - 1, 0)

        active_rows = int(np.count_nonzero(v_applied))
        # Gated-off columns sense nothing, so only active columns burn ADC /
        # output-register / adder energy.
        active_adcs        = min(n_active_cols, xbar.num_adcs)
        active_output_regs = min(n_active_cols, xbar.num_output_regs)
        active_adders      = min(n_active_cols, xbar.num_adders)

        row_power_w = (v_applied ** 2) @ G_used
        crossbar_energy_j = float(
            n_digits * xbar.config.read_time_s * np.sum(row_power_w))
        # The wire capacitance is charged and discharged once per digit read.
        crossbar_energy_j += n_digits * self._parasitic_cap_energy(xbar, v_applied)

        dac_energy_j        = n_digits * active_rows * xbar.dac.energy_nj * 1e-9
        adc_energy_j        = n_digits * active_adcs * xbar.adc.energy_nj * 1e-9
        output_reg_energy_j = n_digits * active_output_regs * xbar.output_reg.energy_nj * 1e-9

        # The activation word is latched once, then decomposed into digits.
        input_reg_energy_j  = active_rows * xbar.input_reg.energy_nj * 1e-9

        # Digit combine: one shift + one add per extra digit, on the output side.
        shift_metrics      = xbar.shiftreg.operation_metrics(digit_combines)
        shiftreg_energy_j  = active_output_regs * shift_metrics.energy_nj * 1e-9
        digit_add_metrics  = xbar.adder.accumulation_metrics(2, use_tree=True)
        digit_add_energy_j = digit_combines * active_adders * digit_add_metrics.energy_nj * 1e-9

        # Cross-tile partial-sum accumulation, on top of the digit combine.
        adder_metrics     = xbar.adder.accumulation_metrics(active_partial_sums, use_tree=True)
        adder_energy_j    = active_adders * adder_metrics.energy_nj * 1e-9 + digit_add_energy_j

        total_energy_j = (
            crossbar_energy_j
            + dac_energy_j + adc_energy_j
            + input_reg_energy_j + output_reg_energy_j
            + shiftreg_energy_j + adder_energy_j
        )

        # Per-digit read latency keeps the max() form (the stages of one read
        # overlap); the digits are sequential, because the next nibble needs the
        # same rows re-driven at a different DAC code.
        per_digit_latency_ns = max(
            xbar.config.read_time_s * 1e9,
            xbar.dac.latency_ns,
            xbar.adc.latency_ns,
            xbar.input_reg.latency_ns,
            xbar.output_reg.latency_ns,
            self._parasitic_rc_latency_ns(xbar),
        )
        clk_ns = 1000.0 / xbar.config.digital_frequency_mhz
        total_latency_ns = (
            n_digits * per_digit_latency_ns
            + digit_combines * 2.0 * clk_ns          # one shift + one add per combine
            + adder_metrics.latency_ns
        )

        tops_per_watt = (
            (2.0 * active_rows * n_active_cols) / (total_energy_j * 1e12)
            if total_energy_j > 0 and active_rows > 0
            else 0.0
        )

        result = IntegratedReadResult(
            input_codes=codes,
            input_voltages_v=v_applied,
            column_currents_a=I_col,
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

        if accumulate_energy:
            xbar.total_read_energy_j += crossbar_energy_j
            xbar.total_peripheral_energy_j += total_energy_j - crossbar_energy_j
        xbar.last_read_result = result
        return result

    # Internal: full MVM non-ideality pipeline

    def _compute_column_currents(
        self,
        v_applied: np.ndarray,
        G: np.ndarray,
        read_voltage_v: float,
        rows: int,
        read_time_s: float = 0.0,
    ) -> np.ndarray:
        """Apply all read-path non-idealities in physical order."""
        G = G.copy()

        # --- 3. Parasitic series + access-transistor resistance: G_eff = G / (1 + G * R_s) ---
        r_series = (
            self.config.parasitic_series_resistance_ohm
            + self.config.access_transistor_on_resistance_ohm
        )
        if r_series > 0.0:
            G = G / (1.0 + G * r_series)

        # 9. Self-heating: dissipation x thermal resistance.
        # dT = R_th_cell * P_cell  +  R_th_array * P_array, both linear in the
        # power actually dissipated, so a lightly driven array runs cool.
        c = self.config
        if c.temperature_coefficient_per_k != 0.0 and (
                c.thermal_resistance_cell_k_per_w > 0.0
                or c.thermal_resistance_array_k_per_w > 0.0):
            P_cell = (v_applied[:, None] ** 2) * G
            P_array = float(np.sum(P_cell))
            delta_T = (c.thermal_resistance_cell_k_per_w * P_cell
                       + c.thermal_resistance_array_k_per_w * P_array)
            if c.max_temperature_rise_k > 0.0:
                # A ceiling (package / runaway limit), applied AFTER the physics.
                delta_T = np.minimum(delta_T, c.max_temperature_rise_k)
            G = G * (1.0 + c.temperature_coefficient_per_k * delta_T)
            G = np.maximum(G, 1e-18)

        # --- 1. IR drop: effective voltage at each cell reduced by wire resistance ---
        if self.config.wire_resistance_ohm > 0.0:
            I_col = self._apply_ir_drop(v_applied, G)
        else:
            I_col = v_applied @ G

        # --- 8. Current saturation: column current ceiling ---
        if self.config.current_saturation_a is not None:
            I_col = np.minimum(I_col, self.config.current_saturation_a)

        # --- 5. Read noise: uncorrelated per-cell sources add as sqrt(rows) ---
        abs_sigma = np.sqrt(rows) * self.config.read_noise_current_a
        rel_sigma = self.config.read_noise_relative * np.abs(I_col)
        sigma = np.hypot(abs_sigma, rel_sigma)

        # 5b. Physical sense noise, integrated over the read window.  Both floors
        # fall as 1/sqrt(t), which is what makes a shorter read cost resolution.
        # `read_time_s` is derived from the ADC aperture and the column RC.
        if self.config.physical_sense_noise and read_time_s and read_time_s > 0.0:
            q = 1.602176634e-19
            kB = 1.380649e-23
            shot = np.sqrt(q * np.abs(I_col) / read_time_s)
            g_col = np.sum(G, axis=0) if G.ndim == 2 else np.asarray(G)
            thermal = np.sqrt(
                2.0 * kB * self.config.sense_temperature_k * np.abs(g_col) / read_time_s
            )
            sigma = np.sqrt(sigma ** 2 + shot ** 2 + thermal ** 2)

        if np.any(sigma > 0.0):
            I_col = I_col + self.rng.normal(0.0, sigma)

        # --- 10. Peripheral comparator offset: systematic bias on all outputs ---
        if self.config.comparator_offset_a != 0.0:
            I_col = I_col + self.config.comparator_offset_a

        return I_col

    def _apply_ir_drop(self, v_applied: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        First-order resistive-network solve of row and column wire IR drop.

        A cell's voltage loss is the sum over every wire segment between it and the
        driver, and each segment carries the current of every cell beyond it:

            I_seg[m] = sum_{k >= m} I[k]              (current still in segment m)
            V_drop[j] = r * sum_{m=1..j} I_seg[m]     (line integral to node j)

        a double cumulative sum whose inner one runs away from the driver.

        Row/word lines are driven from column 0; column/bit lines are sensed at row
        0, where the transimpedance front end holds a virtual ground.  Both are
        integrated from index 0 outward, and node 0 of each carries no drop.

        The drop depends on the currents and vice versa, so this iterates the fixed
        point (`ir_drop_iterations`); it converges in a few passes because the drop
        is a perturbation on the ideal bias.  The returned column current is what
        the final voltage map actually produces.
        """
        r = self.config.wire_resistance_ohm
        v_row = v_applied[:, None]

        def tail_sum(A, axis):
            """sum_{k >= m} A[k] along `axis`, without reversing the array twice."""
            return A.sum(axis=axis, keepdims=True) - np.cumsum(A, axis=axis) + A

        V_eff = np.broadcast_to(v_row, G.shape).astype(float, copy=True)
        # Converge on the COLUMN CURRENT, the observable, not the full voltage map:
        # nodes clamped at 0 V jitter in their last bits forever, so a node-wise
        # test never fires.  `IR_CONVERGENCE_RTOL` is three orders below a 4-bit
        # LSB, so exiting here cannot change a digitised result.
        I_prev = None
        for _ in range(max(1, int(self.config.ir_drop_iterations))):
            I = V_eff * G

            # --- row/word line, driven at column 0 -------------------------
            # seg_row[i, m] = current still flowing in the segment feeding node m.
            seg_row = tail_sum(I, 1)
            # Line integral to node j, excluding the driver node's own segment.
            drop_row = r * (np.cumsum(seg_row, axis=1) - seg_row[:, :1])
            V_row_eff = np.maximum(v_row - drop_row, 0.0)

            # --- column/bit line, sensed at row 0 --------------------------
            seg_col = tail_sum(V_row_eff * G, 0)
            drop_col = r * (np.cumsum(seg_col, axis=0) - seg_col[:1, :])

            V_eff = np.maximum(V_row_eff - drop_col, 0.0)
            I_col = np.sum(V_eff * G, axis=0)
            if I_prev is not None:
                scale = np.maximum(np.abs(I_prev), 1e-30)
                if float(np.max(np.abs(I_col - I_prev) / scale)) < self.IR_CONVERGENCE_RTOL:
                    return I_col
            I_prev = I_col

        return I_prev

    # Internal: capacitance helpers

    def _parasitic_cap_energy(self, xbar, v_applied: np.ndarray) -> float:
        """
        Energy (J) dissipated charging and discharging the wire capacitance per read.

        A full charge-then-discharge cycle dissipates C*V^2, not C*V^2/2: half the
        delivered energy burns in the driver, and the stored half burns on
        discharge.
        """
        C = self.config.parasitic_capacitance_f
        if C == 0.0:
            return 0.0
        active_rows = int(np.count_nonzero(v_applied))
        if active_rows == 0:
            return 0.0
        V_avg = float(np.mean(np.abs(v_applied[v_applied != 0.0])))
        C_row = C * xbar.cols   # one row wire spans `cols` cell pitches
        C_col = C * xbar.rows   # one col wire spans `rows` cell pitches
        return V_avg ** 2 * (C_row * active_rows + C_col * xbar.cols)

    def _parasitic_rc_latency_ns(self, xbar) -> float:
        """
        Settling time of the slower of the two wires, in nanoseconds.

        Elmore delay of a distributed RC line is R_total*C_total/2: each resistive
        segment sees only the capacitance downstream of it.  A row spans `cols`
        pitches and a column spans `rows`, so each gets its own dimension.
        """
        C = self.config.parasitic_capacitance_f
        r = self.config.wire_resistance_ohm
        if C == 0.0 or r == 0.0:
            return 0.0
        tau_row = 0.5 * (r * xbar.cols) * (C * xbar.cols)
        tau_col = 0.5 * (r * xbar.rows) * (C * xbar.rows)
        return 5.0 * max(tau_row, tau_col) * 1e9

    # Reporting

    def nonideal_stats(self, xbar, v_applied: np.ndarray) -> dict[str, float]:
        """
        Compare ideal vs non-ideal column currents for a given input voltage vector.

        Useful for understanding how much each non-ideality degrades the MVM result.
        """
        I_ideal = v_applied @ xbar.conductance_matrix_s
        I_ni = self._compute_column_currents(
            v_applied, xbar.conductance_matrix_s, xbar.config.read_voltage_v, xbar.rows,
            read_time_s=float(getattr(xbar.config, "read_time_s", 0.0) or 0.0),
        )
        mask = I_ideal > 0.0
        abs_err = np.abs(I_ni - I_ideal)
        rel_err = np.where(mask, abs_err / I_ideal, 0.0)

        return {
            "ideal_col_current_mean_a":    float(np.mean(I_ideal[mask])) if mask.any() else 0.0,
            "nonideal_col_current_mean_a": float(np.mean(I_ni)),
            "mean_abs_error_a":            float(np.mean(abs_err)),
            "max_abs_error_a":             float(np.max(abs_err)),
            "mean_relative_error":         float(np.mean(rel_err[mask])) if mask.any() else 0.0,
            "max_relative_error":          float(np.max(rel_err[mask])) if mask.any() else 0.0,
            "rc_latency_overhead_ns":      self._parasitic_rc_latency_ns(xbar),
        }
