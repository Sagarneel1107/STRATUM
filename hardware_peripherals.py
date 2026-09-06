"""
hardware_peripherals.py

Peripheral models integrated into the crossbar simulator.  The crossbar creates
and owns these blocks: ADC, DAC, Reg, ShiftReg, Adder.

Units: area um^2, power W, latency ns, energy_nj nJ.

Energy is the primitive, power is derived.  Dynamic energy per operation is a
property of the gate (~alpha*C*V^2), not of the schedule it runs at, so the
clocked blocks (`Reg`, `ShiftReg`, `Adder`) derive a fixed per-operation energy
from a power figure characterised at `REFERENCE_FREQUENCY_MHZ`:

    energy_per_op = P_ref / f_ref            (constant, set by the technology)
    power_w       = energy_per_op * f_cfg    (scales with the clock)
    latency_ns    = 1e9 / f_cfg

W * ns = nJ still holds for these blocks.

The converters (`ADC`, `DAC`) are not clocked by `frequency_mhz` and are not
purely dynamic.  Their tabulated `power_w` is the power drawn while retiring
`sample_rate_gsps` samples per second, which splits into a continuous bias term
and a per-conversion term:

    P_rated   = P_static + E_dynamic * f_rated
    P_static  = static_power_fraction * P_rated
    E_dynamic = (1 - static_power_fraction) * P_rated / f_rated
    E_conv(T) = P_static * T + E_dynamic      T = the conversion period in use

So the energy of one conversion depends on how often the converter is used: a
4-bit ADC characterised at 1 GS/s but issued once per 7.2 ns read idles, biased,
for 86% of every read window.  `retime(period_s)` sets T, and
`IntegratedMemristorCrossbar` calls it with the array's read window.  Left unset,
T defaults to the rated period 1/f_rated.

`latency_ns` stays the conversion (pipeline) latency and is used for timing only;
for these two blocks `energy_nj != power_w * latency_ns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Optional


@dataclass
class HardwareMetrics:
    area_um2: float = 0.0
    power_w: float = 0.0
    latency_ns: float = 0.0
    energy_nj: float = 0.0

    @property
    def energy_j(self) -> float:
        return self.energy_nj * 1e-9

    def scaled(self, count: int) -> "HardwareMetrics":
        if count < 0:
            raise ValueError("count must be non-negative")
        return HardwareMetrics(
            area_um2=self.area_um2 * count,
            power_w=self.power_w * count,
            latency_ns=self.latency_ns,
            energy_nj=self.energy_nj * count,
        )


class ADC:
    """
    Analog-to-digital converter.  Area is counted per output column, energy during
    every read; the output quantises the column currents or voltages.
    """

    #: Fraction of the tabulated operating power that is continuous analog bias
    #: (comparator preamps, reference ladder and buffers).  The remainder is
    #: switched per conversion.  Converters in this resolution/speed class are
    #: bias-dominated; 0.5 is a deliberately conservative midpoint.
    STATIC_POWER_FRACTION = 0.5

    def __init__(
        self,
        choice: int = 2,
        precision_bits: Optional[int] = None,
        area_um2: Optional[float] = None,
        power_w: Optional[float] = None,
        sample_rate_gsps: Optional[float] = None,
        conversion_period_s: Optional[float] = None,
        static_power_fraction: Optional[float] = None,
    ):
        self.choice = choice
        self.precision_bits = precision_bits
        self.area_um2 = area_um2
        self.power_w = power_w
        self.sample_rate_gsps = sample_rate_gsps
        self.static_power_fraction = (
            self.STATIC_POWER_FRACTION if static_power_fraction is None
            else float(static_power_fraction))
        if not 0.0 <= self.static_power_fraction <= 1.0:
            raise ValueError("static_power_fraction must be in [0, 1]")
        #: Conversion period actually in use (s); None => the rated period.
        self.conversion_period_s = conversion_period_s

        self.latency_ns = 0.0
        self.energy_nj = 0.0
        self.interval_thresholds: list[float] = []

        self.calculate_area()
        self.calculate_precision()
        self.calculate_power()
        self.calculate_sample_rate()
        self.calculate_latency()
        self.calculate_energy()

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            table = {
                1: 1600.0,
                2: 1200.0,
                3: 1650.0,
                4: 580.0,
                5: 1650.0,
                6: 1650.0,
                7: 500.0,
                8: 1.0,
                9: 15899.0,
            }
            self.area_um2 = table[self.choice]
        return self.area_um2

    def calculate_precision(self) -> int:
        if self.precision_bits is None:
            table = {
                1: 10,
                2: 8,
                3: 8,
                4: 6,
                5: 8,
                6: 6,
                7: 4,
                8: 1,
                9: 8,
            }
            self.precision_bits = table[self.choice]
        return self.precision_bits

    def calculate_power(self) -> float:
        if self.power_w is None:
            table = {
                1: 6.92e-3,
                2: 2.00e-3,
                3: 4.00e-3,
                4: 1.26e-3,
                5: 4.00e-3,
                6: 1.26e-3,
                7: 0.70e-3,
                8: 0.1086 * 15e-6,
                9: 8 * 0.0073e-3,
            }
            self.power_w = table[self.choice]
        return self.power_w

    def calculate_sample_rate(self) -> float:
        if self.sample_rate_gsps is None:
            table = {
                1: 1.5,
                2: 1.28,
                3: 1.1,
                4: 1.0,
                5: 1.1,
                6: 1.0,
                7: 1.0,
                8: 1.0,
                9: 6.0,
            }
            self.sample_rate_gsps = table[self.choice]
        return self.sample_rate_gsps

    def calculate_latency(self) -> float:
        if self.precision_bits == 1:
            self.latency_ns = 1.0 / self.sample_rate_gsps
        elif self.choice == 9:
            self.latency_ns = (1.0 / self.sample_rate_gsps) * (2 ** self.precision_bits)
        else:
            self.latency_ns = (1.0 / self.sample_rate_gsps) * (self.precision_bits + 2)
        return self.latency_ns

    def calculate_energy(self) -> float:
        """
        Energy per conversion at the period the converter is actually operated at.

            E_conv = P_static * T + E_dynamic

        `T` is `conversion_period_s` when the array has told the converter its read
        window (`retime`), otherwise the rated period 1/f_s.  The (bits + 2) in
        `calculate_latency` is the pipeline depth: it governs when a result appears,
        not how long the front end is occupied, so it is not used here.
        """
        f_rated_hz = self.sample_rate_gsps * 1e9
        t_rated_s = 1.0 / f_rated_hz
        p_static_w = self.static_power_fraction * self.power_w
        e_dynamic_j = (1.0 - self.static_power_fraction) * self.power_w * t_rated_s

        # A converter cannot be issued faster than its rated period.
        period_s = t_rated_s if self.conversion_period_s is None else max(
            float(self.conversion_period_s), t_rated_s)

        self.energy_nj = (p_static_w * period_s + e_dynamic_j) * 1e9
        return self.energy_nj

    def retime(self, conversion_period_s: Optional[float]) -> float:
        """
        Tell the converter how often it is actually issued, and re-derive its
        per-conversion energy.  `None` restores the rated period.
        """
        self.conversion_period_s = (None if conversion_period_s is None
                                    else float(conversion_period_s))
        return self.calculate_energy()

    def configure_intervals(self, v_min: float, v_max: float) -> list[float]:
        """
        Uniform quantization thresholds between v_min and v_max.
        """
        levels = 2 ** self.precision_bits
        if levels <= 1:
            self.interval_thresholds = [(v_min + v_max) / 2.0]
            return self.interval_thresholds

        step = (v_max - v_min) / levels
        self.interval_thresholds = [v_min + step * (i + 1) for i in range(levels - 1)]
        return self.interval_thresholds

    def quantize(self, values):
        """
        Quantize scalar or numpy array using configured thresholds.
        """
        import numpy as np

        if not self.interval_thresholds:
            raise RuntimeError("ADC intervals must be configured before quantization")

        arr = np.asarray(values, dtype=float)
        return np.digitize(arr, self.interval_thresholds)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)


class DAC:
    """
    Digital-to-analog converter.  Area is counted per input row, energy during
    every read; digital input codes become row voltages.
    """

    #: Row drivers are more bias-dominated than the ADC: a current-steering or
    #: resistive-ladder DAC HOLDS its output level for the whole analog read
    #: window, so its tail currents flow for the entire period and only the code
    #: transition is switched energy.
    STATIC_POWER_FRACTION = 0.8

    def __init__(
        self,
        choice: int = 1,
        precision_bits: Optional[int] = None,
        area_um2: Optional[float] = None,
        power_w: Optional[float] = None,
        sample_rate_gsps: Optional[float] = None,
        conversion_period_s: Optional[float] = None,
        static_power_fraction: Optional[float] = None,
    ):
        self.choice = choice
        self.precision_bits = precision_bits
        self.area_um2 = area_um2
        self.power_w = power_w
        self.sample_rate_gsps = sample_rate_gsps
        self.static_power_fraction = (
            self.STATIC_POWER_FRACTION if static_power_fraction is None
            else float(static_power_fraction))
        if not 0.0 <= self.static_power_fraction <= 1.0:
            raise ValueError("static_power_fraction must be in [0, 1]")
        #: Drive period actually in use (s); None => the rated period.
        self.conversion_period_s = conversion_period_s

        self.latency_ns = 0.0
        self.energy_nj = 0.0

        self.calculate_area()
        self.calculate_precision()
        self.calculate_power()
        self.calculate_sample_rate()
        self.calculate_latency()
        self.calculate_energy()

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            table = {
                1: 0.166,
                2: 0.332,
                3: 0.664,
                4: 1.328,
                5: 5.312,
                6: 21.248,
                7: 0.166,
            }
            self.area_um2 = table[self.choice]
        return self.area_um2

    def calculate_precision(self) -> int:
        if self.precision_bits is None:
            table = {
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 6,
                6: 8,
                7: 1,
            }
            self.precision_bits = table[self.choice]
        return self.precision_bits

    def calculate_power(self) -> float:
        if self.power_w is None:
            table = {
                1: 0.0039e-3,
                2: 0.0078e-3,
                3: 0.0156e-3,
                4: 0.0312e-3,
                5: 0.1248e-3,
                6: 0.4992e-3,
                7: 0.0039e-3,
            }
            self.power_w = table[self.choice]
        return self.power_w

    def calculate_sample_rate(self) -> float:
        if self.sample_rate_gsps is None:
            self.sample_rate_gsps = 1.0
        return self.sample_rate_gsps

    def calculate_latency(self) -> float:
        self.latency_ns = (self.precision_bits + 2) / self.sample_rate_gsps
        return self.latency_ns

    def calculate_energy(self) -> float:
        """
        Energy per row drive, split static/dynamic exactly as `ADC.calculate_energy`.

        A row driver does not retire a sample and switch off - it holds the row at
        its code for the whole analog integration window.
        """
        f_rated_hz = self.sample_rate_gsps * 1e9
        t_rated_s = 1.0 / f_rated_hz
        p_static_w = self.static_power_fraction * self.power_w
        e_dynamic_j = (1.0 - self.static_power_fraction) * self.power_w * t_rated_s

        period_s = t_rated_s if self.conversion_period_s is None else max(
            float(self.conversion_period_s), t_rated_s)

        self.energy_nj = (p_static_w * period_s + e_dynamic_j) * 1e9
        return self.energy_nj

    def retime(self, conversion_period_s: Optional[float]) -> float:
        """Set the drive period actually in use; `None` restores the rated period."""
        self.conversion_period_s = (None if conversion_period_s is None
                                    else float(conversion_period_s))
        return self.calculate_energy()

    def code_to_voltage(self, code: int, v_min: float, v_max: float) -> float:
        max_code = 2 ** self.precision_bits - 1
        if not 0 <= code <= max_code:
            raise ValueError(f"DAC code must be in [0, {max_code}]")
        if max_code == 0:
            return v_min
        return v_min + (v_max - v_min) * code / max_code

    def codes_to_voltages(self, codes, v_min: float, v_max: float):
        import numpy as np

        arr = np.asarray(codes, dtype=int)
        max_code = 2 ** self.precision_bits - 1
        if np.any(arr < 0) or np.any(arr > max_code):
            raise ValueError(f"DAC codes must be in [0, {max_code}]")
        if max_code == 0:
            return np.full_like(arr, v_min, dtype=float)
        return v_min + (v_max - v_min) * arr / max_code

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)


class Reg:
    """
    Register/latch model used for input and output storage.

    Per-operation energy is fixed by the technology; `power_w` is what it costs at
    the configured clock.  See the module docstring.
    """

    #: Clock the tabulated power figures were characterised at.
    REFERENCE_FREQUENCY_MHZ = 100.0

    def __init__(
        self,
        bitwidth: int = 8,
        tech_node_nm: int = 65,
        frequency_mhz: float = 100.0,
        area_um2: Optional[float] = None,
        power_w: Optional[float] = None,
    ):
        if bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")

        self.bitwidth = bitwidth
        self.tech_node_nm = tech_node_nm
        self.frequency_mhz = frequency_mhz
        self.area_um2 = area_um2
        self.power_w = power_w
        self.latency_ns = 0.0
        self.energy_nj = 0.0
        # Fixed per-operation dynamic energy; set by calculate_power().
        self.energy_per_op_j = 0.0

        self.calculate_area()
        self.calculate_power()
        self.calculate_latency()
        self.calculate_energy()

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            area_per_bit_65nm = 1.4256 / 8.0
            scale = (self.tech_node_nm / 65.0) ** 2
            self.area_um2 = area_per_bit_65nm * self.bitwidth * scale
        return self.area_um2

    def calculate_power(self) -> float:
        if self.power_w is None:
            power_per_8bit_65nm = 18.8e-9        # at REFERENCE_FREQUENCY_MHZ
            scale = (self.tech_node_nm / 65.0) ** 2
            reference_power_w = power_per_8bit_65nm * (self.bitwidth / 8.0) * scale
            self.energy_per_op_j = reference_power_w / (self.REFERENCE_FREQUENCY_MHZ * 1e6)
            self.power_w = self.energy_per_op_j * (self.frequency_mhz * 1e6)
        else:
            # An explicit override is read as the power AT the configured clock.
            self.energy_per_op_j = self.power_w / (self.frequency_mhz * 1e6)
        return self.power_w

    def calculate_latency(self) -> float:
        self.latency_ns = 1000.0 / self.frequency_mhz
        return self.latency_ns

    def calculate_energy(self) -> float:
        self.energy_nj = self.energy_per_op_j * 1e9
        return self.energy_nj

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)


class STTMRAMSignCell:
    """
    Spin-transfer-torque MRAM (STT-MRAM) sign-storage cell: the spintronic
    alternative to the volatile CMOS `Reg` sign bit of the 2T1R scheme.

    One magnetic tunnel junction over one selector transistor (1T1MTJ).  The MTJ
    magnetisation - parallel (R_P) or anti-parallel (R_AP = (1+TMR)*R_P) -
    non-volatilely encodes the sign and statically biases the steering transistors,
    so there is no per-cycle latch read and no standby power.

    * AREA   : ~36 F^2 (vs ~140 F^2 for a 6T SRAM bit).
    * ENERGY : per-read sense ~1-2 fJ/bit (V_read^2 / R_avg * t_read), a one-time
               STT write of ~0.3-1.2 pJ/bit, and ~0 standby power.
    * TOPS   : the MTJ read (~1-1.5 ns) is off the analog-read critical path, so
               throughput is unchanged.

    Defaults from embedded 28-65 nm perpendicular-MTJ STT-MRAM macros.  `energy_nj`
    is the per-read sign-access energy, so the object drops into the sign path in
    place of a 1-bit `Reg`.
    """

    nonvolatile: bool = True

    def __init__(
        self,
        tech_node_nm: int = 65,
        frequency_mhz: float = 1000.0,
        cell_area_f2: float = 36.0,            # 1T1MTJ footprint (F = feature size)
        r_parallel_ohm: float = 6.0e3,         # MTJ low-resistance (parallel) state
        tmr_ratio: float = 1.2,                # (R_AP - R_P)/R_P ~ 120%
        read_voltage_v: float = 0.10,          # kept << switching to avoid disturb
        read_time_ns: Optional[float] = None,  # default: one digital clock period
        write_current_a: float = 70e-6,        # STT switching current
        write_voltage_v: float = 0.70,         # write-driver voltage
        write_time_ns: float = 10.0,           # switching pulse width
        standby_power_w: float = 0.0,          # non-volatile => no hold leakage
        area_um2: Optional[float] = None,
    ):
        if tech_node_nm <= 0:
            raise ValueError("tech_node_nm must be positive")
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")

        self.tech_node_nm = tech_node_nm
        self.frequency_mhz = frequency_mhz
        self.cell_area_f2 = cell_area_f2
        self.r_parallel_ohm = r_parallel_ohm
        self.tmr_ratio = tmr_ratio
        self.read_voltage_v = read_voltage_v
        self.read_time_ns = (1000.0 / frequency_mhz) if read_time_ns is None else read_time_ns
        self.write_current_a = write_current_a
        self.write_voltage_v = write_voltage_v
        self.write_time_ns = write_time_ns
        self.standby_power_w = standby_power_w
        self.area_um2 = area_um2

        self.power_w = 0.0
        self.latency_ns = self.read_time_ns
        self.read_energy_nj = 0.0
        self.write_energy_nj = 0.0
        self.energy_nj = 0.0

        self.calculate_area()
        self.calculate_read_energy()
        self.calculate_write_energy()

    @property
    def r_antiparallel_ohm(self) -> float:
        return self.r_parallel_ohm * (1.0 + self.tmr_ratio)

    @property
    def r_average_ohm(self) -> float:
        return 0.5 * (self.r_parallel_ohm + self.r_antiparallel_ohm)

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            # F is the feature size; 1 nm = 1e-3 um, so area = cell_F2 * (node_um)^2.
            feature_um = self.tech_node_nm * 1e-3
            self.area_um2 = self.cell_area_f2 * (feature_um ** 2)
        return self.area_um2

    def calculate_read_energy(self) -> float:
        # Resistive sense of the MTJ over one read window: V_read^2 / R_avg * t_read.
        self.power_w = (self.read_voltage_v ** 2) / self.r_average_ohm
        self.read_energy_nj = self.power_w * self.read_time_ns
        self.energy_nj = self.read_energy_nj
        return self.read_energy_nj

    def calculate_write_energy(self) -> float:
        # One-time STT switching energy to program the sign: V_write * I_write * t_write.
        self.write_energy_nj = self.write_voltage_v * self.write_current_a * self.write_time_ns
        return self.write_energy_nj

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)


class ShiftReg:
    """
    Shift register for bit-serial input feeding and precision expansion.

    Per-shift energy is fixed by the technology; `power_w` is what it costs at the
    configured clock.  See the module docstring.
    """

    #: Clock the tabulated power figures were characterised at.
    REFERENCE_FREQUENCY_MHZ = 100.0

    def __init__(
        self,
        bitwidth: int = 8,
        tech_node_nm: int = 65,
        frequency_mhz: float = 100.0,
        area_um2: Optional[float] = None,
        power_w: Optional[float] = None,
    ):
        if bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")

        self.bitwidth = bitwidth
        self.tech_node_nm = tech_node_nm
        self.frequency_mhz = frequency_mhz
        self.area_um2 = area_um2
        self.power_w = power_w
        self.latency_ns = 0.0
        self.energy_nj = 0.0
        # Fixed per-operation dynamic energy; set by calculate_power().
        self.energy_per_op_j = 0.0

        self.calculate_area()
        self.calculate_power()
        self.calculate_latency(shift_cycles=1)
        self.calculate_energy()

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            area_per_bit_65nm = 1.4256 / 8.0
            scale = (self.tech_node_nm / 65.0) ** 2
            control_overhead = 1.20
            self.area_um2 = area_per_bit_65nm * self.bitwidth * scale * control_overhead
        return self.area_um2

    def calculate_power(self) -> float:
        if self.power_w is None:
            power_per_8bit_65nm = 18.8e-9        # at REFERENCE_FREQUENCY_MHZ
            scale = (self.tech_node_nm / 65.0) ** 2
            switching_overhead = 1.50
            reference_power_w = (
                power_per_8bit_65nm * (self.bitwidth / 8.0) * scale * switching_overhead
            )
            self.energy_per_op_j = reference_power_w / (self.REFERENCE_FREQUENCY_MHZ * 1e6)
            self.power_w = self.energy_per_op_j * (self.frequency_mhz * 1e6)
        else:
            # An explicit override is read as the power AT the configured clock.
            self.energy_per_op_j = self.power_w / (self.frequency_mhz * 1e6)
        return self.power_w

    def calculate_latency(self, shift_cycles: int = 1) -> float:
        if shift_cycles <= 0:
            raise ValueError("shift_cycles must be positive")
        self.latency_ns = shift_cycles * (1000.0 / self.frequency_mhz)
        return self.latency_ns

    def calculate_energy(self) -> float:
        self.energy_nj = self.energy_per_op_j * 1e9
        return self.energy_nj

    def operation_metrics(self, shift_cycles: int) -> HardwareMetrics:
        latency_ns = shift_cycles * (1000.0 / self.frequency_mhz)
        # One shift cycle costs one fixed quantum of energy, whatever the clock.
        energy_nj = shift_cycles * self.energy_per_op_j * 1e9
        return HardwareMetrics(self.area_um2, self.power_w, latency_ns, energy_nj)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)


class Adder:
    """
    Digital adder for partial-sum accumulation: split crossbar outputs,
    positive/negative differential crossbars, and bit-sliced results.

    Per-addition energy is fixed by the technology; `power_w` is what it costs at
    the configured clock.  See the module docstring.
    """

    #: Clock the tabulated power figures were characterised at.
    REFERENCE_FREQUENCY_MHZ = 100.0

    def __init__(
        self,
        bitwidth: int = 16,
        tech_node_nm: int = 65,
        frequency_mhz: float = 100.0,
        area_um2: Optional[float] = None,
        power_w: Optional[float] = None,
    ):
        if bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")

        self.bitwidth = bitwidth
        self.tech_node_nm = tech_node_nm
        self.frequency_mhz = frequency_mhz
        self.area_um2 = area_um2
        self.power_w = power_w
        self.latency_ns = 0.0
        self.energy_nj = 0.0
        # Fixed per-operation dynamic energy; set by calculate_power().
        self.energy_per_op_j = 0.0

        self.calculate_area()
        self.calculate_power()
        self.calculate_latency(addition_cycles=1)
        self.calculate_energy()

    def _closest_tech_node(self, table: dict[int, float]) -> int:
        nodes = sorted(table.keys())
        for node in nodes:
            if self.tech_node_nm <= node:
                return node
        return nodes[-1]

    def calculate_area(self) -> float:
        if self.area_um2 is None:
            table = {
                130: 10 * 14 * 130 * 130 / 1e6,
                65: 1.42,
                55: 10 * 14 * 55 * 55 / 1e6,
                45: 10 * 14 * 45 * 45 / 1e6,
                28: 10 * 14 * 28 * 28 / 1e6,
            }
            node = self._closest_tech_node(table)
            self.area_um2 = table[node] * self.bitwidth
        return self.area_um2

    def calculate_power(self) -> float:
        if self.power_w is None:
            table = {
                130: 2.5e-9,
                65: 3.0e-7,
                55: 2.5e-9,
                45: 2.5e-9,
                28: 2.5e-9,
            }
            node = self._closest_tech_node(table)
            reference_power_w = table[node] * self.bitwidth      # at REFERENCE_FREQUENCY_MHZ
            self.energy_per_op_j = reference_power_w / (self.REFERENCE_FREQUENCY_MHZ * 1e6)
            self.power_w = self.energy_per_op_j * (self.frequency_mhz * 1e6)
        else:
            # An explicit override is read as the power AT the configured clock.
            self.energy_per_op_j = self.power_w / (self.frequency_mhz * 1e6)
        return self.power_w

    def calculate_latency(self, addition_cycles: int = 1) -> float:
        if addition_cycles <= 0:
            raise ValueError("addition_cycles must be positive")
        self.latency_ns = addition_cycles * (1000.0 / self.frequency_mhz)
        return self.latency_ns

    def calculate_energy(self) -> float:
        self.energy_nj = self.energy_per_op_j * 1e9
        return self.energy_nj

    def accumulation_metrics(self, number_of_terms: int, use_tree: bool = True) -> HardwareMetrics:
        if number_of_terms <= 1:
            return HardwareMetrics(self.area_um2, self.power_w, 0.0, 0.0)

        cycles = ceil(log2(number_of_terms)) if use_tree else number_of_terms - 1
        latency_ns = cycles * (1000.0 / self.frequency_mhz)
        # One addition cycle costs one fixed quantum of energy, whatever the clock.
        energy_nj = cycles * self.energy_per_op_j * 1e9
        return HardwareMetrics(self.area_um2, self.power_w, latency_ns, energy_nj)

    def metrics(self) -> HardwareMetrics:
        return HardwareMetrics(self.area_um2, self.power_w, self.latency_ns, self.energy_nj)
