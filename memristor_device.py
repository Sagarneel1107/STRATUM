"""
memristor_device.py

Single pulse-programmed memristor model.

The stored data is the observed conductance level after electrical pulse
programming. Energies are calculated using:

    P = V^2 * G
    E = V^2 * G * t

All internal energy values are returned in Joules unless the method name
explicitly says nJ.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Union
import csv
import json
import numpy as np


Number = Union[int, float]


@dataclass
class PulseParameters:
    pulse_width_s: float = 100e-9
    pulse_interval_s: float = 1e-6
    write_amplitude_v: float = 1.0
    read_voltage_v: float = 0.4
    read_time_s: float = 50e-9


@dataclass
class DeviceState:
    name: str
    current_conductance_s: float
    current_resistance_ohm: float
    conductance_levels_s: list[float]
    resistance_levels_ohm: list[float]
    native_precision_bits: int
    device_area_um2: float
    pulse_parameters: dict


class MemristorDevice:
    """
    Represents one memristive cell with multiple measurable conductance states.

    Parameters
    ----------
    conductance_levels_s:
        Measured/stored conductance states in Siemens.
    pulse_width_s:
        Write pulse width in seconds.
    pulse_interval_s:
        Time gap between programming pulses in seconds.
    write_amplitude_v:
        Write pulse amplitude in volts.
    read_voltage_v:
        Read voltage in volts.
    read_time_s:
        Read duration in seconds.
    device_area_um2:
        Cell area in square micrometers.
    """

    def __init__(
        self,
        conductance_levels_s: Optional[Sequence[Number]] = None,
        pulse_width_s: float = 100e-9,
        pulse_interval_s: float = 1e-6,
        write_amplitude_v: float = 1.0,
        read_voltage_v: float = 0.1,
        read_time_s: float = 50e-9,
        device_area_um2: float = 0.0,
        initial_level_index: int = 0,
        name: str = "memristor",
    ):
        self.name = name
        self.device_area_um2 = float(device_area_um2)

        self.pulse = PulseParameters(
            pulse_width_s=float(pulse_width_s),
            pulse_interval_s=float(pulse_interval_s),
            write_amplitude_v=float(write_amplitude_v),
            read_voltage_v=float(read_voltage_v),
            read_time_s=float(read_time_s),
        )

        self.conductance_levels_s: list[float] = []
        self.current_conductance_s: float = 0.0

        # Each item is one observed conductance after one write pulse.
        self.conductance_trace_s: list[float] = []
        self.pulse_amplitude_trace_v: list[float] = []
        self.pulse_width_trace_s: list[float] = []
        self.pulse_interval_trace_s: list[float] = []
        self.read_time_trace_s: list[float] = []

        if conductance_levels_s is not None:
            self.store_conductance_levels(conductance_levels_s)
            self.set_state_by_index(initial_level_index)

    # State handling

    @property
    def current_resistance_ohm(self) -> float:
        if self.current_conductance_s <= 0:
            return float("inf")
        return 1.0 / self.current_conductance_s

    @property
    def resistance_levels_ohm(self) -> list[float]:
        return [1.0 / g for g in self.conductance_levels_s]

    @property
    def number_of_states(self) -> int:
        return len(self.conductance_levels_s)

    @property
    def native_precision_bits(self) -> int:
        if self.number_of_states <= 1:
            return 0
        return int(np.floor(np.log2(self.number_of_states)))

    def store_conductance_levels(
        self,
        conductance_levels_s: Sequence[Number],
        sort_levels: bool = True,
        remove_duplicates: bool = True,
    ) -> None:
        levels = np.asarray(conductance_levels_s, dtype=float)

        if levels.ndim != 1:
            raise ValueError("conductance_levels_s must be a 1D sequence")

        if len(levels) == 0:
            raise ValueError("at least one conductance level is required")

        if np.any(levels <= 0):
            raise ValueError("all conductance levels must be positive Siemens values")

        if remove_duplicates:
            levels = np.unique(levels)

        if sort_levels:
            levels = np.sort(levels)

        self.conductance_levels_s = levels.tolist()

        if self.current_conductance_s <= 0:
            self.current_conductance_s = self.conductance_levels_s[0]

    def set_state_by_index(self, level_index: int) -> float:
        if not self.conductance_levels_s:
            raise RuntimeError("No conductance levels have been stored")

        if not 0 <= level_index < len(self.conductance_levels_s):
            raise IndexError("level_index is out of range")

        self.current_conductance_s = self.conductance_levels_s[level_index]
        return self.current_conductance_s

    def nearest_level_index(self, conductance_s: Number) -> int:
        if not self.conductance_levels_s:
            raise RuntimeError("No conductance levels have been stored")

        levels = np.asarray(self.conductance_levels_s, dtype=float)
        return int(np.argmin(np.abs(levels - float(conductance_s))))

    def set_state_by_conductance(
        self,
        conductance_s: Number,
        snap_to_nearest: bool = True,
    ) -> float:
        g = float(conductance_s)

        if g <= 0:
            raise ValueError("conductance must be positive")

        if snap_to_nearest:
            self.current_conductance_s = self.conductance_levels_s[self.nearest_level_index(g)]
        else:
            self.current_conductance_s = g

        return self.current_conductance_s

    def conductance_to_code(self, conductance_s: Number) -> int:
        return self.nearest_level_index(conductance_s)

    def code_to_conductance(self, code: int) -> float:
        if not 0 <= code < len(self.conductance_levels_s):
            raise IndexError("conductance code is out of range")
        return self.conductance_levels_s[code]

    # Pulse trace handling

    def reset_trace(self) -> None:
        self.conductance_trace_s.clear()
        self.pulse_amplitude_trace_v.clear()
        self.pulse_width_trace_s.clear()
        self.pulse_interval_trace_s.clear()
        self.read_time_trace_s.clear()

    def record_pulse_observation(
        self,
        conductance_s: Number,
        pulse_amplitude_v: Optional[Number] = None,
        pulse_width_s: Optional[Number] = None,
        pulse_interval_s: Optional[Number] = None,
        read_time_s: Optional[Number] = None,
        update_state: bool = True,
    ) -> None:
        """
        Record measured conductance after one pulse.
        """
        g = float(conductance_s)

        if g <= 0:
            raise ValueError("observed conductance must be positive")

        amp = self.pulse.write_amplitude_v if pulse_amplitude_v is None else float(pulse_amplitude_v)
        width = self.pulse.pulse_width_s if pulse_width_s is None else float(pulse_width_s)
        interval = self.pulse.pulse_interval_s if pulse_interval_s is None else float(pulse_interval_s)
        read_time = self.pulse.read_time_s if read_time_s is None else float(read_time_s)

        if width < 0 or interval < 0 or read_time < 0:
            raise ValueError("times must be non-negative")

        self.conductance_trace_s.append(g)
        self.pulse_amplitude_trace_v.append(amp)
        self.pulse_width_trace_s.append(width)
        self.pulse_interval_trace_s.append(interval)
        self.read_time_trace_s.append(read_time)

        if update_state:
            self.current_conductance_s = g

    def program_with_observed_trace(
        self,
        conductance_trace_s: Sequence[Number],
        pulse_amplitudes_v: Optional[Sequence[Number]] = None,
        pulse_widths_s: Optional[Sequence[Number]] = None,
        pulse_intervals_s: Optional[Sequence[Number]] = None,
        read_times_s: Optional[Sequence[Number]] = None,
        append: bool = False,
    ) -> None:
        """
        Store a full measured conductance-after-pulse trace.
        """
        if not append:
            self.reset_trace()

        conductances = np.asarray(conductance_trace_s, dtype=float)

        if conductances.ndim != 1:
            raise ValueError("conductance_trace_s must be 1D")

        n = len(conductances)

        def expand(values: Optional[Sequence[Number]], default: float, name: str) -> np.ndarray:
            if values is None:
                return np.full(n, default, dtype=float)
            arr = np.asarray(values, dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have length {n}")
            return arr

        amps = expand(pulse_amplitudes_v, self.pulse.write_amplitude_v, "pulse_amplitudes_v")
        widths = expand(pulse_widths_s, self.pulse.pulse_width_s, "pulse_widths_s")
        intervals = expand(pulse_intervals_s, self.pulse.pulse_interval_s, "pulse_intervals_s")
        read_times = expand(read_times_s, self.pulse.read_time_s, "read_times_s")

        for g, amp, width, interval, read_time in zip(conductances, amps, widths, intervals, read_times):
            self.record_pulse_observation(
                conductance_s=g,
                pulse_amplitude_v=amp,
                pulse_width_s=width,
                pulse_interval_s=interval,
                read_time_s=read_time,
                update_state=True,
            )

    # Energy models

    @staticmethod
    def joule_to_nj(energy_j: Number) -> float:
        return float(energy_j) * 1e9

    @staticmethod
    def nj_to_joule(energy_nj: Number) -> float:
        return float(energy_nj) * 1e-9

    def read_current_a(
        self,
        conductance_s: Optional[Number] = None,
        read_voltage_v: Optional[Number] = None,
    ) -> float:
        g = self.current_conductance_s if conductance_s is None else float(conductance_s)
        v = self.pulse.read_voltage_v if read_voltage_v is None else float(read_voltage_v)
        return v * g

    def read_energy_j(
        self,
        conductance_s: Optional[Number] = None,
        read_voltage_v: Optional[Number] = None,
        read_time_s: Optional[Number] = None,
    ) -> float:
        """
        E_read = V_read^2 * G * t_read
        """
        g = self.current_conductance_s if conductance_s is None else float(conductance_s)
        v = self.pulse.read_voltage_v if read_voltage_v is None else float(read_voltage_v)
        t = self.pulse.read_time_s if read_time_s is None else float(read_time_s)

        if g <= 0:
            raise ValueError("conductance must be positive")
        if t < 0:
            raise ValueError("read_time_s must be non-negative")

        return (v ** 2) * g * t

    def write_energy_j(
        self,
        conductance_s: Optional[Number] = None,
        write_amplitude_v: Optional[Number] = None,
        pulse_width_s: Optional[Number] = None,
    ) -> float:
        """
        E_write = V_write^2 * G_observed_after_pulse * pulse_width
        """
        g = self.current_conductance_s if conductance_s is None else float(conductance_s)
        v = self.pulse.write_amplitude_v if write_amplitude_v is None else float(write_amplitude_v)
        t = self.pulse.pulse_width_s if pulse_width_s is None else float(pulse_width_s)

        if g <= 0:
            raise ValueError("conductance must be positive")
        if t < 0:
            raise ValueError("pulse_width_s must be non-negative")

        return (v ** 2) * g * t

    def read_energy_from_trace_j(
        self,
        conductance_trace_s: Optional[Sequence[Number]] = None,
        read_times_s: Optional[Sequence[Number]] = None,
        read_voltage_v: Optional[Number] = None,
        upto: Optional[int] = None,
    ) -> float:
        g = np.asarray(
            self.conductance_trace_s if conductance_trace_s is None else conductance_trace_s,
            dtype=float,
        )

        if upto is not None:
            g = g[:upto]

        if g.size == 0:
            return 0.0
        if np.any(g <= 0):
            raise ValueError("all conductances must be positive")

        if read_times_s is not None:
            t = np.asarray(read_times_s, dtype=float)
        elif len(self.read_time_trace_s) >= len(g):
            t = np.asarray(self.read_time_trace_s[:len(g)], dtype=float)
        else:
            t = np.full(len(g), self.pulse.read_time_s, dtype=float)

        if len(t) != len(g):
            raise ValueError("read_times_s must match conductance trace length")
        if np.any(t < 0):
            raise ValueError("read times must be non-negative")

        v = self.pulse.read_voltage_v if read_voltage_v is None else float(read_voltage_v)
        return float((v ** 2) * np.dot(g, t))

    def read_energy_from_time_trace_j(
        self,
        conductance_trace_s: Sequence[Number],
        times_s: Sequence[Number],
        read_voltage_v: Optional[Number] = None,
    ) -> float:
        """
        Prototype-style integrated read energy:
            E = V_read^2 * sum(G_i * dt_i)
        """
        g = np.asarray(conductance_trace_s, dtype=float)
        times = np.asarray(times_s, dtype=float)

        if g.shape != times.shape:
            raise ValueError("conductance_trace_s and times_s must have the same shape")
        if g.ndim != 1:
            raise ValueError("conductance_trace_s and times_s must be 1D")
        if np.any(g <= 0):
            raise ValueError("all conductances must be positive")
        if np.any(np.diff(times) < 0):
            raise ValueError("times_s must be monotonically increasing")

        dt = np.diff(times, prepend=0.0)
        v = self.pulse.read_voltage_v if read_voltage_v is None else float(read_voltage_v)
        return float((v ** 2) * np.dot(g, dt))

    def write_energy_from_trace_j(
        self,
        conductance_trace_s: Optional[Sequence[Number]] = None,
        pulse_amplitudes_v: Optional[Sequence[Number]] = None,
        pulse_widths_s: Optional[Sequence[Number]] = None,
        upto: Optional[int] = None,
    ) -> float:
        """
        E_write_trace = sum_i V_i^2 * G_i * pulse_width_i
        """
        g = np.asarray(
            self.conductance_trace_s if conductance_trace_s is None else conductance_trace_s,
            dtype=float,
        )

        if upto is not None:
            g = g[:upto]

        if g.size == 0:
            return 0.0
        if np.any(g <= 0):
            raise ValueError("all conductances must be positive")

        n = len(g)

        if pulse_amplitudes_v is not None:
            amp = np.asarray(pulse_amplitudes_v, dtype=float)
        elif len(self.pulse_amplitude_trace_v) >= n:
            amp = np.asarray(self.pulse_amplitude_trace_v[:n], dtype=float)
        else:
            amp = np.full(n, self.pulse.write_amplitude_v, dtype=float)

        if pulse_widths_s is not None:
            width = np.asarray(pulse_widths_s, dtype=float)
        elif len(self.pulse_width_trace_s) >= n:
            width = np.asarray(self.pulse_width_trace_s[:n], dtype=float)
        else:
            width = np.full(n, self.pulse.pulse_width_s, dtype=float)

        if len(amp) != n or len(width) != n:
            raise ValueError("pulse arrays must match conductance trace length")
        if np.any(width < 0):
            raise ValueError("pulse widths must be non-negative")

        return float(np.dot((amp ** 2) * width, g))

    def total_programming_energy_j(self) -> float:
        return self.write_energy_from_trace_j() + self.read_energy_from_trace_j()

    # File IO

    def save_conductance_levels(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["state_index", "conductance_s", "resistance_ohm"])
            for i, g in enumerate(self.conductance_levels_s):
                writer.writerow([i, g, 1.0 / g])

    def load_conductance_levels(self, path: str | Path) -> None:
        path = Path(path)

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "conductance_s" in reader.fieldnames:
                levels = [float(row["conductance_s"]) for row in reader]
            else:
                f.seek(0)
                raw = csv.reader(f)
                levels = [float(row[0]) for row in raw if row]

        self.store_conductance_levels(levels)

    def state(self) -> DeviceState:
        return DeviceState(
            name=self.name,
            current_conductance_s=self.current_conductance_s,
            current_resistance_ohm=self.current_resistance_ohm,
            conductance_levels_s=list(self.conductance_levels_s),
            resistance_levels_ohm=self.resistance_levels_ohm,
            native_precision_bits=self.native_precision_bits,
            device_area_um2=self.device_area_um2,
            pulse_parameters=asdict(self.pulse),
        )

    def save_state_json(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w") as f:
            json.dump(asdict(self.state()), f, indent=2)

    def __repr__(self) -> str:
        return (
            f"MemristorDevice(name={self.name!r}, states={self.number_of_states}, "
            f"current_G={self.current_conductance_s:.4e} S, "
            f"current_R={self.current_resistance_ohm:.4e} ohm)"
        )
