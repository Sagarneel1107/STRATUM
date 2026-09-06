"""
single_memristor_nonidealities.py

Single-memristor non-ideality layer for the memristor/crossbar simulator.

This module does not replace memristor_device.py or integrated_crossbar.py.
Instead, it perturbs the conductance matrix that is passed into the existing
IntegratedMemristorCrossbar object.

Implemented non-idealities
--------------------------
1. device-to-device variation: persistent multiplicative variation per cell
2. cycle-to-cycle/programming variation: random write error per programming event
3. stuck-at faults: persistent cells stuck at HRS or LRS
4. retention/drift: exponential, power-law, or Kohlrausch relaxation
5. read noise: transient conductance/current-equivalent noise during read
6. nonlinear/asymmetric pulse update trace: optional single-device trace helper

Units
-----
Conductance: Siemens
Time: seconds
Voltages and energies are handled by the existing device/crossbar modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import count
from typing import Literal, Optional, Sequence, Union
import weakref

import numpy as np


Number = Union[int, float]
RetentionModel = Literal["none", "exponential", "power_law", "kohlrausch"]
UpdateDirection = Literal["potentiate", "depress"]


@dataclass
class SingleMemristorNonIdealityConfig:
    """
    Configuration for single-device non-idealities.

    The sigma_rel quantities are coefficients of variation. For example,
    device_to_device_sigma_rel=0.05 means approximately 5% cell-to-cell spread.
    """

    enabled: bool = True

    # Persistent variation: each cell gets a fixed multiplicative factor.
    device_to_device_sigma_rel: float = 0.0

    # Random programming errors each time a matrix is programmed.
    programming_error_sigma_rel: float = 0.0
    cycle_to_cycle_sigma_rel: float = 0.0

    # Transient read noise. This is applied only during read, not permanently
    # stored, unless restore_conductance_after_read=False is used.
    read_noise_sigma_rel: float = 0.0
    read_noise_sigma_abs_s: float = 0.0

    # Persistent stuck-at defects. stuck_at_lrs_fraction gives the fraction of
    # stuck cells pinned to the maximum conductance; the rest are pinned to the
    # minimum conductance.
    stuck_at_probability: float = 0.0
    stuck_at_lrs_fraction: float = 0.5

    # Retention/conductance relaxation model.
    # kohlrausch: G(t) = G_floor + [G(0)-G_floor] exp[-(t/tau)^beta]
    # power_law : G(t) = G_floor + [G(0)-G_floor] (1+t/tau)^(-alpha)
    # exponential: same as kohlrausch with beta = 1
    retention_model: RetentionModel = "none"
    retention_tau_s: float = 3600.0
    retention_beta: float = 0.5
    retention_alpha: float = 0.1
    retention_floor_s: Optional[float] = None

    # Conductance clipping.  `clip_to_measured_range` bounds a perturbed cell to
    # what the DEVICE can hold, not to the nominal programmable grid: clipping at
    # [g_min, g_max] deletes the tails that produce near-stuck cells and piles that
    # mass onto the rails.  Give the physical rails explicitly, or leave them None
    # and they are the grid widened by `physical_range_margin` in log conductance.
    clip_to_measured_range: bool = True
    physical_g_min_s: Optional[float] = None
    physical_g_max_s: Optional[float] = None
    physical_range_margin: float = 0.25
    minimum_conductance_s: float = 1e-18

    # Optional pulse-by-pulse update helper parameters.
    potentiation_step_fraction: float = 0.10
    depression_step_fraction: float = 0.10

    def validate(self) -> None:
        nonnegative_fields = [
            "device_to_device_sigma_rel",
            "programming_error_sigma_rel",
            "cycle_to_cycle_sigma_rel",
            "read_noise_sigma_rel",
            "read_noise_sigma_abs_s",
            "stuck_at_probability",
            "retention_tau_s",
            "retention_beta",
            "retention_alpha",
            "minimum_conductance_s",
            "potentiation_step_fraction",
            "depression_step_fraction",
        ]

        for field in nonnegative_fields:
            value = float(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} must be non-negative")

        if not 0.0 <= self.stuck_at_probability <= 1.0:
            raise ValueError("stuck_at_probability must be between 0 and 1")
        if not 0.0 <= self.stuck_at_lrs_fraction <= 1.0:
            raise ValueError("stuck_at_lrs_fraction must be between 0 and 1")
        if self.retention_model not in {"none", "exponential", "power_law", "kohlrausch"}:
            raise ValueError("retention_model must be one of: none, exponential, power_law, kohlrausch")
        if self.retention_model != "none" and self.retention_tau_s <= 0:
            raise ValueError("retention_tau_s must be positive when retention is enabled")
        # beta=0 gives exp(-1) for all t>0: a step, not a valid relaxation.
        if self.retention_model == "kohlrausch" and self.retention_beta <= 0:
            raise ValueError("retention_beta must be strictly positive for the Kohlrausch model")


@dataclass
class NonIdealConductanceState:
    """
    Matrices produced while programming a crossbar.

    ideal_matrix_s:       exact conductance for the requested state codes.
    programmed_matrix_s:  after programming variation and stuck-at faults.
    retained_matrix_s:    after optional retention drift for elapsed_time_s.
    stats:                compact numerical summary for reporting.
    """

    ideal_matrix_s: np.ndarray
    programmed_matrix_s: np.ndarray
    retained_matrix_s: np.ndarray
    stats: dict[str, float | int | str | bool]


class SingleMemristorNonIdealityModel:
    """
    Applies single-memristor non-idealities to conductance values.

    Sits between the target/state-code matrix and the crossbar object:

        model = SingleMemristorNonIdealityModel(levels, config, rng=0)
        state = model.program_from_state_codes(state_codes)   # one array
        other = model.program_from_state_codes(state_codes)   # a DIFFERENT array
        xbar.set_conductance_matrix(state.retained_matrix_s)
        result = model.read_from_digital_input(xbar, input_codes)
    """

    def __init__(
        self,
        conductance_levels_s: Sequence[Number],
        config: Optional[SingleMemristorNonIdealityConfig] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.conductance_levels_s = self._validate_levels(conductance_levels_s)
        self.config = config if config is not None else SingleMemristorNonIdealityConfig()
        self.config.validate()

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # Persistent per-ARRAY draws (D2D spread, stuck-at map): properties of one
        # piece of silicon, fixed for its life and independent between arrays.  The
        # cache is keyed by (shape, array_key), not shape alone, or every 128x128
        # array on the chip would share one variation pattern and defect map.
        # A crossbar OBJECT gets a stable key (weak-referenced), so re-programming
        # the same array keeps the same die; a bare matrix gets a fresh key per
        # call unless `array_key` is passed explicitly.
        self._variation_cache: dict[tuple[tuple[int, int], object], dict[str, np.ndarray]] = {}
        self._object_keys: "weakref.WeakKeyDictionary[object, int]" = weakref.WeakKeyDictionary()
        self._array_key_counter = count()

    # Array identity

    def new_array_key(self) -> int:
        """Allocate a key for one more physical array."""
        return next(self._array_key_counter)

    def array_key_for(self, obj) -> int:
        """
        Stable key for a crossbar object: same object -> same die.

        Falls back to a fresh key for objects that cannot be weak-referenced.
        """
        try:
            key = self._object_keys.get(obj)
            if key is None:
                key = self.new_array_key()
                self._object_keys[obj] = key
            return key
        except TypeError:
            return self.new_array_key()

    # Public matrix creation methods

    def ideal_matrix_from_state_codes(self, state_code_matrix: Sequence[Sequence[int]]) -> np.ndarray:
        """Convert integer state codes to the ideal conductance matrix."""
        codes = np.asarray(state_code_matrix, dtype=int)

        if codes.ndim != 2:
            raise ValueError("state_code_matrix must be 2D")
        if np.any(codes < 0) or np.any(codes >= len(self.conductance_levels_s)):
            raise IndexError("state code is outside the available conductance levels")

        return self.conductance_levels_s[codes]

    def snap_to_nearest_level(self, conductance_matrix_s: Sequence[Sequence[Number]]) -> np.ndarray:
        """Snap arbitrary conductance values to the nearest measured level."""
        matrix = np.asarray(conductance_matrix_s, dtype=float)
        flat = matrix.reshape(-1)
        levels = self.conductance_levels_s
        nearest = np.argmin(np.abs(flat[:, None] - levels[None, :]), axis=1)
        return levels[nearest].reshape(matrix.shape)

    def program_from_state_codes(
        self,
        state_code_matrix: Sequence[Sequence[int]],
        elapsed_time_s: float = 0.0,
        array_key: Optional[int] = None,
    ) -> NonIdealConductanceState:
        """
        Convert state codes to a non-ideal programmed conductance matrix.

        `array_key` identifies which physical array is being programmed, so its D2D
        spread and stuck-at map stay fixed across calls while staying independent of
        every other array.  Omit it and each call is treated as a new array.
        """
        if array_key is None:
            array_key = self.new_array_key()
        ideal = self.ideal_matrix_from_state_codes(state_code_matrix)
        programmed = self.apply_programming_nonidealities(ideal, array_key=array_key)
        retained = self.apply_retention(
            programmed, elapsed_time_s=elapsed_time_s, array_key=array_key)
        stats = self.matrix_error_stats(ideal, retained, array_key=array_key)
        return NonIdealConductanceState(
            ideal_matrix_s=ideal,
            programmed_matrix_s=programmed,
            retained_matrix_s=retained,
            stats=stats,
        )

    def program_from_target_matrix(
        self,
        target_conductance_matrix_s: Sequence[Sequence[Number]],
        elapsed_time_s: float = 0.0,
        snap_to_levels_before_programming: bool = True,
        array_key: Optional[int] = None,
    ) -> NonIdealConductanceState:
        """
        Apply non-idealities to a target conductance matrix.

        Use snap_to_levels_before_programming=True when your hardware stores only
        measured discrete states. Set it to False for an idealized continuous-cell
        experiment.
        """
        target = np.asarray(target_conductance_matrix_s, dtype=float)
        if target.ndim != 2:
            raise ValueError("target_conductance_matrix_s must be 2D")
        if np.any(target <= 0):
            raise ValueError("target conductances must be positive")

        if array_key is None:
            array_key = self.new_array_key()
        ideal = self.snap_to_nearest_level(target) if snap_to_levels_before_programming else target.copy()
        programmed = self.apply_programming_nonidealities(ideal, array_key=array_key)
        retained = self.apply_retention(
            programmed, elapsed_time_s=elapsed_time_s, array_key=array_key)
        stats = self.matrix_error_stats(ideal, retained, array_key=array_key)
        return NonIdealConductanceState(
            ideal_matrix_s=ideal,
            programmed_matrix_s=programmed,
            retained_matrix_s=retained,
            stats=stats,
        )

    def apply_to_crossbar_from_state_codes(
        self,
        xbar,
        state_code_matrix: Sequence[Sequence[int]],
        elapsed_time_s: float = 0.0,
    ) -> NonIdealConductanceState:
        """
        Program an IntegratedMemristorCrossbar with a non-ideal matrix.

        The crossbar object supplies the array identity, so programming the same
        array twice keeps its die properties.
        """
        state = self.program_from_state_codes(
            state_code_matrix,
            elapsed_time_s=elapsed_time_s,
            array_key=self.array_key_for(xbar),
        )
        xbar.set_conductance_matrix(state.retained_matrix_s)
        return state

    # Individual non-ideality transforms

    def apply_programming_nonidealities(
        self,
        ideal_conductance_matrix_s: Sequence[Sequence[Number]],
        array_key: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply D2D spread, programming/cycle variation, and stuck-at faults.

        `array_key` selects which array's persistent draw to use; omit it and
        this matrix is treated as a new array with its own independent draw.
        """
        matrix = np.asarray(ideal_conductance_matrix_s, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("ideal_conductance_matrix_s must be 2D")
        if np.any(matrix <= 0):
            raise ValueError("conductances must be positive")

        if not self.config.enabled:
            return matrix.copy()

        result = matrix.copy()
        shape = result.shape
        if array_key is None:
            array_key = self.new_array_key()
        cache = self._cache_for_shape(shape, array_key)

        # Persistent cell-to-cell variation.
        if self.config.device_to_device_sigma_rel > 0:
            result *= cache["d2d_factor"]

        # Random write/programming variation for this programming event.
        # Multiplicative lognormal, matching the D2D draw above: a conductance is
        # positive by construction and its write scatter is multiplicative.
        total_write_sigma = float(
            np.hypot(self.config.programming_error_sigma_rel, self.config.cycle_to_cycle_sigma_rel)
        )
        if total_write_sigma > 0:
            result = result * self._lognormal_factor(total_write_sigma, shape)

        # Persistent stuck-at defects override normal programming.
        if self.config.stuck_at_probability > 0:
            stuck_mask = cache["stuck_mask"].astype(bool)
            result[stuck_mask] = cache["stuck_values_s"][stuck_mask]

        return self._clip(result)

    def apply_retention(
        self,
        conductance_matrix_s: Sequence[Sequence[Number]],
        elapsed_time_s: float,
        array_key: Optional[int] = None,
    ) -> np.ndarray:
        """Apply time-dependent retention/drift to a conductance matrix."""
        matrix = np.asarray(conductance_matrix_s, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("conductance_matrix_s must be 2D")
        if np.any(matrix <= 0):
            raise ValueError("conductances must be positive")
        if elapsed_time_s < 0:
            raise ValueError("elapsed_time_s must be non-negative")

        if (
            not self.config.enabled
            or self.config.retention_model == "none"
            or elapsed_time_s == 0
        ):
            return matrix.copy()

        floor = self._retention_floor()
        tau = self.config.retention_tau_s
        t = float(elapsed_time_s)

        if self.config.retention_model == "exponential":
            survival = np.exp(-t / tau)
        elif self.config.retention_model == "power_law":
            survival = (1.0 + t / tau) ** (-self.config.retention_alpha)
        elif self.config.retention_model == "kohlrausch":
            survival = np.exp(-((t / tau) ** self.config.retention_beta))
        else:
            raise ValueError(f"Unsupported retention model: {self.config.retention_model}")

        retained = floor + (matrix - floor) * survival

        # Re-pin stuck cells after retention drift.  `_cache_for_shape` guarantees
        # the cache exists and reuses the RNG-seeded positions.  Pass the SAME
        # array_key the matrix was programmed with, or the cells re-pinned here
        # belong to a different array.
        if self.config.stuck_at_probability > 0:
            if array_key is None:
                array_key = self.new_array_key()
            cache = self._cache_for_shape(matrix.shape, array_key)
            stuck_mask = cache["stuck_mask"].astype(bool)
            retained[stuck_mask] = cache["stuck_values_s"][stuck_mask]

        return self._clip(retained)

    def apply_read_noise(self, conductance_matrix_s: Sequence[Sequence[Number]]) -> np.ndarray:
        """Apply transient read noise to a conductance matrix."""
        matrix = np.asarray(conductance_matrix_s, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("conductance_matrix_s must be 2D")
        if np.any(matrix <= 0):
            raise ValueError("conductances must be positive")

        if not self.config.enabled:
            return matrix.copy()

        cfg = self.config
        if cfg.read_noise_sigma_rel == 0.0 and cfg.read_noise_sigma_abs_s == 0.0:
            return matrix.copy()

        # Relative term is multiplicative lognormal (same statistics as D2D and
        # write scatter); the absolute floor is a genuine additive current-equivalent
        # term and stays Gaussian.
        noisy = matrix
        if cfg.read_noise_sigma_rel > 0:
            noisy = noisy * self._lognormal_factor(cfg.read_noise_sigma_rel, matrix.shape)
        if cfg.read_noise_sigma_abs_s > 0:
            noisy = noisy + self.rng.normal(
                loc=0.0, scale=cfg.read_noise_sigma_abs_s, size=matrix.shape)
        return self._clip(noisy)

    # Crossbar read wrappers

    def read_from_digital_input(
        self,
        xbar,
        input_codes: Sequence[int],
        elapsed_time_s: float = 0.0,
        accumulate_energy: bool = True,
        active_partial_sums: int = 1,
        restore_conductance_after_read: bool = True,
    ):
        """
        Read the existing xbar using retention + read-noise non-idealities.

        By default, the stored xbar.conductance_matrix_s is restored after the
        read, because read noise should not permanently change the device state.
        Set restore_conductance_after_read=False if you want the retained/noisy
        matrix to remain in the crossbar after reading.
        """
        stored_matrix = np.asarray(xbar.conductance_matrix_s, dtype=float).copy()
        read_matrix = self.apply_retention(
            stored_matrix,
            elapsed_time_s=elapsed_time_s,
            array_key=self.array_key_for(xbar),
        )
        read_matrix = self.apply_read_noise(read_matrix)

        xbar.set_conductance_matrix(read_matrix)
        try:
            result = xbar.read_from_digital_input(
                input_codes=input_codes,
                accumulate_energy=accumulate_energy,
                active_partial_sums=active_partial_sums,
            )
        finally:
            if restore_conductance_after_read:
                xbar.set_conductance_matrix(stored_matrix)

        return result

    def read_from_analog_voltages(
        self,
        xbar,
        input_voltages_v: Sequence[Number],
        elapsed_time_s: float = 0.0,
        accumulate_energy: bool = True,
        active_partial_sums: int = 1,
        restore_conductance_after_read: bool = True,
    ):
        """Analog-voltage version of read_from_digital_input(...)."""
        stored_matrix = np.asarray(xbar.conductance_matrix_s, dtype=float).copy()
        read_matrix = self.apply_retention(
            stored_matrix,
            elapsed_time_s=elapsed_time_s,
            array_key=self.array_key_for(xbar),
        )
        read_matrix = self.apply_read_noise(read_matrix)

        xbar.set_conductance_matrix(read_matrix)
        try:
            result = xbar.read_from_analog_voltages(
                input_voltages_v=input_voltages_v,
                accumulate_energy=accumulate_energy,
                active_partial_sums=active_partial_sums,
            )
        finally:
            if restore_conductance_after_read:
                xbar.set_conductance_matrix(stored_matrix)

        return result

    # Optional single-device pulse update helper

    def simulate_pulse_update_trace(
        self,
        initial_conductance_s: Number,
        pulse_count: int,
        direction: UpdateDirection = "potentiate",
    ) -> np.ndarray:
        """
        Generate a simple nonlinear/asymmetric conductance update trace.

        This is useful for studying single-device potentiation/depression curves.
        It is not automatically used by matrix programming, because your current
        crossbar code programs cells by target state codes.
        """
        if pulse_count < 0:
            raise ValueError("pulse_count must be non-negative")
        if direction not in {"potentiate", "depress"}:
            raise ValueError("direction must be 'potentiate' or 'depress'")

        g_min = float(np.min(self.conductance_levels_s))
        g_max = float(np.max(self.conductance_levels_s))
        g = float(initial_conductance_s)
        if g <= 0:
            raise ValueError("initial_conductance_s must be positive")

        trace = []
        for _ in range(pulse_count):
            if direction == "potentiate":
                delta = self.config.potentiation_step_fraction * (g_max - g)
            else:
                delta = -self.config.depression_step_fraction * (g - g_min)

            if self.config.cycle_to_cycle_sigma_rel > 0:
                delta += self.rng.normal(
                    loc=0.0,
                    scale=self.config.cycle_to_cycle_sigma_rel * max(abs(delta), self.config.minimum_conductance_s),
                )

            g = float(self._clip(np.array([g + delta]))[0])
            trace.append(g)

        return np.asarray(trace, dtype=float)

    # Reporting

    def matrix_error_stats(
        self,
        ideal_matrix_s: Sequence[Sequence[Number]],
        actual_matrix_s: Sequence[Sequence[Number]],
        array_key: Optional[int] = None,
    ) -> dict[str, float | int | str | bool]:
        """Return compact stats comparing ideal and actual conductance matrices."""
        ideal = np.asarray(ideal_matrix_s, dtype=float)
        actual = np.asarray(actual_matrix_s, dtype=float)
        if ideal.shape != actual.shape:
            raise ValueError("ideal and actual matrices must have the same shape")

        abs_err = np.abs(actual - ideal)
        rel_err = abs_err / np.maximum(np.abs(ideal), self.config.minimum_conductance_s)
        cache = self._variation_cache.get((tuple(ideal.shape), array_key))
        stuck_count = int(np.sum(cache["stuck_mask"])) if cache is not None and "stuck_mask" in cache else 0

        return {
            "enabled": self.config.enabled,
            "retention_model": self.config.retention_model,
            "shape_rows": int(ideal.shape[0]),
            "shape_cols": int(ideal.shape[1]),
            "ideal_min_s": float(np.min(ideal)),
            "ideal_max_s": float(np.max(ideal)),
            "actual_min_s": float(np.min(actual)),
            "actual_max_s": float(np.max(actual)),
            "ideal_mean_s": float(np.mean(ideal)),
            "actual_mean_s": float(np.mean(actual)),
            "actual_std_s": float(np.std(actual)),
            "mean_abs_error_s": float(np.mean(abs_err)),
            "max_abs_error_s": float(np.max(abs_err)),
            "mean_relative_abs_error": float(np.mean(rel_err)),
            "max_relative_abs_error": float(np.max(rel_err)),
            "stuck_cell_count": stuck_count,
        }

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    # Internal helpers

    @staticmethod
    def _validate_levels(conductance_levels_s: Sequence[Number]) -> np.ndarray:
        levels = np.asarray(conductance_levels_s, dtype=float)
        if levels.ndim != 1:
            raise ValueError("conductance_levels_s must be a 1D sequence")
        if levels.size == 0:
            raise ValueError("at least one conductance level is required")
        if np.any(levels <= 0):
            raise ValueError("all conductance levels must be positive")
        return np.sort(np.unique(levels))

    def _lognormal_factor(self, cv: float, shape) -> np.ndarray:
        """
        Multiplicative variation factor: positive, unit mean, coefficient of
        variation `cv`.

        One definition, used by every conductance perturbation in this module (D2D,
        write scatter, relative read noise).  `hwat_core` draws its buffers the same
        way.
        """
        cv = float(cv)
        if cv <= 0:
            return np.ones(shape, dtype=float)
        sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
        mu_ln = -0.5 * sigma_ln ** 2
        return self.rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=shape)

    def _cache_for_shape(self, shape: tuple[int, int],
                         array_key: object) -> dict[str, np.ndarray]:
        cache_key = (tuple(shape), array_key)
        if cache_key in self._variation_cache:
            return self._variation_cache[cache_key]

        rows, cols = shape
        cache: dict[str, np.ndarray] = {}

        cache["d2d_factor"] = self._lognormal_factor(
            self.config.device_to_device_sigma_rel, shape)

        p_stuck = self.config.stuck_at_probability
        if p_stuck > 0:
            stuck_mask = self.rng.random(size=shape) < p_stuck
            stuck_lrs = self.rng.random(size=shape) < self.config.stuck_at_lrs_fraction
            g_min = float(np.min(self.conductance_levels_s))
            g_max = float(np.max(self.conductance_levels_s))
            stuck_values = np.where(stuck_lrs, g_max, g_min)
            cache["stuck_mask"] = stuck_mask
            cache["stuck_values_s"] = stuck_values
        else:
            cache["stuck_mask"] = np.zeros(shape, dtype=bool)
            cache["stuck_values_s"] = np.zeros(shape, dtype=float)

        self._variation_cache[cache_key] = cache
        return cache

    def _retention_floor(self) -> float:
        if self.config.retention_floor_s is not None:
            floor = float(self.config.retention_floor_s)
            if floor <= 0:
                raise ValueError("retention_floor_s must be positive")
            return floor
        return float(np.min(self.conductance_levels_s))

    def physical_conductance_rails_s(self) -> tuple[float, float]:
        """
        The device's physical HRS/LRS rails, which BRACKET the programmable grid.

        The grid is where the programming algorithm aims, not where the physics
        stops, so clipping at the grid would delete the tails that produce
        near-stuck cells.  Unless given explicitly the rails are the grid widened by
        `physical_range_margin` in log conductance.
        """
        cfg = self.config
        g_lo = float(np.min(self.conductance_levels_s))
        g_hi = float(np.max(self.conductance_levels_s))
        m = max(float(cfg.physical_range_margin), 0.0)
        lo = g_lo / (1.0 + m) if cfg.physical_g_min_s is None else float(cfg.physical_g_min_s)
        hi = g_hi * (1.0 + m) if cfg.physical_g_max_s is None else float(cfg.physical_g_max_s)
        if lo <= 0 or hi <= lo:
            raise ValueError("physical conductance rails must satisfy 0 < g_min < g_max")
        return lo, hi

    def _clip(self, matrix: np.ndarray) -> np.ndarray:
        # np.clip already returns a new array, so no .copy() is needed.
        low = self.config.minimum_conductance_s
        high = None

        if self.config.clip_to_measured_range:
            rail_lo, rail_hi = self.physical_conductance_rails_s()
            low = max(low, rail_lo)
            high = rail_hi

        return np.clip(np.asarray(matrix, dtype=float), low, high)
