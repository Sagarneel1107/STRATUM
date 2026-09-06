"""
stratum_platform.py

Tkinter GUI front-end for the memristor IMC workflows, developed by IMChip B.V.

A STUDY selector chooses which workload to deploy on the chip: the MNIST MLP run
as a RISC-V-like instruction program (example_usage_riscv_workflow.py), or the
ResNet-18 / VGG-16 classification backbones scored by top-1 (resnet_vgg_study.py).

Every parameter of `simulation_config.py` the selected study consumes, plus its
workflow-local settings, appears in grouped editable tabs.  Edits apply live and
can be saved to JSON profiles.  The simulation runs in a background thread with
its log streamed into the GUI; results come back as KPI cards, a comparison
table, the embedded matplotlib dashboard and a self-contained HTML report, with
run-to-run deltas on the headline metrics.

The platform never writes to simulation_config.py: overrides are applied to the
imported module object at run time.
"""

from __future__ import annotations

import base64
import html as _html
import json
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

import simulation_config as simcfg


PROFILE_PATH = "stratum_profile.json"
# Fallbacks only - each Study carries its own output paths (see Study below).
REPORT_PATH = "riscv_workflow_report.html"
FIGURE_PATH = "riscv_workflow_results.png"

# Palette (shared by the GUI and the HTML report).
COL_HEADER = "#16324f"
COL_ACCENT = "#3d6bb3"
COL_GOOD = "#2e8b57"
COL_BAD = "#c44e52"
COL_MUTED = "#6b7a88"
COL_BG = "#f2f5f8"
COL_CARD = "#ffffff"
COL_MOD = "#b26a00"          # modified-parameter highlight
COL_LOG_BG = "#16232f"
COL_LOG_FG = "#c7d5e0"


# Parameter registry

@dataclass
class Param:
    """One editable parameter (backed by simulation_config or run-local)."""
    key: str                       # attribute in simulation_config, or run key
    label: str                     # human name shown in the GUI
    kind: str                      # int | float | bool | choice | float_list
    target: str = "simcfg"         # "simcfg" | "run"
    unit: str = ""
    help: str = ""
    choices: tuple = ()            # for kind == "choice" (or allowed int set)
    default: Any = None            # captured at registry build time
    # (controller_key, allowed_values): row is shown only when the controller
    # parameter currently holds one of the allowed values (else it is hidden).
    visible_if: Optional[tuple] = None

    def is_modified(self, value: Any) -> bool:
        return value != self.default


def make_param(key: str, label: str, kind: str, target: str = "simcfg",
               unit: str = "", help: str = "", choices: tuple = (),
               visible_if: Optional[tuple] = None,
               run_defaults: Optional[Dict[str, Any]] = None) -> Param:
    """Build one Param, resolving its default from simulation_config or the
    study's DEFAULT_RUN_SETTINGS."""
    if target == "simcfg":
        default = getattr(simcfg, key)
    else:
        default = (run_defaults or {})[key]
    if kind in ("float_list", "multichoice"):
        default = list(default)
    return Param(key=key, label=label, kind=kind, target=target,
                 unit=unit, help=help, choices=choices, default=default,
                 visible_if=visible_if)


# A study-specific parameter group is declared as data: (tab, title, [kwargs])
# where each kwargs dict is the argument list of make_param().
GroupSpec = Tuple[str, str, List[Dict[str, Any]]]


def build_registry(
    run_defaults: Dict[str, Any],
    extra_groups: Sequence[GroupSpec] = (),
    exclude_keys: Sequence[str] = (),
) -> List[Tuple[str, str, List[Param]]]:
    """
    Grouped parameter registry: (tab name, full group title, params).

    Defaults for "simcfg" params come from the live simulation_config module;
    "run" params from the study's DEFAULT_RUN_SETTINGS.  A study only sees
    parameters it consumes; study-only tabs are appended from `extra_groups`.
    """
    drop = set(exclude_keys)

    def P(key, label, kind, target="simcfg", unit="", help="", choices=(),
          visible_if=None):
        if key in drop or (target == "run" and key not in run_defaults):
            return None
        return make_param(key=key, label=label, kind=kind, target=target,
                          unit=unit, help=help, choices=choices,
                          visible_if=visible_if, run_defaults=run_defaults)

    # MRAM rows appear inside the Sign / 2T1R tab only for the spintronic mode.
    _mram_if = ("SIGN_STORAGE_MODE", ("spintronic_mram",))

    groups: List[Tuple[str, str, List[Param]]] = [
        ("Run", "Run & analysis settings", [
            P("test_subset_n", "Test images (hardware path)", "int", "run",
              unit="imgs", help="Test images pushed through the full hardware path"),
            P("calib_n", "ADC calibration images", "int", "run",
              unit="imgs", help="Images used to calibrate each tile's ADC full-scale"),
            P("NN_DECAY_TIME_POINTS_S", "Retention time points", "float_list",
              unit="s", help="Elapsed times of the accuracy-retention sweep"),
        ]),
        ("Device", "Memristor device", [
            # The relaxation time constant is a property of the CELL, not of the
            # run, so it belongs beside the other device numbers rather than in
            # the run settings.  Both studies read it from here.
            P("NN_DECAY_RETENTION_TAU_S", "Retention tau (Kohlrausch)", "float",
              unit="s", help="Characteristic conductance-relaxation time. With "
                             "the retention model and beta on the 'Non-ideal' "
                             "tab this fixes the drift law; how far along that "
                             "law a run sits is the retention age / time points"),
            P("CONDUCTANCE_LEVELS_S", "Conductance states", "float_list",
              unit="S", help="Programmable states; 16 states => 4-bit device"),
            P("READ_VOLTAGE_V", "Read voltage", "float", unit="V",
              help="Row read voltage applied by the DAC at full scale"),
            P("WIRE_RESISTANCE_OHM", "Wire resistance", "float", unit="ohm",
              help="Bit/word line resistance per cell pitch. With the "
                   "capacitance below this sets the column settling time, "
                   "which is one of the two constraints the read time is "
                   "derived from"),
            P("WIRE_CAPACITANCE_F", "Wire capacitance", "float", unit="F",
              help="Bit/word line capacitance per cell pitch"),
            P("READ_TIME_MARGIN", "Read-time margin", "float", unit="x",
              help="Headroom on the derived read time (1.0 = settle exactly to "
                   "1 LSB). The read duration itself is DERIVED - the larger of "
                   "the ADC's conversion aperture and the column RC settling - "
                   "so it is not edited directly; change the ADC or the wire RC "
                   "to move it"),
            P("DEVICE_AREA_UM2", "Device area", "float", unit="um^2",
              help="Area of a single memristor cell"),
        ]),
        ("Crossbar", "Crossbar & datapath", [
            P("XBAR_ROWS", "Crossbar rows", "int",
              help="Physical rows of one array (inputs per tile)"),
            P("XBAR_COLS", "Crossbar columns", "int",
              help="Physical columns of one array (outputs per tile)"),
            P("TECH_NODE_NM", "Technology node", "int", unit="nm",
              help="CMOS node for the digital periphery models"),
            P("INPUT_PRECISION_BITS", "Input precision", "int", unit="bit",
              help="Activation precision fed digit-serially to the DAC"),
            P("digit_bits", "DAC digit width", "int", "run", unit="bit",
              choices=(1, 2, 4, 8),
              help="Bits per digit-serial slice; 4 => two nibbles per 8-bit input"),
            # Series resistance in the read path, OHMS PER CELL.  Contact+via 100
            # and access transistor 500 make the 600 ohm/cell path; the builders
            # divide by the row count on the way into the array model.
            P("CB_NI_PARASITIC_SERIES_RESISTANCE_OHM",
              "Contact + via resistance", "float", unit="ohm/cell",
              help="Contact and via in series with each cell. Default 100, "
                   "which an advanced-node via budget supports (<50 ohm per "
                   "via, about two per cell)"),
            P("CB_NI_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM",
              "Access transistor R_on (single crossbar)", "float",
              unit="ohm/cell",
              help="1T1R/2T1R select-transistor on-resistance for the "
                   "single-crossbar path. Default 500, from published W/L >= 5 "
                   "sizing (210 nm / 40 nm for >150 uA) at gate overdrive"),
            P("WORKFLOW_ACCESS_TRANSISTOR_ON_RESISTANCE_OHM",
              "Access transistor R_on (studies)", "float", unit="ohm/cell",
              help="The same transistor as seen by the MLP / ResNet / VGG / "
                   "detector studies, which build their array through "
                   "build_workflow_crossbar_ni_config. Separate knob because "
                   "that builder overrides the single-crossbar value above. "
                   "Under-driving the word line pushes R_on past 1e5 ohm/cell "
                   "and the array stops working - keep it under ~10 kohm/cell"),
        ]),
        ("Sign / 2T1R", "Sign storage & 2T1R cell", [
            P("SIGN_STORAGE_MODE", "Sign-storage technology", "choice",
              choices=("register_2t1r", "spintronic_mram"),
              help="Volatile CMOS register vs non-volatile STT-MRAM sign cell "
                   "(selecting the spintronic option reveals its MTJ parameters "
                   "below)"),
            # --- spintronic STT-MRAM MTJ parameters (shown only when the
            #     spintronic sign-storage mode is selected above) ---
            P("MRAM_CELL_AREA_F2", "MTJ cell area", "float", unit="F^2",
              visible_if=_mram_if,
              help="1T1MTJ footprint in F^2 (F = feature size)"),
            P("MRAM_R_PARALLEL_OHM", "MTJ R_parallel", "float", unit="ohm",
              visible_if=_mram_if,
              help="Low-resistance (parallel) MTJ state"),
            P("MRAM_TMR_RATIO", "TMR ratio", "float", unit="frac",
              visible_if=_mram_if,
              help="(R_AP - R_P)/R_P tunnel magnetoresistance"),
            P("MRAM_READ_VOLTAGE_V", "MTJ read voltage", "float", unit="V",
              visible_if=_mram_if,
              help="Sense voltage (kept << switching to avoid read disturb)"),
            P("MRAM_WRITE_CURRENT_A", "STT write current", "float", unit="A",
              visible_if=_mram_if,
              help="Spin-transfer-torque switching current"),
            P("MRAM_WRITE_VOLTAGE_V", "Write driver voltage", "float", unit="V",
              visible_if=_mram_if, help="Write-driver voltage"),
            P("MRAM_WRITE_TIME_NS", "Write pulse width", "float", unit="ns",
              visible_if=_mram_if, help="STT switching pulse width"),
            P("t2_on_resistance_ohm", "Access transistor R_on", "float", "run",
              unit="ohm", help="Series on-resistance of the selected branch"),
            P("t2_off_resistance_ohm", "Access transistor R_off", "float", "run",
              unit="ohm", help="Wrong-rail leakage path (inf disables)"),
            P("t2_rail_gain_mismatch", "Rail gain mismatch", "float", "run",
              unit="frac", help="+/- differential sense-path gain mismatch"),
            P("t2_rail_offset_a", "Rail offset current", "float", "run",
              unit="A", help="Systematic differential sense offset"),
            P("t2_transistor_area_um2", "Transistor area", "float", "run",
              unit="um^2", help="Area of ONE access transistor (two per cell)"),
            P("t2_transistor_switch_energy_nj", "Transistor switch energy",
              "float", "run", unit="nJ",
              help="Gate-drive energy per access transistor per read"),
        ]),
        ("Non-ideal", "Device non-idealities", [
            P("SM_NI_ENABLED", "Non-idealities enabled", "bool",
              help="Master switch for the single-device non-ideality model"),
            P("SM_NI_DEVICE_TO_DEVICE_SIGMA_REL", "D2D variation sigma", "float",
              unit="frac", help="Persistent cell-to-cell spread (CV)"),
            P("SM_NI_PROGRAMMING_ERROR_SIGMA_REL", "Write error sigma", "float",
              unit="frac", help="Random error per programming event"),
            P("SM_NI_CYCLE_TO_CYCLE_SIGMA_REL", "Cycle-to-cycle sigma", "float",
              unit="frac", help="Cycle-to-cycle programming variation"),
            P("SM_NI_READ_NOISE_SIGMA_REL", "Read noise sigma", "float",
              unit="frac", help="Transient read noise (re-drawn each read)"),
            P("SM_NI_STUCK_AT_PROBABILITY", "Stuck-at probability", "float",
              unit="frac", help="Fraction of cells stuck at LRS/HRS"),
            P("SM_NI_RETENTION_MODEL", "Retention model", "choice",
              choices=("none", "exponential", "power_law", "kohlrausch"),
              help="Conductance drift law used for the retention sweep"),
            P("SM_NI_RETENTION_BETA", "Retention beta", "float",
              help="Stretched-exponential exponent (Kohlrausch)"),
        ]),
        ("Chip", "Chip architecture", [
            P("CHIP_NUM_TILES", "Tiles per chip", "int",
              help="Number of tiles on the chip"),
            P("TILE_NUM_PUS", "PUs per tile", "int",
              help="Processing units per tile (used for array placement)"),
            P("PU_NUM_ARRAYS", "Arrays per PU", "int",
              help="Crossbar arrays per processing unit"),
        ]),
        ("MNIST", "MNIST network & training", [
            P("MNIST_HIDDEN_UNITS", "Hidden units", "int",
              help="MLP hidden width (change triggers a retrain)"),
            P("MNIST_EPOCHS", "Training epochs", "int",
              help="Epochs used when (re)training the MLP"),
            P("MNIST_BATCH_SIZE", "Batch size", "int",
              help="Training mini-batch size"),
            P("MNIST_LEARNING_RATE", "Learning rate", "float",
              help="Adam learning rate"),
            P("force_retrain", "Force retrain", "bool", "run",
              help="Retrain the MLP even if cached weights exist"),
        ]),
    ]
    for tab, title, specs in extra_groups:
        groups.append((tab, title,
                       [make_param(run_defaults=run_defaults, **spec)
                        for spec in specs
                        if spec["key"] not in drop]))
    # Drop the parameters this study does not consume, then any group left empty.
    return [(tab, title, [p for p in plist if p is not None])
            for tab, title, plist in groups
            if any(p is not None for p in plist)]


# Studies: what the platform can deploy on the chip

@dataclass
class Kpi:
    """One headline metric shown as a card (and tracked run-to-run)."""
    key: str                                   # id used for the "vs prev" delta
    label: str
    getter: Callable[[Dict[str, Any]], float]  # summary -> number
    fmt: str = "{:.3f}"
    unit: str = ""
    higher_is_better: Optional[bool] = None    # None => no good/bad colouring

    def value(self, summary: Dict[str, Any]) -> float:
        try:
            v = self.getter(summary)
        except (KeyError, TypeError, IndexError, ZeroDivisionError):
            return 0.0
        return 0.0 if v is None else float(v)


@dataclass
class Table:
    """A comparison table: column headers plus the rows to show under them."""
    title: str
    headers: List[str]
    rows: List[List[str]]
    highlight_row: int = -1                    # index rendered as the reference
    note: str = ""


@dataclass
class Panel:
    """
    One card in the Comparison tab.

    A titled block of label/value rows.  A row whose label and value are both
    empty renders as a horizontal rule, which is how a card separates its
    accuracy half from its cost half.
    """
    title: str
    subtitle: str = ""
    rows: List[Tuple[str, str]] = field(default_factory=list)
    accent: str = ""                           # header colour; "" => COL_ACCENT
    footer: str = ""


@dataclass
class PanelSet:
    """A Comparison tab rendered as cards rather than a grid of numbers."""
    title: str
    panels: List[Panel]
    note: str = ""
    columns: int = 2


@dataclass
class Study:
    """
    One runnable workload: what to execute, what to plot, which parameters it
    consumes and how its results are presented.
    """
    key: str
    label: str                                 # shown in the study selector
    subtitle: str                              # header line under the app title
    run_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    fig_builder: Callable[..., Any]
    run_defaults: Dict[str, Any]
    # Either a fixed list, or summary -> list: a study whose headline
    # metrics depend on what the run produced (the classifier study shows
    # one set of tiles per network it mapped) passes a callable.
    kpis: Any
    figure_path: str
    report_path: str
    summary_path: str
    report_title: str
    extra_groups: Sequence[GroupSpec] = ()
    exclude_keys: Sequence[str] = ()
    # summary -> the comparison table shown in the GUI and the HTML report
    table_fn: Optional[Callable[[Dict[str, Any]], Optional[Table]]] = None
    # summary -> comparison CARDS for the GUI tab.  When this returns a
    # PanelSet the Comparison tab shows it instead of the table; the HTML
    # report keeps using table_fn either way, so nothing is lost from it.
    panels_fn: Optional[Callable[[Dict[str, Any]], Optional["PanelSet"]]] = None
    # (summary, elapsed_s, run_count) -> the one-line note under the KPI cards
    note_fn: Optional[Callable[[Dict[str, Any], float, int], str]] = None
    # summary -> extra HTML sections appended to the report
    sections_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    # summary -> the sub-title line of the HTML report header
    report_subtitle_fn: Optional[Callable[[Dict[str, Any]], str]] = None

    def kpi_list(self, s: Dict[str, Any]) -> List[Kpi]:
        """The KPI cards for one summary (resolving a callable `kpis`)."""
        if callable(self.kpis):
            try:
                return list(self.kpis(s))
            except Exception:
                return []
        return list(self.kpis)

    def build_groups(self) -> List[Tuple[str, str, List[Param]]]:
        return build_registry(self.run_defaults, self.extra_groups,
                              self.exclude_keys)


# Value formatting / parsing

def _fmt_num(x: float) -> str:
    if x == 0:
        return "0"
    if x != x or abs(x) == float("inf"):
        return str(x)
    ax = abs(x)
    if ax >= 1e5 or ax < 1e-3:
        return f"{x:.6g}"
    return f"{x:g}"


def _fmt_value(p: Param, v: Any) -> str:
    """Round-trippable text form of a parameter value (used in entry fields)."""
    if p.kind == "bool":
        return "on" if v else "off"
    if p.kind == "float":
        return _fmt_num(float(v))
    if p.kind == "float_list":
        return ", ".join(_fmt_num(float(x)) for x in v)
    if p.kind == "multichoice":
        return ", ".join(str(x) for x in v)
    return str(v)


def _parse_value(p: Param, s: str) -> Any:
    s = str(s).strip()
    if not s:
        raise ValueError("empty value")
    if p.kind == "int":
        v = int(float(s))
        if p.choices and v not in p.choices:
            raise ValueError(f"must be one of {p.choices}")
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v
    if p.kind == "float":
        return float(s)
    if p.kind == "bool":
        low = s.lower()
        if low in ("1", "y", "yes", "true", "on"):
            return True
        if low in ("0", "n", "no", "false", "off"):
            return False
        raise ValueError("use on/off")
    if p.kind == "choice":
        if s in [str(c) for c in p.choices]:
            return s
        raise ValueError(f"choose one of: {', '.join(map(str, p.choices))}")
    if p.kind == "float_list":
        vals = [float(t) for t in s.replace(",", " ").split()]
        if not vals:
            raise ValueError("give at least one number")
        return vals
    if p.kind == "multichoice":
        picked = [t for t in s.replace(",", " ").split()]
        allowed = [str(c) for c in p.choices]
        bad = [t for t in picked if t not in allowed]
        if bad:
            raise ValueError(f"unknown: {', '.join(bad)} "
                             f"(choose from {', '.join(allowed)})")
        if not picked:
            raise ValueError("select at least one")
        # canonical order, no duplicates
        return [c for c in allowed if c in picked]
    if p.kind == "path":
        # Free text, checked here rather than failing deep inside a run.  May be a
        # list of directories separated by ';' or ',': the two backbones come out
        # of separate GPU runs, and the first folder holding a file wins.
        import os as _os
        parts = [t.strip() for t in s.replace(",", ";").split(";") if t.strip()]
        if not parts:
            raise ValueError("enter at least one directory")
        found = [t for t in parts if _os.path.isdir(_os.path.expanduser(t))]
        if not found:
            raise ValueError("no such director" + ("y: " if len(parts) == 1
                                                   else "ies: ") + ", ".join(parts))
        # Some listed folders may legitimately not exist yet - the run that
        # produces them may not have happened. Accept as long as one does; the
        # study logs by name whichever weights it could not find.
        return s
    return s


# The tkinter platform

class _QueueWriter:
    """File-like stdout replacement that forwards text to a queue."""

    def __init__(self, q: "queue.Queue[str]"):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:
        pass


class Platform:
    """
    Tkinter GUI around one or more studies.

    Each Study supplies:
        run_fn(run_settings: dict) -> summary dict   (prints its engineering log)
        fig_builder(summary, fig) -> fig             (draws the dashboard)
    plus the parameters it consumes, its KPI cards and its report sections.
    """

    def __init__(
        self,
        studies: Sequence[Study],
        initial: Optional[str] = None,
    ):
        if not studies:
            raise ValueError("the platform needs at least one study")
        self.studies: List[Study] = list(studies)
        self._by_key = {s.key: s for s in self.studies}
        self.study: Study = self._by_key.get(initial or "", self.studies[0])

        self.run_count = 0
        # Per-study state: previous KPIs (for the deltas) and the last summary.
        self.prev_kpis: Dict[str, Dict[str, float]] = {}
        self.last_summary: Optional[Dict[str, Any]] = None
        self._summaries: Dict[str, Dict[str, Any]] = {}

        self._running = False
        self._t0 = 0.0
        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._result_q: "queue.Queue[tuple]" = queue.Queue()

        # UI handles (filled by _init_ui)
        self.root: Optional[tk.Tk] = None
        self._vars: Dict[str, tk.Variable] = {}
        self._labels: Dict[str, ttk.Label] = {}
        self._rowwidgets: Dict[str, list] = {}   # key -> [label, input, unit, help]
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._config_nb = None

        self._load_study_params()

    # Study state

    def _load_study_params(self) -> None:
        """(Re)build the parameter registry for the currently selected study."""
        self.groups = self.study.build_groups()
        self.params: Dict[str, Param] = {
            p.key: p for _, _, plist in self.groups for p in plist}
        # Params whose value controls the visibility of other rows.
        self._controllers = {p.visible_if[0] for p in self.params.values()
                             if p.visible_if}
        self._last_values: Dict[str, Any] = {
            p.key: p.default for p in self.params.values()}

    @property
    def title(self) -> str:
        return self.study.subtitle

    # UI construction

    def launch(self) -> None:
        self._init_ui()
        self.root.mainloop()

    def _init_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("STRATUM - Memristor PIM Simulation Studio")
        self.root.geometry("1380x880")
        self.root.configure(bg=COL_BG)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=COL_BG, font=("Segoe UI", 9))
        style.configure("TNotebook", background=COL_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 5), font=("Segoe UI", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", COL_CARD)],
                  foreground=[("selected", COL_ACCENT)])
        style.configure("Card.TFrame", background=COL_CARD)
        style.configure("Help.TLabel", foreground=COL_MUTED,
                        font=("Segoe UI", 8))
        style.configure("Unit.TLabel", foreground=COL_MUTED,
                        font=("Segoe UI", 8, "italic"))
        style.configure("GroupTitle.TLabel", foreground=COL_ACCENT,
                        font=("Segoe UI", 11, "bold"))

        self._build_header()

        body = tk.PanedWindow(self.root, orient="horizontal", sashwidth=5,
                              bg=COL_BG, bd=0)
        body.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        left = self._build_config_panel(body)
        right = self._build_results_panel(body)
        body.add(left, minsize=430)
        body.add(right, minsize=640)

        self._status = tk.Label(
            self.root, anchor="w", bg=COL_HEADER, fg="#dbe6f2",
            font=("Segoe UI", 9), padx=10,
            text="Ready. Edit any parameter, then press Run simulation.")
        self._status.pack(fill="x", side="bottom", pady=(6, 0))

        self._apply_conditional_visibility()     # hide MRAM rows unless spintronic
        self.root.after(100, self._poll)

    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=COL_HEADER)
        hdr.pack(fill="x")

        top = tk.Frame(hdr, bg=COL_HEADER)
        top.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(top, text="STRATUM", bg=COL_HEADER, fg="white",
                 font=("Segoe UI", 17, "bold")).pack(side="left")
        tk.Label(top, text="developed by IMChip B.V.", bg=COL_HEADER,
                 fg="#a9c0d8", font=("Segoe UI", 10)).pack(side="left",
                                                           padx=(8, 0), pady=(7, 0))

        # Study selector: which workload gets deployed on the chip.
        sel = tk.Frame(top, bg=COL_HEADER)
        sel.pack(side="right")
        tk.Label(sel, text="STUDY", bg=COL_HEADER, fg="#a9c0d8",
                 font=("Segoe UI", 8, "bold")).pack(anchor="e")
        self._study_var = tk.StringVar(value=self.study.label)
        self._study_cb = ttk.Combobox(
            sel, textvariable=self._study_var, state="readonly", width=42,
            font=("Segoe UI", 9),
            values=[s.label for s in self.studies])
        self._study_cb.pack()
        self._study_cb.bind("<<ComboboxSelected>>",
                            lambda _e: self._on_study_change())

        self._subtitle = tk.Label(
            hdr, text="Memristor PIM Simulation Studio   -   " + self.title,
            bg=COL_HEADER, fg="#a9c0d8", font=("Segoe UI", 10))
        self._subtitle.pack(anchor="w", padx=16, pady=(0, 10))

    def _on_study_change(self) -> None:
        """Switch to the study picked in the header selector."""
        if self._running:                     # never swap mid-run
            self._study_var.set(self.study.label)
            self._set_status("A simulation is running - wait for it to finish "
                             "before switching study.")
            return
        label = self._study_var.get()
        study = next((s for s in self.studies if s.label == label), None)
        if study is None or study is self.study:
            return
        self.study = study
        self._load_study_params()
        self._subtitle.configure(
            text="Memristor PIM Simulation Studio   -   " + self.title)
        self._populate_config_tabs()
        self._apply_conditional_visibility()
        self._show_last_results()
        self._set_status(f"Study: {study.label}. Edit any parameter, then press "
                         "Run simulation.")

    # ---- left: configuration -----------------------------------------

    def _build_config_panel(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=COL_BG)

        tk.Label(panel, text="CONFIGURATION", bg=COL_BG, fg=COL_ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w",
                                                     padx=4, pady=(2, 4))
        nb = ttk.Notebook(panel)
        nb.pack(fill="both", expand=True)
        self._config_nb = nb
        self._populate_config_tabs()

        # Profile / reset toolbar.
        bar = tk.Frame(panel, bg=COL_BG)
        bar.pack(fill="x", pady=(8, 4))
        ttk.Button(bar, text="Save profile...",
                   command=self._save_profile).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Load profile...",
                   command=self._load_profile).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Reset to defaults",
                   command=self._reset_defaults).pack(side="left")

        # Report options.
        opt = tk.Frame(panel, bg=COL_BG)
        opt.pack(fill="x", pady=(0, 4))
        self._auto_open_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Open HTML report after each run",
                        variable=self._auto_open_var).pack(side="left")
        ttk.Button(opt, text="Open last report",
                   command=self._open_report).pack(side="right")

        # Run button + progress.
        self._run_btn = tk.Button(
            panel, text="▶   Run simulation", command=self._start_run,
            bg=COL_GOOD, fg="white", activebackground="#257247",
            activeforeground="white", font=("Segoe UI", 12, "bold"),
            relief="flat", pady=8, cursor="hand2")
        self._run_btn.pack(fill="x", pady=(4, 4))
        self._progress = ttk.Progressbar(panel, mode="indeterminate")
        self._progress.pack(fill="x", pady=(0, 6))
        return panel

    def _populate_config_tabs(self) -> None:
        """(Re)build the configuration notebook for the current study."""
        nb = self._config_nb
        for tab in nb.tabs():                      # clear the previous study
            nb.forget(tab)
            nb.nametowidget(tab).destroy()
        self._vars.clear()
        self._labels.clear()
        self._rowwidgets.clear()
        for tab_name, group_title, plist in self.groups:
            outer = tk.Frame(nb, bg=COL_CARD)
            nb.add(outer, text=tab_name)
            inner = self._scrollable(outer)        # tab content scrolls if tall
            self._build_group_tab(inner, group_title, plist)

    def _build_group_tab(self, frame: tk.Frame, group_title: str,
                         plist: List[Param]) -> None:
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=group_title, style="GroupTitle.TLabel",
                  background=COL_CARD).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 6))
        row = 1
        for p in plist:
            lbl = ttk.Label(frame, text=p.label, background=COL_CARD,
                            font=("Segoe UI", 9))
            lbl.grid(row=row, column=0, sticky="w", padx=(12, 8), pady=(4, 0))
            self._labels[p.key] = lbl

            if p.kind == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(p.default))
                w: tk.Widget = ttk.Checkbutton(frame, variable=var)
            elif p.kind == "multichoice":
                # A row of checkbuttons that together drive one list-valued
                # StringVar (e.g. which backbones take part in the run).
                var = tk.StringVar(value=_fmt_value(p, p.default))
                w = self._multichoice_widget(frame, p, var)
            elif p.kind == "choice" or (p.kind == "int" and p.choices):
                var = tk.StringVar(value=str(p.default))
                w = ttk.Combobox(frame, textvariable=var, state="readonly",
                                 width=20,
                                 values=[str(c) for c in p.choices])
            else:
                var = tk.StringVar(value=_fmt_value(p, p.default))
                w = ttk.Entry(frame, textvariable=var, width=26)
            w.grid(row=row, column=1, sticky="we", pady=(4, 0))
            unit_lbl = ttk.Label(frame, text=p.unit, style="Unit.TLabel",
                                 background=COL_CARD)
            unit_lbl.grid(row=row, column=2, sticky="w", padx=(6, 12))
            self._vars[p.key] = var
            var.trace_add("write",
                          lambda *_a, key=p.key: self._on_var_change(key))

            help_lbl = ttk.Label(frame, text=p.help, style="Help.TLabel",
                                 background=COL_CARD, wraplength=380,
                                 justify="left")
            help_lbl.grid(row=row + 1, column=0, columnspan=3, sticky="w",
                          padx=(12, 12), pady=(0, 2))
            # Remember every grid-managed widget of this parameter so rows can be
            # shown/hidden as a unit (conditional MRAM parameters).
            self._rowwidgets[p.key] = [lbl, w, unit_lbl, help_lbl]
            row += 2

    def _multichoice_widget(self, frame: tk.Frame, p: Param,
                            var: tk.StringVar) -> tk.Widget:
        """
        One checkbutton per allowed value; ticking any of them rewrites `var`
        with the selected values in the parameter's canonical order.  Editing
        `var` from elsewhere (profile load, reset) re-ticks the boxes.
        """
        box = tk.Frame(frame, bg=COL_CARD)
        picked = {str(c): tk.BooleanVar(value=str(c) in
                                        [str(d) for d in p.default])
                  for c in p.choices}
        syncing = {"on": False}

        def push(*_a):
            if syncing["on"]:
                return
            sel = [c for c in picked if picked[c].get()]
            var.set(", ".join(str(c) for c in p.choices if str(c) in sel))

        def pull(*_a):
            syncing["on"] = True
            try:
                cur = {t for t in str(var.get()).replace(",", " ").split()}
                for c, bv in picked.items():
                    if bv.get() != (c in cur):
                        bv.set(c in cur)
            finally:
                syncing["on"] = False

        for c, bv in picked.items():
            ttk.Checkbutton(box, text=str(c),
                            variable=bv).pack(side="left", padx=(0, 8))
            bv.trace_add("write", push)
        var.trace_add("write", pull)
        return box

    def _scrollable(self, parent) -> tk.Frame:
        """Wrap `parent` in a vertically scrollable canvas; return the inner
        frame that content should be built into."""
        canvas = tk.Canvas(parent, bg=COL_CARD, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=COL_CARD)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        def _on_inner(_e):                     # content resized -> update region
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas(e):                     # stretch inner to canvas width
            canvas.itemconfigure(win, width=e.width)

        inner.bind("<Configure>", _on_inner)
        canvas.bind("<Configure>", _on_canvas)

        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")

        # Scope the wheel to when the pointer is over this canvas.
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return inner

    def _apply_conditional_visibility(self) -> None:
        """Show or hide rows whose `visible_if` controller does not currently
        match (e.g. the spintronic MTJ parameters inside the Sign / 2T1R tab)."""
        for key, p in self.params.items():
            if not p.visible_if:
                continue
            ctrl_key, allowed = p.visible_if
            ctrl = self._vars.get(ctrl_key)
            show = ctrl is not None and str(ctrl.get()) in [str(a) for a in allowed]
            for w in self._rowwidgets.get(key, []):
                if show:
                    w.grid()
                else:
                    w.grid_remove()

    # ---- right: results -------------------------------------------------

    def _build_results_panel(self, parent) -> tk.Frame:
        panel = tk.Frame(parent, bg=COL_BG)
        nb = ttk.Notebook(panel)
        nb.pack(fill="both", expand=True)
        self._results_nb = nb

        # Results tab: KPI cards + dashboard figure.
        res = tk.Frame(nb, bg=COL_BG)
        nb.add(res, text="Results")
        self._tab_results = res
        self._kpi_frame = tk.Frame(res, bg=COL_BG)
        self._kpi_frame.pack(fill="x", padx=6, pady=(8, 2))
        self._kpi_note = tk.Label(res, text="", bg=COL_BG, fg=COL_MUTED,
                                  font=("Segoe UI", 9), anchor="w",
                                  justify="left")
        self._kpi_note.pack(fill="x", padx=10)
        self._fig_frame = tk.Frame(res, bg=COL_BG)
        self._fig_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self._fig_placeholder = tk.Label(
            self._fig_frame,
            text="Run a simulation to see the results dashboard here.",
            bg=COL_BG, fg=COL_MUTED, font=("Segoe UI", 12))
        self._fig_placeholder.pack(expand=True)

        # Comparison tab: the study's side-by-side metric table.
        cmpf = tk.Frame(nb, bg=COL_BG)
        nb.add(cmpf, text="Comparison")
        self._tab_compare = cmpf
        self._cmp_title = tk.Label(cmpf, text="", bg=COL_BG, fg=COL_ACCENT,
                                   font=("Segoe UI", 11, "bold"), anchor="w")
        self._cmp_title.pack(fill="x", padx=10, pady=(10, 2))
        self._cmp_note = tk.Label(cmpf, text="", bg=COL_BG, fg=COL_MUTED,
                                  font=("Segoe UI", 9), anchor="w",
                                  justify="left", wraplength=700)
        self._cmp_note.pack(fill="x", padx=10, pady=(0, 6))
        # Keep the explanatory note wrapped to whatever width the pane has.
        cmpf.bind("<Configure>",
                  lambda e: self._cmp_note.configure(
                      wraplength=max(320, e.width - 30)))
        self._cmp_holder = tk.Frame(cmpf, bg=COL_BG)
        self._cmp_holder.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self._cmp_tree = None
        self._cmp_panels = None
        self._cmp_placeholder = tk.Label(
            self._cmp_holder,
            text="Run a simulation to see the side-by-side comparison here.",
            bg=COL_BG, fg=COL_MUTED, font=("Segoe UI", 12))
        self._cmp_placeholder.pack(expand=True)

        # Log tab.
        logf = tk.Frame(nb, bg=COL_BG)
        nb.add(logf, text="Simulation log")
        self._tab_log = logf
        self._log = scrolledtext.ScrolledText(
            logf, state="disabled", bg=COL_LOG_BG, fg=COL_LOG_FG,
            insertbackground=COL_LOG_FG, font=("Consolas", 9),
            relief="flat", padx=10, pady=8, wrap="none")
        self._log.pack(fill="both", expand=True, padx=4, pady=4)
        return panel

    # ---- comparison table ------------------------------------------------

    def _update_comparison(self, summary: Optional[Dict[str, Any]]) -> None:
        """
        Fill the Comparison tab: the study's panels when it defines them, its
        table otherwise.
        """
        panels = None
        if summary is not None and self.study.panels_fn is not None:
            try:
                panels = self.study.panels_fn(summary)
            except Exception:
                self._append_log("\n! comparison panels failed:\n"
                                 + traceback.format_exc() + "\n")
        table = None
        if (panels is None and summary is not None
                and self.study.table_fn is not None):
            try:
                table = self.study.table_fn(summary)
            except Exception:
                self._append_log("\n! comparison table failed:\n"
                                 + traceback.format_exc() + "\n")
        if self._cmp_tree is not None:
            self._cmp_tree.master.destroy()
            self._cmp_tree = None
        if self._cmp_panels is not None:
            self._cmp_panels.destroy()
            self._cmp_panels = None
        if self._cmp_placeholder is not None:
            self._cmp_placeholder.destroy()
            self._cmp_placeholder = None

        if panels is not None:
            self._cmp_title.configure(text=panels.title)
            self._cmp_note.configure(text=panels.note)
            self._render_panels(panels)
            return

        if table is None:
            self._cmp_title.configure(text="")
            self._cmp_note.configure(text="")
            self._cmp_placeholder = tk.Label(
                self._cmp_holder,
                text=("This study has no side-by-side comparison - "
                      "see the Results tab."),
                bg=COL_BG, fg=COL_MUTED, font=("Segoe UI", 12))
            self._cmp_placeholder.pack(expand=True)
            return

        self._cmp_title.configure(text=table.title)
        self._cmp_note.configure(text=table.note)
        wrap = tk.Frame(self._cmp_holder, bg=COL_BG)
        wrap.pack(fill="both", expand=True)
        tree = ttk.Treeview(wrap, columns=list(range(len(table.headers))),
                            show="headings", height=min(len(table.rows) + 1, 14))
        for i, h in enumerate(table.headers):
            tree.heading(i, text=h)
            # Narrow enough that a wide comparison mostly fits; every column
            # stays draggable, and the pane scrolls horizontally if it doesn't.
            tree.column(i, anchor="e" if i else "w",
                        width=118 if i == 0 else 86,
                        minwidth=60 if i else 84, stretch=True)
        tree.tag_configure("ref", background="#e8f0fb", font=("Segoe UI", 9, "bold"))
        for r, row in enumerate(table.rows):
            tree.insert("", "end", values=row,
                        tags=("ref",) if r == table.highlight_row else ())
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="we")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self._cmp_tree = tree

    def _render_panels(self, ps: PanelSet) -> None:
        """The Comparison tab as cards: one titled block per panel."""
        wrap = tk.Frame(self._cmp_holder, bg=COL_BG)
        wrap.pack(fill="both", expand=True)
        canvas = tk.Canvas(wrap, bg=COL_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        inner = tk.Frame(canvas, bg=COL_BG)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Keep the cards as wide as the pane so they share its width evenly.
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))

        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")

        def _bind_wheel(w):
            w.bind("<MouseWheel>", _wheel)
            for child in w.winfo_children():
                _bind_wheel(child)

        cols = max(1, int(ps.columns))
        for c in range(cols):
            inner.columnconfigure(c, weight=1, uniform="cmp")
        for i, p in enumerate(ps.panels):
            card = tk.Frame(inner, bg=COL_CARD, highlightbackground="#d7dee6",
                            highlightthickness=1)
            card.grid(row=i // cols, column=i % cols, sticky="nsew",
                      padx=5, pady=5)
            accent = p.accent or COL_ACCENT
            tk.Label(card, text=p.title, bg=COL_CARD, fg=accent, anchor="w",
                     font=("Segoe UI", 12, "bold")).pack(
                fill="x", padx=12, pady=(9, 0))
            if p.subtitle:
                tk.Label(card, text=p.subtitle, bg=COL_CARD, fg=COL_MUTED,
                         anchor="w", justify="left",
                         font=("Segoe UI", 8)).pack(fill="x", padx=12,
                                                    pady=(1, 4))
            for label, value in p.rows:
                if not label and not value:
                    tk.Frame(card, bg="#e6ebf0", height=1).pack(
                        fill="x", padx=12, pady=5)
                    continue
                row = tk.Frame(card, bg=COL_CARD)
                row.pack(fill="x", padx=12, pady=1)
                tk.Label(row, text=label, bg=COL_CARD, fg=COL_MUTED,
                         anchor="w", font=("Segoe UI", 9)).pack(side="left")
                tk.Label(row, text=value, bg=COL_CARD, fg=COL_HEADER,
                         anchor="e", font=("Segoe UI", 10, "bold")).pack(
                    side="right")
            tk.Label(card, text=p.footer, bg=COL_CARD, fg=accent, anchor="w",
                     font=("Segoe UI", 8, "bold")).pack(
                fill="x", padx=12, pady=(6, 9))
        _bind_wheel(inner)
        canvas.bind("<MouseWheel>", _wheel)
        self._cmp_panels = wrap

    # Parameter state

    def _on_var_change(self, key: str) -> None:
        p = self.params[key]
        lbl = self._labels.get(key)
        if lbl is None:
            return
        try:
            v = self._read_var(p)
        except ValueError:
            lbl.configure(foreground=COL_BAD)
            return
        lbl.configure(foreground=COL_MOD if p.is_modified(v) else "black")
        # A controller (e.g. SIGN_STORAGE_MODE) changed -> refresh which rows show.
        if key in self._controllers:
            self._apply_conditional_visibility()

    def _read_var(self, p: Param) -> Any:
        var = self._vars[p.key]
        if p.kind == "bool":
            return bool(var.get())
        return _parse_value(p, str(var.get()))

    def _set_var(self, p: Param, value: Any) -> None:
        var = self._vars[p.key]
        if p.kind == "bool":
            var.set(bool(value))
        else:
            var.set(_fmt_value(p, value))

    def _collect_values(self) -> Dict[str, Any]:
        values, errors = {}, []
        for p in self.params.values():
            try:
                values[p.key] = self._read_var(p)
            except ValueError as e:
                errors.append(f"{p.label}: {e}")
        if errors:
            raise ValueError("\n".join(errors))
        return values

    def _reset_defaults(self) -> None:
        for p in self.params.values():
            self._set_var(p, p.default)
        self._set_status("All parameters reset to defaults.")

    # Profiles

    def _save_profile(self) -> None:
        try:
            values = self._collect_values()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e), parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save configuration profile",
            initialfile=PROFILE_PATH, defaultextension=".json",
            filetypes=[("JSON profile", "*.json")])
        if not path:
            return
        with open(path, "w") as f:
            json.dump(values, f, indent=2)
        self._set_status(f"Saved {len(values)} parameters -> {path}")

    def _load_profile(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Load configuration profile",
            filetypes=[("JSON profile", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Profile error",
                                 f"Could not read profile:\n{e}",
                                 parent=self.root)
            return
        n = 0
        for k, v in data.items():
            if k in self.params:
                self._set_var(self.params[k], v)
                n += 1
        self._set_status(f"Loaded {n} parameters from {path}")

    # Run orchestration

    def _set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def _append_log(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _start_run(self) -> None:
        if self._running:
            return
        try:
            values = self._collect_values()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e), parent=self.root)
            return

        # Apply simulation_config overrides; build the run-settings dict.
        run_settings: Dict[str, Any] = {}
        for p in self.params.values():
            if p.target == "simcfg":
                setattr(simcfg, p.key, values[p.key])
            else:
                run_settings[p.key] = values[p.key]
        self._last_values = values

        mods = [p for p in self.params.values()
                if p.is_modified(values[p.key])]
        header = (f"\n{'='*78}\nRUN #{self.run_count + 1}  -  {self.study.label}"
                  f"  -  {time.strftime('%Y-%m-%d %H:%M:%S')}  -  "
                  f"{len(mods)} parameter(s) modified\n")
        for p in mods:
            header += (f"    {p.label}: {_fmt_value(p, p.default)} -> "
                       f"{_fmt_value(p, values[p.key])} {p.unit}\n")
        header += "=" * 78 + "\n"
        self._append_log(header)
        self._results_nb.select(self._tab_log)      # show the live log

        self._running = True
        self._t0 = time.time()
        self._run_btn.configure(state="disabled", text="Running...",
                                bg=COL_MUTED)
        self._progress.start(12)
        self._set_status("Simulation running - see the Simulation log tab.")

        threading.Thread(target=self._worker, args=(run_settings,),
                         daemon=True).start()

    def _worker(self, run_settings: Dict[str, Any]) -> None:
        old_out = sys.stdout
        sys.stdout = _QueueWriter(self._log_q)
        try:
            summary = self.study.run_fn(run_settings)
            self._result_q.put(("ok", summary))
        except Exception:
            self._result_q.put(("err", traceback.format_exc()))
        finally:
            sys.stdout = old_out

    def _poll(self) -> None:
        # Stream the engineering log into the GUI.
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        # Handle a finished run.
        try:
            status, payload = self._result_q.get_nowait()
        except queue.Empty:
            pass
        else:
            self._progress.stop()
            self._running = False
            self._run_btn.configure(state="normal",
                                    text="▶   Run simulation",
                                    bg=COL_GOOD)
            if status == "ok":
                self._finish_run(payload)
            else:
                self._append_log("\n! SIMULATION FAILED\n" + payload + "\n")
                self._set_status("Simulation failed - see the log for the "
                                 "traceback.")
                messagebox.showerror(
                    "Simulation failed",
                    "The run raised an error - the full traceback is in the "
                    "Simulation log tab.\nAdjust the configuration and try "
                    "again.", parent=self.root)
        self.root.after(100, self._poll)

    def _finish_run(self, summary: Dict[str, Any]) -> None:
        elapsed = time.time() - self._t0
        self.run_count += 1
        study = self.study

        # Dashboard figure (embedded + saved to PNG for the HTML report).
        try:
            fig = Figure(figsize=(13.8, 8.0), dpi=100)
            study.fig_builder(summary, fig)
            fig.savefig(study.figure_path, dpi=150, bbox_inches="tight")
            self._embed_figure(fig)
        except Exception:
            self._append_log("\n! plotting failed:\n"
                             + traceback.format_exc() + "\n")

        self._update_kpis(summary, elapsed)
        self._update_comparison(summary)

        # HTML report.
        try:
            generate_html_report(summary, self, study.figure_path,
                                 study.report_path)
            if self._auto_open_var.get():
                self._open_report()
        except Exception:
            self._append_log("\n! HTML report generation failed:\n"
                             + traceback.format_exc() + "\n")

        self.last_summary = summary
        self._summaries[study.key] = summary
        self.prev_kpis[study.key] = self._kpis(summary)
        self._results_nb.select(self._tab_results)   # jump to the visual results
        self._set_status(
            f"Run #{self.run_count} ({study.label}) complete in {elapsed:.1f} s"
            f"  -  results saved to {study.figure_path}, {study.report_path}, "
            f"{study.summary_path}")

    def _show_last_results(self) -> None:
        """Re-present whatever the newly selected study last produced."""
        summary = self._summaries.get(self.study.key)
        self.last_summary = summary
        if summary is None:
            for w in self._kpi_frame.winfo_children():
                w.destroy()
            self._kpi_note.configure(
                text=f"{self.study.label}: no run yet - press Run simulation.")
            self._clear_figure()
            self._update_comparison(None)
            return
        try:
            fig = Figure(figsize=(13.8, 8.0), dpi=100)
            self.study.fig_builder(summary, fig)
            self._embed_figure(fig)
        except Exception:
            self._append_log("\n! plotting failed:\n"
                             + traceback.format_exc() + "\n")
        self._update_kpis(summary, 0.0)
        self._update_comparison(summary)

    def _clear_figure(self) -> None:
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._fig_placeholder is None:
            self._fig_placeholder = tk.Label(
                self._fig_frame,
                text="Run a simulation to see the results dashboard here.",
                bg=COL_BG, fg=COL_MUTED, font=("Segoe UI", 12))
            self._fig_placeholder.pack(expand=True)

    def _embed_figure(self, fig) -> None:
        if self._fig_placeholder is not None:
            self._fig_placeholder.destroy()
            self._fig_placeholder = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self._fig_frame)
        self._canvas.draw()
        self._toolbar = NavigationToolbar2Tk(self._canvas, self._fig_frame,
                                             pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side="bottom", fill="x")
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _open_report(self) -> None:
        path = self.study.report_path
        if os.path.exists(path):
            webbrowser.open("file:///"
                            + os.path.abspath(path).replace("\\", "/"))
        else:
            self._set_status(f"No report for {self.study.label} yet - "
                             "run a simulation first.")

    # KPI cards

    def _kpis(self, s: Dict[str, Any]) -> Dict[str, float]:
        """The current study's headline metrics, read out of a run summary."""
        return {k.key: k.value(s) for k in self.study.kpi_list(s)}

    def _update_kpis(self, s: Dict[str, Any], elapsed: float) -> None:
        for w in self._kpi_frame.winfo_children():
            w.destroy()
        k = self._kpis(s)
        prev = self.prev_kpis.get(self.study.key)
        cols = 4
        for c in range(cols):
            self._kpi_frame.columnconfigure(c, weight=1, uniform="kpi")
        for i, kpi in enumerate(self.study.kpi_list(s)):
            card = tk.Frame(self._kpi_frame, bg=COL_CARD,
                            highlightbackground="#d7dee6",
                            highlightthickness=1)
            card.grid(row=i // cols, column=i % cols, sticky="nsew",
                      padx=3, pady=3)
            val_txt = kpi.fmt.format(k[kpi.key]) + (f" {kpi.unit}"
                                                    if kpi.unit else "")
            tk.Label(card, text=val_txt, bg=COL_CARD, fg=COL_ACCENT,
                     font=("Segoe UI", 13, "bold")).pack(anchor="w",
                                                         padx=10, pady=(6, 0))
            delta_txt, delta_col = "", COL_MUTED
            if prev is not None and kpi.key in prev:
                d = k[kpi.key] - prev[kpi.key]
                if abs(d) > 1e-12:
                    sign = "+" if d >= 0 else ""
                    delta_txt = f"{sign}{kpi.fmt.format(d)} vs prev"
                    if kpi.higher_is_better is not None:
                        delta_col = (COL_GOOD if (d > 0) == kpi.higher_is_better
                                     else COL_BAD)
            row2 = tk.Frame(card, bg=COL_CARD)
            row2.pack(anchor="w", fill="x", padx=10, pady=(0, 6))
            tk.Label(row2, text=kpi.label.upper(), bg=COL_CARD, fg=COL_MUTED,
                     font=("Segoe UI", 7)).pack(side="left")
            if delta_txt:
                tk.Label(row2, text="  " + delta_txt, bg=COL_CARD,
                         fg=delta_col, font=("Segoe UI", 7, "bold")).pack(
                    side="left")

        note = ""
        if self.study.note_fn is not None:
            try:
                note = self.study.note_fn(s, elapsed, self.run_count)
            except Exception:
                note = ""
        self._kpi_note.configure(text=note)


# HTML report generation

_CSS = """
:root { --accent:#3d6bb3; --good:#2e8b57; --bad:#c44e52; --ink:#22303c;
        --muted:#6b7a88; --bg:#f2f5f8; --card:#ffffff; }
* { box-sizing: border-box; }
body { margin:0; font-family: 'Segoe UI', system-ui, sans-serif;
       background:var(--bg); color:var(--ink); }
.header { background:linear-gradient(120deg,#16324f,#3d6bb3); color:#fff;
          padding:28px 40px; }
.header h1 { margin:0 0 6px 0; font-size:24px; letter-spacing:0.5px; }
.header h1 .by { font-size:14px; font-weight:400; opacity:0.85; letter-spacing:0.3px; }
.header .sub { opacity:0.85; font-size:13px; }
.wrap { max-width:1200px; margin:0 auto; padding:24px 40px 60px; }
h2 { font-size:16px; text-transform:uppercase; letter-spacing:1.2px;
     color:var(--accent); border-bottom:2px solid var(--accent);
     padding-bottom:6px; margin:36px 0 16px; }
.cards { display:flex; flex-wrap:wrap; gap:14px; }
.card { background:var(--card); border-radius:10px; padding:16px 20px;
        min-width:170px; flex:1; box-shadow:0 1px 4px rgba(20,40,70,.12); }
.card .v { font-size:26px; font-weight:600; color:var(--accent); }
.card .l { font-size:12px; color:var(--muted); margin-top:4px;
           text-transform:uppercase; letter-spacing:.8px; }
.card .u { font-size:13px; color:var(--muted); font-weight:400; }
/* Wide comparisons scroll inside their own box, never the whole page. */
.tablewrap { overflow-x:auto; border-radius:10px;
             box-shadow:0 1px 4px rgba(20,40,70,.12); }
table { border-collapse:collapse; width:100%; background:var(--card);
        border-radius:10px; overflow:hidden; font-size:13.5px; }
th { background:#e6ecf3; text-align:left; padding:9px 14px; font-size:12px;
     text-transform:uppercase; letter-spacing:.6px; color:#3b4a5a;
     white-space:nowrap; }
td { padding:8px 14px; border-top:1px solid #edf1f5; }
/* Numbers never wrap, so a metric column keeps one row per record ... */
td.num { white-space:nowrap; }
/* ... except in the configuration snapshot, whose "value" column holds long
   lists (the conductance levels, the retention time points). */
table.cfg td.num { white-space:normal; word-break:break-word; }
tr:hover td { background:#f7fafc; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.imp { color:var(--good); font-weight:600; }
.mod { background:#fff7df !important; }
.badge { display:inline-block; background:#f2c94c; color:#5a4500;
         font-size:10px; padding:1px 7px; border-radius:8px; margin-left:8px;
         vertical-align:middle; font-weight:600; }
.fig { background:var(--card); border-radius:10px; padding:14px;
       box-shadow:0 1px 4px rgba(20,40,70,.12); }
.fig img { width:100%; height:auto; border-radius:6px; }
pre.prog { background:#16232f; color:#c7d5e0; padding:16px 20px;
           border-radius:10px; overflow-x:auto; font-size:12.5px;
           line-height:1.55; }
.foot { margin-top:40px; color:var(--muted); font-size:12px;
        text-align:center; }
"""


def _kpi_card(value: str, label: str, unit: str = "") -> str:
    u = f' <span class="u">{unit}</span>' if unit else ""
    return (f'<div class="card"><div class="v">{value}{u}</div>'
            f'<div class="l">{label}</div></div>')


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]],
           num_cols: Sequence[int] = ()) -> str:
    th = "".join(
        f'<th class="num">{h}</th>' if i in num_cols else f"<th>{h}</th>"
        for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(
            f'<td class="num">{c}</td>' if i in num_cols else f"<td>{c}</td>"
            for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _comparison_html(table: Optional[Table]) -> str:
    """Render a Study.Table (the side-by-side comparison) as an HTML section."""
    if table is None or not table.rows:
        return ""
    rows = []
    for i, r in enumerate(table.rows):
        cells = [_html.escape(str(c)) for c in r]
        if i == table.highlight_row:
            cells = [f'<span class="imp">{c}</span>' for c in cells]
        rows.append(cells)
    note = f"<p>{_html.escape(table.note)}</p>" if table.note else ""
    return (f"<h2>{_html.escape(table.title)}</h2>{note}"
            + _table(table.headers, rows,
                     num_cols=tuple(range(1, len(table.headers)))))


def commercial_html(s: Dict[str, Any], subject: str = "this design") -> str:
    """The peak-INT8 CPU/GPU context table (shared by every study)."""
    esc = _html.escape
    comp = s.get("commercial_comparison")
    if not comp:
        return ""
    td = comp["this_design"]
    rows = [[f'<span class="imp">{esc(td["name"])}</span>',
             f'<span class="imp">{esc(str(td.get("kind","")))}</span>', "-",
             f'<span class="imp">{td["tops_per_watt"]:.2f}</span>',
             '<span class="imp">1.0&times;</span>']]
    for r in sorted(comp["references"],
                    key=lambda d: d["tops_per_watt"], reverse=True):
        rows.append([esc(r["name"]), esc(r["kind"]), f"{r['power_w']:.0f}",
                     f"{r['tops_per_watt']:.2f}",
                     f"{r['efficiency_ratio']:.1f}&times;"])
    return (
        "<h2>Energy efficiency vs commercial CPUs / GPUs</h2>"
        "<p>Peak-INT8 datasheet figures for familiar parts, shown for "
        "context only (they are not run through the simulator). Absolute "
        f"TOPS differ because {esc(subject)} fills only a handful of the "
        "chip&rsquo;s arrays, not a full "
        "accelerator &mdash; the fair comparison is energy efficiency "
        "(TOPS/W). &lsquo;This vs it&rsquo; is how many times more "
        "energy-efficient this design is.</p>"
        + _table(["Device", "Class", "TDP (W)", "TOPS/W", "This vs it"],
                 rows, num_cols=(2, 3, 4)))


def generate_html_report(
    s: Dict[str, Any], platform: "Platform",
    png_path: Optional[str] = None, out_path: Optional[str] = None,
) -> str:
    """Render the run summary as a self-contained HTML dashboard."""
    esc = _html.escape
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    study = platform.study
    png_path = png_path or study.figure_path
    out_path = out_path or study.report_path

    # ---- KPI cards (the study's own headline metrics) -------------------
    cards = "".join(
        _kpi_card(k.fmt.format(k.value(s)), k.label, k.unit)
        for k in study.kpi_list(s))

    # ---- figure --------------------------------------------------------
    fig_html = "<p>(figure not available)</p>"
    if os.path.exists(png_path):
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        fig_html = (f'<div class="fig"><img alt="results dashboard" '
                    f'src="data:image/png;base64,{b64}"></div>')

    # ---- the study's comparison table + its own sections -----------------
    cmp_html = ""
    if study.table_fn is not None:
        try:
            cmp_html = _comparison_html(study.table_fn(s))
        except Exception:
            cmp_html = ""
    sections_html = ""
    if study.sections_fn is not None:
        try:
            sections_html = study.sections_fn(s)
        except Exception:
            sections_html = ""
    subtitle = (study.report_subtitle_fn(s)
                if study.report_subtitle_fn is not None else study.subtitle)

    # ---- configuration snapshot ----------------------------------------------
    cfg_rows = []
    for _tab, gname, plist in platform.groups:
        for p in plist:
            v = platform._last_values.get(p.key, p.default)
            mod = p.is_modified(v)
            badge = '<span class="badge">modified</span>' if mod else ""
            row = (f'<tr{" class=" + chr(34) + "mod" + chr(34) if mod else ""}>'
                   f"<td>{esc(gname)}</td>"
                   f"<td>{esc(p.label)}{badge}</td>"
                   f'<td class="num">{esc(_fmt_value(p, v))}</td>'
                   f"<td>{esc(p.unit)}</td>"
                   f'<td class="num">{esc(_fmt_value(p, p.default))}</td></tr>')
            cfg_rows.append(row)
    cfg_tbl = ('<div class="tablewrap"><table class="cfg"><thead><tr>'
               "<th>Group</th><th>Parameter</th>"
               '<th class="num">Value used</th><th>Unit</th>'
               '<th class="num">Default</th></tr></thead>'
               f"<tbody>{''.join(cfg_rows)}</tbody></table></div>")

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>STRATUM - {esc(study.report_title)}</title>
<style>{_CSS}</style></head>
<body>
<div class="header">
  <h1>STRATUM <span class="by">developed by IMChip B.V.</span> &mdash; {esc(study.report_title)}</h1>
  <div class="sub">{subtitle} &middot; generated {ts}</div>
</div>
<div class="wrap">
  <h2>Key performance indicators</h2>
  <div class="cards">{cards}</div>

  {cmp_html}

  <h2>Results dashboard</h2>
  {fig_html}

  {sections_html}

  <h2>Configuration snapshot</h2>
  {cfg_tbl}

  <div class="foot">STRATUM &middot; integrated memristor PIM toolkit
  &middot; {esc(study.label)} &middot; run #{platform.run_count}
  &middot; {ts}</div>
</div>
</body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path


# Study 1: MNIST MLP deployed as a RISC-V instruction program

_MLP_KPIS = [
    Kpi("hw_acc", "Hardware accuracy",
        lambda s: s["hardware_accuracy_4bit_adc"], "{:.4f}", "", True),
    Kpi("sw_acc", "Software accuracy",
        lambda s: s["software_accuracy"], "{:.4f}", "", True),
    Kpi("ideal_acc", "Ideal-datapath accuracy",
        lambda s: s["ideal_datapath_accuracy"], "{:.4f}", "", True),
    Kpi("e_pj", "Energy / inference",
        lambda s: s["energy_per_inference_j"] * 1e12, "{:.1f}", "pJ", False),
    Kpi("lat_ns", "Latency / inf (serial)",
        lambda s: s["latency_per_inference_ns"], "{:.1f}", "ns", False),
    Kpi("lat_par_ns", "Latency / inf (parallel)",
        lambda s: s.get("latency_parallel_ns", s["latency_per_inference_ns"]),
        "{:.1f}", "ns", False),
    Kpi("pow_mw", "Power (serial)", lambda s: s["power_w"] * 1e3, "{:.2f}", "mW", False),
    Kpi("tops", "Throughput (serial)", lambda s: s["tops"], "{:.4f}", "TOPS", True),
    # The chip reads a layer's independent tiles together; this is the number
    # that describes the machine rather than a single-issue sequencer.
    Kpi("tops_par", "Throughput (parallel)",
        lambda s: s.get("tops_parallel", s["tops"]), "{:.4f}", "TOPS", True),
    Kpi("par_speedup", "Parallel speed-up",
        lambda s: s.get("parallel_speedup", 1.0), "{:.2f}", "x", True),
    Kpi("tops_w", "Energy efficiency",
        lambda s: s["tops_per_watt"], "{:.2f}", "TOPS/W", True),
    Kpi("reads", "Analog reads / inf",
        lambda s: s["analog_reads_per_inference"], "{:.0f}", "", False),
    Kpi("arrays", "Crossbar arrays", lambda s: s["n_arrays"], "{:.0f}", "", None),
    Kpi("area_mm2", "Total array area",
        lambda s: s["total_area_um2"] / 1e6, "{:.4f}", "mm²", False),
    Kpi("instr", "Program length",
        lambda s: s["instruction_count"], "{:.0f}", "instr", None),
]


def mlp_table(s: Dict[str, Any]) -> Optional[Table]:
    """The MLP study's side-by-side: CMOS sign register vs spintronic MRAM."""
    ss = s.get("sign_storage_comparison")
    if not ss or "register_2t1r" not in ss or "spintronic_mram" not in ss:
        return None
    reg, mram = ss["register_2t1r"], ss["spintronic_mram"]
    fields = [
        ("Non-volatile", "nonvolatile", lambda v: "yes" if v else "no"),
        ("Sign area / bit (um^2)", "area_per_bit_um2", lambda v: f"{v:.4f}"),
        ("Total array area (mm^2)", "total_area_um2", lambda v: f"{v/1e6:.4f}"),
        ("Read energy / bit (J)", "read_energy_per_bit_j", lambda v: f"{v:.2e}"),
        ("Sign hold power, all bits (W)", "standby_power_total_w", lambda v: f"{v:.2e}"),
        ("Energy / inf, dynamic (J)", "energy_per_inference_j", lambda v: f"{v:.3e}"),
        ("Energy / inf incl. hold (J)", "energy_incl_hold_j", lambda v: f"{v:.3e}"),
        ("TOPS", "tops", lambda v: f"{v:.4f}"),
        ("TOPS/W (dynamic)", "tops_per_watt", lambda v: f"{v:.2f}"),
        ("TOPS/W (incl. hold leakage)", "tops_per_watt_incl_hold",
         lambda v: f"{v:.2f}"),
    ]
    return Table(
        title="Sign-storage architecture study",
        headers=["Metric", "CMOS register (2T1R)", "Spintronic STT-MRAM"],
        rows=[[lbl, fmt(reg[k]), fmt(mram[k])] for lbl, k, fmt in fields],
        note=("The sign bit of every weight is held either in a volatile CMOS "
              "register or in a non-volatile STT-MRAM cell. Both drive the same "
              "2T1R read path, so the trade is area and hold power against "
              "non-volatility."))


def mlp_sections(s: Dict[str, Any]) -> str:
    """Chip area, CPU/GPU context, retention and the compiled program."""
    esc = _html.escape
    n = s["n_arrays"]
    one = s["array_area_um2_per_array"]
    tran = s["transistor_area_um2_per_array"]
    sign = s["sign_area_um2_per_array"]
    total = s["total_area_um2"]
    area_rows = [
        ["Memristor crossbar arrays", f"{one:.1f}", f"{one*n/1e6:.4f}",
         f"{100*one*n/total:.1f} %"],
        ["Access transistors (2 per cell)", f"{tran:.1f}",
         f"{tran*n/1e6:.4f}", f"{100*tran*n/total:.1f} %"],
        [f"Sign cells ({esc(str(s.get('sign_storage_mode','')))})",
         f"{sign:.1f}", f"{sign*n/1e6:.4f}", f"{100*sign*n/total:.1f} %"],
        [f'<span class="imp">Total ({n} arrays)</span>', "",
         f'<span class="imp">{total/1e6:.4f}</span>',
         '<span class="imp">100 %</span>'],
    ]
    out = ["<h2>Chip area breakdown</h2>",
           _table(["Component", "Per array (um&sup2;)", "Total (mm&sup2;)",
                   "Share"], area_rows, num_cols=(1, 2, 3)),
           commercial_html(s, "this design")]

    ret = s.get("retention_over_time") or []
    if ret:
        rows = [[_fmt_num(r["time_s"]), f"{r['conductance_decay_pct']:.3f}",
                 f"{r['weight_domain_accuracy']:.4f}",
                 f"{r['hardware_accuracy']:.4f}",
                 f"{r['energy_per_inference_j']:.3e}",
                 f"{r['tops_per_watt']:.2f}"] for r in ret]
        out.append("<h2>Accuracy retention over time</h2>"
                   f"<p>Kohlrausch relaxation, &tau; = "
                   f"{_fmt_num(s.get('retention_tau_s', 0))} s. Signs are held "
                   "in the sign register, so drift loses amplitude but never "
                   "flips a weight's sign.</p>"
                   + _table(["Time (s)", "G decay (%)", "Weight-domain acc",
                             "Hardware acc", "Energy/inf (J)", "TOPS/W"],
                            rows, num_cols=(0, 1, 2, 3, 4, 5)))

    tail = s.get("program_tail") or []
    if tail:
        listing = "\n".join(esc(line) for line in tail)
        out.append("<h2>Compiled instruction program</h2>"
                   f"<p>{s.get('instruction_count', '?')} instructions total; "
                   "the tail of the program (fc2) is shown.</p>"
                   f'<pre class="prog">{listing}</pre>')
    return "".join(out)


def mlp_note(s: Dict[str, Any], elapsed: float, run_count: int) -> str:
    ret = s.get("retention_over_time") or []
    note = (f"Run #{run_count}: {s.get('test_subset_n', '?')} test images, "
            f"{elapsed:.1f} s simulation time.")
    if len(ret) >= 2:
        drop = 100 * (ret[0]["hardware_accuracy"] - ret[-1]["hardware_accuracy"])
        note += (f"   Retention: {drop:.2f} pt accuracy drop after "
                 f"{_fmt_num(ret[-1]['time_s'])} s "
                 f"({ret[-1]['conductance_decay_pct']:.1f} % mean "
                 "conductance decay).")
    return note


def mlp_report_subtitle(s: Dict[str, Any]) -> str:
    esc = _html.escape
    return (f"MNIST MLP ({esc(str(s.get('architecture','')))}) deployed as an "
            f"instruction program &middot; digit-serial "
            f"{s.get('digit_serial_bits','?')}-bit input &middot; "
            f"sign-magnitude weights "
            f"({esc(str(s.get('sign_storage_mode','')))} sign storage) "
            f"&middot; 1&nbsp;GHz digital clock")


# Study 2: hardware-aware training on a real detection task

def _det(s: Dict[str, Any], key: str) -> Dict[str, Any]:
    for r in s.get("results") or []:
        if r["key"] == key:
            return r
    return {}


def _dep(s: Dict[str, Any], key: str) -> Dict[str, Any]:
    """The deployable score: output-calibrated if that technique was applied."""
    r = _det(s, key)
    return r.get("hardware_calibrated") or r.get("hardware") or {}


_DET_KPIS = [
    Kpi("map_hw_hwat", "mAP on chip (HW-aware)",
        lambda s: _dep(s, "hwat")["map50"] * 100, "{:.2f}", "%", True),
    Kpi("map_hw_float", "mAP on chip (float-trained)",
        lambda s: _dep(s, "float")["map50"] * 100, "{:.2f}", "%", True),
    Kpi("gain", "Accuracy gain",
        lambda s: s["deployed_gain_factor"], "{:.2f}", "x", True),
    Kpi("ceiling", "Float ceiling (software)",
        lambda s: s["ceiling_map50"] * 100, "{:.2f}", "%", True),
    Kpi("kept_hwat", "Ceiling kept (HW-aware)",
        lambda s: _dep(s, "hwat")["map50"] / max(s["ceiling_map50"], 1e-9)
        * 100, "{:.1f}", "%", True),
    Kpi("kept_float", "Ceiling kept (float-trained)",
        lambda s: _dep(s, "float")["map50"] / max(s["ceiling_map50"], 1e-9)
        * 100, "{:.1f}", "%", True),
    Kpi("prec", "Precision on chip",
        lambda s: _dep(s, "hwat")["precision"] * 100, "{:.1f}", "%", True),
    Kpi("rec", "Recall on chip",
        lambda s: _dep(s, "hwat")["recall"] * 100, "{:.1f}", "%", True),
    Kpi("map_sw_hwat", "mAP software (HW-aware)",
        lambda s: _det(s, "hwat")["software"]["map50"] * 100, "{:.2f}", "%", True),
    Kpi("arrays", "Crossbar arrays",
        lambda s: _det(s, "hwat")["arrays"], "{:.0f}", "", None),
    Kpi("reads", "Analog reads / scene",
        lambda s: _det(s, "hwat")["reads_per_scene"], "{:.0f}", "", False),
    Kpi("energy", "Energy / scene",
        lambda s: _det(s, "hwat")["energy_per_scene_j"] * 1e6, "{:.2f}", "uJ", False),
]


def det_table(s: Dict[str, Any]) -> Optional[Table]:
    """The four numbers: each training regime scored in software and on chip."""
    rows = []
    ceiling = max(s.get("ceiling_map50", 0.0), 1e-9)
    hi = -1
    regimes = [("software", "software (float64)"), ("hardware", "on chip"),
               ("hardware_calibrated", "on chip + output calibration")]
    for r in s.get("results") or []:
        for regime, rlabel in regimes:
            m = r.get(regime)
            if not m:
                continue
            if r["key"] == "hwat" and regime == regimes[-1][0]:
                hi = len(rows)
            rows.append([
                r["label"],
                rlabel,
                f"{m['map50']*100:.2f}",
                f"{m['precision']*100:.1f}",
                f"{m['recall']*100:.1f}",
                f"{m['map50']/ceiling*100:.1f}",
            ])
    if not rows:
        return None
    return Table(
        title=f"Hardware-aware training: {s.get('task','detection task')}",
        headers=["Training", "Where it ran", "mAP@0.5 (%)", "Precision (%)",
                 "Recall (%)", "Share of float ceiling (%)"],
        rows=rows,
        highlight_row=hi,
        note=("Same architecture, data, schedule and seed for both training "
              "regimes - the only difference is whether the chip's arithmetic "
              "(4-bit sign-magnitude weights, 8-bit digit-serial inputs, a "
              "4-bit ADC on every 128-row array's partial sum, device spread "
              "and read noise) was inside the training forward pass. Every "
              "'on chip' row is a real run through IMCProcessor with all "
              "non-idealities. The highlighted row is the deployable number; "
              "the gap between it and the float-trained on-chip row is what "
              "the methodology buys."))


def det_sections(s: Dict[str, Any]) -> str:
    esc = _html.escape
    cfg = s.get("hw_config") or {}
    out = [
        "<h2>The methodology</h2>"
        "<p>A network trained in float is never asked to tolerate 4-bit weights "
        "or a coarse ADC, so it does not. Hardware-aware training puts the "
        "chip&rsquo;s arithmetic inside the forward pass and lets gradients "
        "flow through it with straight-through estimators, so the weights it "
        "converges on are ones that survive deployment. The piece that matters "
        "most is quantising <em>per array tile</em>: a layer wider than "
        f"{cfg.get('tile_rows', 128)} rows is split across arrays and each "
        "array&rsquo;s partial sum is digitised by its own ADC before the "
        "partials are added. Quantising only the final sum would model a "
        "machine that does not exist and would train the wrong robustness.</p>",
        "<h2>What was modelled during training</h2>"
        + _table(["Effect", "Setting"],
                 [["Weight levels (sign-magnitude)",
                   str(cfg.get("weight_levels", "-"))],
                  ["Input codes (digit-serial DAC)",
                   f"{cfg.get('input_bits', '-')}-bit"],
                  ["ADC per array partial sum",
                   f"{cfg.get('adc_bits', '-')}-bit"],
                  ["Array rows per tile", str(cfg.get("tile_rows", "-"))],
                  ["Device-to-device spread",
                   f"{cfg.get('d2d_sigma', 0)*100:.1f} %"],
                  ["Read noise (per forward)",
                   f"{cfg.get('read_sigma', 0)*100:.2f} %"]],
                 num_cols=(1,))]

    nc = len(((_det(s, "hwat").get("hardware") or {}).get("ap_per_class")) or [])
    if nc:
        apf = _det(s, "float")["hardware"]["ap_per_class"]
        aph = _det(s, "hwat")["hardware"]["ap_per_class"]
        out.append(
            "<h2>Per-class accuracy on chip</h2>"
            + _table(["Digit", "AP float-trained (%)", "AP hardware-aware (%)"],
                     [[str(i), f"{apf[i]*100:.2f}", f"{aph[i]*100:.2f}"]
                      for i in range(nc)], num_cols=(1, 2)))

    tr = (_det(s, "hwat").get("train") or {})
    if tr.get("seconds"):
        out.append(
            "<h2>Training</h2>"
            f"<p>Both detectors were trained on the same synthesised scenes. "
            f"Hardware-aware training took {tr.get('seconds', 0):.0f} s on CPU "
            f"(the float baseline "
            f"{(_det(s,'float').get('train') or {}).get('seconds', 0):.0f} s); "
            f"weights are cached, so re-running the study reuses them unless "
            f"&lsquo;Retrain detectors&rsquo; is ticked.</p>")
    return "".join(out)


def det_note(s: Dict[str, Any], elapsed: float, run_count: int) -> str:
    fl, hw = _det(s, "float"), _det(s, "hwat")
    return (f"Run #{run_count}: {s.get('scenes','?')} scenes / "
            f"{s.get('ground_truth_objects','?')} digits through the full "
            f"non-ideal datapath, {elapsed:.0f} s.   Deployed accuracy "
            f"{fl['hardware']['map50']*100:.2f} % -> "
            f"{hw['hardware']['map50']*100:.2f} % mAP@0.5 "
            f"({s.get('deployed_gain_factor',1):.2f}x) from hardware-aware "
            f"training.   This is a real detection metric, not a fidelity proxy.")


def det_report_subtitle(s: Dict[str, Any]) -> str:
    return (f"{_html.escape(str(s.get('task','')))} &middot; "
            f"{_html.escape(str(s.get('metric','')))} &middot; "
            f"{s.get('scenes','?')} scenes &middot; every convolution on the "
            f"real 2T1R datapath with all non-idealities")


_DET_GROUPS: List[GroupSpec] = [
    ("Detection", "Hardware-aware training", [
        dict(key="det_scenes", label="Validation scenes", kind="int",
             target="run", unit="scenes",
             help="Scenes pushed through the hardware per training regime. "
                  "~6 k analog reads each, so 150 scenes takes a few minutes"),
        dict(key="det_calib_scenes", label="ADC calibration scenes", kind="int",
             target="run", unit="scenes",
             help="Scenes used to set each tile's ADC full-scale before it is "
                  "frozen"),
        dict(key="det_retrain", label="Retrain detectors", kind="bool",
             target="run",
             help="Retrain both the float and hardware-aware detectors instead "
                  "of loading the cached weights (needs PyTorch)"),
        dict(key="det_epochs", label="Training epochs", kind="int",
             target="run", unit="epochs",
             help="Epochs for each detector when retraining"),
        dict(key="det_train_scenes", label="Training scenes", kind="int",
             target="run", unit="scenes",
             help="Synthesised MNIST-detection scenes used for training"),
        dict(key="det_seed", label="Device seed", kind="int", target="run",
             help="Seeds the device-to-device spread and stuck-at draws"),
    ]),
]

_DET_EXCLUDES = (
    "NN_DECAY_TIME_POINTS_S", "NN_DECAY_RETENTION_TAU_S",
    "MNIST_HIDDEN_UNITS", "MNIST_EPOCHS", "MNIST_BATCH_SIZE",
    "MNIST_LEARNING_RATE",
)


# Study 3: ResNet-18 / VGG-16 classifiers.  Kept apart from the detection panel:
# that one asks whether the datapath survives a detection head, this one asks what
# the two standard backbones cost in arrays, TOPS and TOPS/W, and whether the
# network fits on the chip.

def _rv_first(s: Dict[str, Any], net: str) -> Dict[str, Any]:
    """Lowest-resolution result for one net (CIFAR when both are present)."""
    res = s.get("nets", {}).get(net, {}).get("resolutions", {})
    if not res:
        return {}
    return res[sorted(res, key=lambda k: int(k))[0]]


def _rv_map(s: Dict[str, Any], net: str, idx: int = 0) -> Dict[str, Any]:
    ms = _rv_first(s, net).get("mappings", [])
    return ms[idx] if len(ms) > idx else {}


def _rv_any(s: Dict[str, Any]) -> str:
    """
    The network the headline KPI tiles describe.

    Prefers one that actually has an accuracy measurement: the two backbones are
    trained in separate GPU runs, so one may have cost numbers and no top-1.
    """
    nets = list(s.get("nets", {}))
    if not nets:
        return "resnet18"
    scored = [n for n in nets if _rv_first(s, n).get("accuracy")]
    return scored[0] if scored else nets[0]


def _rv_scored(s: Dict[str, Any]) -> List[str]:
    """Every network with an accuracy measurement, in study order."""
    return [n for n in s.get("nets", {}) if _rv_first(s, n).get("accuracy")]


def _rv_acc(s: Dict[str, Any], net: str, key: str) -> float:
    return _rv_first(s, net).get("accuracy", {}).get(key, float("nan")) * 100


_RV_LABEL = {"resnet18": "ResNet-18", "vgg16": "VGG-16"}
_RV_COLOUR = {"resnet18": "#2b6cb0", "vgg16": "#c05621"}


def _rv_name(net: str) -> str:
    return _RV_LABEL.get(net, net)


def _rv_best_chip(s: Dict[str, Any], net: str) -> Dict[str, Any]:
    """
    The chip row the TOPS / throughput tiles quote.

    The study now costs exactly ONE chip - the one configured on the 'Chip' tab -
    so this is that row.  The `fits` preference is kept for summaries written
    before that change, which still carry the three fixed sizes.
    """
    chips = _rv_map(s, net).get("chips", [])
    if len(chips) == 1:
        return chips[0]
    return next((c for c in chips if c.get("fits")), chips[0] if chips else {})


def _rv_pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.2f} %"


def _rv_kpis(s: Dict[str, Any]) -> List[Kpi]:
    """
    Headline tiles, one set per network the run actually mapped.

    Built from the summary so every mapped network stays visible and no tile
    exists for a number that was never measured.
    """
    nets = list(s.get("nets", {}))
    scored = _rv_scored(s)

    def tiles(key, label, getter, fmt, unit, hib, on=None):
        return [Kpi(f"{key}_{n}", f"{_rv_name(n)} {label}",
                    (lambda ss, net=n: getter(ss, net)), fmt, unit, hib)
                for n in (nets if on is None else on)]

    k: List[Kpi] = []
    k += tiles("rv_topsw", "TOPS/W",
               lambda ss, n: _rv_map(ss, n).get("tops_per_watt", 0.0),
               "{:.2f}", "", True)
    # THE headline throughput: replicated weights, images pipelined through the
    # layer chain - the two assumptions the roofline makes silently, both
    # switched on and both costed.  This is what the chip reaches, so it leads.
    k += tiles("rv_tops_pipe", "TOPS",
               lambda ss, n: _rv_best_chip(ss, n).get("tops_pipelined", 0.0),
               "{:,.0f}", "", True)
    # One image end to end on replicated weights: the latency number.
    k += tiles("rv_tops_ach", "TOPS (one image)",
               lambda ss, n: _rv_best_chip(ss, n).get("tops_achievable", 0.0),
               "{:,.1f}", "", True)
    k += tiles("rv_repl", "weight replication",
               lambda ss, n: _rv_best_chip(ss, n).get("replication_factor", 1.0),
               "{:,.1f}", "x", None)
    # Reference only, and last: the ceiling the datapath is measured against.
    k += tiles("rv_tops", "TOPS (roofline ref.)",
               lambda ss, n: _rv_best_chip(ss, n).get("tops", 0.0),
               "{:,.0f}", "", True)
    k += tiles("rv_acc_hwat", "on chip, HW-aware",
               lambda ss, n: _rv_acc(ss, n, "hwat/chip"),
               "{:.2f}", "%", True, on=scored)
    k += tiles("rv_acc_float", "on chip, float-trained",
               lambda ss, n: _rv_acc(ss, n, "float/chip"),
               "{:.2f}", "%", True, on=scored)
    k += tiles("rv_arrays", "crossbar arrays",
               lambda ss, n: _rv_map(ss, n).get("arrays", 0),
               "{:,.0f}", "", None)
    k += tiles("rv_area", "array area",
               lambda ss, n: _rv_map(ss, n).get("area_mm2", 0.0),
               "{:.1f}", "mm^2", False)
    k.append(Kpi("rv_readtime", "Derived read time",
                 lambda ss: ss.get("config", {}).get("read_time_ns", 0.0),
                 "{:.2f}", "ns", False))
    return k


def rv_panels(s: Dict[str, Any]) -> Optional[PanelSet]:
    """
    The Comparison tab as one card per mapped network: what it scores, then
    what it costs, then the chip it is resident on.
    """
    panels: List[Panel] = []
    for net, blob in s.get("nets", {}).items():
        for res_key, res in sorted(blob.get("resolutions", {}).items(),
                                   key=lambda kv: int(kv[0])):
            ms = res.get("mappings", [])
            if not ms:
                continue
            m = ms[0]
            chips = m.get("chips", [])
            best = next((c for c in chips if c.get("fits")),
                        chips[0] if chips else {})
            acc = res.get("accuracy") or {}
            rows: List[Tuple[str, str]] = []
            if acc:
                fc = acc.get("float/software")
                fl = acc.get("float/chip")
                hw = acc.get("hwat/chip")
                rows += [("Float ceiling (software)", _rv_pct(fc)),
                         ("On chip, float-trained", _rv_pct(fl)),
                         ("On chip, HW-aware", _rv_pct(hw))]
                if fl is not None and hw is not None:
                    rows.append(("What HW-aware training buys",
                                 f"{(hw - fl) * 100:+.2f} pts"))
                if fc and hw is not None:
                    rows.append(("Ceiling kept on chip", _rv_pct(hw / fc)))
            else:
                rows.append(("Accuracy", "not scored - no trained weights"))
            rows += [("", ""),
                     ("Crossbar arrays", f"{m.get('arrays', 0):,}"),
                     ("Silicon area", f"{m.get('area_mm2', 0):.1f} mm^2"),
                     ("Analog reads / image", f"{m.get('reads', 0):,}"),
                     ("Energy / inference",
                      f"{m.get('energy_j', 0) * 1e6:.2f} uJ"),
                     ("TOPS (peak, replicated + pipelined)",
                      f"{best.get('tops_pipelined', 0):,.0f}"),
                     ("TOPS (one image, replicated)",
                      f"{best.get('tops_achievable', 0):,.1f}"),
                     ("TOPS (roofline, reference)",
                      f"{best.get('tops', 0):,.0f}"
                      + ("" if best.get("tops_roofline_valid_for_chip", True)
                         else "  (assumes enough arrays)")),
                     ("Weight replication",
                      f"{best.get('replication_factor', 1):,.1f}x "
                      f"({best.get('arrays_used', 0):,} of "
                      f"{best.get('arrays_available', 0):,} arrays)"),
                     ("TOPS/W", f"{m.get('tops_per_watt', 0):.2f}"),
                     ("Throughput",
                      f"{best.get('fps_achievable', 0):,.0f} / s"),
                     ("Cells gated off (sparsity)",
                      _rv_pct(m.get("sparsity_exploited")))]
            # What the ISA's operation modes buy.  The top figure is this chip,
            # whose mode bits let the decoder drive only the lines the operation
            # needs; the second is the same mapping on a machine whose only
            # instruction is the full matrix read.
            am = m.get("all_matrix")
            if am:
                modes = ", ".join(f"{k} x{v:,}" for k, v in
                                  sorted(m.get("mode_counts", {}).items()))
                rows += [("", ""),
                         ("ISA modes issued", modes or "-"),
                         ("Energy, this chip",
                          f"{m.get('energy_j', 0) * 1e6:.2f} uJ"),
                         ("Energy, no sub-mode opcodes",
                          f"{am.get('energy_j', 0) * 1e6:.2f} uJ"),
                         ("TOPS/W, no sub-mode opcodes",
                          f"{am.get('tops_per_watt', 0):.2f}"),
                         ("What the mode bits save",
                          f"{am.get('energy_ratio', 0):.2f}x")]
            sub = (f"{res.get('params', 0) / 1e6:.2f} M weights   |   "
                   f"{res.get('macs', 0) / 1e6:,.1f} MMAC / image   |   "
                   f"{m.get('adc_bits', '-')}-bit ADC")
            foot = (f"Resident on {best.get('chip', '-')}   |   "
                    f"{best.get('reprogram_groups', 1)} weight group"
                    if best.get("fits")
                    else "Not resident on any listed chip size")
            panels.append(Panel(
                title=f"{_rv_name(net)}   @{res_key} px",
                subtitle=sub, rows=rows,
                accent=_RV_COLOUR.get(net, COL_ACCENT), footer=foot))
    if not panels:
        return None
    note = ("Accuracy comes from the trained weights scored through the "
            "hardware forward; cost comes from the mapping. Both backbones are "
            "read the same way, so the cards are directly comparable. The "
            "roofline TOPS is the array's ceiling - every read spread over every "
            "array, nothing waiting. Reaching it needs weight replication (an "
            "array holds one copy of a layer's weights, so its output positions "
            "stream past the same cells one at a time) and images pipelined "
            "across the layer chain; TOPS (peak) turns both on. TOPS (one image) "
            "is the single-inference latency on replicated weights. TOPS/W is "
            "the same under all of them: concurrency changes when the joules are "
            "spent, not how many.")
    return PanelSet(title="ResNet-18 vs VGG-16 on the chip",
                    panels=panels, note=note, columns=2)


def rv_table(s: Dict[str, Any]) -> Optional[Table]:
    """Every (net, resolution, ADC) mapping on the chips that hold it resident."""
    # Only the configured chip is costed now, so nothing gets filtered.
    resident_only = bool(s.get("resident_only", False))
    rows: List[List[str]] = []
    for net, blob in s.get("nets", {}).items():
        for res_key, res in sorted(blob.get("resolutions", {}).items(),
                                   key=lambda kv: int(kv[0])):
            acc = res.get("accuracy", {})
            for m in res.get("mappings", []):
                chips = m.get("chips", [])
                best = next((c for c in chips if c["fits"]), chips[0] if chips else {})
                if resident_only and not best:
                    continue          # nothing holds it; the row would be noise
                row = [
                    net,
                    f"{res_key}px",
                    f"{m['adc_bits']}b",
                    f"{m['arrays']:,}",
                    f"{m.get('area_mm2', 0):.1f}",
                    f"{m['energy_j']*1e6:.2f}",
                    f"{m['tops_per_watt']:.2f}",
                    f"{best.get('tops_pipelined', 0):,.0f}",
                    f"{best.get('tops_achievable', 0):,.1f}",
                    f"{best.get('fps_achievable', 0):,.0f}",
                    # Reference, last.  Starred when the chosen chip does not
                    # hold the network - there it describes a larger machine.
                    (f"{best.get('tops', 0):.0f}"
                     + ("" if best.get("tops_roofline_valid_for_chip", True) else "*")),
                    best.get("chip", "-"),
                ]
                # Accuracy is measured once per resolution, at the first ADC in
                # the sweep, so only that row can carry it honestly.
                on_chip = acc.get("hwat/chip")
                row.append(f"{on_chip*100:.2f}%" if on_chip is not None
                           and m["adc_bits"] == res["mappings"][0]["adc_bits"] else "-")
                rows.append(row)
    if not rows:
        return None
    note = ("TOPS (roofline) spreads every analog read over every array the "
            "network needs and lets nothing wait. It is reachable, but only with "
            "the two things it assumes silently: weight replication, because an "
            "array holds one copy of a layer's weights and that layer's output "
            "positions must stream past the same cells one at a time; and "
            "pipelining, because for a single image only one layer's arrays can "
            "be busy. TOPS (peak) turns both on - spare arrays spent on extra "
            "copies of the position-heavy layers, images in flight at every "
            "stage - and lands close to the roofline. TOPS (one image) is the "
            "latency number: replicated weights, one inference end to end, with "
            "im2col delivery, activation movement and pipeline fill charged. A "
            "starred roofline is for a chip that cannot hold one copy of the "
            "weights, so that figure describes a larger machine than the one in "
            "the 'chip' column. TOPS/W is the same under all of them - "
            "concurrency changes when the joules are spent, not how many; the "
            "same network moves an order of magnitude across the ADC sweep while "
            "its array footprint never changes.")
    note += (" The 'chip' column is the configuration set on the Chip tab "
             "(tiles x PUs x arrays per PU); it is the only one costed. If it "
             "cannot hold one copy of the weights the network is time-"
             "multiplexed through reprogramming groups. That is a programming "
             "cost, not a read-path cost, so it is reported in its own column "
             "and never folded into TOPS/W.")
    return Table(
        title="VGG / ResNet on the chip - accuracy from the trained weights, cost from the mapping",
        headers=["net", "input", "ADC", "arrays", "mm^2", "uJ/inf",
                 "TOPS/W", "TOPS", "TOPS (one image)", "FPS (one image)",
                 "TOPS (roofline ref.)", "chip", "top-1 on chip"],
        rows=rows,
        note=note,
    )


def rv_note(s: Dict[str, Any], elapsed: float, run_count: int) -> str:
    net = _rv_any(s)
    m4, m8 = _rv_map(s, net, 0), _rv_map(s, net, 1)
    hw = _rv_acc(s, net, "hwat/chip")
    fl = _rv_acc(s, net, "float/chip")
    acc = ""
    if hw == hw:                                   # not NaN
        acc = (f"  {net}: hardware-aware {hw:.2f}% on chip against {fl:.2f}% for "
               f"the same network trained in float, a {hw-fl:+.2f} point "
               f"difference bought by putting the converter in the training "
               f"forward.")
    # With both backbones scored, the comparison is the result - say it here
    # rather than leaving it to be read off the table.
    scored = _rv_scored(s)
    if len(scored) > 1:
        parts = [f"{n} {_rv_acc(s, n, 'hwat/chip'):.2f}%" for n in scored]
        acc += f"  On chip, hardware-aware: {', '.join(parts)}."
    missing = [n for n in s.get("nets", {}) if n not in scored]
    if missing:
        acc += (f"  No trained weights found for {', '.join(missing)} - "
                f"cost only for those; point 'Trained weights folder' at the "
                f"run that produced them.")
    return (f"{len(s.get('nets', {}))} classifier(s) mapped in {elapsed:.1f}s "
            f"(run {run_count}).  {m4.get('tops_per_watt', 0):.1f} TOPS/W at 4-bit ADC "
            f"vs {m8.get('tops_per_watt', 0):.1f} at 8-bit - the converter is the "
            f"efficiency lever.{acc}")


def _rv_throughput_section(s: Dict[str, Any]) -> str:
    """Roofline against the corners it assumes, with this run's own numbers."""
    esc = _html.escape
    rows = []
    for net in s.get("nets", {}):
        c = _rv_best_chip(s, net)
        if not c or "tops_no_replication_single" not in c:
            continue
        rows.append([
            esc(_rv_name(net)), esc(c.get("chip", "-")),
            f"{c.get('tops_pipelined', 0):,.0f}",
            f"{c.get('tops_achievable', 0):,.1f}",
            f"{c.get('replication_factor', 1):,.1f}&times;",
            f"{c.get('tops_per_watt', 0):,.2f}",
            f"{c.get('tops', 0):,.0f}",
            f"{c.get('tops_no_replication_single', 0):,.1f}",
            f"{c.get('tops_no_replication_pipelined', 0):,.0f}",
        ])
    if not rows:
        return ""
    return (
        "<h2>What the roofline assumes</h2>"
        "<p>The <strong>roofline</strong> divides every analog read of the "
        "network by every array the network needs. It is not wrong, but it is "
        "reachable only under two assumptions it never states &mdash; and until "
        "both are modelled the number floats free of the machine.</p>"
        "<p><strong>1. Weight replication.</strong> An array holds <em>one</em> "
        "copy of a layer&rsquo;s weights, so that layer&rsquo;s H&times;W output "
        "positions stream past the same cells one at a time. ResNet-18&rsquo;s "
        "stem puts 1024 positions on 1 array; its last block puts 16 positions "
        "on 144. Costed with one copy each, the stem takes 14.7&nbsp;&micro;s and "
        "the deep block 0.23&nbsp;&micro;s, and on average only about 7 of 698 "
        "arrays are doing anything at all. The fix is the one a real mapper uses: "
        "spend the chip&rsquo;s <em>spare</em> arrays on extra copies of the "
        "position-heavy layers so the stages balance. ResNet-18 needs 698 of "
        "2048, which is room for 2.9&times;.</p>"
        "<p><strong>2. Pipelining.</strong> Layers are sequential for one image, "
        "so a single inference can only ever have one layer&rsquo;s arrays busy. "
        "Filling the chip means images in flight at every stage at once, and then "
        "the rate is set by the slowest stage rather than the sum of them.</p>"
        "<p>With both switched on the datapath lands close to the roofline, so "
        "the roofline is a real peak &mdash; it simply is not a single-image "
        "latency. On top of the array reads the model also charges im2col "
        "delivery (only the part not hidden behind the reads), activation "
        "movement over the NoC, the adder-tree fill, and the pipeline fill; "
        "together these are a small correction next to the two above. "
        "<strong>TOPS/W is identical in every column</strong>, which is the check "
        "that the model only moved time and never energy.</p>"
        + _table(["Network", "chip", "TOPS (peak)", "TOPS (one image)",
                  "replication", "TOPS/W",
                  "roofline (ref.)", "1 copy, 1 image", "1 copy, pipelined"],
                 rows, num_cols=tuple(range(2, 9))))


def _rv_ni_section(s: Dict[str, Any]) -> str:
    """The device the run was scored on - written from the summary, not assumed."""
    cfg = s.get("config", {})
    ni = cfg.get("non_idealities") or {}
    if not ni:
        return ""
    esc = _html.escape
    LABEL = [
        ("d2d_sigma", "Device-to-device spread", "sigma",
         "persistent, drawn once when the array is programmed"),
        ("c2c_sigma", "Write / cycle-to-cycle spread", "sigma",
         "redrawn at every programming event"),
        ("stuck_at_prob", "Stuck-at defect probability", "fraction",
         "cells pinned at LRS or HRS; a gross error, so it does not average down"),
        ("read_sigma", "Read noise", "sigma",
         "transient, re-drawn per array read"),
        ("read_noise_at_inference", "Read noise at inference", "",
         "off makes a scored run bit-for-bit repeatable"),
        ("retention_time_s", "Retention age", "s",
         "uniform gain on the weights; the signal falls as gain^depth"),
        ("bias_on_array", "Bias rides the array row", "",
         "on = the bias drifts with the weights, as the chip does"),
    ]
    rows = [[esc(lbl), f"{ni[k]}", esc(unit), esc(note)]
            for k, lbl, unit, note in LABEL if k in ni]
    rows.append(["ADC ranges refitted before scoring",
                 str(cfg.get("adc_recalibrated", True)), "",
                 "off models a chip calibrated once at t=0 and never re-ranged"])
    return (
        "<h2>The device this run was scored on</h2>"
        "<p>Every value below perturbs the stored <strong>cell "
        "conductance</strong>. The converter&rsquo;s own error is the ADC "
        "resolution and is a separate axis entirely &mdash; measured on this "
        "chip, a 5% device spread reaches the ADC as ~0.6%, one twentieth of a "
        "4-bit code, so the converter is the limiter and the memristors are "
        "not.</p>"
        + _table(["Non-ideality", "Value", "Unit", "What it is"], rows,
                 num_cols=(1,)))


def rv_sections(s: Dict[str, Any]) -> str:
    cfg = s.get("config", {})
    return (
        _rv_ni_section(s) +
        _rv_throughput_section(s) +
        "<h2>What this study measures</h2>"
        "<p>Two standard classification backbones on the same chip every "
        "other study uses, at both CIFAR (32 px) and ImageNet (224 px) "
        "resolution. "
        "The hardware-aware forward pass <em>is</em> the chip datapath - "
        "sign-magnitude weights on "
        f"{cfg.get('conductance_states', 16)} conductance states plus a separate "
        "STT-MRAM sign bit, 8-bit activations as two 4-bit nibbles, and a "
        "per-array-tile ADC on every 128-row partial sum - so the on-chip score "
        "is the trained model evaluated with the hardware enabled.</p>"
        "<h2>Three things worth reading off the table</h2>"
        "<p><strong>TOPS/W barely moves between the two networks</strong> but "
        "changes by an order of magnitude across the ADC sweep. Efficiency here "
        "is a per-read property of the converter, not a property of the model.</p>"
        "<p><strong>ResNet-18 is the smaller network by silicon and the larger "
        "one by compute.</strong> It needs 698 arrays to VGG-16's 906 (47.5 mm^2 "
        "against 61.6) while doing 1.8x the MACs - 555 M against 313 M per "
        "image - because a residual network reuses each weight across a larger "
        "feature map. It also pays for that: 92,680 analog reads per image "
        "against 45,960, and 21 mapped layers against 14, so twice as many "
        "quantisation events stand between the input and the answer. Fewer "
        "arrays does not mean less work.</p>"
        "<p><strong>VGG-16 at 224 px does not fit.</strong> It needs roughly "
        "8.2k arrays (~560 mm^2), so on any chip smaller than 32x32x16 it is "
        "time-multiplexed through reprogramming groups - and the reprogramming "
        "column shows what that costs. ResNet-18 is resident on 16x16x8 at "
        "<em>both</em> resolutions and pays nothing: its 224 px footprint is 699 "
        "arrays, one more than at 32 px, because array count follows the weights "
        "and a convolution's weights do not depend on the image size.</p>"
        "<h2>Activation sparsity is measured, not assumed</h2>"
        "<p>A zero activation leaves its row's DAC off and none of that row's "
        "cells are read, which is what the ISA's row decoder exists to exploit. "
        "That fraction used to be a single configured number (50%) applied to "
        "every layer of every network, so the reported 'cells gated off' was an "
        "echo of the setting. It is now measured per layer from the trained "
        "weights. On VGG-16 the real figure is 60% on average and spans 4-81% "
        "across layers; the energy it saves is still only about 1%, because the "
        "ADC is ~95% of the budget and scales with columns, not rows.</p>"
        f"<p>Read time is derived ({cfg.get('read_time_ns', 0):.2f} ns), not "
        "assumed: it is the larger of the converter's own aperture and the "
        "column RC settling to 1 LSB, with wire resistance "
        f"{cfg.get('wire_resistance_ohm', 0)} ohm and capacitance "
        f"{cfg.get('wire_capacitance_f', 0)*1e15:.2f} fF per cell pitch.</p>"
    )


def rv_report_subtitle(s: Dict[str, Any]) -> str:
    nets = ", ".join(s.get("nets", {}))
    return (f"{nets} on the memristor PIM chip - accuracy, array footprint, "
            f"TOPS and TOPS/W")


_RV_GROUPS: List[GroupSpec] = [
    ("Classifiers", "ResNet / VGG classification study", [
        dict(key="rv_nets", label="Networks to map", kind="multichoice",
             target="run", choices=("resnet18", "vgg16"),
             help="Which classification backbones take part in the comparison"),
        dict(key="rv_img_sizes", label="Input resolutions", kind="float_list",
             target="run", unit="px",
             help="32 = CIFAR variant (3x3 stem, no pool); 224 = ImageNet "
                  "variant (7x7/2 stem + max-pool, and VGG's three 4096-wide "
                  "FC layers, which are 124M of its 138M weights)"),
        dict(key="rv_adc_bits", label="ADC sweep", kind="float_list",
             target="run", unit="bit",
             help="Converter resolutions to sweep. This is the dominant "
                  "efficiency lever: ~95% of read energy is the ADC"),
        dict(key="rv_accuracy", label="Score accuracy", kind="bool",
             target="run",
             help="Evaluate the trained weights through the hardware forward. "
                  "Needs torch and trained .npz weights from resnet_vgg_hwat.py; "
                  "turn off for a cost-only run"),
        dict(key="rv_test_subset", label="Test images", kind="int",
             target="run", unit="imgs",
             help="Test images used for the top-1 score. A hardware evaluation "
                  "runs the full chip datapath, so this dominates the runtime"),
        dict(key="rv_weights_dir", label="Trained weights folder", kind="path",
             target="run",
             help="Directory holding the trained <net>_hwat_float.npz and "
                  "<net>_hwat_hwat.npz. Read in place; nothing is copied. "
                  "Several folders may be listed separated by ';' - ResNet-18 "
                  "and VGG-16 come out of two separate GPU runs, and the first "
                  "folder holding a given file wins"),
        dict(key="rv_measure_sparsity", label="Measure activation sparsity",
             kind="bool", target="run",
             help="Measure the real per-layer zero fraction from the trained "
                  "weights instead of assuming one number for every layer. "
                  "Measured on VGG-16 it is 60% on average and spans 4-81% "
                  "across layers, so the single 50% setting below was never "
                  "right for any particular layer. Falls back to that setting "
                  "when no trained weights are present"),
        dict(key="rv_input_sparsity", label="Assumed activation sparsity",
             kind="float", target="run", unit="frac",
             help="Fraction of activation rows that are zero after ReLU. A zero "
                  "row drives no DAC and no cell, so the array read skips it. "
                  "Used only as the fallback when the sparsity is not measured"),
        dict(key="rv_sparse_modes", label="Exploit ISA sub-modes", kind="bool",
             target="run",
             help="Address reads with the point/row/column decoders when the "
                  "active set collapses to a single line, instead of always "
                  "issuing a full matrix read"),
    ]),
    ("Deployment", "Deployment conditions - the device itself is on the "
                   "'Non-ideal' tab", [
        dict(key="rv_retention_s", label="Retention age", kind="float",
             target="run", unit="s",
             help="How long the array has been sitting programmed before this "
                  "inference. The magnitude conductance relaxes toward the "
                  "g_min floor under the law chosen on the 'Non-ideal' tab "
                  "(SM_NI_RETENTION_MODEL / beta, with the time constant from "
                  "'Retention tau'); the sign lives in the STT-MRAM cell and "
                  "never drifts, so a weight loses amplitude but cannot flip. "
                  "It is a UNIFORM gain on the weights, so the signal reaching "
                  "the converter falls as gain^depth - which is why it costs a "
                  "14-layer network far more than a 2-layer one. 0 = score the "
                  "array as freshly programmed"),
        dict(key="rv_recalibrate", label="Refit ADC ranges before scoring",
             kind="bool", target="run",
             help="On: the converter is re-ranged against the array as it "
                  "actually is - what a maintained deployment does. Off: the "
                  "ranges stay as calibrated at t=0, so a relaxing signal falls "
                  "below the first code and is quantised away. The single most "
                  "consequential switch here: measured on VGG-16 at a 7200 s "
                  "retention age, 8.8% frozen against 81.6% refitted, so ~90% "
                  "of what looks like retention damage is the converter rather "
                  "than the devices"),
        dict(key="rv_bias_on_array", label="Bias rides the array row",
             kind="bool", target="run",
             help="On: the bias is stored in memristors on the array's bias row "
                  "and relaxes with everything else - what the chip actually "
                  "does. Off: it is a digital value added after the ADC and "
                  "stays put while the weights fade, so a fixed offset "
                  "eventually dominates the layer. Only matters at a nonzero "
                  "retention age; measured 22 points apart at 7200 s"),
        dict(key="rv_read_at_inference", label="Read noise at inference",
             kind="bool", target="run",
             help="Off: the read noise set on the 'Non-ideal' tab is applied "
                  "only while calibrating, so a scored run is bit-for-bit "
                  "repeatable - one particular chip, measured exactly. On: it "
                  "is applied to deployed reads too, which is what a real chip "
                  "does and what makes repeated runs differ. Turn it on if you "
                  "want an error bar rather than a single number"),
        dict(key="rv_redraw_d2d", label="Redraw device spread", kind="bool",
             target="run",
             help="The trained .npz stores the device-to-device pattern the "
                  "network was fitted against, and loading the weights restores "
                  "it - correct for deploying a hardware-aware network on the "
                  "array it was trained for, but it means the 'D2D variation "
                  "sigma' on the 'Non-ideal' tab has NO effect here. Turn this "
                  "on to draw a fresh pattern at that sigma instead, which is "
                  "what a device-spread sweep needs"),
    ]),
]

# The classifier study does not consume the retention sweep or the MNIST knobs.
_RV_EXCLUDES = (
    # The retention TIME POINTS are the MLP study's sweep and mean nothing here -
    # this study scores one age, set by 'Retention age'.  The TAU is kept: it is
    # the device's time constant and the retention model needs it.
    "NN_DECAY_TIME_POINTS_S",
    "test_subset_n", "calib_n",
    "MNIST_HIDDEN_UNITS", "MNIST_EPOCHS", "MNIST_BATCH_SIZE",
    "MNIST_LEARNING_RATE",
)


# The study registry

def build_default_studies() -> List[Study]:
    """
    The workloads the platform can deploy on the chip.  Imported lazily so this
    module stays importable on its own (and so a missing workflow only breaks
    its own study).
    """
    import example_usage_riscv_workflow as mlp_wf
    import resnet_vgg_study as rv_wf

    return [
        Study(
            key="mlp",
            label="MNIST MLP  -  RISC-V instruction program",
            subtitle="RISC-V IMC workflow - MNIST MLP as an instruction program",
            run_fn=mlp_wf.run_workflow,
            fig_builder=mlp_wf.build_results_figure,
            run_defaults=mlp_wf.DEFAULT_RUN_SETTINGS,
            kpis=_MLP_KPIS,
            figure_path="riscv_workflow_results.png",
            report_path="riscv_workflow_report.html",
            summary_path="riscv_workflow_summary.json",
            report_title="RISC-V IMC Workflow Report",
            table_fn=mlp_table,
            note_fn=mlp_note,
            sections_fn=mlp_sections,
            report_subtitle_fn=mlp_report_subtitle,
        ),
        Study(
            key="resnet_vgg",
            label="ResNet-18 / VGG-16  -  classification backbones",
            subtitle=("ResNet-18 and VGG-16 on the chip: accuracy, array "
                      "footprint, TOPS and TOPS/W"),
            run_fn=rv_wf.run_resnet_vgg_benchmark,
            fig_builder=rv_wf.build_results_figure,
            run_defaults=rv_wf.DEFAULT_RUN_SETTINGS,
            kpis=_rv_kpis,
            figure_path=rv_wf.FIGURE_PATH,
            report_path="resnet_vgg_report.html",
            summary_path=rv_wf.SUMMARY_PATH,
            report_title="ResNet / VGG Classification Report",
            extra_groups=_RV_GROUPS,
            exclude_keys=_RV_EXCLUDES,
            table_fn=rv_table,
            panels_fn=rv_panels,
            note_fn=rv_note,
            sections_fn=rv_sections,
            report_subtitle_fn=rv_report_subtitle,
        ),
    ]
