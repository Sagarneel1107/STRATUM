"""
crossbar_user_config.py

DEPRECATED as the place to edit parameters.

All tweakable parameters now live in ONE file: simulation_config.py.
This module simply re-exports the device/crossbar builders and constants from
there, so older scripts that did

    from crossbar_user_config import build_device, build_crossbar, LAYER_ROWS, LAYER_COLS

keep working unchanged. To change any value, edit simulation_config.py.
"""

from __future__ import annotations

# Re-export everything public from the single control panel.
from simulation_config import *           # noqa: F401,F403
from simulation_config import (            # noqa: F401  (explicit for the common names)
    build_device,
    build_crossbar,
    LAYER_ROWS,
    LAYER_COLS,
)


if __name__ == "__main__":
    device = build_device()
    crossbar = build_crossbar()

    print(device)
    print(crossbar)

    print("\nArea breakdown:")
    for key, value in crossbar.area_breakdown_um2().items():
        print(f"{key}: {value:.6e} um^2")

    print("\nLayer mapping:")
    mapping = crossbar.estimate_layer_mapping(LAYER_ROWS, LAYER_COLS)
    for key, value in mapping.__dict__.items():
        print(f"{key}: {value}")
