"""Sampling and explicit reference limits, independent of robot qualification."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TraversabilityConfig:
    enabled: bool = True
    resolution_m: float = 0.25
    robot_length_m: float = 0.7
    robot_width_m: float = 0.5
    robot_height_m: float = 0.5
    margin_m: float = 0.02
    max_slope_deg: float = 20.0
    max_step_m: float = 0.1
    max_cells: int = 12_000_000

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("traversability.enabled must be boolean")
        if type(self.max_cells) is not int or self.max_cells < 1:
            raise ValueError("traversability.max_cells must be a positive integer")
        for key, value in asdict(self).items():
            if key in {"enabled", "max_cells"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"traversability.{key} must be a finite number")
            if value < 0 or (key != "margin_m" and value == 0):
                raise ValueError(f"traversability.{key} must be positive (margin may be zero)")
        if self.max_slope_deg >= 90:
            raise ValueError("traversability.max_slope_deg must be below 90")
        if self.resolution_m > min(self.robot_length_m, self.robot_width_m) / 2:
            raise ValueError(
                "traversability.resolution_m must sample each reference footprint axis at least twice"
            )
        # Bound the footprint operator as well as the output grid.
        if self.radius_m / self.resolution_m > 32:
            raise ValueError("Traversability footprint exceeds 32 pixels; increase resolution_m")

    @property
    def radius_m(self):
        return float(np.hypot(self.robot_length_m, self.robot_width_m) / 2 + self.margin_m)
