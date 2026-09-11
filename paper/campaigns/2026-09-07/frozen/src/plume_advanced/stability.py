"""Conservative roof-span screening, not a calibrated rock-mechanics solver.

For a simply supported, unit-width roof beam under its own weight,
sigma_max = 3 rho g width**2 / (4 roof_thickness).  A safety factor and
effective fractured-rock tensile strength define the admissible envelope.
At a FIXED floor elevation, increasing cavity height reduces roof thickness;
the height and width limits therefore describe the same coupled constraint.
Arching, stress confinement, layering and nearby support pillars need a more
complete structural model; this screening model deliberately claims none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RoofStabilityAssessment:
    roof_thickness_m: float
    required_roof_thickness_m: float
    maximum_width_m: float
    maximum_height_m: float
    demand_ratio: float
    failed: bool


@dataclass(frozen=True)
class RoofStabilityModel:
    gravity_m_s2: float = 9.80665
    rock_density_kg_m3: float = 2900.0
    effective_tensile_strength_pa: float = 3_000_000.0
    safety_factor: float = 1.5

    def __post_init__(self) -> None:
        for name, value in (
            ("gravity_m_s2", self.gravity_m_s2),
            ("rock_density_kg_m3", self.rock_density_kg_m3),
            ("effective_tensile_strength_pa", self.effective_tensile_strength_pa),
            ("safety_factor", self.safety_factor),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"stability {name} must be finite and positive")
        if self.safety_factor < 1.0:
            raise ValueError("stability safety_factor must be at least 1")

    @property
    def load_coefficient(self) -> float:
        return (
            0.75 * self.safety_factor * self.rock_density_kg_m3 * self.gravity_m_s2
            / self.effective_tensile_strength_pa
        )

    def assess(
        self, *, width_m: float, height_m: float, floor_depth_m: float
    ) -> RoofStabilityAssessment:
        if not all(math.isfinite(v) for v in (width_m, height_m, floor_depth_m)):
            raise ValueError("stability dimensions must be finite")
        if width_m < 0.0 or height_m < 0.0:
            raise ValueError("stability width and height cannot be negative")
        roof = floor_depth_m - height_m
        required = self.load_coefficient * width_m**2
        # A finite sentinel keeps JSON reports strict even for an exposed roof.
        demand = required / max(roof, 1e-9)
        return RoofStabilityAssessment(
            roof_thickness_m=roof,
            required_roof_thickness_m=required,
            maximum_width_m=math.sqrt(max(roof, 0.0) / self.load_coefficient),
            maximum_height_m=max(floor_depth_m - required, 0.0),
            demand_ratio=demand,
            failed=roof <= 0.0 or required > roof * (1.0 + 1e-12),
        )
