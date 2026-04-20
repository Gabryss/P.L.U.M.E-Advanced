"""Stage A: low-frequency host field generation.

The host field defines the broad geological constraints that later stages will
use to grow believable conduit centerlines. This stage intentionally avoids
high-frequency noise and instead focuses on smooth, readable proxies:

- terrain elevation
- slope
- cover thickness
- roof competence
- growth cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import NDArray

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


def _default_waves() -> tuple["TerrainWave", ...]:
    return (
        TerrainWave(amplitude=12.0, wavelength=1600.0, angle_degrees=20.0, phase=0.25),
        TerrainWave(amplitude=8.0, wavelength=1050.0, angle_degrees=74.0, phase=1.05),
        TerrainWave(amplitude=4.5, wavelength=620.0, angle_degrees=-32.0, phase=-0.65),
    )


@dataclass(frozen=True)
class GridConfig:
    """Regular 2D grid used to evaluate the host field."""

    width: float = 3000.0
    height: float = 2400.0
    nx: int = 220
    ny: int = 180

    @property
    def spacing_x(self) -> float:
        return self.width / (self.nx - 1)

    @property
    def spacing_y(self) -> float:
        return self.height / (self.ny - 1)


@dataclass(frozen=True)
class TerrainWave:
    """A directional low-frequency wave used to deform the terrain smoothly."""

    amplitude: float
    wavelength: float
    angle_degrees: float
    phase: float = 0.0


@dataclass(frozen=True)
class HostFieldConfig:
    """Parameters controlling stage-A host field generation."""

    grid: GridConfig = field(default_factory=GridConfig)
    seed_point: tuple[float, float] = (-1200.0, 0.0)
    high_side_elevation: float = 182.0
    longitudinal_drop: float = 84.0
    flow_angle_degrees: float = 0.0
    corridor_depth: float = 12.0
    corridor_width: float = 520.0
    volcanic_layer_thickness: float = 64.0
    minimum_stable_cover: float = 18.0
    roof_competence_baseline: float = 0.72
    roof_competence_variation: float = 0.18
    fracture_zone_angle_degrees: float = 82.0
    fracture_zone_center_offset: float = -140.0
    fracture_zone_width: float = 240.0
    waves: tuple[TerrainWave, ...] = field(default_factory=_default_waves)


@dataclass(frozen=True)
class HostFieldSample:
    """Bilinear sample of the host field at a single point."""

    elevation: float
    slope_degrees: float
    cover_thickness: float
    roof_competence: float
    growth_cost: float
    gradient_x: float
    gradient_y: float


@dataclass(frozen=True)
class HostField:
    """Generated stage-A outputs, ready for later graph growth stages."""

    config: HostFieldConfig
    x_coords: Array1D
    y_coords: Array1D
    elevation: Array2D
    slope_degrees: Array2D
    cover_thickness: Array2D
    roof_competence: Array2D
    growth_cost: Array2D
    gradient_x: Array2D
    gradient_y: Array2D

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            float(self.x_coords[0]),
            float(self.x_coords[-1]),
            float(self.y_coords[0]),
            float(self.y_coords[-1]),
        )

    def summary(self) -> dict[str, float]:
        """Return small scalar summaries for logging and quick inspection."""

        return {
            "elevation_min": float(self.elevation.min()),
            "elevation_max": float(self.elevation.max()),
            "slope_mean_deg": float(self.slope_degrees.mean()),
            "cover_thickness_mean": float(self.cover_thickness.mean()),
            "roof_competence_mean": float(self.roof_competence.mean()),
            "growth_cost_mean": float(self.growth_cost.mean()),
        }

    def sample(self, x_coord: float, y_coord: float) -> HostFieldSample:
        """Sample all fields at one position for future graph growth logic."""

        return HostFieldSample(
            elevation=self._bilinear_sample(self.elevation, x_coord, y_coord),
            slope_degrees=self._bilinear_sample(self.slope_degrees, x_coord, y_coord),
            cover_thickness=self._bilinear_sample(self.cover_thickness, x_coord, y_coord),
            roof_competence=self._bilinear_sample(self.roof_competence, x_coord, y_coord),
            growth_cost=self._bilinear_sample(self.growth_cost, x_coord, y_coord),
            gradient_x=self._bilinear_sample(self.gradient_x, x_coord, y_coord),
            gradient_y=self._bilinear_sample(self.gradient_y, x_coord, y_coord),
        )

    def contains(self, x_coord: float, y_coord: float, margin: float = 0.0) -> bool:
        """Return whether a coordinate is inside the host field bounds."""

        min_x, max_x, min_y, max_y = self.extent
        return (
            min_x + margin <= x_coord <= max_x - margin
            and min_y + margin <= y_coord <= max_y - margin
        )

    def downhill_direction(
        self,
        x_coord: float,
        y_coord: float,
        fallback_angle_degrees: float | None = None,
    ) -> tuple[float, float]:
        """Sample the downhill direction from the terrain gradient."""

        sample = self.sample(x_coord, y_coord)
        downhill_x = -sample.gradient_x
        downhill_y = -sample.gradient_y
        length = math.hypot(downhill_x, downhill_y)

        if math.isclose(length, 0.0):
            if fallback_angle_degrees is None:
                return 0.0, 0.0

            fallback_radians = math.radians(fallback_angle_degrees)
            return math.cos(fallback_radians), math.sin(fallback_radians)

        return downhill_x / length, downhill_y / length

    def _bilinear_sample(self, values: Array2D, x_coord: float, y_coord: float) -> float:
        x_position = self._coordinate_to_fractional_index(self.x_coords, x_coord)
        y_position = self._coordinate_to_fractional_index(self.y_coords, y_coord)

        x0 = min(int(math.floor(x_position)), len(self.x_coords) - 2)
        y0 = min(int(math.floor(y_position)), len(self.y_coords) - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        tx = x_position - x0
        ty = y_position - y0

        top = (1.0 - tx) * values[y0, x0] + tx * values[y0, x1]
        bottom = (1.0 - tx) * values[y1, x0] + tx * values[y1, x1]
        return float((1.0 - ty) * top + ty * bottom)

    @staticmethod
    def _coordinate_to_fractional_index(coords: Array1D, value: float) -> float:
        if not float(coords[0]) <= value <= float(coords[-1]):
            raise ValueError(
                f"Coordinate {value} is outside the host field bounds "
                f"[{float(coords[0])}, {float(coords[-1])}]"
            )

        if math.isclose(value, float(coords[-1])):
            return float(len(coords) - 1)

        spacing = float(coords[1] - coords[0])
        return (value - float(coords[0])) / spacing


class HostFieldGenerator:
    """Generate the low-frequency host field that constrains lava tube growth."""

    def __init__(self, config: HostFieldConfig | None = None) -> None:
        self.config = config or HostFieldConfig()

    def generate(self) -> HostField:
        x_coords, y_coords = self._build_axes()
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        elevation = self._build_terrain(x_grid, y_grid)
        gradient_y, gradient_x = self._build_gradient(elevation)
        slope_degrees = self._build_slope_degrees(gradient_x, gradient_y)
        cover_thickness = self._build_cover_thickness(elevation, slope_degrees)
        roof_competence = self._build_roof_competence(x_grid, y_grid)
        growth_cost = self._build_growth_cost(
            slope_degrees=slope_degrees,
            cover_thickness=cover_thickness,
            roof_competence=roof_competence,
        )

        return HostField(
            config=self.config,
            x_coords=x_coords,
            y_coords=y_coords,
            elevation=elevation,
            slope_degrees=slope_degrees,
            cover_thickness=cover_thickness,
            roof_competence=roof_competence,
            growth_cost=growth_cost,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )

    def _build_axes(self) -> tuple[Array1D, Array1D]:
        grid = self.config.grid
        x_coords = np.linspace(-grid.width / 2.0, grid.width / 2.0, grid.nx, dtype=float)
        y_coords = np.linspace(-grid.height / 2.0, grid.height / 2.0, grid.ny, dtype=float)
        return x_coords, y_coords

    def _build_terrain(self, x_grid: Array2D, y_grid: Array2D) -> Array2D:
        seed_x, seed_y = self.config.seed_point
        relative_x = x_grid - seed_x
        relative_y = y_grid - seed_y

        flow_projection = self._project_along_angle(
            x_grid,
            y_grid,
            self.config.flow_angle_degrees,
        )
        cross_projection = self._project_along_angle(
            relative_x,
            relative_y,
            self.config.flow_angle_degrees + 90.0,
        )

        flow_min, flow_max = self._projected_bounds(self.config.flow_angle_degrees)
        normalized_flow = (flow_projection - flow_min) / max(flow_max - flow_min, 1.0)

        terrain = (
            self.config.high_side_elevation
            - self.config.longitudinal_drop * normalized_flow
        )
        terrain -= self.config.corridor_depth * np.exp(
            -np.square(cross_projection / self.config.corridor_width)
        )

        for wave in self.config.waves:
            directional_offset = self._project_along_angle(
                relative_x,
                relative_y,
                wave.angle_degrees,
            )
            terrain += wave.amplitude * np.sin(
                2.0 * math.pi * directional_offset / wave.wavelength + wave.phase
            )

        return terrain

    def _build_gradient(self, elevation: Array2D) -> tuple[Array2D, Array2D]:
        grid = self.config.grid
        return np.gradient(
            elevation,
            grid.spacing_y,
            grid.spacing_x,
        )

    def _build_slope_degrees(self, gradient_x: Array2D, gradient_y: Array2D) -> Array2D:
        slope_rise = np.hypot(gradient_x, gradient_y)
        return np.degrees(np.arctan(slope_rise))

    def _build_cover_thickness(
        self,
        elevation: Array2D,
        slope_degrees: Array2D,
    ) -> Array2D:
        relief_bonus = 0.18 * (elevation - float(elevation.mean()))
        slope_penalty = 0.45 * slope_degrees
        cover = self.config.volcanic_layer_thickness + relief_bonus - slope_penalty

        minimum_cover = self.config.minimum_stable_cover
        maximum_cover = self.config.volcanic_layer_thickness * 1.35
        return np.clip(cover, minimum_cover, maximum_cover)

    def _build_roof_competence(self, x_grid: Array2D, y_grid: Array2D) -> Array2D:
        """Build an explicit roof-stability field for later geometry and texturing."""

        seed_x, seed_y = self.config.seed_point
        relative_x = x_grid - seed_x
        relative_y = y_grid - seed_y

        drainage = self._project_along_angle(
            relative_x,
            relative_y,
            self.config.flow_angle_degrees,
        )
        cross_drainage = self._project_along_angle(
            relative_x,
            relative_y,
            self.config.flow_angle_degrees + 90.0,
        )
        fracture_axis = self._project_along_angle(
            relative_x,
            relative_y,
            self.config.fracture_zone_angle_degrees,
        )

        structural_bands = (
            0.55 * np.sin(2.0 * math.pi * drainage / 1550.0 + 0.35)
            + 0.45 * np.cos(2.0 * math.pi * cross_drainage / 980.0 - 0.2)
        )
        fracture_zone = np.exp(
            -np.square(
                (fracture_axis - self.config.fracture_zone_center_offset)
                / self.config.fracture_zone_width
            )
        )
        edge_weathering = np.clip(
            np.abs(cross_drainage) / (self.config.grid.width * 0.5),
            0.0,
            1.0,
        )

        competence = (
            self.config.roof_competence_baseline
            + self.config.roof_competence_variation * structural_bands
            - 0.32 * fracture_zone
            - 0.10 * edge_weathering
        )
        return np.clip(competence, 0.0, 1.0)

    def _build_growth_cost(
        self,
        *,
        slope_degrees: Array2D,
        cover_thickness: Array2D,
        roof_competence: Array2D,
    ) -> Array2D:
        slope_penalty = self._normalize_percentile(slope_degrees)
        cover_penalty = 1.0 - np.clip(
            (cover_thickness - self.config.minimum_stable_cover)
            / max(
                self.config.volcanic_layer_thickness - self.config.minimum_stable_cover,
                1.0,
            ),
            0.0,
            1.0,
        )
        competence_penalty = 1.0 - roof_competence

        growth_cost = 0.38 * slope_penalty + 0.22 * cover_penalty + 0.40 * competence_penalty
        return np.clip(growth_cost, 0.0, 1.0)

    @staticmethod
    def _project_along_angle(
        x_values: Array2D,
        y_values: Array2D,
        angle_degrees: float,
    ) -> Array2D:
        angle_radians = math.radians(angle_degrees)
        return math.cos(angle_radians) * x_values + math.sin(angle_radians) * y_values

    def _projected_bounds(self, angle_degrees: float) -> tuple[float, float]:
        half_width = self.config.grid.width / 2.0
        half_height = self.config.grid.height / 2.0

        x_corners = np.array(
            [-half_width, -half_width, half_width, half_width],
            dtype=float,
        )
        y_corners = np.array(
            [-half_height, half_height, -half_height, half_height],
            dtype=float,
        )
        projected = self._project_along_angle(x_corners, y_corners, angle_degrees)
        return float(projected.min()), float(projected.max())

    @staticmethod
    def _normalize_percentile(
        values: Array2D,
        *,
        lower: float = 5.0,
        upper: float = 95.0,
    ) -> Array2D:
        lower_value, upper_value = np.percentile(values, [lower, upper])
        if math.isclose(float(lower_value), float(upper_value)):
            return np.zeros_like(values, dtype=float)

        clipped = np.clip(values, lower_value, upper_value)
        return (clipped - lower_value) / (upper_value - lower_value)
