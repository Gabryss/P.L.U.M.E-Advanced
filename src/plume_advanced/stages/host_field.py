"""Stage A: low-frequency host field generation.

The host field defines the broad geological constraints that later stages will
use to grow believable conduit centerlines. This stage intentionally avoids
high-frequency noise and instead focuses on smooth, readable proxies:

- terrain elevation
- slope
- emplacement and cover thickness
- lithology, fracture, cooling, deposit, and erosion proxies
- flow capacity and gravity-aware roof stability
- one explicit routing cost with inspectable component penalties
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from plume_advanced.procedural import procedural_rng

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


class HostVariation(TypedDict):
    wave_phase_offsets: tuple[float, ...]
    corridor_depth_scale: Array2D
    corridor_width_scale: Array2D
    competence_bias: Array2D
    fracture_center_offset: float
    fracture_width_scale: float
    drainage_phase_offset: float
    cross_drainage_phase_offset: float


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
class RoutingWeights:
    """Inspectable weights for the process-informed routing surrogate.

    The defaults are the baseline Stage-A weights.  ``resolved`` is the only
    path used by generation so evaluation ablations cannot accidentally apply
    a different normalization convention.
    """

    slope: float = 0.12
    cover: float = 0.10
    fracture: float = 0.22
    capacity: float = 0.28
    stability: float = 0.28
    normalize: bool = True
    enabled: bool = True

    def resolved(self) -> dict[str, float]:
        values = {
            "slope": float(self.slope),
            "cover": float(self.cover),
            "fracture": float(self.fracture),
            "capacity": float(self.capacity),
            "stability": float(self.stability),
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
            raise ValueError("host_field.routing_weights must be finite and non-negative")
        total = sum(values.values())
        if self.enabled and total <= 0.0:
            raise ValueError("at least one host_field.routing_weights value must be positive")
        if not self.enabled:
            return {name: 0.0 for name in values}
        if self.normalize:
            return {name: value / total for name, value in values.items()}
        return values

    def without(self, component: str) -> "RoutingWeights":
        """Return a single-term ablation with the remaining weights normalized."""

        if component not in {"slope", "cover", "fracture", "capacity", "stability"}:
            raise ValueError(f"Unknown routing component: {component}")
        values = {
            "slope": self.slope,
            "cover": self.cover,
            "fracture": self.fracture,
            "capacity": self.capacity,
            "stability": self.stability,
        }
        values[component] = 0.0
        return RoutingWeights(**values, normalize=True, enabled=self.enabled)


@dataclass(frozen=True)
class HostFieldConfig:
    """Parameters controlling stage-A host field generation."""

    grid: GridConfig = field(default_factory=GridConfig)
    random_seed: int | None = None
    body_spatial_scale: float = 1.0
    body_vertical_scale: float = 1.0
    body_fracture_scale: float = 1.0
    target_route_length_m: float = 5_000.0
    seed_point: tuple[float, float] = (-1200.0, 0.0)
    high_side_elevation: float = 182.0
    longitudinal_drop: float = 84.0
    flow_angle_degrees: float = 0.0
    corridor_depth: float = 12.0
    corridor_width: float = 520.0
    corridor_count: int = 1
    corridor_spacing: float = 60.0
    corridor_lateral_variation: float = 25.0
    corridor_correlation_length: float = 250.0
    volcanic_layer_thickness: float = 64.0
    minimum_stable_cover: float = 18.0
    roof_competence_baseline: float = 0.72
    roof_competence_variation: float = 0.18
    fracture_zone_angle_degrees: float = 82.0
    fracture_zone_center_offset: float = -140.0
    fracture_zone_width: float = 240.0
    gravity_m_s2: float = 9.80665
    rock_density_kg_m3: float = 2_900.0
    effective_tensile_strength_pa: float = 3_000_000.0
    material_quality: float = 0.55
    material_weathering: float = 0.30
    characteristic_passage_span_m: float = 10.0
    routing_weights: RoutingWeights = field(default_factory=RoutingWeights)
    waves: tuple[TerrainWave, ...] = field(default_factory=_default_waves)

    def __post_init__(self):
        if type(self.corridor_count) is not int or not 1 <= self.corridor_count <= 8:
            raise ValueError("host_field.corridor_count must be an integer in [1, 8]")
        for name in ("corridor_spacing", "corridor_lateral_variation", "corridor_correlation_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"host_field.{name} must be finite and positive")


@dataclass(frozen=True)
class HostFieldSample:
    """Bilinear sample of the host field at a single point."""

    elevation: float
    slope_degrees: float
    cover_thickness: float
    roof_competence: float
    growth_cost: float
    emplacement_thickness: float
    lithology_quality: float
    fracture_intensity: float
    cooling_index: float
    flow_capacity: float
    deposit_thickness: float
    erosion_index: float
    roof_stability: float
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
    emplacement_thickness: Array2D
    lithology_quality: Array2D
    fracture_intensity: Array2D
    cooling_index: Array2D
    flow_capacity: Array2D
    deposit_thickness: Array2D
    erosion_index: Array2D
    roof_stability: Array2D
    routing_slope_penalty: Array2D
    routing_cover_penalty: Array2D
    routing_fracture_penalty: Array2D
    routing_capacity_penalty: Array2D
    routing_stability_penalty: Array2D
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
            "fracture_intensity_mean": float(self.fracture_intensity.mean()),
            "flow_capacity_mean": float(self.flow_capacity.mean()),
            "roof_stability_mean": float(self.roof_stability.mean()),
        }

    def routing_influence_summary(self) -> dict[str, dict[str, float]]:
        """Quantify whether every named routing term has a measurable effect."""

        terms = {
            "slope": self.routing_slope_penalty,
            "cover": self.routing_cover_penalty,
            "fracture": self.routing_fracture_penalty,
            "capacity": self.routing_capacity_penalty,
            "stability": self.routing_stability_penalty,
        }
        result: dict[str, dict[str, float]] = {}
        cost_flat = self.growth_cost.ravel()
        for name, term in terms.items():
            flat = term.ravel()
            correlation = (
                float(np.corrcoef(flat, cost_flat)[0, 1]) if float(np.std(flat)) > 1e-12 else 0.0
            )
            result[name] = {
                "minimum": float(np.min(flat)),
                "maximum": float(np.max(flat)),
                "mean": float(np.mean(flat)),
                "standard_deviation": float(np.std(flat)),
                "mean_absolute_contribution": float(np.mean(np.abs(flat))),
                "correlation_with_routing_cost": correlation,
            }
        return result

    def sample(self, x_coord: float, y_coord: float) -> HostFieldSample:
        """Sample all fields at one position for future graph growth logic."""

        return HostFieldSample(
            elevation=self._bilinear_sample(self.elevation, x_coord, y_coord),
            slope_degrees=self._bilinear_sample(self.slope_degrees, x_coord, y_coord),
            cover_thickness=self._bilinear_sample(self.cover_thickness, x_coord, y_coord),
            roof_competence=self._bilinear_sample(self.roof_competence, x_coord, y_coord),
            growth_cost=self._bilinear_sample(self.growth_cost, x_coord, y_coord),
            emplacement_thickness=self._bilinear_sample(
                self.emplacement_thickness, x_coord, y_coord
            ),
            lithology_quality=self._bilinear_sample(self.lithology_quality, x_coord, y_coord),
            fracture_intensity=self._bilinear_sample(self.fracture_intensity, x_coord, y_coord),
            cooling_index=self._bilinear_sample(self.cooling_index, x_coord, y_coord),
            flow_capacity=self._bilinear_sample(self.flow_capacity, x_coord, y_coord),
            deposit_thickness=self._bilinear_sample(self.deposit_thickness, x_coord, y_coord),
            erosion_index=self._bilinear_sample(self.erosion_index, x_coord, y_coord),
            roof_stability=self._bilinear_sample(self.roof_stability, x_coord, y_coord),
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
        variation = self._build_variation_fields(x_grid, y_grid)

        elevation = self._build_terrain(x_grid, y_grid, variation)
        gradient_y, gradient_x = self._build_gradient(elevation)
        slope_degrees = self._build_slope_degrees(gradient_x, gradient_y)
        process = self._build_process_layers(
            x_grid=x_grid,
            y_grid=y_grid,
            slope_degrees=slope_degrees,
            variation=variation,
        )
        cover_thickness = process["cover_thickness"]
        roof_competence = process["roof_competence"]
        growth_cost, routing_terms = self._build_growth_cost(
            slope_degrees=slope_degrees,
            cover_thickness=cover_thickness,
            fracture_intensity=process["fracture_intensity"],
            flow_capacity=process["flow_capacity"],
            roof_stability=process["roof_stability"],
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
            emplacement_thickness=process["emplacement_thickness"],
            lithology_quality=process["lithology_quality"],
            fracture_intensity=process["fracture_intensity"],
            cooling_index=process["cooling_index"],
            flow_capacity=process["flow_capacity"],
            deposit_thickness=process["deposit_thickness"],
            erosion_index=process["erosion_index"],
            roof_stability=process["roof_stability"],
            routing_slope_penalty=routing_terms["slope"],
            routing_cover_penalty=routing_terms["cover"],
            routing_fracture_penalty=routing_terms["fracture"],
            routing_capacity_penalty=routing_terms["capacity"],
            routing_stability_penalty=routing_terms["stability"],
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )

    def _build_axes(self) -> tuple[Array1D, Array1D]:
        grid = self.config.grid
        x_coords = np.linspace(-grid.width / 2.0, grid.width / 2.0, grid.nx, dtype=float)
        y_coords = np.linspace(-grid.height / 2.0, grid.height / 2.0, grid.ny, dtype=float)
        return x_coords, y_coords

    def _build_variation_fields(
        self,
        x_grid: Array2D,
        y_grid: Array2D,
    ) -> HostVariation:
        rng = procedural_rng(self.config.random_seed, "host-field-variation")
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

        wave_phase_offsets = tuple(float(rng.uniform(-0.25, 0.25)) for _ in self.config.waves)

        long_wavelength = float(rng.uniform(1800.0, 3400.0))
        cross_wavelength = float(rng.uniform(950.0, 1800.0))
        corridor_depth_scale = np.clip(
            1.0
            + 0.10
            * np.sin(2.0 * math.pi * drainage / long_wavelength + rng.uniform(-math.pi, math.pi))
            + 0.03
            * np.cos(
                2.0 * math.pi * cross_drainage / cross_wavelength + rng.uniform(-math.pi, math.pi)
            ),
            0.82,
            1.18,
        )
        corridor_width_scale = np.clip(
            1.0
            + 0.08
            * np.cos(
                2.0 * math.pi * drainage / (long_wavelength * 0.82) + rng.uniform(-math.pi, math.pi)
            )
            + 0.03
            * np.sin(
                2.0 * math.pi * cross_drainage / (cross_wavelength * 1.35)
                + rng.uniform(-math.pi, math.pi)
            ),
            0.88,
            1.18,
        )

        fracture_center_offset = float(rng.normal(0.0, self.config.fracture_zone_width * 0.12))
        fracture_width_scale = float(np.clip(rng.normal(1.0, 0.08), 0.88, 1.18))
        drainage_phase_offset = float(rng.uniform(-0.25, 0.25))
        cross_drainage_phase_offset = float(rng.uniform(-0.20, 0.20))

        competence_bias = np.zeros_like(x_grid, dtype=float)
        along_min = float(drainage.min())
        along_max = float(drainage.max())
        cross_span = max(self.config.grid.width * 0.42, 1.0)
        pod_count = int(rng.integers(1, 4))
        for _ in range(pod_count):
            along_center = float(rng.uniform(along_min, along_max))
            cross_center = float(rng.uniform(-cross_span, cross_span))
            along_sigma = float(rng.uniform(320.0, 880.0))
            cross_sigma = float(rng.uniform(110.0, 300.0))
            amplitude = float(rng.uniform(-0.05, 0.05))
            competence_bias += amplitude * np.exp(
                -np.square((drainage - along_center) / along_sigma)
                - np.square((cross_drainage - cross_center) / cross_sigma)
            )

        if rng.random() < 0.7:
            band_center = float(
                self.config.fracture_zone_center_offset
                + rng.normal(0.0, self.config.fracture_zone_width * 0.25)
            )
            band_width = float(self.config.fracture_zone_width * rng.uniform(0.85, 1.45))
            band_strength = float(rng.uniform(-0.03, 0.04))
            competence_bias += band_strength * np.exp(
                -np.square((fracture_axis - band_center) / max(band_width, 1.0))
            )

        return {
            "wave_phase_offsets": wave_phase_offsets,
            "corridor_depth_scale": corridor_depth_scale,
            "corridor_width_scale": corridor_width_scale,
            "competence_bias": competence_bias,
            "fracture_center_offset": fracture_center_offset,
            "fracture_width_scale": fracture_width_scale,
            "drainage_phase_offset": drainage_phase_offset,
            "cross_drainage_phase_offset": cross_drainage_phase_offset,
        }

    def _build_terrain(
        self,
        x_grid: Array2D,
        y_grid: Array2D,
        variation: HostVariation,
    ) -> Array2D:
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

        terrain = self.config.high_side_elevation - self.config.longitudinal_drop * normalized_flow
        corridor_width = self.config.corridor_width * variation["corridor_width_scale"]
        corridor_depth = self.config.corridor_depth * variation["corridor_depth_scale"]
        along = self._project_along_angle(relative_x, relative_y, self.config.flow_angle_degrees)
        terrain -= corridor_depth * self._corridor_envelope(along, cross_projection, corridor_width)

        wave_phase_offsets = variation["wave_phase_offsets"]
        for index, wave in enumerate(self.config.waves):
            directional_offset = self._project_along_angle(
                relative_x,
                relative_y,
                wave.angle_degrees,
            )
            terrain += wave.amplitude * np.sin(
                2.0 * math.pi * directional_offset / wave.wavelength
                + wave.phase
                + wave_phase_offsets[index]
            )

        return terrain

    def _corridor_envelope(self, along, cross, width):
        """One shared terrain/process field; corridors are not network paths.

        The single-corridor expression is unchanged. Multiple broad, independently
        varying troughs may overlap; a smooth bounded union avoids multiplying
        excavation depth at a confluence. No network seed or event enters Stage A.
        """
        if self.config.corridor_count == 1:
            return np.exp(-np.square(cross / np.maximum(width, 1.0)))
        from scipy.interpolate import CubicSpline

        cfg = self.config
        lo, hi = float(along.min()), float(along.max())
        knots = np.linspace(lo, hi, max(4, int(np.ceil((hi-lo) / cfg.corridor_correlation_length))+1))
        complement = np.ones_like(cross)
        for i in range(cfg.corridor_count):
            rng = procedural_rng(cfg.random_seed, "host-corridor", i)
            offsets = rng.uniform(-cfg.corridor_lateral_variation, cfg.corridor_lateral_variation, len(knots))
            centre = (i - (cfg.corridor_count-1)/2) * cfg.corridor_spacing
            centre = centre + CubicSpline(knots, offsets, bc_type="natural")(along)
            complement *= 1 - np.exp(-np.square((cross-centre) / np.maximum(width, 1.0)))
        return 1 - complement

    def _build_gradient(self, elevation: Array2D) -> tuple[Array2D, Array2D]:
        grid = self.config.grid
        gradient_y, gradient_x = np.gradient(
            elevation,
            grid.spacing_y,
            grid.spacing_x,
        )
        return (
            np.asarray(gradient_y, dtype=np.float64),
            np.asarray(gradient_x, dtype=np.float64),
        )

    def _build_slope_degrees(self, gradient_x: Array2D, gradient_y: Array2D) -> Array2D:
        slope_rise = np.hypot(gradient_x, gradient_y)
        return np.degrees(np.arctan(slope_rise))

    def _build_process_layers(
        self,
        *,
        x_grid: Array2D,
        y_grid: Array2D,
        slope_degrees: Array2D,
        variation: HostVariation,
    ) -> dict[str, Array2D]:
        """Build distinct causal proxies before combining them for routing."""

        seed_x, seed_y = self.config.seed_point
        relative_x = x_grid - seed_x
        relative_y = y_grid - seed_y
        along = self._project_along_angle(relative_x, relative_y, self.config.flow_angle_degrees)
        cross = self._project_along_angle(
            relative_x, relative_y, self.config.flow_angle_degrees + 90.0
        )
        fracture_axis = self._project_along_angle(
            relative_x, relative_y, self.config.fracture_zone_angle_degrees
        )
        along_norm = self._normalize_percentile(along, lower=0.0, upper=100.0)
        edge_norm = np.clip(np.abs(cross) / max(0.5 * self.config.grid.width, 1.0), 0.0, 1.0)
        corridor = self._corridor_envelope(
            along, cross, self.config.corridor_width * variation["corridor_width_scale"]
        )

        emplacement_factor = np.clip(
            0.72
            + 0.25 * corridor
            + 0.10 * np.sin(2.0 * math.pi * along / 1850.0 + 0.4)
            + 0.06 * np.cos(2.0 * math.pi * cross / 730.0 - 0.8),
            0.48,
            1.22,
        )
        emplacement = self.config.volcanic_layer_thickness * emplacement_factor

        fracture_zone = np.exp(
            -np.square(
                (
                    fracture_axis
                    - self.config.fracture_zone_center_offset
                    - float(variation["fracture_center_offset"])
                )
                / max(
                    self.config.fracture_zone_width * float(variation["fracture_width_scale"]),
                    1.0,
                )
            )
        )
        joint_bands = 0.5 + 0.5 * np.sin(
            2.0 * math.pi * along / 1080.0
            - 2.0 * math.pi * cross / 1460.0
            + float(variation["drainage_phase_offset"])
        )
        fracture_intensity = np.clip(
            0.12 + 0.62 * fracture_zone + 0.16 * joint_bands + 0.08 * edge_norm,
            0.0,
            1.0,
        )

        cooling_index = np.clip(
            0.18
            + 0.52 * along_norm
            + 0.25 * edge_norm
            + 0.08 * np.sin(2.0 * math.pi * along / 920.0),
            0.0,
            1.0,
        )
        erosion_index = np.clip(
            0.08
            + 0.62 * self.config.material_weathering
            + 0.18 * self._normalize_percentile(slope_degrees)
            + 0.15 * edge_norm,
            0.0,
            1.0,
        )
        deposit_thickness = np.clip(
            self.config.volcanic_layer_thickness
            * (0.015 + 0.10 * erosion_index * (0.35 + 0.65 * (1.0 - corridor))),
            0.0,
            0.18 * self.config.volcanic_layer_thickness,
        )
        cover = np.clip(
            emplacement
            - erosion_index * 0.16 * self.config.volcanic_layer_thickness
            + 0.25 * deposit_thickness,
            self.config.minimum_stable_cover,
            1.35 * self.config.volcanic_layer_thickness,
        )

        lithology_seed = self._build_roof_competence(x_grid, y_grid, variation)
        cooling_quality = 1.0 - 0.55 * np.abs(cooling_index - 0.52)
        lithology_quality = np.clip(
            0.46 * lithology_seed + 0.34 * self.config.material_quality + 0.20 * cooling_quality,
            0.0,
            1.0,
        )
        roof_competence = np.clip(
            lithology_quality * (1.0 - 0.68 * fracture_intensity) * (1.0 - 0.28 * erosion_index),
            0.0,
            1.0,
        )
        emplacement_norm = np.clip(
            emplacement / max(self.config.volcanic_layer_thickness, 1.0),
            0.0,
            1.4,
        )
        flow_capacity = np.clip(
            0.46 * corridor
            + 0.34 * emplacement_norm
            + 0.20 * (1.0 - fracture_intensity)
            - 0.18 * deposit_thickness / max(self.config.volcanic_layer_thickness, 1.0),
            0.0,
            1.0,
        )
        demand_ratio = (
            self.config.rock_density_kg_m3
            * self.config.gravity_m_s2
            * self.config.characteristic_passage_span_m**2
            / np.maximum(cover, 0.1)
            / max(self.config.effective_tensile_strength_pa, 1.0)
        )
        roof_stability = np.clip(
            roof_competence / (1.0 + demand_ratio),
            0.0,
            1.0,
        )
        return {
            "emplacement_thickness": emplacement,
            "cover_thickness": cover,
            "lithology_quality": lithology_quality,
            "fracture_intensity": fracture_intensity,
            "cooling_index": cooling_index,
            "flow_capacity": flow_capacity,
            "deposit_thickness": deposit_thickness,
            "erosion_index": erosion_index,
            "roof_competence": roof_competence,
            "roof_stability": roof_stability,
        }

    def _build_roof_competence(
        self,
        x_grid: Array2D,
        y_grid: Array2D,
        variation: HostVariation,
    ) -> Array2D:
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

        structural_bands = 0.55 * np.sin(
            2.0 * math.pi * drainage / 1550.0 + 0.35 + variation["drainage_phase_offset"]
        ) + 0.45 * np.cos(
            2.0 * math.pi * cross_drainage / 980.0 - 0.2 + variation["cross_drainage_phase_offset"]
        )
        fracture_zone = np.exp(
            -np.square(
                (
                    fracture_axis
                    - (
                        self.config.fracture_zone_center_offset
                        + variation["fracture_center_offset"]
                    )
                )
                / max(
                    self.config.fracture_zone_width * variation["fracture_width_scale"],
                    1.0,
                )
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
            + variation["competence_bias"]
        )
        return np.clip(competence, 0.0, 1.0)

    def _build_growth_cost(
        self,
        *,
        slope_degrees: Array2D,
        cover_thickness: Array2D,
        fracture_intensity: Array2D,
        flow_capacity: Array2D,
        roof_stability: Array2D,
    ) -> tuple[Array2D, dict[str, Array2D]]:
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
        terms = {
            name: self.config.routing_weights.resolved()[name] * penalty
            for name, penalty in {
                "slope": slope_penalty,
                "cover": cover_penalty,
                "fracture": fracture_intensity,
                "capacity": 1.0 - flow_capacity,
                "stability": 1.0 - roof_stability,
            }.items()
        }
        growth_cost = (
            sum(terms.values())
            if self.config.routing_weights.enabled
            else np.full_like(slope_penalty, 0.5, dtype=float)
        )
        return np.clip(growth_cost, 0.0, 1.0), terms

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


def export_host_influence_report(
    host_field: HostField,
    output_path: str | Path,
    *,
    generation_context: dict[str, object] | None = None,
) -> Path:
    """Write the measurable contribution of every routing term."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "plume.host-routing-influence.v2",
        "seed": host_field.config.random_seed,
        "grid": {
            "nx": host_field.config.grid.nx,
            "ny": host_field.config.grid.ny,
            "spacing_x_m": host_field.config.grid.spacing_x,
            "spacing_y_m": host_field.config.grid.spacing_y,
        },
        "routing_enabled": host_field.config.routing_weights.enabled,
        "routing_formula": host_field.config.routing_weights.resolved(),
        "generation_context": generation_context or {},
        "field_summary": {
            name: {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
            }
            for name, values in {
                "elevation": host_field.elevation,
                "slope_degrees": host_field.slope_degrees,
                "cover_thickness": host_field.cover_thickness,
                "roof_competence": host_field.roof_competence,
                "fracture_intensity": host_field.fracture_intensity,
                "flow_capacity": host_field.flow_capacity,
                "roof_stability": host_field.roof_stability,
                "routing_cost": host_field.growth_cost,
            }.items()
        },
        "influence": host_field.routing_influence_summary(),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


__all__ = [
    "GridConfig",
    "HostField",
    "HostFieldConfig",
    "HostFieldGenerator",
    "HostFieldSample",
    "RoutingWeights",
    "TerrainWave",
    "export_host_influence_report",
]
