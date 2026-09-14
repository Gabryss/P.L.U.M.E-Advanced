"""Explicit acceptance requirements shared by generation, export and evaluation.

Policies describe the checks needed for publication, independently of resolution
selection (run.quality). Missing evidence never satisfies a required check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plume_advanced.stages.geometry_types import CaveGeometry, GeometryConfig
    from plume_advanced.world import ExportConfig


_PROFILE_REQUIREMENTS = {
    "research": {},
    "inspection": dict(require_clearance=True, require_collision=True, require_export_budgets=True),
    "simulation": dict(require_clearance=True, require_collision=True,
                       require_export_budgets=True, require_resolution=True),
}


@dataclass(frozen=True)
class AcceptancePolicy:
    profile: str = "research"
    require_clearance: bool = False
    require_collision: bool = False
    require_export_budgets: bool = False
    require_resolution: bool = False
    require_textures: bool = False
    require_native: bool = False
    route_height_m: float = 0.5
    route_width_m: float = 0.5
    route_margin_m: float = 0.02
    minimum_relief_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or self.profile not in _PROFILE_REQUIREMENTS:
            raise ValueError("acceptance.profile must be research, inspection or simulation")
        for field in fields(self):
            if field.name.startswith("require_") and type(getattr(self, field.name)) is not bool:
                raise ValueError(f"acceptance.{field.name} must be a boolean")
        for name in ("route_height_m", "route_width_m", "route_margin_m", "minimum_relief_scale"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"acceptance.{name} must be finite and nonnegative")
        if not 0 < self.route_width_m <= self.route_height_m:
            raise ValueError("acceptance requires 0 < route_width_m <= route_height_m")
        if self.minimum_relief_scale > 1:
            raise ValueError("acceptance.minimum_relief_scale must be in [0, 1]")
        for name in _PROFILE_REQUIREMENTS[self.profile]:
            if not getattr(self, name):
                raise ValueError(f"acceptance.{name} cannot be disabled for {self.profile}")
        if self.profile == "simulation" and self.minimum_relief_scale <= 0:
            raise ValueError("simulation requires a positive acceptance.minimum_relief_scale")


def build_acceptance_policy(raw: dict | None = None) -> AcceptancePolicy:
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("acceptance must be a TOML table")
    data = {} if raw is None else dict(raw)
    unknown = set(data) - {field.name for field in fields(AcceptancePolicy)}
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(
            f"acceptance.{name}" for name in sorted(unknown)))
    profile = data.get("profile", "research")
    if not isinstance(profile, str) or profile not in _PROFILE_REQUIREMENTS:
        raise ValueError("acceptance.profile must be research, inspection or simulation")
    defaults: dict[str, Any] = dict(_PROFILE_REQUIREMENTS[profile])
    if profile == "simulation":
        defaults["minimum_relief_scale"] = 1.0
    return AcceptancePolicy(**(defaults | data))


def apply_acceptance_defaults(policy: AcceptancePolicy, geometry: dict, export: dict) -> tuple[dict, dict]:
    """Fill absent controls only; explicit contradictory settings are rejected later."""
    geometry, export = dict(geometry), dict(export)
    if policy.require_clearance:
        for key, value in (("required_route_height_m", policy.route_height_m),
                           ("required_route_width_m", policy.route_width_m),
                           ("route_clearance_margin_m", policy.route_margin_m)):
            geometry.setdefault(key, value)
    if policy.require_collision:
        export.setdefault("generate_collision", True)
    if policy.require_export_budgets:
        export.setdefault("max_visual_triangles", 2_000_000)
        export.setdefault("max_asset_bytes", 256 * 1024 * 1024)
    if policy.require_resolution:
        geometry.setdefault("resolution_refinement_attempts", 2)
    return geometry, export


def validate_acceptance_configuration(
    policy: AcceptancePolicy, geometry: GeometryConfig, export: ExportConfig,
) -> None:
    """Also called on direct export requests and effective campaign overrides."""
    if policy.require_clearance:
        for key, minimum in (("required_route_height_m", policy.route_height_m),
                             ("required_route_width_m", policy.route_width_m),
                             ("route_clearance_margin_m", policy.route_margin_m)):
            if getattr(geometry, key) < minimum:
                raise ValueError(f"geometry.{key} must be at least {minimum:g} for the acceptance policy")
    if policy.require_collision and not export.generate_collision:
        raise ValueError("acceptance.require_collision needs export.generate_collision = true")
    if policy.require_export_budgets:
        for key in ("max_visual_triangles", "max_asset_bytes"):
            if type(getattr(export, key)) is not int or getattr(export, key) <= 0:
                raise ValueError(f"acceptance.require_export_budgets needs positive export.{key}")
    if policy.require_textures and not all(getattr(geometry, f"cave_{role}_texture")
                                          for role in ("diffuse", "normal", "roughness")):
        raise ValueError("acceptance.require_textures needs diffuse, normal and roughness maps")


class AcceptanceError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        failures = [f"{name}: {row['status']} ({row['detail']})"
                    for name, row in report["checks"].items()
                    if row["required"] and row["status"] != "passed"]
        super().__init__("Acceptance requirements not met: " + "; ".join(failures))


def _check(required: bool, passed: bool | None, detail: str, **evidence) -> dict:
    status = "not_requested" if not required else (
        "unavailable" if passed is None else "passed" if passed else "failed")
    return dict(required=required, status=status, detail=detail, **evidence)


def require_available_acceptance(policy: AcceptancePolicy) -> None:
    """Native checks are not yet part of the atomic publication transaction.

    Do not accept a hand-supplied or stale editor report to bypass that limitation.
    Separate native campaigns remain useful evidence for their recorded assets.
    """
    if policy.require_native:
        raise AcceptanceError(dict(schema="plume.acceptance.v1", policy=asdict(policy), passed=False,
            checks={"native": _check(True, None,
                "Native validation is not integrated into publication; this requirement cannot yet be satisfied")},
            phase="preflight"))


def _resolution_check(resolution: dict | None, geometry: CaveGeometry) -> tuple[bool | None, str]:
    if not resolution or type(resolution.get("section_count")) is not int:
        return None, "Input-profile resolution evidence is missing"
    count, under = resolution["section_count"], resolution.get("under_resolved_count")
    if count <= 0 or type(under) is not int or not 0 <= under <= count:
        return False, "Input-profile resolution evidence is invalid or empty"
    if type(resolution.get("minimum_samples")) is not int:
        return None, "Input-profile resolution evidence has no screening threshold"
    if resolution["minimum_samples"] < 8:
        return False, "Input-profile resolution requires at least eight samples"
    if under == 0:
        return True, "All input profiles pass the eight-sample screen; not a full contour-convergence proof"
    repair: dict[str, Any] = dict(geometry.resolution_repair)
    attempts = repair.get("attempts", [])
    last = attempts[-1] if attempts else {}
    comparison = last.get("comparison", {})
    passed = (repair.get("passed") is True and repair.get("outcome") == "converged"
              and last.get("accepted") is True and comparison.get("passed") is True
              and comparison.get("topology_equal") is True
              and comparison.get("compared_samples", 0) > 0
              and not comparison.get("blocked_samples"))
    return passed, ("Under-resolved inputs require successful measured floor/roof convergence; "
                    "the study does not certify every surface feature")


def evaluate_acceptance(
    policy: AcceptancePolicy, geometry: CaveGeometry, package: dict,
    resolution: dict | None, export: ExportConfig | None,
) -> dict:
    """Evaluate current evidence, before publication and again at run completion."""
    checks = {}
    checks["mesh"] = _check(True, all(package.get(key, {}).get("passed") is True
                                     for key in ("raw", "visual", "serialized")),
                            "Raw, visual and serialized geometry checks")
    route_reports = [package.get(key, {}).get("traversal", {}) for key in ("raw", "visual")]
    if policy.require_collision:
        route_reports.append(package.get("collision", {}).get("inspection", {}).get("traversal", {}))
    expected_ids = list(geometry.route_path_segment_ids)
    route_passed = all(r.get("enabled") is True and r.get("passed") is True and r.get("paths")
                       and len(r["paths"]) == len(geometry.required_route_paths)
                       and [p.get("segment_id") for p in r["paths"]] == expected_ids
                       and all(p.get("passed") is True and p.get("samples", 0) >= 2 for p in r["paths"])
                       and r.get("height_m", 0) >= policy.route_height_m
                       and r.get("width_m", 0) >= policy.route_width_m
                       and r.get("margin_m", -1) >= policy.route_margin_m for r in route_reports)
    checks["clearance"] = _check(policy.require_clearance, bool(route_passed),
        "Continuous upright-body checks on raw/visual geometry and the required collider",
        height_m=policy.route_height_m, width_m=policy.route_width_m, margin_m=policy.route_margin_m)
    collision = package.get("collision", {})
    checks["collision"] = _check(policy.require_collision,
        collision.get("enabled") is True and collision.get("inspection", {}).get("passed") is True,
        "Dedicated collider with actual-mesh inspection and serialized package checks")
    resolution_passed, resolution_detail = (
        _resolution_check(resolution, geometry) if policy.require_resolution
        else (None, "Resolution acceptance not requested")
    )
    checks["resolution"] = _check(policy.require_resolution, resolution_passed, resolution_detail,
        input_profiles=resolution if policy.require_resolution else None,
        refinement=dict(geometry.resolution_repair) if policy.require_resolution else None)

    requested_relief = any(getattr(geometry.config, f"surface_{part}_relief_m") > 0
                           for part in ("wall", "roof", "floor", "crust"))
    regions: list[dict[str, Any]] = [dict(region) for region in geometry.effective_local_relief_regions]
    local = [float(region.get("scale", 0.)) for region in regions]
    scale = geometry.effective_surface_relief_scale * min([1., *local])
    checks["relief"] = _check(policy.minimum_relief_scale > 0 and requested_relief,
        math.isfinite(scale) and scale >= policy.minimum_relief_scale,
        "Lower bound on global/local relief setting factors; not measured surface-area or roughness fidelity",
        minimum_scale=policy.minimum_relief_scale, effective_minimum_scale=scale,
        global_scale=geometry.effective_surface_relief_scale, local_regions=len(local))
    texture = package.get("textures", {})
    checks["texture_integrity"] = _check(True, texture.get("passed") is True,
        "Integrity of declared maps and the material package, including neutral exports")
    checks["pbr_textures"] = _check(policy.require_textures,
        all(getattr(geometry.config, f"cave_{role}_texture")
            for role in ("diffuse", "normal", "roughness")) and texture.get("passed") is True,
        "Required diffuse, normal and roughness inputs with successful texture inspection")
    limits = None if export is None or not policy.require_export_budgets else dict(
        visual_triangles=export.max_visual_triangles, asset_bytes=export.max_asset_bytes)
    checks["export_budgets"] = _check(policy.require_export_budgets,
        None if limits is None else all(type(v) is int and v > 0 for v in limits.values()),
        "Finite export limits; the exporter enforces actual counts/bytes before publication", limits=limits)
    checks["native"] = _check(policy.require_native, None,
        "No integrated native-engine validation; separate editor evidence does not satisfy this requirement")
    return dict(schema="plume.acceptance.v1", policy=asdict(policy), checks=checks,
                passed=all(not row["required"] or row["status"] == "passed" for row in checks.values()),
                scope="Declared numerical acceptance; not scientific, ground-contact or runtime-performance certification")


def enforce_acceptance(report: dict) -> None:
    if not report["passed"]:
        raise AcceptanceError(report)
