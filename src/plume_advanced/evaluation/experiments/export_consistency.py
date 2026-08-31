"""Automatic canonical-identity and target-package consistency checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import geometry_semantic_hash
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, generate_sections
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.exporters import export_target_asset
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.world import ExportConfig


def run_export_consistency(config: EvaluationConfig, *, force: bool = False) -> dict:
    section = config.section("export_consistency")
    targets = tuple(section.get("targets", ("blender", "ue5", "unity", "gazebo", "omniverse")))
    base = load_project_config(config.project_config, world_body="earth", dev_mode=True)
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section},
        inputs=(config.project_config, config.seed_file("export_consistency")),
    )
    store = ResultStore(config.output_root, "export_consistency")
    asset_root = store.root / "packages"
    for seed in config.seeds("export_consistency"):
        project = for_seed(base, seed)
        template = ExperimentResult(
            experiment_name="export_consistency",
            run_id=f"seed-{seed:06d}-all-targets",
            condition_id="all_targets",
            seed=seed,
            status="complete",
            git_commit=str(provenance["git_commit"]),
            git_dirty=bool(provenance["git_dirty"]),
            resolved_config_sha256=config_hash(project),
        )

        def operation(project=project, seed=seed):
            _host, network, sections = generate_sections(project)
            geometry = GeometryGenerator(project.geometry).build_base_volume(network, sections)
            canonical_hash = geometry_semantic_hash(geometry)
            canonical_bounds = np.asarray(geometry.voxel_grid.bounds, dtype=float)
            output = asset_root / f"seed-{seed:06d}"
            result = export_target_asset(
                geometry,
                ExportConfig(target="all", file_format="auto", generate_collision=True),
                output,
                asset_name=f"plume_seed_{seed:06d}",
            )
            manifest = json.loads(result.primary_asset.read_text(encoding="utf-8"))
            checks = [
                _target_check(output, target, manifest["targets"].get(target), canonical_bounds)
                for target in targets
            ]
            return {
                "canonical_scene_hash": canonical_hash,
                "canonical_bbox_m": canonical_bounds.tolist(),
                "target_checks": checks,
                "all_targets_present": all(check["visual_asset_present"] for check in checks),
                "all_collisions_present": all(check["collision_present"] for check in checks),
            }

        run_case(store, template, operation, force=force)
    rows = store.rows()
    complete = [row for row in rows if row["status"] == "complete"]
    summary = {
        "schema": "plume.export-consistency-summary.v1",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "all_packages_present": all(row["all_targets_present"] for row in complete),
        "all_collisions_present": all(row["all_collisions_present"] for row in complete),
        "manual_imports_completed": 0,
        "manual_import_protocol": "docs/paper/EXPORT_VERIFICATION_PROTOCOL.md",
    }
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _target_check(
    output: Path,
    target: str,
    record: dict[str, Any] | None,
    canonical_bounds: np.ndarray,
) -> dict[str, Any]:
    if record is None:
        return {
            "target": target,
            "visual_asset_present": False,
            "collision_present": False,
            "failure_reason": "target missing from all-exports manifest",
        }
    primary = output / record["primary_asset"]
    files = [output / value for value in record.get("files", ())]
    collision = any("collision" in path.name.lower() and path.is_file() for path in files)
    if target == "omniverse" and primary.is_file():
        collision = collision or "CaveCollision" in primary.read_text(
            encoding="utf-8", errors="ignore"
        )
    bbox_error = None
    exported_bounds = _parse_bounds(primary)
    if exported_bounds is not None:
        canonical_extent = np.sort(np.ptp(canonical_bounds, axis=0))
        exported_extent = np.sort(np.ptp(exported_bounds, axis=0))
        bbox_error = float(
            np.max(np.abs(exported_extent - canonical_extent) / np.maximum(canonical_extent, 1e-9))
        )
    descriptor = next(
        (path for path in files if path.suffix == ".json" and "validation" not in path.name),
        None,
    )
    descriptor_payload = (
        json.loads(descriptor.read_text(encoding="utf-8"))
        if descriptor is not None and descriptor.is_file()
        else None
    )
    return {
        "target": target,
        "primary_asset": str(primary),
        "visual_asset_present": primary.is_file(),
        "collision_present": collision,
        "target_descriptor_present": descriptor_payload is not None,
        "target_descriptor": descriptor_payload,
        "exported_bbox_m": exported_bounds.tolist() if exported_bounds is not None else None,
        "bbox_relative_extent_error": bbox_error,
        "warnings": record.get("warnings", ()),
    }


def _parse_bounds(path: Path) -> np.ndarray | None:
    if path.suffix.lower() not in {".glb", ".obj"} or not path.is_file():
        return None
    try:
        loaded = trimesh.load(path, force="scene")
        return np.asarray(loaded.bounds, dtype=float)
    except (OSError, ValueError):
        return None


__all__ = ["run_export_consistency"]
