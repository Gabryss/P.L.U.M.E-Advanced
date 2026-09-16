"""Automatic canonical-identity and target-package consistency checks."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from plume_advanced.acceptance import require_available_acceptance
from plume_advanced.evaluation.artifacts import geometry_semantic_hash
from plume_advanced.evaluation.config import EvaluationConfig
from plume_advanced.evaluation.experiments.common import config_hash, for_seed, generate_sections
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.evaluation.provenance import capture_provenance
from plume_advanced.evaluation.reliability_state import project_inputs
from plume_advanced.evaluation.runner import ResultStore, run_case
from plume_advanced.evaluation.schema import ExperimentResult
from plume_advanced.exporters import export_target_asset
from plume_advanced.identity import sha256_file
from plume_advanced.pipeline.recovery import build_accepted_base
from plume_advanced.stages.geometry import GeometryGenerator


def run_export_consistency(config: EvaluationConfig, *, force: bool = False) -> dict:
    section = config.section("export_consistency")
    targets = tuple(section.get("targets", ("blender", "ue5", "unity", "gazebo", "omniverse")))
    if not targets:
        raise ValueError("export_consistency.targets must contain at least one target")
    base = config.load_project(world_body="earth", dev_mode=True)
    provenance = capture_provenance(
        config.project_config.parent.parent,
        resolved_config={"experiment": section, "project": config_hash(base)},
        inputs=(*project_inputs(base, config.project_config), config.seed_file("export_consistency")),
    )
    identity = str(provenance["identity_sha256"])
    store = ResultStore(
        config.output_root, "export_consistency", provenance_sha256=identity,
        validate_cached=lambda row: _valid_package_receipt(
            config.output_root / "export_consistency/packages" / f"seed-{row['seed']:06d}", row),
    )
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
            provenance_sha256=identity,
        )

        def operation(project=project, seed=seed):
            require_available_acceptance(project.acceptance)
            host, network, sections = generate_sections(project)
            accepted = build_accepted_base(project, host, network, sections)
            geometry = GeometryGenerator(accepted.geometry.config).finalize(accepted.geometry)
            canonical_hash = geometry_semantic_hash(geometry)
            vertices = np.asarray(geometry.assembled_vertices, dtype=float)
            base_bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
            output = asset_root / f"seed-{seed:06d}"
            result = export_target_asset(
                geometry,
                replace(project.export, target="all", file_format="auto", generate_collision=True),
                output,
                asset_name=f"plume_seed_{seed:06d}",
                acceptance=project.acceptance,
                resolution=section_resolution_report(accepted.sections, geometry.voxel_grid.voxel_size),
            )
            manifest = json.loads(result.primary_asset.read_text(encoding="utf-8"))
            canonical_bounds = np.asarray(manifest["canonical_visual_bbox_m"], dtype=float)
            checks = [
                _target_check(output, target, manifest["targets"].get(target), canonical_bounds)
                for target in targets
            ]
            metrics = {
                "base_bbox_m": base_bounds.tolist(),
                "acceptance": json.loads((output / "pipeline_inspection.json").read_text())["acceptance"],
                "canonical_scene_hash": canonical_hash,
                "canonical_bbox_m": canonical_bounds.tolist(),
                "target_checks": checks,
                "all_targets_present": all(check["visual_asset_present"] for check in checks),
                "all_collisions_present": all(check["collision_present"] for check in checks),
                "target_checks_passed": all(check["passed"] for check in checks),
            }
            if not metrics["target_checks_passed"]:
                raise ExportConsistencyError(metrics)
            metrics["artifact_receipts"] = _package_receipt(output)
            return metrics

        run_case(store, template, operation, force=force)
    rows = store.rows()
    complete = [row for row in rows if row["status"] == "complete"]
    all_completed = bool(rows) and len(complete) == len(rows)
    summary = {
        "schema": "plume.export-consistency-summary.v1",
        "planned_n": len(rows),
        "complete_n": len(complete),
        "failed_n": sum(row["status"] in {"failed", "timeout"} for row in rows),
        "all_packages_present": all_completed
        and all(row["all_targets_present"] for row in complete),
        "all_collisions_present": all_completed
        and all(row["all_collisions_present"] for row in complete),
        "manual_imports_completed": 0,
        "manual_import_protocol": "README.md#native-engine-qualification",
    }
    summary["passed"] = bool(
        summary["all_packages_present"] and summary["all_collisions_present"]
        and all(row.get("target_checks_passed") is True for row in complete)
    )
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


class ExportConsistencyError(ValueError):
    def __init__(self, metrics: dict[str, Any]):
        self.metrics = metrics
        super().__init__("Target export consistency checks failed")


def _package_receipt(output: Path) -> dict[str, str]:
    root = output.resolve()
    receipts = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Export package cannot contain linked or external files")
        if path.is_file():
            receipts[path.relative_to(output).as_posix()] = sha256_file(path)
    if not receipts:
        raise ValueError("Export package is empty")
    return receipts


def _valid_package_receipt(output: Path, row: dict[str, Any]) -> bool:
    try:
        return bool(row.get("target_checks_passed") is True
                    and row.get("artifact_receipts") == _package_receipt(output))
    except (OSError, ValueError):
        return False


def _target_check(
    output: Path,
    target: str,
    record: dict[str, Any] | None,
    canonical_bounds: np.ndarray,
) -> dict[str, Any]:
    check: dict[str, Any] = dict(target=target, passed=False, visual_asset_present=False,
                                collision_present=False, target_descriptor_present=False)
    try:
        if record is None:
            raise ValueError("target missing from all-exports manifest")
        primary = output / record["primary_asset"]
        files = [output / value for value in record.get("files", ())]
        if any(not p.resolve().is_relative_to(output.resolve()) for p in [primary, *files]):
            raise ValueError("target references an external file")
        check["primary_asset"] = str(primary)
        check["visual_asset_present"] = primary.is_file()
        check["collision_present"] = any(
            "collision" in path.name.lower() and path.is_file() for path in files)
        if target == "omniverse" and primary.is_file():
            check["collision_present"] = 'def Mesh "CaveCollision"' in primary.read_text()
        descriptor_payload = None
        for path in files:
            if path.suffix == ".json" and path.is_file():
                payload = json.loads(path.read_text())
                if (isinstance(payload, dict) and payload.get("schema") == "plume.target_export.v1"
                        and payload.get("target") == target):
                    descriptor_payload = payload
                    break
        check["target_descriptor_present"] = descriptor_payload is not None
        check["target_descriptor"] = descriptor_payload
        visual = primary
        if target == "gazebo":
            visual = next(p for p in files if p.suffix == ".obj" and "collision" not in p.name)
        exported_bounds = _parse_bounds(visual)
        if exported_bounds is None:
            raise ValueError("cannot read target visual bounds")
        canonical_bounds = np.asarray(canonical_bounds, dtype=float)
        if canonical_bounds.shape != (2, 3) or not np.isfinite(canonical_bounds).all():
            raise ValueError("invalid canonical bounds")
        extent = np.ptp(canonical_bounds, axis=0)
        error = float(np.max(np.abs(np.ptp(exported_bounds, axis=0) - extent)
                             / np.maximum(extent, 1e-9)))
        # Same inspected visual surface, allowing only target float32 serialization.
        tolerance = max(1e-5, 8 * float(np.finfo(np.float32).eps)
                        * max(1., float(np.max(np.abs(canonical_bounds)))))
        check.update(exported_bbox_m=exported_bounds.tolist(), bbox_relative_extent_error=error,
                     bbox_tolerance_m=tolerance,
                     bounds_match=bool(np.all(np.abs(exported_bounds-canonical_bounds) <= tolerance)),
                     warnings=record.get("warnings", ()))
        check["passed"] = bool(check["visual_asset_present"] and check["collision_present"]
                               and check["target_descriptor_present"] and check["bounds_match"]
                               and all(path.is_file() for path in files))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        check["failure_reason"] = str(error)
    return check


def _parse_bounds(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() in {".usd", ".usda"}:
            text = path.read_text()
            match = re.search(r'def Mesh "CaveWall".*?point3f\[\] points = \[(.*?)\]', text, re.S)
            if match is None:
                return None
            flat = np.fromstring(match[1].translate(str.maketrans("(),", "   ")), sep=" ")
            points = flat.reshape(-1, 3).astype(np.float32)
            bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        elif path.suffix.lower() in {".glb", ".obj"}:
            loaded = trimesh.load(path, force="scene", process=False)
            bounds = np.asarray(loaded.bounds, dtype=float)
            if path.suffix.lower() == ".glb":
                # glTF Y-up -> PLUME Z-up: (x, y, z) -> (x, -z, y).
                bounds = np.array([[bounds[0, 0], -bounds[1, 2], bounds[0, 1]],
                                   [bounds[1, 0], -bounds[0, 2], bounds[1, 1]]])
        else:
            return None
        return bounds if bounds.shape == (2, 3) and np.isfinite(bounds).all() else None
    except (OSError, ValueError, TypeError, IndexError):
        return None


__all__ = ["run_export_consistency"]
