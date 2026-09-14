"""Deterministic upstream recovery after surface acceptance is exhausted.

The host, root seed and quality thresholds never change. Resolution changes
only within the explicitly configured refinement budget. Local
candidates are independent edits of the original realization. Regeneration is
explicit, uses a versioned seed sequence, and shares the original host field.
Only the accepted network/sections/base triple may feed floors, events or export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from plume_advanced.evaluation.artifacts import (
    host_semantic_hash,
    network_semantic_hash,
    section_semantic_hash,
)
from plume_advanced.pipeline.inspection import _write
from plume_advanced.pipeline.resolution import build_with_resolution_checks
from plume_advanced.procedural import canonical_seed, derive_subseed
from plume_advanced.progress import report_progress
from plume_advanced.stages.geometry_types import CaveGeometry
from plume_advanced.stages.mesh_inspection import inspect_surface, route_inspection_arguments
from plume_advanced.stages.network import CaveNetwork, CaveNetworkGenerator
from plume_advanced.stages.network_acceptance import repair_network
from plume_advanced.stages.network_quality import NetworkQualityError, assess_network
from plume_advanced.stages.route_clearance import fit_required_sections
from plume_advanced.stages.section_field import SectionField, SectionFieldGenerator
from plume_advanced.stages.surface_topology import SurfaceTopologyError

RECOVERY_VERSION = "plume.pipeline-recovery.v1"
SEED_DOMAIN = "pipeline-mesh-recovery-v1"


class PipelineRecoveryError(SurfaceTopologyError):
    """All explicitly budgeted local repairs and regenerations were rejected."""


@dataclass(frozen=True)
class AcceptedBase:
    network: CaveNetwork
    sections: SectionField
    geometry: CaveGeometry
    report: dict

    @property
    def context_sha256(self) -> str:
        # Downstream checkpoints cannot be reused for a different realization.
        return hashlib.sha256(
            json.dumps(self.report["accepted_identity"], sort_keys=True).encode()
        ).hexdigest()


def write_recovery_report(result: AcceptedBase, path: Path) -> Path:
    """Also republish evidence when the accepted triple came from a checkpoint."""
    return _write(path, result.report)


def _identity(network, sections):
    return dict(network=network_semantic_hash(network), sections=section_semantic_hash(sections))


def locate_affected_sections(network, sections, failure: dict, voxel_size: float) -> dict:
    """Associate measured regions/blocked points with section envelopes.

    When patch localization is inconclusive, flag under-resolved junction
    profiles as hypotheses. These do not claim to locate an unwanted handle.
    """
    regions = list(failure.get("defect_regions", []))
    for attempt in failure.get("attempts", []):
        inspection = attempt.get("inspection", {})
        for point in inspection.get("blocked_points_m", []):
            regions.append(
                dict(
                    kind="blocked_route_center",
                    lower_m=list(point),
                    upper_m=list(point),
                    center_m=list(point),
                )
            )
        for row in inspection.get("measurements", []):
            if not row["inside"]:
                regions.append(
                    dict(
                        kind="outside_mesh_center",
                        lower_m=row["point_m"],
                        upper_m=row["point_m"],
                        center_m=row["point_m"],
                    )
                )
    affected = set()
    for region in regions:
        lower, upper = np.asarray(region["lower_m"]), np.asarray(region["upper_m"])
        hits = []
        for field in sections.segment_fields:
            for sample in field.samples:
                point = np.asarray((sample.x, sample.y, sample.z))
                padding = max(sample.tube_width, sample.tube_height) * 0.5 + 2 * voxel_size
                if np.all(point >= lower - padding) and np.all(point <= upper + padding):
                    hits.append(field.segment_id)
                    break
        region["segment_ids"] = sorted(hits)
        affected.update(hits)
    hypothesis = []
    if not affected:
        for field in sections.segment_fields:
            if any(
                s.junction_blend_weight > 0.1 and min(s.tube_width, s.tube_height) < 8 * voxel_size
                for s in field.samples
            ):
                hypothesis.append(field.segment_id)
        affected.update(hypothesis)
    junction_ids = sorted(
        j.junction_id for j in network.junctions if affected.intersection(j.segment_ids)
    )
    return dict(
        regions=regions,
        segment_ids=sorted(affected),
        junction_ids=junction_ids,
        under_resolved_junction_hypotheses=hypothesis,
        limitation="Patch localization is partial; a handle in a cyclic graph may be intentional. Full acceptance gates decide repairs.",
    )


def _resample_sections(network, sections, affected):
    config = replace(
        sections.config,
        minimum_sample_spacing=sections.config.minimum_sample_spacing * 0.5,
        maximum_sample_spacing=sections.config.maximum_sample_spacing * 0.5,
        uniform_sample_spacing=sections.config.uniform_sample_spacing * 0.5,
        reference_sample_spacing=sections.config.reference_sample_spacing * 0.5,
    )
    generated = SectionFieldGenerator(config).generate(network)
    fields = {field.segment_id: field for field in generated.segment_fields}
    # Retain the requested config, and record the local override in the journal.
    return replace(
        sections,
        segment_fields=tuple(
            fields[field.segment_id] if field.segment_id in affected else field
            for field in sections.segment_fields
        ),
    )


def _local_candidate(kind, project, host, network, sections, affected):
    if kind == "local_section_resampling":
        return network, _resample_sections(network, sections, affected)
    repaired = repair_network(
        CaveNetworkGenerator(network.config),
        host,
        network,
        0,
        failed_checks=[dict(name="nonlocal_passage_overlap", segment_ids=affected)],
    )
    # This existing width repair preserves graph nodes and routes, enforces
    # minimum widths/gradients and rebuilds host samples and flow bookkeeping.
    return repaired, SectionFieldGenerator(project.section_field).generate(repaired)


def build_accepted_base(
    project, host, network, sections, *, report_path: Path | None = None, progress=None
) -> AcceptedBase:
    """Shared by normal generation and full evaluation workers; finite and fail closed."""
    controls = project.geometry
    before = host_semantic_hash(host)
    original_identity = _identity(network, sections)
    original_network, original_sections = network, sections
    required_centers = tuple((s.x, s.y, s.z) for f in sections.segment_fields for s in f.samples)
    journal = dict(
        schema=RECOVERY_VERSION,
        status="running",
        accepted=False,
        root_seed=project.procedural_seed,
        original_network_seed=canonical_seed(project.network.random_seed),
        host_semantic_sha256=before,
        original_identity=original_identity,
        budgets=dict(
            local_attempts=controls.recovery_local_attempts,
            network_attempts=controls.recovery_network_attempts,
            route_attempts=controls.route_repair_attempts if controls.required_route_height_m else 0,
            resolution_refinements=controls.resolution_refinement_attempts,
        ),
        seed_policy=f"derive_subseed(original network stage seed, {SEED_DOMAIN!r}, retry starting at 1)",
        attempts=[],
        limitation="Finite numerical recovery; not a guarantee for every seed or geological realism.",
    )

    def publish(detail):
        if report_path is not None:
            _write(report_path, journal)
        if progress is not None:
            total = (1 + controls.recovery_local_attempts + controls.recovery_network_attempts
                     + (controls.route_repair_attempts if controls.required_route_height_m else 0))
            completed = (
                total
                if journal["status"] in {"accepted", "exhausted"}
                else len(journal["attempts"])
            )
            progress("pipeline-recovery", completed, total, detail)
        else:
            report_progress("Pipeline recovery", detail=detail)

    candidates = [("original", 0)]
    if controls.required_route_height_m:
        candidates += [("route_clearance_repair", i) for i in range(1, controls.route_repair_attempts+1)]
    if project.network.quality.enabled:
        candidates += [
            (kind, i + 1)
            for i, kind in enumerate(
                ("local_section_resampling", "local_width_clearance")[
                    : controls.recovery_local_attempts
                ]
            )
        ]
        candidates += [
            ("network_regeneration", i) for i in range(1, controls.recovery_network_attempts + 1)
        ]
    localization = None
    seen = set()
    try:
        for kind, index in candidates:
            record: dict = dict(kind=kind, index=index, status="running")
            journal["attempts"].append(record)
            publish(f"{kind}: preparing candidate {len(journal['attempts'])}/{len(candidates)}")
            try:
                if kind == "route_clearance_repair":
                    if localization is None or not localization["segment_ids"]:
                        record.update(status="skipped", reason="No measured route-clearance repair target")
                        publish(record["reason"])
                        continue
                    network, sections = original_network, original_sections
                    record["localization"] = localization
                elif kind.startswith("local_"):
                    if localization is None or not localization["segment_ids"]:
                        record.update(
                            status="skipped",
                            reason="No localized repair target; do not edit unrelated passages",
                        )
                        publish(record["reason"])
                        continue
                    record["localization"] = localization
                    record["change"] = (
                        "Halve section sampling intervals on affected segments"
                        if kind == "local_section_resampling"
                        else "Target 15% narrower conflicting widths within original minimum-width/gradient limits; resample routes and rebuild sections and flow state"
                    )
                    network, sections = _local_candidate(
                        kind,
                        project,
                        host,
                        original_network,
                        original_sections,
                        localization["segment_ids"],
                    )
                    original_edges = [
                        (s.segment_id, s.start_node_id, s.end_node_id)
                        for s in original_network.segments
                    ]
                    if original_edges != [
                        (s.segment_id, s.start_node_id, s.end_node_id) for s in network.segments
                    ]:
                        raise AssertionError("Local repair changed required graph edges")
                elif kind == "network_regeneration":
                    seed = derive_subseed(project.network.random_seed, SEED_DOMAIN, index)
                    record["network_seed"] = seed
                    network = CaveNetworkGenerator(
                        replace(project.network, random_seed=seed)
                    ).generate(
                        host,
                        section_config=project.section_field,
                        quality_progress=lambda detail: publish(detail),
                    )
                    sections = SectionFieldGenerator(project.section_field).generate(network)
                sections, route_repair = fit_required_sections(
                    network, sections, controls, host,
                    extra_margin_m=index*controls.voxel_size if kind == "route_clearance_repair" else 0.,
                    affected=localization["segment_ids"] if kind == "route_clearance_repair" and localization is not None else None,
                )
                record["route_section_repair"] = route_repair
                identity = _identity(network, sections)
                record["identity"] = identity
                key = (identity["network"], identity["sections"])
                if key in seen:
                    record.update(status="skipped", reason="Identical candidate already rejected")
                    publish(record["reason"])
                    continue
                seen.add(key)
                assessment = {}
                if project.network.quality.enabled:
                    assessment = assess_network(network, host, sections)
                    record["assessment"] = assessment
                    if not assessment["accepted"]:
                        raise NetworkQualityError(assessment)
                geometry = build_with_resolution_checks(network, sections, controls, progress=progress)
                record["resolution_repair"] = dict(geometry.resolution_repair)
                if kind.startswith("local_") or kind == "route_clearance_repair":
                    # In addition to all new centres, protect the original local
                    # realization's routes. A repair cannot hide a lost passage
                    # by shifting the samples inspected by the next stage.
                    extra = tuple(sorted(set(required_centers) - set(geometry.route_centers)))
                    all_centers = (*geometry.route_centers, *extra)
                    inspection = inspect_surface(
                        geometry.assembled_vertices,
                        geometry.assembled_faces,
                        points=all_centers,
                        expected_genus=geometry.expected_surface_genus,
                        **route_inspection_arguments(geometry),
                    )
                    record["preserved_original_centers"] = len(required_centers)
                    original_protected = {
                        (s.x, s.y, s.z)
                        for f in original_sections.segment_fields
                        if f.segment_id in original_sections.dominant_route_segment_ids
                        for s in f.samples
                    }
                    geometry = replace(
                        geometry,
                        route_centers=all_centers,
                        protected_route_points=tuple(
                            sorted(set(geometry.protected_route_points) | original_protected)
                        ),
                        mesh_inspection=tuple(inspection.items()),
                    )
                    network = replace(
                        network,
                        quality_report={
                            **network.quality_report,
                            "selected_shape_sha256": assessment.get("shape_sha256"),
                            "post_mesh_repair": dict(
                                kind=kind, assessment=assessment, parent_identity=original_identity
                            ),
                        },
                    )
                if host_semantic_hash(host) != before:
                    raise AssertionError("Recovery mutated its input host field")
                record.update(
                    status="accepted",
                    surface_attempts=[dict(r) for r in geometry.surface_quality_records],
                )
                journal.update(
                    status="accepted",
                    accepted=True,
                    outcome=(
                        "unchanged"
                        if kind == "original" and not route_repair["changed_samples"]
                        else "regenerated"
                        if kind == "network_regeneration"
                        else "locally_repaired"
                    ),
                    accepted_identity={**identity, "geometry_config": asdict(geometry.config)},
                    selected_attempt=len(journal["attempts"]) - 1,
                    effective_network_seed=canonical_seed(network.config.random_seed),
                    host_unchanged=True,
                )
                publish(f"Accepted {kind}; downstream stages will use this realization")
                return AcceptedBase(network, sections, geometry, journal)
            except (SurfaceTopologyError, NetworkQualityError) as error:
                record.update(
                    status="rejected",
                    error_type=type(error).__name__,
                    reason=str(error),
                    inspection=getattr(error, "report", {}),
                )
                if kind == "original" and isinstance(error, SurfaceTopologyError):
                    localization = locate_affected_sections(
                        network, sections, record["inspection"], controls.voxel_size
                    )
                if host_semantic_hash(host) != before:
                    raise AssertionError("Recovery mutated its input host field") from error
                publish(f"Rejected {kind}: {error}")
        journal.update(status="exhausted", host_unchanged=True)
        publish("Recovery budget exhausted; downstream generation and export are blocked")
    except Exception as error:
        journal.update(status="interrupted", error_type=type(error).__name__, error=str(error))
        publish("Recovery stopped on an error that is not safe to treat as a seed failure")
        raise
    raise PipelineRecoveryError(
        "Pipeline recovery exhausted; no candidate passed all checks", report=journal
    )
