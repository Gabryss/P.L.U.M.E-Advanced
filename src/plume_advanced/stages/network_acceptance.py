"""Bounded candidate search and deterministic, topology-preserving repairs."""

from __future__ import annotations

import hashlib
import platform
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from plume_advanced.procedural import canonical_seed, derive_subseed
from plume_advanced.stages.network_quality import (
    BLIND_KINDS,
    QUALITY_VERSION,
    NetworkQualityError,
    assess_network,
    write_quality_report,
)


def limit_width_gradient(widths: np.ndarray, arc: np.ndarray, gradient: float) -> np.ndarray:
    """Largest non-expanding width envelope with a bounded spatial slope."""
    if widths.shape != arc.shape or not np.isfinite(widths).all() or not np.isfinite(arc).all():
        raise ValueError("Width constraints require matching finite arrays")
    if np.any(widths <= 0) or np.any(np.diff(arc) < 0) or not np.isfinite(gradient) or gradient <= 0:
        raise ValueError("Width constraints require positive widths/gradient and ordered distances")
    if not len(widths):
        return widths.copy()
    distance = gradient * (arc - arc[0])
    left = distance + np.minimum.accumulate(widths - distance)
    right = -distance + np.minimum.accumulate((widths + distance)[::-1])[::-1]
    return np.minimum(widths, np.minimum(left, right))


def repair_network(generator, host, network, repair_pass, *, failed_checks=None):
    """Keep graph nodes and topology fixed; rebuild all dependent geometry/state.

    A global displacement bound avoids the sharp derivative discontinuities of
    independently clipping each smoothing offset. No branch is silently deleted.
    """
    config = generator.config
    section_clearance_checks = {
        "nonlocal_passage_overlap", "section_island_clearance", "section_footprint_islands",
    }
    preserve_routes = bool(failed_checks) and all(
        check["name"] in section_clearance_checks for check in failed_checks
    )
    affected = {sid for check in (failed_checks or []) for sid in check.get("segment_ids", [])}
    nodes = {n.node_id: n for n in network.nodes}
    outgoing = {s.start_node_id for s in network.segments}
    degrees = network._degrees()
    repaired = []
    desired_widths: dict[int, np.ndarray | list[float]] = {}
    for segment in network.segments:
        raw = np.array([[p.x, p.y] for p in segment.points])
        if len(raw) < 2 or not np.isfinite(raw).all():
            raise ValueError(f"Segment {segment.segment_id} has no finite repairable route")
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        if arc[-1] <= 1e-9:
            raise ValueError(f"Segment {segment.segment_id} has zero length; try a fresh candidate")
        spacing = min(2.0, 0.2 * segment.mean_width)
        distances = np.linspace(0, arc[-1], max(5, int(np.ceil(arc[-1] / spacing)) + 1))
        linear = np.column_stack([np.interp(distances, arc, raw[:, k]) for k in range(2)])
        sigma = min((1.5 + repair_pass) * segment.mean_width, 0.18 * arc[-1])
        coords = linear.copy() if preserve_routes else gaussian_filter1d(
            linear, sigma / (distances[1] - distances[0]), axis=0, mode="nearest"
        )
        t = distances / arc[-1]
        # Restore exact endpoints with a smooth, full-span affine correction.
        coords += (1 - t[:, None]) * (raw[0] - coords[0]) + t[:, None] * (raw[-1] - coords[-1])
        delta = coords - linear
        bound = (2 + repair_pass) * segment.mean_width
        coords = linear + delta * min(
            1.0, bound / max(float(np.max(np.linalg.norm(delta, axis=1))), 1e-9)
        )
        coords[:, 0] = np.clip(coords[:, 0], host.x_coords[0], host.x_coords[-1])
        coords[:, 1] = np.clip(coords[:, 1], host.y_coords[0], host.y_coords[-1])
        coords[0] = [nodes[segment.start_node_id].x, nodes[segment.start_node_id].y]
        coords[-1] = [nodes[segment.end_node_id].x, nodes[segment.end_node_id].y]
        new_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        old_width = np.interp(distances, arc, [p.width for p in segment.points])
        # Bounded discharge response replaces ceiling saturation. It is a
        # morphology heuristic, not a hydraulic cross-section solver.
        flux = max(segment.mean_flux, 1e-9) / max(config.source_flux, 1e-9)
        target = 2 * config.minimum_passage_radius + 2 * (
            config.maximum_passage_radius - config.minimum_passage_radius
        ) * np.sqrt(flux / (1 + flux))
        widths = target * np.clip(old_width / max(float(np.median(old_width)), 1e-9), 0.8, 1.1)
        widths = np.minimum(widths, 1.9 * config.maximum_passage_radius)
        if config.topology.style in {"trunk_dominated", "interconnected"}:
            # Keep the gallery's metric width envelope and narrow island arms.
            # Replacing every segment with a new discharge-derived width can
            # erase the rock island between two valid routes.
            widths = gaussian_filter1d(old_width, max(1., sigma / (distances[1] - distances[0])), mode="nearest")
            widths = np.minimum(widths, 1.9 * config.maximum_passage_radius)
        if preserve_routes:
            # Narrow conflicting envelopes before touching an already accepted
            # route. Smoothing island arms toward their chord closes the solid
            # island and can make a section-clearance failure worse.
            widths = old_width.copy()
            if not affected or segment.segment_id in affected:
                widths = np.minimum(widths, np.maximum(2 * config.minimum_passage_radius, .85 * widths))
        metadata = dict(segment.metadata, quality_repair_pass=repair_pass + 1,
                        quality_repair_kind="section_width" if preserve_routes else "route")
        if (
            segment.kind in BLIND_KINDS
            and segment.end_node_id not in outgoing
            and degrees[segment.end_node_id] == 1
        ):
            reach = min(0.45 * new_arc[-1], 6 * target)
            u = np.clip((new_arc - (new_arc[-1] - reach)) / max(reach, 1e-9), 0, 1)
            factor = 1 - (1 - config.quality.terminal_width_ratio * 0.8) * u * u * (3 - 2 * u)
            # Reapply an absolute cap, rather than compounding the taper on
            # every repair. This also applies to general lobe growth.
            widths = np.minimum(widths, float(max(widths)) * factor)
            metadata["quality_terminal_taper"] = True
        # Centerline smoothing changes station distances. Width interpolation
        # on the original arc alone can leave arbitrarily steep transitions.
        widths = limit_width_gradient(widths, new_arc, 0.95 * config.quality.maximum_width_gradient)
        points = []
        for index, (xy, distance, width) in enumerate(zip(coords, new_arc, widths)):
            substrate = host.sample(float(xy[0]), float(xy[1]))
            points.append(
                replace(
                    segment.points[0],
                    index=index,
                    x=float(xy[0]),
                    y=float(xy[1]),
                    arc_length=float(distance),
                    width=float(width),
                    elevation=substrate.elevation,
                    slope_degrees=substrate.slope_degrees,
                    cover_thickness=substrate.cover_thickness,
                    roof_competence=substrate.roof_competence,
                    growth_cost=substrate.growth_cost,
                )
            )
        desired_widths[segment.segment_id] = widths
        repaired.append(replace(segment, points=tuple(points), metadata=metadata))
    # Recompute cooling/travel ages and flux against the repaired lengths.
    if config.topology.style == "interconnected" and not preserve_routes:
        from plume_advanced.stages.network_interconnected import smooth_routes
        repaired = smooth_routes(host, repaired, network.backend_provenance["flow_direction"])
        desired_widths = {s.segment_id: [p.width for p in s.points] for s in repaired}
    repaired = generator._assign_conserved_flow(list(network.nodes), repaired)
    repaired = [
        replace(
            s,
            points=tuple(
                replace(p, width=float(w)) for p, w in zip(s.points, desired_widths[s.segment_id])
            ),
        )
        for s in repaired
    ]
    if config.topology.generation_mode == "independent_growth":
        from plume_advanced.stages.network_gallery_growth import refresh_phase_discharge
        repaired = refresh_phase_discharge(generator, list(network.nodes), repaired)
    repaired = generator._annotate_emplacement_flux_history(repaired)
    route = generator._dominant_route(list(network.nodes), repaired)
    generator._validate_generated_graph(list(network.nodes), repaired, route)
    geometry = generator._build_flow_geometry(host)
    occupancy = np.zeros_like(host.growth_cost, dtype=bool)
    width_field = np.zeros_like(host.growth_cost)
    for segment in repaired:
        generator._rasterize_segment(host, occupancy, width_field, segment)
    generator._paint_structural_chambers(
        host, occupancy, width_field, list(network.nodes), repaired
    )
    positions, counts = generator._measure_parallel_channels(
        geometry=geometry, segments=repaired, include_passage_width=False
    )
    _, visible = generator._measure_parallel_channels(
        geometry=geometry, segments=repaired, include_passage_width=True
    )
    return replace(
        network,
        segments=tuple(repaired),
        junctions=tuple(generator._build_junctions(list(network.nodes), repaired)),
        occupancy=occupancy,
        width_field=width_field,
        dominant_route_node_ids=route,
        slice_along_positions=positions,
        slice_channel_counts=counts,
        slice_visible_channel_counts=visible,
    )


def generate_accepted_network(
    generator, host, *, section_config=None, report_path=None, progress=None
):
    from plume_advanced.stages.network_systems import GenerationDomainError
    from plume_advanced.stages.section_field import SectionFieldGenerator

    config = generator.config
    source_digest = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for source in sorted(package.rglob("*.py")):
        source_digest.update(source.relative_to(package).as_posix().encode())
        source_digest.update(source.read_bytes())
    import scipy

    from plume_advanced.evaluation.artifacts import host_semantic_hash

    report: dict[str, Any] = {
        "schema": QUALITY_VERSION,
        "status": "searching",
        "accepted": False,
        "base_seed": canonical_seed(config.random_seed),
        "host_semantic_sha256": host_semantic_hash(host) if host is not None else None,
        "generation_config": asdict(config),
        "section_config": asdict(section_config) if section_config is not None else None,
        "implementation_sha256": source_digest.hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "thresholds": asdict(config.quality),
        "attempts": [],
        "scope": "network_and_sections" if section_config is not None else "network_only",
        "seed_policy": "attempt 0 = base; later = derive_subseed(base, network-quality-v1, attempt)",
        "limitation": "Heuristic morphology screening; not geological validation.",
    }

    def publish(message):
        if report_path is not None:
            write_quality_report(report, report_path)
        if progress is not None:
            progress(message)

    for attempt in range(config.quality.max_attempts):
        seed = (
            config.random_seed
            if attempt == 0
            else derive_subseed(config.random_seed, "network-quality-v1", attempt)
        )
        # Preserve injected backends and subclass behavior across retries.
        # The caller's config and random state remain untouched.
        worker = copy(generator)
        worker.config = replace(config, random_seed=seed)
        candidate = None
        previous_failed_checks: list[dict[str, Any]] = []
        for repair_pass in range(config.quality.repair_passes + 1):
            try:
                if repair_pass == 0:
                    publish(
                        f"Network candidate {attempt + 1}/{config.quality.max_attempts}: seed {seed}"
                    )
                    candidate = worker._generate_candidate(host)
                else:
                    candidate = repair_network(worker, host, candidate, repair_pass - 1,
                                               failed_checks=previous_failed_checks)
                assessment = assess_network(candidate, host)
                # Section generation is much cheaper than meshing, but skip it
                # while a candidate still has known network construction defects.
                if assessment["accepted"] and section_config is not None:
                    sections = SectionFieldGenerator(section_config).generate(candidate)
                    assessment = assess_network(candidate, host, sections)
                record = {
                    "attempt": attempt,
                    "seed": canonical_seed(seed),
                    "repair_pass": repair_pass,
                    **assessment,
                }
            except GenerationDomainError as error:
                report.update(status="invalid_input", failure_reason=str(error))
                publish(str(error))
                raise
            except (ValueError, ArithmeticError) as error:
                record = {
                    "attempt": attempt,
                    "seed": canonical_seed(seed),
                    "repair_pass": repair_pass,
                    "accepted": False,
                    "checks": [
                        dict(
                            name="candidate_generation",
                            passed=False,
                            severity="error",
                            value=str(error),
                            limit=None,
                            segment_ids=[],
                        )
                    ],
                }
            report["attempts"].append(record)
            failed = [c["name"] for c in record["checks"] if not c["passed"]]
            previous_failed_checks = [c for c in record["checks"] if not c["passed"]]
            if record["accepted"]:
                report.update(
                    status="accepted",
                    accepted=True,
                    selected_attempt=attempt,
                    selected_seed=canonical_seed(seed),
                    selected_repair_pass=repair_pass,
                    selected_shape_sha256=record["shape_sha256"],
                )
                publish(
                    f"Accepted network candidate {attempt + 1}, repair {repair_pass}; {len(record['checks'])} checks passed"
                )
                assert candidate is not None
                return replace(candidate, quality_report=report)
            publish(f"Rejected candidate {attempt + 1}, repair {repair_pass}: {', '.join(failed)}")
            if candidate is None:
                break
    report["status"] = "rejected"
    publish("No candidate passed morphology screening; meshing is blocked")
    raise NetworkQualityError(report)
