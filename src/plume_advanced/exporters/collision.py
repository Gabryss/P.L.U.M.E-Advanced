"""Deterministic collision decimation with measured error and topology gates."""

from __future__ import annotations

import threading

import fast_simplification
import numpy as np

from plume_advanced.exporters.precision import repair_face_rounding
from plume_advanced.progress import report_progress
from plume_advanced.stages.mesh_inspection import MeshInspectionError, inspect_surface
from plume_advanced.stages.triangle_queries import TriangleIndex

# The native wrapper owns a mutable global mesh; threads must not interleave it.
_SIMPLIFIER_LOCK = threading.Lock()


def passage_deviation(reference, candidate):
    errors = [abs(a[key]-b[key])
              for a, b in zip(reference["measurements"], candidate["measurements"], strict=True)
              for key in ("floor_distance_m", "roof_distance_m")
              if a[key] is not None and b[key] is not None]
    return max(errors, default=0.)


def surface_deviation(reference, candidate, limit, *, purpose="Collision"):
    """Bidirectional vertex/face-centre distances; not a continuous Hausdorff bound."""
    worst = 0.
    checked = 0
    total = sum(len(vertices)+len(faces) for vertices, faces in (reference, candidate))
    for source, target in ((reference, candidate), (candidate, reference)):
        vertices, faces = source
        index = TriangleIndex(*target)
        for points in (vertices, np.asarray(vertices)[faces].mean(axis=1)):
            for start in range(0, len(points), 8192):
                batch = points[start:start+8192]
                values = index.distances(batch)
                checked += len(batch)
                if checked % 65536 < len(batch):
                    report_progress(f"{purpose} surface comparison", checked, total,
                                    "nearest-triangle distances for every vertex and face centre")
                worst = max(worst, float(values.max(initial=0)))
                if worst > limit:
                    return dict(passed=False, max_sampled_error_m=worst, checked_points=checked)
    return dict(passed=True, max_sampled_error_m=worst, checked_points=checked)



def stabilize_collision_precision(vertices, faces, error_limit):
    """Check metre and centimetre float32 representations before publication.

    Tiny local neighbour relaxation can separate vertices lost to rounding.
    Residual slivers receive bounded, monotonic face-area repair. No faces are
    removed and every face must preserve orientation in both engine encodings.
    The caller subsequently checks complete topology, distances and traversal.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    original = vertices[faces]
    normals = np.cross(original[:, 1]-original[:, 0], original[:, 2]-original[:, 0])
    bound = min(.001, error_limit*.1)
    report = dict(method="local_precision_relaxation", passed=False, attempts=[],
                  relaxation_bound_m=bound, encodings=["float32_metres", "float32_centimetres"])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        report["failure"] = "Candidate already contains zero-area triangles before quantization"
        return vertices, report

    def inspect_encoding(candidate):
        quantized = candidate.astype(np.float32).astype(float)
        invalid = np.zeros(len(faces), bool)
        counts = {}
        for scale, name in ((1., "float32_metres"), (100., "float32_centimetres")):
            encoded = (quantized*scale).astype(np.float32).astype(float)
            triangles = encoded[faces]
            cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            # Includes both zero area and faces whose normal flips during rounding.
            bad = np.sum(normals*cross, axis=1) <= 0
            counts[name] = int(bad.sum())
            invalid |= bad
        return quantized, invalid, counts

    quantized, invalid, counts = inspect_encoding(vertices)
    report["attempts"].append(dict(fraction=0., invalid_faces=counts))
    if not invalid.any():
        report.update(passed=True, relaxed_vertices=0,
                      maximum_vertex_change_m=float(np.linalg.norm(quantized-vertices, axis=1).max()))
        return quantized, report
    selected = np.unique(faces[invalid])
    if len(selected) > max(64, min(10000, len(vertices)//100)):
        report["failure"] = "Float32 damage is not localized; cannot safely apply a small precision repair"
        return vertices, report
    neighbours: dict[int, set[int]] = {int(i): set() for i in selected}
    for face in faces[np.any(np.isin(faces, selected), axis=1)]:
        for i in face:
            if int(i) in neighbours:
                neighbours[int(i)].update(int(j) for j in face if j != i)
    directions = np.array([vertices[sorted(neighbours[int(i)])].mean(axis=0)-vertices[i]
                           for i in selected])
    best, best_invalid = quantized, invalid
    for fraction in (.001, .01, .05):
        move = directions*fraction
        lengths = np.linalg.norm(move, axis=1)
        move *= np.minimum(1., np.divide(bound, lengths, out=np.ones(len(lengths)),
                                         where=lengths > 0))[:, None]
        trial = vertices.copy()
        trial[selected] += move
        quantized, invalid, counts = inspect_encoding(trial)
        report["attempts"].append(dict(fraction=fraction, invalid_faces=counts))
        if not invalid.any():
            report.update(passed=True, relaxed_vertices=len(selected),
                          maximum_vertex_change_m=float(np.linalg.norm(quantized-vertices, axis=1).max()))
            return quantized, report
        if invalid.sum() < best_invalid.sum():
            best, best_invalid = quantized, invalid
    repaired, directed = repair_face_rounding(vertices, faces, best, normals, best_invalid, bound)
    quantized, invalid, counts = inspect_encoding(repaired)
    report["face_area_repair"] = dict(directed, invalid_faces=counts)
    if not invalid.any():
        changed = np.any(quantized != vertices.astype(np.float32), axis=1)
        report.update(passed=True, relaxed_vertices=int(changed.sum()),
                      maximum_vertex_change_m=float(np.linalg.norm(quantized-vertices, axis=1).max()))
        return quantized, report
    report["failure"] = "Bounded float32 precision repair could not preserve every triangle orientation"
    return vertices, report

def simplify_collision(vertices, faces, config, arguments, *, report=None):
    return simplify_surface(vertices, faces, arguments,
        target_reduction=config.collision_target_reduction,
        max_error_m=config.collision_max_error_m, attempts=config.collision_repair_attempts,
        report=report)


def simplify_surface(vertices, faces, arguments, *, target_reduction, max_error_m,
                     attempts, report=None, triangle_budget=0, purpose="Collision"):
    """Restart each bounded QEM candidate from the same accepted source surface.

    Global extent never sets the simplification tolerance. Full topology,
    passage measurements and any required capsule route are rechecked, then
    bidirectional surface distances reject candidates that drift elsewhere.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("Cannot create collision geometry from an empty cave mesh")
    report = {} if report is None else report
    reference = inspect_surface(vertices, faces, **arguments)
    report.update(method="quadric_edge_collapse", source_triangles=len(faces),
                  max_error_m=max_error_m, attempts=[], used_raw_fallback=False,
                  error_scope="All vertices and face centres in both directions, plus complete required capsule sweeps")
    for attempt in range(attempts):
        # Increase the retained face count gradually: halving the reduction
        # skips useful candidates (80% -> 40% drops straight from 20% to 60%
        # retained). Lower aggressiveness also changes unsafe collapse choices.
        reduction = target_reduction if triangle_budget else 1-(1-target_reduction)*1.5**attempt
        if reduction <= 0:
            break
        aggressiveness = max(1., 5.-attempt)
        if int(len(faces)*(1-reduction)) < 12:
            continue
        report_progress(f"{purpose} reduction", attempt, attempts,
                        f"target {reduction:.0%} fewer triangles; error limit {max_error_m:g} m")
        with _SIMPLIFIER_LOCK:
            new_vertices, new_faces = fast_simplification.simplify(
                vertices, faces, target_reduction=reduction,
                agg=aggressiveness)
        new_faces = np.asarray(new_faces, np.int64)
        record = dict(target_reduction=reduction, aggressiveness=aggressiveness,
                      triangles=len(new_faces), accepted=False)
        report["attempts"].append(record)
        if triangle_budget:
            # QEM can leave exactly collapsed coplanar faces. Remove only
            # zero-area faces and weld exact duplicate vertices, then subject
            # the result to every topology, clearance and distance gate below.
            new_vertices, inverse = np.unique(new_vertices, axis=0, return_inverse=True)
            new_faces = inverse[new_faces]
            tri = new_vertices[new_faces]
            keep = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) > 0
            record["collapsed_faces_removed"] = int((~keep).sum())
            new_faces = new_faces[keep]
            used, inverse = np.unique(new_faces, return_inverse=True)
            new_vertices, new_faces = new_vertices[used], inverse.reshape(-1, 3)
            record["triangles"] = len(new_faces)
        if triangle_budget and len(new_faces) > triangle_budget:
            record["reason"] = "Reduced surface exceeds the triangle budget"
            continue
        if len(new_faces) >= len(faces):
            record["reason"] = "No reduction achievable at this tolerance"
            continue
        new_vertices, precision = stabilize_collision_precision(new_vertices, new_faces,
                                                                 max_error_m)
        record["precision"] = precision
        if not precision["passed"]:
            record["reason"] = precision["failure"]
            continue
        try:
            inspected = inspect_surface(new_vertices, new_faces, **arguments)
            if any(inspected["topology"][key] != reference["topology"][key]
                   for key in ("components", "euler")):
                raise MeshInspectionError(dict(failures=["Collision reduction changed topology"]))
            # Opposite changes to roof/floor can cancel in total height. Check each.
            record["max_passage_error_m"] = passage_deviation(reference, inspected)
            if record["max_passage_error_m"] > max_error_m:
                record["reason"] = "Reduced collider changed measured floor/roof distances"
                continue
            report_progress(f"{purpose} surface comparison", detail="checking every vertex and face centre in both directions")
            deviation = surface_deviation((vertices, faces), (new_vertices, new_faces),
                                          max_error_m, purpose=purpose)
            record["surface_deviation"] = deviation
            if not deviation["passed"]:
                record["reason"] = "Reduced collider exceeds surface error limit"
                continue
        except MeshInspectionError as error:
            record.update(reason=str(error), inspection=error.report)
            continue
        record["accepted"] = True
        report.update(inspection=inspected, output_triangles=len(new_faces),
                      achieved_reduction=1-len(new_faces)/len(faces))
        report_progress(f"{purpose} acceptance", len(report["attempts"]), len(report["attempts"]),
                        f"accepted {len(new_faces):,} triangles ({report['achieved_reduction']:.0%} reduction)")
        return new_vertices, new_faces
    if triangle_budget and len(faces) > triangle_budget:
        raise MeshInspectionError(dict(passed=False,
            failures=[f"No {purpose.lower()} reduction satisfies both the triangle and surface-error budgets"],
            reduction=report))
    fallback, precision = stabilize_collision_precision(vertices, faces, max_error_m)
    if not precision["passed"]:
        raise MeshInspectionError(dict(failures=[precision["failure"]], collision_precision=precision))
    fallback_inspection = inspect_surface(fallback, faces, **arguments)
    if passage_deviation(reference, fallback_inspection) > max_error_m:
        raise MeshInspectionError(dict(failures=["Full collider changed floor/roof distances after float32 conversion"],
                                       collision_precision=precision))
    deviation = surface_deviation((vertices, faces), (fallback, faces), max_error_m, purpose=purpose)
    if not deviation["passed"]:
        raise MeshInspectionError(dict(failures=["Full collider exceeds error budget after float32 conversion"],
                                       collision_precision=precision, surface_deviation=deviation))
    report.update(used_raw_fallback=True, inspection=fallback_inspection, precision=precision,
                  surface_deviation=deviation, output_triangles=len(faces),
                  achieved_reduction=0., fallback_reason="No smaller collider passed the bounded topology, passage and surface-error checks")
    report_progress("Collision fallback", detail=report["fallback_reason"])
    return fallback, faces
