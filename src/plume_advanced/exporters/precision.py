"""Bounded local repair of face orientations lost to engine coordinate rounding."""

from __future__ import annotations

import numpy as np


def repair_face_rounding(vertices, faces, candidate, normals, invalid, bound):
    """Move one vertex along a face-area gradient without damaging its neighbours.

    Every accepted move strictly removes an invalid face, introduces none, and
    stays within ``bound`` of the source vertex. Incidence is built only for the
    affected stars; no full-mesh scan occurs inside the trial loop. Both float32
    metres and float32 centimetres after metre serialization must retain the
    original orientation. The caller still checks the entire mesh afterwards.
    """
    result = candidate.copy()
    affected = np.flatnonzero(invalid)
    selected = np.unique(faces[affected])
    incidence: dict[int, list[int]] = {int(i): [] for i in selected}
    for face_id in np.flatnonzero(np.any(np.isin(faces, selected), axis=1)):
        for vertex_id in faces[face_id]:
            if int(vertex_id) in incidence:
                incidence[int(vertex_id)].append(int(face_id))
    stars = {i: np.asarray(ids, dtype=np.int64) for i, ids in incidence.items()}
    normal_lengths = np.linalg.norm(normals, axis=1)
    moves = []

    def invalid_faces(indices):
        triangles = result[faces[indices]]
        unit_normals = normals[indices] / normal_lengths[indices, None]
        bad = np.zeros(len(indices), dtype=bool)
        for scale in (1., 100.):
            encoded = (triangles * scale).astype(np.float32).astype(float) / scale
            cross = np.cross(encoded[:, 1] - encoded[:, 0], encoded[:, 2] - encoded[:, 0])
            bad |= np.sum(unit_normals * cross, axis=1) <= 0
        return bad

    for _ in range(4):
        remaining = affected[invalid_faces(affected)]
        if not len(remaining):
            break
        improved = False
        for face_id in remaining:
            for slot, vertex_id in enumerate(faces[face_id]):
                star = stars[int(vertex_id)]
                before = invalid_faces(star)
                if not before.any():
                    break
                edge = (result[faces[face_id, (slot + 1) % 3]]
                        - result[faces[face_id, (slot + 2) % 3]])
                direction = np.cross(edge, normals[face_id] / normal_lengths[face_id])
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                direction /= length
                previous = result[vertex_id].copy()
                accepted = False
                for fraction in (.002, .005, .01, .03, .1, .3, 1.):
                    trial = (previous + bound * fraction * direction).astype(np.float32).astype(float)
                    if np.linalg.norm(trial - vertices[vertex_id]) > bound:
                        continue
                    result[vertex_id] = trial
                    after = invalid_faces(star)
                    if after.sum() < before.sum() and not np.any(after & ~before):
                        moves.append(dict(vertex=int(vertex_id), face=int(face_id),
                                          step_fraction=fraction,
                                          repaired_faces=int(before.sum() - after.sum())))
                        improved = accepted = True
                        break
                if accepted:
                    break
                result[vertex_id] = previous
        if not improved:
            break
    return result, dict(method="signed_face_area_gradient", moves=moves,
                        affected_faces=len(affected), remaining_faces=int(invalid_faces(affected).sum()))
