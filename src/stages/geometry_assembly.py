"""Mesh assembly and connectivity reporting for Stage D geometry."""

from __future__ import annotations

from collections import defaultdict

from stages.geometry_types import CaveGeometry, GeometryConfig


def assemble_geometry(
    *,
    config: GeometryConfig,
    meshes,
    junction_patches,
    skylights,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...], int]:
    """Weld geometry elements into one assembled vertex/face set and count components."""

    tolerance = max(config.weld_tolerance, 1e-9)
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    element_vertex_sets: list[set[int]] = []

    def weld_vertex(vertex: tuple[float, float, float]) -> int:
        key = (
            int(round(vertex[0] / tolerance)),
            int(round(vertex[1] / tolerance)),
            int(round(vertex[2] / tolerance)),
        )
        existing = vertex_lookup.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        vertex_lookup[key] = index
        vertices.append(vertex)
        return index

    for element in list(meshes) + list(junction_patches) + list(skylights):
        remapped: list[int] = [weld_vertex(vertex) for vertex in element.vertices]
        element_vertex_sets.append(set(remapped))
        for face in element.faces:
            faces.append(tuple(remapped[index] for index in face))

    component_count = _count_element_components(element_vertex_sets)
    return tuple(vertices), tuple(faces), component_count


def _count_element_components(element_vertex_sets: list[set[int]]) -> int:
    if not element_vertex_sets:
        return 0

    vertex_to_elements: dict[int, list[int]] = defaultdict(list)
    for element_index, vertex_ids in enumerate(element_vertex_sets):
        for vertex_id in vertex_ids:
            vertex_to_elements[vertex_id].append(element_index)

    adjacency: list[set[int]] = [set() for _ in element_vertex_sets]
    for element_ids in vertex_to_elements.values():
        if len(element_ids) < 2:
            continue
        for element_id in element_ids:
            adjacency[element_id].update(other_id for other_id in element_ids if other_id != element_id)

    visited: set[int] = set()
    component_count = 0
    for start in range(len(element_vertex_sets)):
        if start in visited:
            continue
        component_count += 1
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
    return component_count
