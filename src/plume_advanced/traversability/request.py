"""Small chart descriptions shared by generation and saved-run mapping."""

from dataclasses import dataclass

import numpy as np

from .config import TraversabilityConfig


@dataclass(frozen=True)
class TraversabilityRequest:
    config: TraversabilityConfig
    charts: tuple[dict, ...]
    provenance: dict
    segments: tuple[dict, ...]


def from_paths(config, segments, paths, *, layer_count=1, layered=False, provenance=None):
    graph: list[dict] = []
    seen = set()
    charts: list[dict] = [
        dict(id=f"layer_{i}", kind="layer", layer=i, segment_ids=[], paths=[], portals=[])
        for i in range(layer_count)
    ]
    for sid, start_node, end_node, metadata in sorted(segments, key=lambda s: s[0]):
        if sid in seen:
            raise ValueError("Map segment identifiers must be unique")
        seen.add(sid)
        path = np.asarray(paths[sid], float)
        if (
            path.ndim != 2
            or path.shape[1] != 5
            or len(path) < 2
            or not np.isfinite(path).all()
            or (path[:, 3:] <= 0).any()
        ):
            raise ValueError("Traversability requires finite XYZ/width/height section paths")
        a, b = (
            (int(metadata[f"regional_{key}_layer"]) for key in ("start", "end"))
            if layered
            else (0, 0)
        )
        if not (0 <= a < layer_count and 0 <= b < layer_count):
            raise ValueError("Traversability layer identifiers are outside the declared layout")
        graph.append(dict(id=sid, start_node=start_node, end_node=end_node,
                          from_layer=a, to_layer=b, path=path,
                          chart_id=f"layer_{a}" if a == b else f"ramp_{sid}"))
        if a == b:
            chart = charts[a]
            chart["segment_ids"].append(sid)
            chart["paths"].append(path)
        else:
            portals = []
            for layer, node, point in ((a, start_node, path[0]), (b, end_node, path[-1])):
                portal = dict(node_id=node, xyz_m=point[:3].tolist())
                charts[layer]["portals"].append(dict(portal, connects_to=f"ramp_{sid}"))
                portals.append(dict(portal, connects_to=f"layer_{layer}"))
            charts.append(
                dict(
                    id=f"ramp_{sid}",
                    kind="ramp",
                    segment_ids=[sid],
                    paths=[path],
                    from_layer=a,
                    to_layer=b,
                    start_node=start_node,
                    end_node=end_node,
                    start_xyz_m=path[0, :3].tolist(),
                    end_xyz_m=path[-1, :3].tolist(),
                    portals=portals,
                )
            )
    if any(not c["paths"] for c in charts):
        raise ValueError("A declared traversability layer has no section samples")
    return TraversabilityRequest(config, tuple(charts), provenance or {}, tuple(graph))


def from_generation(config, network, sections):
    from plume_advanced.evaluation.artifacts import network_semantic_hash, section_semantic_hash

    return from_paths(
        config,
        [(s.segment_id, s.start_node_id, s.end_node_id, s.metadata) for s in network.segments],
        {
            f.segment_id: np.array(
                [(s.x, s.y, s.z, s.tube_width, s.tube_height) for s in f.samples]
            )
            for f in sections.segment_fields
        },
        layer_count=network.config.layers.count if network.config.layers.enabled else 1,
        layered=network.config.layers.enabled,
        provenance=dict(
            network_semantic_sha256=network_semantic_hash(network),
            section_semantic_sha256=section_semantic_hash(sections),
        ),
    )
