"""Turn compatible local passage encounters into explicit graph junctions.

This is a bounded topology transaction, not a blanket overlap exemption. The
caller must inspect the entire returned network before committing the edit.
"""

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from plume_advanced.stages.network_layers import segment_xyz
from plume_advanced.stages.network_neighborhoods import PassageNeighborhoods
from plume_advanced.stages.network_quality import _crossings


def _contacts(network, changed):
    """Exact crossings first, then touching width envelopes on a 0.5 m grid."""
    segments = sorted(network.segments, key=lambda s: s.segment_id)
    chains = {s.segment_id: (np.array([(p.x, p.y) for p in s.points]),
                              np.array([p.width for p in s.points])) for s in segments}
    lookup = {s.segment_id: s for s in segments}
    contacts = []
    for hit in _crossings(chains, network):
        a, b = hit["segments"]
        if a == b or not changed.intersection((a, b)):
            continue
        stations = []
        for label, sid in zip(("first", "second"), (a, b)):
            s = lookup[sid]
            i, t = hit[f"{label}_piece"], hit[f"{label}_fraction"]
            stations.append(s.points[i].arc_length*(1-t)+s.points[i+1].arc_length*t)
        contacts.append((0., a, b, *stations))
    arrays, stations, widths = [], [], []
    for s in segments:
        arc = np.array([p.arc_length for p in s.points])
        at = np.linspace(0, arc[-1], max(2, int(np.ceil(arc[-1]/.5))+1))
        xyz = segment_xyz(s, network.config.layers)
        arrays.append((s, np.column_stack([np.interp(at, arc, v) for v in xyz.T])))
        stations.append(at)
        widths.append(np.interp(at, arc, chains[s.segment_id][1]))
    neighborhoods = PassageNeighborhoods(arrays, widths=widths)
    trees = [cKDTree(xyz[:, :2]) for _, xyz in arrays]
    for i, (a, xyz) in enumerate(arrays):
        for j in range(i+1, len(arrays)):
            b, other = arrays[j]
            if not changed.intersection((a.segment_id, b.segment_id)):
                continue
            # A crossing already gives a precise encounter for this pair.
            if any(c[1:3] == (a.segment_id, b.segment_id) for c in contacts):
                continue
            best = None
            limit = .5*(widths[i].max()+widths[j].max())
            for offset in range(0, len(xyz), 512):
                neighbors = trees[j].query_ball_point(xyz[offset:offset+512, :2], limit)
                counts = np.fromiter(map(len, neighbors), dtype=int, count=len(neighbors))
                if not counts.sum():
                    continue
                first = np.repeat(np.arange(offset, offset+len(neighbors)), counts)
                second = np.concatenate(neighbors).astype(int)
                distance = np.linalg.norm(xyz[first, :2]-other[second, :2], axis=1)
                eligible = distance <= .5*(widths[i][first]+widths[j][second])
                eligible &= ~neighborhoods.local(i, first, j, second)
                indices = np.flatnonzero(eligible)
                if len(indices):
                    k = indices[np.argmin(distance[indices])]
                    contact = (float(distance[k]), a.segment_id, b.segment_id,
                               float(stations[i][first[k]]), float(stations[j][second[k]]))
                    if best is None or contact < best:
                        best = contact
            if best is not None:
                contacts.append(best)
    return sorted(set(contacts))


def _at(segment, station, layers):
    arc = np.array([p.arc_length for p in segment.points])
    xyz = segment_xyz(segment, layers)
    position = np.array([np.interp(station, arc, col) for col in xyz.T])
    a, b = max(0., station-1), min(arc[-1], station+1)
    tangent = np.array([np.interp(b, arc, col)-np.interp(a, arc, col) for col in xyz[:, :2].T])
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    return position, tangent


def _split(segment, station, node, potential, xyz, next_id, host, config):
    """Keep original approaches; ease a local encounter into one shared XYZ."""
    arc = np.array([p.arc_length for p in segment.points])
    # A support of three widths keeps snapping a touching footprint gradual.
    support = min(3*segment.mean_width, station-2*segment.mean_width,
                  arc[-1]-station-2*segment.mean_width)
    if support <= segment.mean_width:
        raise ValueError("encounter_too_close_to_existing_node")
    at = np.unique(np.r_[arc, station, np.linspace(station-support, station+support,
                                                  int(np.ceil(2*support/.5))+1)])
    # Original and dense stations can differ only by round-off. Such slivers
    # create effectively zero chords and spurious 180-degree turns after snap.
    at = at[(abs(at-station) > 1e-6) | (at == station)]
    at = at[np.r_[True, np.diff(at) > 1e-6]]
    original_xyz = segment_xyz(segment, config.layers)
    values = np.column_stack([np.interp(at, arc, col) for col in original_xyz.T])
    center, _ = _at(segment, station, config.layers)
    t = np.clip(1-abs(at-station)/support, 0, 1)
    taper = t*t*t*(t*(6*t-15)+10)
    values += taper[:, None]*(xyz-center)
    # Exact shared endpoint, independent of floating point interpolation order.
    cut = int(np.searchsorted(at, station))
    values[cut] = xyz
    widths = np.interp(at, arc, [p.width for p in segment.points])
    pieces = []
    for part, indices in enumerate((np.arange(cut+1), np.arange(cut, len(at)))):
        coords = values[indices]
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1))]
        samples = [host.sample(float(x), float(y)) for x, y in coords[:, :2]]
        points = tuple(replace(segment.points[0], index=i, x=float(v[0]), y=float(v[1]),
                               elevation=samples[i].elevation, arc_length=float(lengths[i]),
                               width=float(widths[k]), slope_degrees=samples[i].slope_degrees,
                               cover_thickness=samples[i].cover_thickness,
                               roof_competence=samples[i].roof_competence, growth_cost=samples[i].growth_cost)
                       for i, (k, v) in enumerate(zip(indices, coords)))
        metadata = dict(segment.metadata)
        metadata.pop("network_detail_vertical_offsets_m", None)
        # Coarse planner-cell indices no longer describe a subdivided route.
        metadata.pop("regional_cell_path", None)
        metadata.pop("regional_cell_fractions", None)
        metadata["detail_parent_segment_id"] = metadata.get("detail_parent_segment_id", segment.segment_id)
        metadata["regional_end_potential" if part == 0 else "regional_start_potential"] = potential
        if part:
            metadata.pop("source_system_id", None)
        elif "regional_blind_terminal" in metadata:
            metadata["regional_blind_terminal"] = False
        piece = replace(segment, segment_id=segment.segment_id if part == 0 else next_id,
                        start_node_id=segment.start_node_id if part == 0 else node.node_id,
                        end_node_id=node.node_id if part == 0 else segment.end_node_id,
                        points=points, metadata=metadata)
        if config.layers.enabled:
            metadata["network_detail_vertical_offsets_m"] = (coords[:, 2]-segment_xyz(piece, config.layers)[:, 2]).tolist()
        pieces.append(piece)
    return pieces


def connect_overlaps(generator, host, network, changed, *, maximum_junctions=8):
    """Propose shared nodes before the caller rejects an overlapping edit.

    Do not join overpasses, ramps, opposed flows, old node approaches or flow
    cycles. Extended overlap still has to pass the ordinary separation check.
    Every unsuccessful encounter is recorded and leaves its input untouched.
    """
    from plume_advanced.stages.network import CaveNode
    from plume_advanced.stages.network_acceptance import rebuild_network_geometry

    current = network
    changed = set(changed)
    audit = dict(junctions=[], rejected=[], limit=maximum_junctions)
    attempted = set()
    for _ in range(maximum_junctions):
        joined = False
        lookup = {s.segment_id: s for s in current.segments}
        for contact in _contacts(current, changed):
            if contact in attempted:
                continue
            attempted.add(contact)
            distance, aid, bid, sa, sb = contact
            a, b = lookup[aid], lookup[bid]
            try:
                if current.config.layers.enabled:
                    layers = {s.metadata.get(f"regional_{side}_layer") for s in (a, b) for side in ("start", "end")}
                    if len(layers) != 1:
                        raise ValueError("separate_layers_or_ramp")
                pa, ta = _at(a, sa, current.config.layers)
                pb, tb = _at(b, sb, current.config.layers)
                if abs(pa[2]-pb[2]) > .5:
                    raise ValueError("different_elevations")
                if ta @ tb < np.cos(np.radians(current.config.interconnection.maximum_junction_angle_degrees)):
                    raise ValueError("incompatible_flow_directions")
                low = max(float(s.metadata["regional_end_potential"]) for s in (a, b))
                high = min(float(s.metadata["regional_start_potential"]) for s in (a, b))
                if high-low <= 1e-6:
                    raise ValueError("incompatible_directed_potential")
                potential = np.mean([float(s.metadata["regional_start_potential"])*(1-t/s.total_length)
                                     + float(s.metadata["regional_end_potential"])*t/s.total_length
                                     for s, t in ((a, sa), (b, sb))])
                potential = float(np.clip(potential, low+.25*(high-low), high-.25*(high-low)))
                xyz = (pa+pb)/2
                geometry = generator._build_flow_geometry(host)
                node = CaveNode(max(n.node_id for n in current.nodes)+1, float(xyz[0]), float(xyz[1]),
                                generator._project_along(geometry, *xyz[:2]),
                                float((xyz[0]-geometry.seed_x)*geometry.cross_x+(xyz[1]-geometry.seed_y)*geometry.cross_y),
                                "junction")
                next_id = max(lookup)+1
                pieces = (_split(a, sa, node, potential, xyz, next_id, host, current.config)
                          + _split(b, sb, node, potential, xyz, next_id+1, host, current.config))
                segments = tuple(s for s in current.segments if s.segment_id not in (aid, bid)) + tuple(pieces)
                candidate = replace(current, nodes=(*current.nodes, node), segments=tuple(sorted(segments, key=lambda s: s.segment_id)))
                # Topology edits must refresh flows before quality inspection;
                # otherwise even a valid confluence has stale flux/route data.
                current = rebuild_network_geometry(generator, host, candidate, list(candidate.segments))
            except ValueError as error:
                audit["rejected"].append(dict(segments=[aid, bid], reason=str(error)))
                continue
            audit["junctions"].append(dict(node_id=node.node_id, segments=[aid, bid],
                                           child_segments=[s.segment_id for s in pieces], xyz_m=xyz.tolist(),
                                           contact_distance_m=distance))
            changed.update(s.segment_id for s in pieces)
            joined = True
            break
        if not joined:
            break
    return current, audit


def original_routes_preserved(original, candidate):
    """Old nodes and each directed route survive, possibly subdivided by joins."""
    nodes = {n.node_id: n for n in candidate.nodes}
    if any(nodes.get(n.node_id) != n for n in original.nodes):
        return False
    groups: dict[int, list[Any]] = {}
    for s in candidate.segments:
        parent = s.metadata.get("detail_parent_segment_id", s.segment_id)
        groups.setdefault(parent, []).append(s)
    if groups.keys() != {s.segment_id for s in original.segments}:
        return False
    for s in original.segments:
        pieces = groups[s.segment_id]
        by_start = {p.start_node_id: p for p in pieces}
        if len(by_start) != len(pieces) or any((p.kind, p.z_level) != (s.kind, s.z_level) for p in pieces):
            return False
        node, visited = s.start_node_id, set()
        while node in by_start and node not in visited:
            visited.add(node)
            node = by_start[node].end_node_id
        if node != s.end_node_id or len(visited) != len(pieces):
            return False
    return True
