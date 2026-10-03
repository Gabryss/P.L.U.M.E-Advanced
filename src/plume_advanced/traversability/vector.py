"""Local-metre vector geometry and intended topology with measured witnesses."""

from collections import defaultdict

import numpy as np
import trimesh

from plume_advanced.progress import report_progress

from .boundaries import mask_rings
from .connectivity import ConnectionInspector

COORDINATES = dict(frame="PLUME canonical right-handed Z-up", units="metres",
                   geographic=False)


def chart_vectors(chart, fields):
    """Cell outlines are sampled geometry, never exact mesh-wall contours."""
    origin, r = fields["origin_xy_m"], float(fields["resolution_m"])
    sampled = fields["sampled_cavity"]
    return dict(
        schema="plume.vector-chart.v1", chart_id=chart["id"], kind=chart["kind"],
        coordinates=COORDINATES, resolution_m=r, origin_xy_m=origin.tolist(),
        shape_yx=list(sampled.shape),
        cavity_rings=mask_rings(sampled, origin, r),
        obstacle_rings=mask_rings(fields["obstacle"] & sampled, origin, r),
        domain_rings=mask_rings(fields["chart_domain"], origin, r),
        scope="Exact boundaries of sampled cell masks, clipped to the chart domain. Not exact mesh contours. Obstacle masks conservatively cover projected prop triangles.",
        ring_convention="Closed XY rings; CCW outer, CW hole. Pair rings by component_id within each mask; diagonal contacts do not merge components. No Z inferred for a boundary.",
    )


def network_vectors(request, vertices, faces):
    """Keep source IDs and XYZ stations; never join paths merely by XY overlap."""
    report_progress("Vector boundary check", detail="checking closure of delivered cave surface")
    closed = bool(trimesh.Trimesh(vertices, faces, process=False).is_watertight)
    if not closed:
        # Visual GLB surfaces can duplicate vertices at UV seams. Exact welding
        # is diagnostic only; it never changes the delivered arrays or hashes.
        welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
        closed = bool(trimesh.Trimesh(welded, inverse[faces], process=False).is_watertight)
    inspector = ConnectionInspector(vertices, faces)
    edges = []
    incidents = defaultdict(list)
    for number, segment in enumerate(request.segments):
        report_progress("Vector passages", number, len(request.segments),
                        f"segment {segment['id']}; continuous cavity witness")
        xyz = segment["path"][:, :3]
        distance = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if distance.sum() <= max(inspector.vertical.tolerance, 1e-6):
            connection = dict(status="unresolved", connected=None, reason="zero-length witness")
        else:
            connection = inspector.connection(xyz)
        if not closed:
            connection.update(status="unresolved", connected=None,
                              reason="cave boundary is not closed after exact vertex welding")
        edge = {k: v for k, v in segment.items() if k != "path"}
        edge.update(xyz_m=xyz.tolist(), length_m=float(distance.sum()),
                    kind="ramp" if segment["from_layer"] != segment["to_layer"] else "passage",
                    cavity_witness=connection)
        edges.append(edge)
        for endpoint, i, key in (("start", 0, "from_layer"), ("end", -1, "to_layer")):
            incidents[segment[f"{endpoint}_node"]].append(dict(
                segment_id=segment["id"], endpoint=endpoint, layer=segment[key],
                xyz_m=xyz[i].tolist(), chart_id=segment["chart_id"]))
    nodes = []
    for number, (node_id, ports) in enumerate(sorted(incidents.items())):
        report_progress("Vector junctions", number, len(incidents),
                        "check incident endpoints without snapping across rock")
        anchor = np.array(ports[0]["xyz_m"])
        links = []
        for port in ports:
            point = np.array(port["xyz_m"])
            gap = float(np.linalg.norm(point - anchor))
            # Tiny differences are still checked for surface contact. A stationary
            # point must be inside; equal XYZ is not enough to verify a junction.
            if gap <= max(inspector.vertical.tolerance, 1e-6):
                tol = max(inspector.vertical.tolerance, 1e-6)
                clear = (inspector.vertical.measure(anchor)["inside"]
                         and inspector.vertical.measure(point)["inside"]
                         and inspector.index.swept_capsule_distance(anchor, point, 0., tol) > tol)
                witness = dict(status="verified" if clear else "unresolved",
                               connected=True if clear else None)
            else:
                witness = inspector.connection(np.array([anchor, point]))
            if not closed:
                witness.update(status="unresolved", connected=None, reason="cave boundary is not closed")
            links.append(dict(segment_id=port["segment_id"], endpoint=port["endpoint"],
                              endpoint_gap_m=gap, cavity_witness=witness))
        incoming = sum(p["endpoint"] == "end" for p in ports)
        outgoing = len(ports) - incoming
        role = ("source_terminal" if outgoing else "sink_terminal") if len(ports) == 1 else (
            "junction" if len(ports) > 2 else "through")
        nodes.append(dict(id=node_id, xyz_m=anchor.tolist(), role=role, degree=len(ports),
                          incoming=incoming, outgoing=outgoing,
                          layers=sorted({p["layer"] for p in ports}), ports=ports,
                          endpoint_links=links,
                          cavity_witness_status="verified" if all(
                              p["cavity_witness"]["status"] == "verified" for p in links
                          ) else "unresolved"))
    report_progress("Vector junctions", len(nodes), len(nodes), "endpoint checks complete")
    verified = sum(e["cavity_witness"]["status"] == "verified" for e in edges)
    return dict(schema="plume.vector-network.v1", coordinates=COORDINATES,
                closed_cave_boundary=closed,
                provenance=request.provenance, nodes=nodes, edges=edges,
                summary=dict(nodes=len(nodes), edges=len(edges),
                             verified_edge_witnesses=verified, unresolved_edge_witnesses=len(edges)-verified,
                             unresolved_junctions=sum(n["cavity_witness_status"] != "verified" for n in nodes)),
                scope="Intended section graph plus continuous point-path witnesses against the closed delivered cave boundary. Props and robot feasibility are separate. An unresolved witness is not proof of disconnection.",
                topology="Node IDs define intended incidence; XY overlap never creates a junction. Traverse a verified edge only through verified endpoint_links at its nodes. Direction is generation order, not a navigation restriction. Terminals are graph endpoints, not certified openings to the exterior.",
                boundary_assumption="The supplied cave surface encloses cavity air. Vertical parity and segment/triangle contact tests require a closed boundary.")
