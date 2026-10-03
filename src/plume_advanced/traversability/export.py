"""Export numeric terrain truth, reference classifications and readable maps."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from plume_advanced.identity import sha256_file
from plume_advanced.progress import report_progress

from .plots import render_views
from .raster import cavity_fields, chart_reference, classify, obstacle_cells, vertical_hits
from .vector import chart_vectors, network_vectors
from .vector_plots import render_chart, render_network


class TraversabilityBudgetError(ValueError):
    """A map allocation limit, independent of whether the cave is exportable."""


def surface_hash(vertices, faces):
    h = hashlib.sha256()
    for array, dtype in ((vertices, "<f8"), (faces, "<i8")):
        a = np.ascontiguousarray(array, dtype=dtype)
        h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def export_traversability(
    request, vertices, faces, output, *, obstacles=(), surface_kind="collision", source=None
):
    """Use the accepted export arrays, never the schematic network or base atlas."""
    config = request.config
    if not config.enabled:
        return ()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Traversability needs a finite exported surface")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Traversability needs valid surface triangles")
    r = config.resolution_m
    origin = np.floor(vertices[:, :2].min(axis=0) / r) * r - r
    size = np.ceil((vertices[:, :2].max(axis=0) - origin) / r).astype(int) + 1
    shape = tuple(size[::-1])
    if int(size.prod()) > config.max_cells:
        raise TraversabilityBudgetError(
            f"Traversability grid needs {size.prod():,} cells; limit is {config.max_cells:,}. "
            "Reduce map extent or explicitly raise traversability.max_cells."
        )
    pixels, heights = vertical_hits(vertices, faces, origin, shape, r)
    # Props are serialized as float32 positions by the scene exporters.
    obstacle_meshes = tuple(
        (np.asarray(v, np.float32).astype(float), np.asarray(f, int)) for v, f in obstacles
    )
    records: list[dict] = []
    files: list[Path] = []
    network = network_vectors(request, vertices, faces)
    network["surface"] = dict(kind=surface_kind, sha256=surface_hash(vertices, faces))
    network_path = output / "network_vectors.json"
    network_path.write_text(json.dumps(network, indent=2, allow_nan=False) + "\n")
    global_names = [network_path.name, render_network(output, network)]
    for number, chart in enumerate(request.charts):
        report_progress(
            "Traversability charts",
            number,
            len(request.charts),
            chart["id"] + "; cavity, support, slope, step, headroom",
        )
        samples = np.concatenate(chart["paths"])
        # Fixed padding keeps physical data and vector outlines independent of
        # the chosen robot. chart_reference covers half-width + two cells.
        margin = samples[:, 3].max() / 2 + 3 * r
        lo = np.maximum(np.floor((samples[:, :2].min(axis=0) - margin - origin) / r).astype(int), 0)
        hi = np.minimum(
            np.ceil((samples[:, :2].max(axis=0) + margin - origin) / r).astype(int), size
        )
        local_origin = origin + lo * r
        local_shape = tuple((hi - lo)[::-1])
        x, y = pixels % shape[1], pixels // shape[1]
        selected = (x >= lo[0]) & (x < hi[0]) & (y >= lo[1]) & (y < hi[1])
        local_pixels = (y[selected] - lo[1]) * local_shape[1] + x[selected] - lo[0]
        hint = chart_reference(chart["paths"], local_origin, local_shape, r)
        floor, roof, uncertain = cavity_fields(local_pixels, heights[selected], hint)
        blocked, obstacle_height = obstacle_cells(obstacle_meshes, floor, roof, local_origin, r)
        fields = classify(floor, roof, uncertain, blocked, config)
        fields.update(
            floor_z_m=floor.astype(np.float32),
            ceiling_z_m=roof.astype(np.float32),
            vertical_clearance_m=(roof - floor).astype(np.float32),
            obstacle=blocked,
            obstacle_height_m=obstacle_height,
            origin_xy_m=local_origin,
            resolution_m=np.asarray(r),
            chart_domain=np.isfinite(hint),
            sampled_cavity=np.isfinite(floor) & np.isfinite(roof) & ~uncertain,
        )
        sampled = fields["sampled_cavity"]
        fields["cavity_component_id"] = ndimage.label(sampled)[0].astype(np.int32)
        physical = sampled.astype(np.uint8)
        physical[sampled & blocked] = 2
        fields["physical_state"] = physical
        stem = output / chart["id"]
        np.savez_compressed(stem.with_suffix(".npz"), **fields)
        physical_path = stem.with_name(stem.name + "_physical.png")
        Image.fromarray(np.flipud(physical)).save(physical_path)
        report_progress("Vector chart boundaries", number, len(request.charts), chart["id"])
        vectors = chart_vectors(chart, fields)
        vector_path = stem.with_name(stem.name + "_vectors.json")
        vector_path.write_text(json.dumps(vectors, indent=2, allow_nan=False) + "\n")
        geometry_preview = render_chart(output, chart, fields, vectors, network)
        # Raw aligned raster: PNG row zero is the NORTH edge, NPZ row zero SOUTH.
        occupancy = np.full(fields["status"].shape, 205, np.uint8)
        occupancy[np.isin(fields["status"], [0, 2])] = 0
        occupancy[fields["status"] == 1] = 254
        image_path = stem.with_name(stem.name + "_occupancy.png")
        Image.fromarray(np.flipud(occupancy)).save(image_path)
        record = {k: v for k, v in chart.items() if k != "paths"}
        # Topological links are useful for layered planners, but are not promises
        # that this reference footprint can cross the connector.
        record["portals"] = []
        for portal in chart["portals"]:
            col, row = np.floor((np.array(portal["xyz_m"][:2]) - local_origin) / r).astype(int)
            inside = 0 <= row < local_shape[0] and 0 <= col < local_shape[1]
            record["portals"].append(
                dict(
                    portal,
                    cell_row_col=[int(row), int(col)] if inside else None,
                    sampled_status=int(fields["status"][row, col]) if inside else 255,
                    component_id=int(fields["component_id"][row, col]) if inside else 0,
                )
            )
        record.update(
            origin_xy_m=local_origin.tolist(),
            shape_yx=[int(value) for value in local_shape],
            bounds_xy_m=[
                local_origin.tolist(),
                (local_origin + np.array(local_shape[::-1]) * r).tolist(),
            ],
            status_counts={str(k): int(np.sum(fields["status"] == k)) for k in (0, 1, 2, 255)},
            npz=stem.with_suffix(".npz").name,
            preview=stem.with_suffix(".png").name,
            occupancy=image_path.name,
            physical_raster=physical_path.name,
            vectors=vector_path.name,
            geometry_preview=geometry_preview,
        )
        if chart["kind"] == "ramp":
            record["centreline_xyz_m"] = chart["paths"][0][:, :3].tolist()
        records.append(record)
    display = render_views(output, records, config)
    for record in records:
        names = [
            record["npz"],
            record["occupancy"],
            record["physical_raster"],
            record["vectors"],
            record["geometry_preview"],
            record["overview"],
            *(view["file"] for view in record["views"].values()),
        ]
        record["files_sha256"] = {name: sha256_file(output / name) for name in names}
        files.extend(output / name for name in names)
    files.extend(output / name for name in global_names)
    implementation = hashlib.sha256()
    for module in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(module.name.encode())
        implementation.update(module.read_bytes())
    metadata = dict(
        schema="plume.traversability.v1",
        status="complete",
        config=asdict(config),
        provenance=request.provenance,
        implementation_sha256=implementation.hexdigest(),
        surface=dict(
            kind=surface_kind,
            sha256=surface_hash(vertices, faces),
            triangles=len(faces),
            source=source,
            obstacle_meshes=len(obstacle_meshes),
            obstacle_sha256=[surface_hash(v, f) for v, f in obstacle_meshes],
        ),
        coordinates=dict(
            frame="PLUME canonical right-handed Z-up",
            units="metres",
            array_order="[row_y, column_x]; row 0 is south; column 0 is west",
            cell_centre="origin_xy_m + resolution_m * [column + 0.5, row + 0.5]",
            occupancy_png="row 0 is north; values 0 occupied, 254 passes reference limits, 205 unknown",
        ),
        status_codes={
            "0": "outside mapped cavity",
            "1": "passes sampled reference limits",
            "2": "blocked",
            "255": "unknown",
        },
        reason_bits={
            "1": "incomplete footprint support",
            "2": "slope",
            "4": "detrended step/roughness",
            "8": "body headroom",
            "16": "placed obstacle",
            "32": "uncertain surface",
        },
        method=dict(
            reference_footprint="circumscribed horizontal disk plus margin and half-cell diagonal",
            support_radius_m=config.radius_m + np.sqrt(2) * r / 2,
            slope="least-squares floor plane over the footprint",
            step="peak-to-peak residual after subtracting the floor plane",
            headroom="minimum roof minus maximum floor over the footprint",
            props="conservative projected triangle bounding boxes; not treated as climbable supports",
        ),
        robot_qualification=False,
        charts=records,
        vector_network=dict(file=network_path.name, preview=global_names[1], **network["summary"]),
        files_sha256={name: sha256_file(output / name) for name in global_names},
        physical_state_codes={"0": "unmeasured or ambiguous", "1": "sampled cavity",
                              "2": "conservative placed-prop mask in sampled cavity"},
        display_scales=display,
        preview_convention="Annotated maps use world XY axes; use NPZ or unannotated occupancy PNG for pixel indexing.",
        limitations=[
            "Sampled static geometry, not wheel/track dynamics, friction, steering or continuous path qualification.",
            "Features between cell-centre rays can be missed. Resolution and source mesh convergence must be assessed for each experiment.",
            "Reference limits classify the map only; they never repair, reject or regenerate the cave.",
            "Different passing-cell components do not prove physical disconnection: footprint dilation and reference limits can remove open passages.",
            "Layer and ramp charts are separate. Connect only through the recorded ramp endpoints, never through XY overlap.",
            "Portals describe network topology. Their sampled status and component identifiers do not certify the whole ramp or a traversable transition.",
            "Event props use exported geometry as conservative obstacles; native engine collider settings may differ.",
            "Physical cavity components and vector outlines follow cell samples within a restricted chart domain; holes and gaps are not proof of solid rock or physical disconnection.",
            "Vector centreline witnesses test the closed cave boundary only, without props or robot limits; their failures remain unresolved and do not discard the cave.",
        ],
    )
    path = output / "manifest.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    report_progress(
        "Traversability charts",
        len(records),
        len(records),
        "numeric maps, legends, occupancy rasters and connections saved",
    )
    return (*files, path)
