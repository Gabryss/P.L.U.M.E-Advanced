#!/usr/bin/env python3
"""Bounded seed/body checks: full A-C networks, local junction meshes, fixed-physics collapse."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import trimesh
from generate_tube_only import export_surface

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_network_artifact, export_section_artifact
from plume_advanced.evaluation.experiments.common import for_seed, generate_sections
from plume_advanced.evaluation.local_geometry import (
    local_geometry,
    section_contour,
    section_resolution_report,
)
from plume_advanced.evaluation.metrics.contours import self_intersection_count
from plume_advanced.evaluation.metrics.morphometry import contour_morphometry
from plume_advanced.evaluation.metrics.network import network_metrics
from plume_advanced.stability import RoofStabilityModel
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.geometry_types import GeometryConfig
from plume_advanced.stages.network import CaveNetwork, CaveNetworkConfig
from plume_advanced.stages.section_field import (
    SectionField,
    SectionFieldConfig,
    SectionSample,
    SegmentSectionField,
)

ROOT = Path(__file__).resolve().parents[1]


def mesh_junction_region(network, sections, config, sample, extent, folder):
    """Expand any clipped axis before accepting a local cross-section."""
    center = np.array([sample.x, sample.y, sample.z])
    extent = np.asarray(extent, float).copy()
    voxel = config.voxel_size
    for attempt in range(9):
        (folder / "region_progress.json").write_text(
            json.dumps(
                {
                    "attempt": attempt,
                    "center_xyz_m": center.tolist(),
                    "half_extent_m": extent.tolist(),
                    "voxel_size_m": voxel,
                },
                indent=2,
            )
            + "\n"
        )
        geometry = local_geometry(network, sections, config, center=center, half_extent=extent)
        export_surface(geometry, folder)
        scene = trimesh.load(folder / "lava_tube_geometry.glb", force="scene", process=False)
        mesh = next(iter(scene.geometry.values())).copy(include_cache=True)
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
        bounds = np.array(
            [
                geometry.voxel_grid.origin,
                np.asarray(geometry.voxel_grid.origin)
                + (np.asarray(geometry.voxel_grid.shape) - 1) * voxel,
            ]
        )
        contour = section_contour(mesh, sample)
        points = (
            center + contour[:, 0, None] * sample.normal + contour[:, 1, None] * sample.binormal
        )
        clipped = np.any(
            (points <= bounds[0] + 2 * voxel) | (points >= bounds[1] - 2 * voxel), axis=0
        )
        if np.any(clipped):
            extent[clipped] = 1.8 * extent[clipped] + 2 * voxel
            del geometry, mesh, scene
            continue
        section_contour(mesh, sample, region_bounds=bounds, clip_margin_m=2 * voxel)
        np.save(folder / "junction_contour.npy", contour)
        return {
            "checks_passed": True,
            "closed_axis_section": True,
            "contour_clear_of_artificial_caps": True,
            "region_bounds_xyz_m": bounds.tolist(),
            "minimum_contour_distance_to_clip_m": float(
                min((points - bounds[0]).min(), (bounds[1] - points).min())
            ),
            "box_expansion_count": attempt,
            "cut": contour_morphometry(contour),
            "cut_scope": "Junction-plane envelope; an intersecting branch can run along the plane. Not an ordinary passage width or unsupported roof span.",
            "surface_components": geometry.component_count,
            "components": [
                {
                    "triangles": len(part.faces),
                    "touches_clip_region": bool(
                        np.any(part.bounds[0] - bounds[0] < 2 * voxel)
                        or np.any(bounds[1] - part.bounds[1] < 2 * voxel)
                    ),
                }
                for part in mesh.split(only_watertight=False)
            ],
            "component_scope": "Box clipping may separate unrelated nearby branches; not a full-network connectivity measurement.",
            "stability_collapse_count": geometry.summary()["stability_collapse_count"],
        }
    raise ValueError("The selected cut remains clipped after eight box expansions")


def collapse_counterfactual():
    """Hold roof, passage shape and rock fixed; vary gravity alone."""
    angle = np.linspace(0, 2 * np.pi, 41)
    profile = tuple(zip(10 * np.cos(angle), 3 * np.sin(angle)))
    samples = tuple(
        SectionSample(
            index=i,
            segment_id=0,
            segment_arc_length=10.0 * i,
            x=10.0 * i,
            y=0.0,
            z=0.0,
            surface_z=4.0,
            cover_thickness=10.0,
            roof_thickness=1.0,
            centerline_depth=4.0,
            tangent=(1.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            binormal=(0.0, 0.0, 1.0),
            tube_width=20.0,
            tube_height=6.0,
            floor_flatness=0.6,
            roof_arch=1.0,
            lateral_skew=0.0,
            junction_blend_weight=0.0,
            junction_influences=(),
            profile_points=profile,
        )
        for i in range(3)
    )
    network = CaveNetwork(
        config=CaveNetworkConfig(),
        nodes=(),
        segments=(),
        junctions=(),
        occupancy=np.zeros((2, 2)),
        width_field=np.zeros((2, 2)),
        dominant_route_node_ids=(),
        slice_along_positions=(),
        slice_channel_counts=(),
        slice_visible_channel_counts=(),
    )
    config = GeometryConfig(
        voxel_size=0.5,
        storage_mode="dense",
        wall_roughness_amplitude=0.0,
        tunnel_radius_scale=1.0,
        density_margin=2.0,
    )
    results = []
    for body, gravity in (("earth", 9.80665), ("mars", 3.71), ("moon", 1.62)):
        model = RoofStabilityModel(gravity_m_s2=gravity)
        prediction = model.assess(width_m=20.0, height_m=6.0, floor_depth_m=7.0)
        sections = SectionField(
            config=SectionFieldConfig(gravity_m_s2=gravity),
            segment_fields=(SegmentSectionField(0, (), samples),),
            dominant_route_segment_ids=(0,),
        )
        geometry = GeometryGenerator(config).generate(network, sections)
        axis_open = bool(geometry.voxel_grid.sample_density((10.0, 0.0, 0.0)) > config.iso_level)
        results.append(
            {
                "body": body,
                "model": asdict(model),
                "assessment": asdict(prediction),
                "mesh_has_faces": bool(geometry.assembled_faces),
                "axis_open": axis_open,
                "collapse_count": geometry.summary()["stability_collapse_count"],
                "consistent_with_screen": axis_open != prediction.failed,
            }
        )
    return {
        "width_m": 20.0,
        "height_m": 6.0,
        "floor_depth_m": 7.0,
        "roof_thickness_m": 1.0,
        "scope": "Implementation behavior of the conservative beam screen; not empirical collapse validation.",
        "results": results,
        "all_passed": all(r["consistent_with_screen"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/earth_tube_only.toml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/geometry_validation/body_matrix"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "bodies": ["earth", "mars", "moon"],
        "seeds": [1, 2, 3],
        "scope": "Full host/network/section generation with shared Earth-scenario controls and body preset overrides. One local junction mesh per run; not a whole-mesh or geological validation of nine networks.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for body in plan["bodies"]:
        for seed in plan["seeds"]:
            start = time.monotonic()
            folder = out / f"{body}_seed{seed}"
            folder.mkdir(exist_ok=True)
            project = for_seed(load_project_config(args.config, world_body=body), seed)
            host, network, sections = generate_sections(project)
            (folder / "resolved_config.json").write_text(
                json.dumps(asdict(project), default=str, indent=2) + "\n"
            )
            export_network_artifact(network, folder / "network.json")
            export_section_artifact(sections, folder / "sections.npz")
            samples = [s for f in sections.segment_fields for s in f.samples]
            measures = [contour_morphometry(s.profile_points) for s in samples]
            network_report = network_metrics(network, host)
            profile_checks = {
                "finite_positive_dimensions": all(
                    np.isfinite([m["width_m"], m["height_m"]]).all()
                    and min(m["width_m"], m["height_m"]) > 0
                    for m in measures
                ),
                "no_self_intersecting_profiles": all(
                    self_intersection_count(s.profile_points) == 0 for s in samples
                ),
                "connected_network": network_report["connected_component_count"] == 1,
                "all_nodes_source_reachable": network_report["source_unreachable_node_count"] == 0,
                "all_entries_have_exit_path": network_report["entries_without_exit_path_count"]
                == 0,
                "flux_conservation": network_report["max_relative_flux_conservation_residual"]
                <= 1e-6,
                "temperature_monotonicity": network_report[
                    "temperature_monotonicity_violation_count"
                ]
                == 0,
                "lava_age_monotonicity": network_report["lava_age_monotonicity_violation_count"]
                == 0,
            }
            # Strongest split/merge influence, deterministic ties in original
            # sample order; deliberately excludes aesthetic hand-picking.
            selected = max(
                range(len(samples)),
                key=lambda i: max(
                    [j.weight for j in samples[i].junction_influences if j.kind != "chamber"]
                    or [0.0]
                ),
            )
            s = samples[selected]
            height = measures[selected]["height_m"]
            voxel = float(np.clip(height / 10.0, 0.05, 0.5))
            c = replace(
                project.geometry,
                voxel_size=voxel,
                storage_mode="dense",
                target_samples_across_passage=project.geometry.characteristic_passage_width_m
                / voxel,
            )
            extent = (
                0.65 * s.tube_width * np.abs(s.normal)
                + min(10.0, s.tube_width) * np.abs(s.tangent)
                + np.array([2.0, 2.0, 0.75 * height + 2.0])
            )
            mesh_result = {
                "selected_sample_index": selected,
                "selection": "maximum non-chamber junction influence",
                "voxel_size_m": voxel,
            }
            try:
                mesh_result.update(mesh_junction_region(network, sections, c, s, extent, folder))
            except (RuntimeError, ValueError) as error:
                mesh_result.update({"checks_passed": False, "error": str(error)})
            row = {
                "body": body,
                "seed": seed,
                "profile_checks": profile_checks,
                "network": network_report,
                "section_count": len(samples),
                "profile_height_m": {
                    "min": min(m["height_m"] for m in measures),
                    "median": float(np.median([m["height_m"] for m in measures])),
                    "max": max(m["height_m"] for m in measures),
                },
                "profile_width_m": {
                    "min": min(m["width_m"] for m in measures),
                    "median": float(np.median([m["width_m"] for m in measures])),
                    "max": max(m["width_m"] for m in measures),
                },
                "input_stability_failures": sum(s.collapse_required for s in samples),
                "configured_resolution": section_resolution_report(
                    sections, project.geometry.voxel_size
                ),
                "local_mesh": mesh_result,
                "elapsed_seconds": time.monotonic() - start,
            }
            row["passed"] = all(profile_checks.values()) and mesh_result["checks_passed"]
            rows.append(row)
            (folder / "checks.json").write_text(json.dumps(row, indent=2) + "\n")
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "plan": plan,
                        "complete": len(rows) == 9,
                        "all_passed": len(rows) == 9 and all(r["passed"] for r in rows),
                        "results": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"{body} seed {seed}: {row['passed']}; {len(samples)} profiles; local mesh {mesh_result}",
                flush=True,
            )
    (out / "collapse_counterfactual.json").write_text(
        json.dumps(collapse_counterfactual(), indent=2) + "\n"
    )
    (out / "clip_audit.json").write_text(
        json.dumps(
            {
                "scope": "Cuts are accepted only beyond a two-voxel margin from the artificial box boundary.",
                "results": [
                    {
                        "run": f"{r['body']}_seed{r['seed']}",
                        "contour_clear_of_artificial_caps": r["local_mesh"].get(
                            "contour_clear_of_artificial_caps", False
                        ),
                        "minimum_contour_distance_to_clip_m": r["local_mesh"].get(
                            "minimum_contour_distance_to_clip_m"
                        ),
                        "box_expansion_count": r["local_mesh"].get("box_expansion_count"),
                        "components": r["local_mesh"].get("components", []),
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
