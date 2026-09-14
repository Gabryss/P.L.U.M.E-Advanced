#!/usr/bin/env python3
"""Complete a trusted local inspection checkpoint and validate its portable export.

Input pickles must be produced locally; never open checkpoints from third parties.
The recorded inputs and per-stage source identities distinguish generation from
later packaging. This script does not change the stored mesh or choose new seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plume-matplotlib")
import numpy as np
import trimesh

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import export_geometry_report
from plume_advanced.evaluation.reliability import write_json
from plume_advanced.exporters import export_target_asset
from plume_advanced.identity import package_source_hash, sha256_file
from plume_advanced.progress import progress_scope
from plume_advanced.run_manifest import write_run_manifest
from plume_advanced.stages.events import GeologicalEventGenerator
from plume_advanced.stages.floor_map import FloorMapGenerator, export_floor_atlas
from plume_advanced.stages.geometry import GeometryGenerator
from plume_advanced.stages.surface_topology import check_closed_surface_topology, component_count
from plume_advanced.validation import PortableAssetValidator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--resume-neutral",
        action="store_true",
        help="Reuse a verified neutral export for a material-only revision",
    )
    parser.add_argument(
        "--texture-directory",
        type=Path,
        help="Reuse prepared 4K base-color, normal, and grayscale roughness PNGs",
    )
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root / "export_blender").exists() and not args.resume_neutral:
        raise FileExistsError("Choose an unexported case; existing assets are preserved")
    started = time.monotonic()
    source_hash = package_source_hash()
    with (root / "inputs.pickle").open("rb") as file:
        project, host, network, sections = pickle.load(file)
    with (root / "base.pickle").open("rb") as file:
        base = pickle.load(file)
    # Both source modes in this campaign use the same portable 4K material.
    # The maintained single-source geometry preset intentionally has no maps.
    material = load_project_config(
        Path(__file__).resolve().parents[1] / "config/earth_short_interconnected_full.toml"
    ).geometry
    material_fields = (
        "cave_diffuse_texture",
        "cave_normal_texture",
        "cave_roughness_texture",
        "embedded_texture_max_size",
        "cave_texture_scale_m",
        "cave_normal_scale",
    )
    geometry_config = replace(
        project.geometry, **{k: getattr(material, k) for k in material_fields}
    )
    texture_directory = args.texture_directory or root.parent.parent / "shared_texture_sources"
    if args.texture_directory and not texture_directory.is_dir():
        raise FileNotFoundError(texture_directory)
    if texture_directory.is_dir():
        for name in ("cave_base_color.png", "cave_normal.png", "cave_roughness.png"):
            if not (texture_directory / name).is_file():
                raise FileNotFoundError(texture_directory / name)
        geometry_config = replace(
            geometry_config,
            cave_diffuse_texture=str((texture_directory / "cave_base_color.png").resolve()),
            cave_normal_texture=str((texture_directory / "cave_normal.png").resolve()),
            cave_roughness_texture=str((texture_directory / "cave_roughness.png").resolve()),
        )
    project = replace(project, geometry=geometry_config)
    base = replace(base, config=geometry_config)
    if project.events.enabled or project.events.include_rock_props:
        raise ValueError("This inspection campaign requires events and rocks disabled")
    last = 0.0
    trace = (root / "completion_progress.jsonl").open("w")

    def progress(step, current=0, total=1, detail=""):
        nonlocal last
        trace.write(json.dumps(dict(step=step, current=current, total=total, detail=detail)) + "\n")
        trace.flush()
        if time.monotonic() - last > 10 or current == total:
            print(step, current, total, detail, flush=True)
            last = time.monotonic()

    with progress_scope(progress):
        floors = FloorMapGenerator(project.floor_map)
        progress("Base floor atlas")
        atlas = floors.generate(network, sections, base)
        events = GeologicalEventGenerator(project.events).generate(
            sections, base, atlas, progress=progress
        )
        geometry = GeometryGenerator(project.geometry).finalize(base, events, progress=progress)
        expected = len(network.segments) - len(network.nodes) + 1
        raw_report = check_closed_surface_topology(
            geometry.assembled_vertices,
            geometry.assembled_faces,
            geometry.component_count,
            expected,
        )
        progress("Final floor atlas")
        floor = floors.revalidate(network, sections, geometry, atlas, events)
        export_floor_atlas(floor, root / "stage_c_floor_map.npz")
        export_geometry_report(geometry, root / "stage_d_geometry_report.json")
        progress("Export")
        # This is the full, unsimplified visual mesh; no collision duplication.
        if args.resume_neutral:
            from plume_advanced.exporters.materials import apply_cave_material
            from plume_advanced.exporters.projected_materials import write_projected_material_bundle
            from plume_advanced.exporters.targets import ExportResult
            from plume_advanced.stages.geometry_export import _write_geometry_manifest
            from plume_advanced.validation import GlbAsset

            folder = root / "export_blender"
            source = next(folder.glob("*.glb"))
            recorded = json.loads((root / "run_manifest.json").read_text())
            assert any(
                r["path"] == str(source.relative_to(root)) and r["sha256"] == sha256_file(source)
                for r in recorded["outputs"]
            )
            if GlbAsset(source).document.get("images"):
                raise ValueError("Resume-neutral only accepts an original untextured export")
            old_metadata = json.loads(source.with_suffix(".manifest.json").read_text())
            diagnostic = root / "export_neutral_diagnostic"
            if diagnostic.exists():
                raise FileExistsError(diagnostic)
            folder.rename(diagnostic)
            source = diagnostic / source.name
            write_json(diagnostic / "run_manifest_before_material.json", recorded)
            folder.mkdir()
            target = folder / source.name
            revision = apply_cave_material(
                source,
                target,
                geometry_config,
                source_tile_size_m=old_metadata["cave"]["material"]["uv_scale_m"],
            )
            write_json(folder / "material_revision.json", revision)
            _write_geometry_manifest(
                geometry,
                target.with_suffix(".manifest.json"),
                displacement=dict(
                    baked=False,
                    scale_m=0.0,
                    midlevel=0.5,
                    minimum_offset_m=0.0,
                    maximum_offset_m=0.0,
                    mean_offset_m=0.0,
                ),
            )
            write_projected_material_bundle(
                target,
                folder / "continuous_material",
                tile_size_m=geometry_config.cave_texture_scale_m,
                normal_strength=geometry_config.cave_normal_scale,
            )
            export = ExportResult(
                "blender", target, tuple(p for p in folder.rglob("*") if p.is_file())
            )
        else:
            export = export_target_asset(
                geometry,
                replace(
                    project.export,
                    target="blender",
                    file_format="glb",
                    generate_collision=False,
                    max_visual_triangles=3_000_000,
                    max_asset_bytes=1_500_000_000,
                ),
                root / "export_blender",
            )
        write_json(
            root / "stage_sources.json",
            dict(
                geometry=json.loads((root / "geometry_check.json").read_text())["source"],
                completion=source_hash,
                input_checkpoints={
                    name: sha256_file(root / name) for name in ("inputs.pickle", "base.pickle")
                },
                note="The cave was generated in the mesh pass; floor, event-disabled finalization, and export complete it here.",
            ),
        )
        outputs = [*export.files, *root.glob("stage_*.json"), *root.glob("stage_*.npz")]
        manifest = write_run_manifest(
            project,
            root / "run_manifest.json",
            outputs=outputs,
            elapsed_seconds=time.monotonic() - started,
            source_root=Path(__file__).resolve().parents[1],
            inputs=[root / "inputs.pickle", root / "base.pickle"],
        )
        checks = PortableAssetValidator(
            export.primary_asset, run_manifest_path=manifest, material_profile="textured"
        ).validate()
        write_json(root / "portable_checks.json", [asdict(c) for c in checks])
        failed = [c.name for c in checks if not c.passed]
        if failed:
            raise ValueError(f"Portable export failed: {failed}")
        progress("Exported surface connectivity")
        scene = trimesh.load(export.primary_asset, force="scene", process=False)
        if len(scene.geometry) != 1 or geometry.event_meshes:
            raise ValueError("Expected exactly one cave and no rocks")
        mesh = next(iter(scene.geometry.values()))
        # Rejoin only identical exported positions, without rounding away
        # narrow real triangles near the coordinate origin.
        positions, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        mesh = trimesh.Trimesh(positions, inverse[mesh.faces], process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or np.any(mesh.area_faces <= 0):
            raise ValueError("Exported mesh is not a closed oriented nondegenerate surface")
        exported_report = check_closed_surface_topology(
            mesh.vertices, mesh.faces, component_count(mesh.faces), expected
        )
        write_json(
            root / "mesh_topology_check.json",
            dict(passed=True, raw=raw_report, exported=exported_report),
        )
        del scene, mesh
        progress("Stage figures")
        from plume_advanced.visualization.floor_map import FloorMapPlotter
        from plume_advanced.visualization.host_field import HostFieldPlotter
        from plume_advanced.visualization.inspection import (
            InspectionSectionPlotter,
            SavedGeometryPlotter,
        )
        from plume_advanced.visualization.network import CaveNetworkPlotter
        from plume_advanced.visualization.network_topology import render_topology_footprint

        figures = [
            ("stage_a_host_field.png", lambda path: HostFieldPlotter().render(host, path)),
            (
                "stage_b_cave_network.png",
                lambda path: CaveNetworkPlotter().render(host, network, path),
            ),
            (
                "stage_c_section_field.png",
                lambda path: InspectionSectionPlotter().render(network, sections, path),
            ),
            (
                "stage_c_floor_map.png",
                lambda path: FloorMapPlotter().render(floor, path, events, network),
            ),
            (
                "stage_d_geometry.png",
                lambda path: SavedGeometryPlotter(geometry).render_debug(network, geometry, path),
            ),
        ]
        if network.config.systems.count > 1:
            from plume_advanced.evaluation.visualization.emplacement import (
                render_emplacement_phase_activity,
            )
            from plume_advanced.visualization.interconnected import render_interconnected

            figures.extend(
                [
                    (
                        "stage_bc_interconnected.png",
                        lambda path: render_interconnected(host, network, sections, path),
                    ),
                    (
                        "stage_b_emplacement_history.png",
                        lambda path: render_emplacement_phase_activity(network, path),
                    ),
                ]
            )
        else:
            figures.append(
                (
                    "stage_bc_topology_footprint.png",
                    lambda path: render_topology_footprint(network, sections, path),
                )
            )
        for i, (name, render) in enumerate(figures, 1):
            render(root / name)
            progress("Stage figures", i, len(figures), name)
        write_json(
            root / "stage_figures_manifest.json",
            dict(
                figures=[dict(file=name, sha256=sha256_file(root / name)) for name, _ in figures],
                events="disabled; no event-placement illustration",
            ),
        )
        write_json(
            root / "completion.json",
            dict(
                passed=True,
                source_unchanged=source_hash == package_source_hash(),
                elapsed_seconds=time.monotonic() - started,
                portable_checks=len(checks),
                triangles=len(geometry.assembled_faces),
                floor_cells=len(floor.cells),
                relief_scale=geometry.effective_surface_relief_scale,
                surface_attempts=[dict(r) for r in geometry.surface_quality_records],
                glb_sha256=sha256_file(export.primary_asset),
            ),
        )
    trace.close()


if __name__ == "__main__":
    main()
