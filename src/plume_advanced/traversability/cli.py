"""Export vector and raster maps from a completed run without remeshing the cave."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from plume_advanced.asset_paths import find_export_asset
from plume_advanced.config import load_project_config
from plume_advanced.exporters.atomic import atomic_output_directory
from plume_advanced.identity import sha256_file
from plume_advanced.progress import TerminalProgress
from plume_advanced.validation import GlbAsset

from .config import TraversabilityConfig
from .export import export_traversability
from .request import from_paths


def saved_inputs(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Traversability requires a completed, exported run")
    known = {row["path"]: row["sha256"] for row in manifest["outputs"]}

    def verify(path):
        relative = path.resolve().relative_to(root).as_posix()
        if relative not in known or sha256_file(path) != known[relative]:
            raise ValueError(
                f"Run provenance mismatch for {relative}; use the matching unmodified export"
            )

    selected = None
    for target in ("blender", "unity", "ue5", "neutral"):
        try:
            selected = find_export_asset(root, target=target)
            break
        except FileNotFoundError:
            pass
    if selected is None:
        raise ValueError(
            "Saved-run mapping needs a neutral, Blender, Unity or Unreal GLB package. New generations map every export target directly."
        )
    paths = [
        root / "stage_b_network.json",
        root / "stage_c_sections.npz",
        root / "stage_c_sections.json",
        selected,
    ]
    for path in paths:
        verify(path)
    network = json.loads(paths[0].read_text())
    metadata = json.loads(paths[2].read_text())
    with np.load(paths[1], allow_pickle=False) as sections:
        section_paths = {
            s["segment_id"]: np.column_stack(
                (
                    sections["center_xyz_m"][sections["segment_id"] == s["segment_id"]],
                    sections["width_m"][sections["segment_id"] == s["segment_id"]],
                    sections["height_m"][sections["segment_id"] == s["segment_id"]],
                )
            )
            for s in network["segments"]
        }
    glb = GlbAsset(selected)
    obstacles = []
    cave = None
    transforms: dict[str, list[float]] = dict(
        translation=[0, 0, 0],
        rotation=[0, 0, 0, 1],
        scale=[1, 1, 1],
        matrix=np.eye(4).ravel().tolist(),
    )
    for node in glb.document["nodes"]:
        for key, expected in transforms.items():
            if not np.allclose(node.get(key, expected), expected, atol=1e-12, rtol=0):
                raise ValueError("Saved PLUME map inputs must retain canonical node transforms")
        if "mesh" not in node:
            continue
        mesh = glb.document["meshes"][node["mesh"]]
        for primitive in mesh["primitives"]:
            v = glb.accessor(primitive["attributes"]["POSITION"]).astype(float)[:, [0, 2, 1]] * [
                1,
                -1,
                1,
            ]
            f = glb.accessor(primitive["indices"]).reshape(-1, 3)
            if node.get("name") == "cave_wall":
                if cave is not None:
                    raise ValueError("Expected one canonical cave primitive")
                cave = (v, f)
            elif node.get("name", "").startswith("event_"):
                obstacles.append((v, f))
    if cave is None:
        raise ValueError("No cave surface in saved GLB")
    collision = selected.with_name(selected.stem + "_collision.obj")
    kind = "visual"
    if collision.exists():
        verify(collision)
        mesh = trimesh.load_mesh(collision, process=False)
        cave = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        kind = "collision"
        paths.append(collision)
    elif manifest["resolved_config"]["export"]["generate_collision"]:
        raise ValueError("Declared collision surface is missing; refusing a silent visual fallback")
    provenance = dict(
        run_manifest_sha256=sha256_file(root / "run_manifest.json"),
        inputs={p.relative_to(root).as_posix(): sha256_file(p) for p in paths},
        network_semantic_sha256=network["semantic_sha256"],
        section_semantic_sha256=metadata["semantic_sha256"],
    )
    return manifest, network, section_paths, cave, obstacles, kind, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config", type=Path, help="Read only [traversability] settings from this project recipe"
    )
    parser.add_argument("--resolution", type=float, help="Override map cell size in metres")
    args = parser.parse_args(argv)
    output = args.output or args.source / "traversability"
    # Atomic publication must never replace the input run or one of its packages.
    if (
        args.source.resolve().is_relative_to(output.resolve())
        or output.resolve().is_relative_to(args.source.resolve() / "export_all")
        or any(
            output.resolve().is_relative_to(p.resolve())
            for p in args.source.glob("export_*")
            if p.is_dir()
        )
    ):
        parser.error("Use a separate map directory, not the source run or an export package")
    progress = TerminalProgress(total_stages=1)
    try:
        progress.start("Traversability maps", "verify exported geometry and layer provenance")
        manifest, network, paths, cave, obstacles, kind, provenance = saved_inputs(args.source)
        config = (
            load_project_config(args.config).traversability
            if args.config
            else TraversabilityConfig(**manifest["resolved_config"].get("traversability", {}))
        )
        if args.resolution is not None:
            config = replace(config, resolution_m=args.resolution)
        if not config.enabled:
            raise ValueError("Traversability is disabled in the selected configuration")
        controls = network.get("layers", {}).get("controls", {})
        request = from_paths(
            config,
            [
                (s["segment_id"], s["source_node_id"], s["target_node_id"], s["metadata"])
                for s in network["segments"]
            ],
            paths,
            layered=controls.get("enabled", False),
            layer_count=controls["count"] if controls.get("enabled") else 1,
            provenance=provenance,
        )
        with atomic_output_directory(output) as staging:
            export_traversability(
                request,
                cave[0],
                cave[1],
                staging,
                obstacles=obstacles,
                surface_kind=kind,
                source=str(args.source.resolve()),
            )
        progress.finish(f"vector network and {len(request.charts)} raster/vector chart sets; {output.resolve()}")
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    finally:
        TerminalProgress.close_active()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
