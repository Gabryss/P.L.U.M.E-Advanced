#!/usr/bin/env python3
"""Generate the saved Stage-B/C artifacts used by the Manim project."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from plume_advanced.config import load_project_config, project_config_manifest
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    export_section_artifact,
    host_semantic_hash,
)
from plume_advanced.evaluation.provenance import semantic_hash, sha256_file
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionFieldGenerator

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "project.toml")
    parser.add_argument("--body", choices=("earth", "mars", "moon"), default="earth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "media" / "video" / "assets" / "hero",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Replace an existing prepared-artifact directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    existing = [path for path in output_dir.glob("*") if path.is_file()]
    if existing and not args.force_overwrite:
        names = ", ".join(path.name for path in existing[:4])
        raise FileExistsError(
            f"Refusing to replace prepared Manim assets in {output_dir} ({names}). "
            "Pass --force-overwrite after reviewing the directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    config = load_project_config(config_path, world_body=args.body)
    host = HostFieldGenerator(config.host_field).generate()
    network = CaveNetworkGenerator(config.network).generate(host)
    sections = SectionFieldGenerator(config.section_field).generate(network)
    network_path = export_network_artifact(network, output_dir / "stage_b_network.json")
    host_npz_path, host_json_path = _export_host_fields(host, output_dir)
    section_npz_path, section_json_path = export_section_artifact(
        sections,
        output_dir / "stage_c_sections",
    )
    manifest = {
        "schema": "plume.manim-inputs.v1",
        "body": args.body,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": semantic_hash(project_config_manifest(config)),
        "git_commit": _git_commit(),
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                host_npz_path,
                host_json_path,
                network_path,
                section_npz_path,
                section_json_path,
            )
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared Manim artifacts in {output_dir}")
    return 0


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _export_host_fields(host, output_dir: Path) -> tuple[Path, Path]:
    field_names = (
        "elevation",
        "cover_thickness",
        "fracture_intensity",
        "roof_stability",
        "routing_cost",
    )
    npz_path = output_dir / "stage_a_host_fields.npz"
    json_path = output_dir / "stage_a_host_fields.json"
    np.savez_compressed(
        npz_path,
        x_coords_m=host.x_coords,
        y_coords_m=host.y_coords,
        **{name: np.asarray(getattr(host, name), dtype=float) for name in field_names},
    )
    metadata = {
        "schema": "plume.manim-host-fields.v1",
        "coordinate_system": {"length_unit": "metre", "up_axis": "Z"},
        "field_names": list(field_names),
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "semantic_sha256": host_semantic_hash(host),
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path


if __name__ == "__main__":
    raise SystemExit(main())
