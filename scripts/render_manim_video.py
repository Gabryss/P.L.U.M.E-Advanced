#!/usr/bin/env python3
"""Render a PLUME Manim scene from explicitly prepared artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from plume_advanced.evaluation.provenance import sha256_file
from plume_advanced.media.manim_data import load_manim_prototype_data

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "paper" / "media" / "video"
STAGE_SCENES = (
    "StageAHostField",
    "StageBSemanticFlow",
    "StageCAdaptiveSections",
    "StageDMarchingCubes",
)
AVAILABLE_SCENES = (*STAGE_SCENES, "GraphToGeometryPrototype")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=VIDEO_ROOT / "assets" / "hero",
        help="Directory containing stage_b_network.json and stage_c_sections.json.",
    )
    parser.add_argument(
        "--scene",
        default="GraphToGeometryPrototype",
        choices=AVAILABLE_SCENES,
        help="Manim scene class to render.",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Render Stage A, B, C, and D as separate video assets.",
    )
    parser.add_argument("--hero-segment", type=int, default=None)
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "production"),
        default="medium",
    )
    parser.add_argument("--format", choices=("mp4", "webm", "mov"), default="mp4")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if importlib.util.find_spec("manim") is None:
        raise RuntimeError(
            "Manim is not installed. Install the project video extra with "
            "`.venv/bin/pip install -e '.[video]'`."
        )
    assets_dir = args.assets_dir.resolve()
    host_path = assets_dir / "stage_a_host_fields.json"
    network_path = assets_dir / "stage_b_network.json"
    section_path = assets_dir / "stage_c_sections.json"
    for path in (host_path, network_path, section_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/prepare_manim_assets.py first."
            )
    data = load_manim_prototype_data(
        network_path,
        section_path,
        hero_segment_id=args.hero_segment,
    )
    selected_hero_segment = data.hero_segment_id
    quality_flags = {
        "low": "-ql",
        "medium": "-qm",
        "high": "-qh",
        "production": "-qk",
    }
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/plume-advanced-matplotlib")
    environment["PLUME_VIDEO_HOST_ARTIFACT"] = str(host_path)
    environment["PLUME_VIDEO_NETWORK_ARTIFACT"] = str(network_path)
    environment["PLUME_VIDEO_SECTION_ARTIFACT"] = str(section_path)
    environment["PLUME_VIDEO_HERO_SEGMENT"] = str(selected_hero_segment)
    scenes = STAGE_SCENES if args.all_stages else (args.scene,)
    for scene in scenes:
        command = (
            sys.executable,
            "-m",
            "manim",
            "--config_file",
            str(VIDEO_ROOT / "manim.cfg"),
            quality_flags[args.quality],
            "--fps",
            "30",
            "--format",
            args.format,
            str(VIDEO_ROOT / "scenes" / "prototype.py"),
            scene,
        )
        result = subprocess.run(command, cwd=VIDEO_ROOT, env=environment, check=False)
        if result.returncode != 0:
            return result.returncode
        scene_segment_id = (
            data.longest_section_segment_id
            if scene in {"StageCAdaptiveSections", "StageDMarchingCubes"}
            else selected_hero_segment
            if scene == "GraphToGeometryPrototype"
            else None
        )
        _write_render_provenance(
            args,
            command,
            scene,
            host_path,
            network_path,
            section_path,
            scene_segment_id,
        )
    return 0


def _write_render_provenance(
    args: argparse.Namespace,
    command: tuple[str, ...],
    scene: str,
    host_path: Path,
    network_path: Path,
    section_path: Path,
    selected_segment_id: int | None,
) -> None:
    video_path = VIDEO_ROOT / "renders" / "video" / f"{scene}.{args.format}"
    if not video_path.is_file():
        raise FileNotFoundError(f"Manim reported success but did not create {video_path}")
    payload = {
        "schema": "plume.manim-render.v1",
        "scene": scene,
        "quality": args.quality,
        "frame_rate": 30,
        "format": args.format,
        "selected_segment_id": selected_segment_id,
        "manim_version": version("manim"),
        "command": list(command),
        "inputs": {
            str(host_path): sha256_file(host_path),
            str(host_path.with_name("stage_a_host_fields.npz")): sha256_file(
                host_path.with_name("stage_a_host_fields.npz")
            ),
            str(network_path): sha256_file(network_path),
            str(section_path): sha256_file(section_path),
            str(section_path.with_name("stage_c_sections.npz")): sha256_file(
                section_path.with_name("stage_c_sections.npz")
            ),
        },
        "output": {
            "path": str(video_path),
            "sha256": sha256_file(video_path),
        },
    }
    video_path.with_suffix(".provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
