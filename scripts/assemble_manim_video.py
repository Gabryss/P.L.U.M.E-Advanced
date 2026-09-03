#!/usr/bin/env python3
"""Join the current overview and A–F videos without re-encoding their frames."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from render_manim_video import STAGE_SCENES, VIDEO_ROOT

from plume_advanced.evaluation.provenance import sha256_file

CHAPTER_TITLES = (
    "Pipeline overview",
    "Stage A - Host field",
    "Stage B - Semantic cave network",
    "Stage C - Adaptive cross-sections",
    "Stage D - Marching cubes",
    "Stage E - Grounded cave rocks",
    "Stage F - Surface preparation and export",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_ROOT / "renders" / "video")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()
    inputs = [(args.video_dir / f"{scene}.mp4").resolve() for scene in STAGE_SCENES]
    output = (args.output or args.video_dir / "FullPipeline.mp4").resolve()
    if output in inputs:
        raise ValueError("The combined output must not replace an individual stage.")
    if output.suffix.lower() != ".mp4":
        raise ValueError("The combined output must be an MP4 file.")
    provenance = output.with_suffix(".provenance.json")
    if not args.force_overwrite and (output.exists() or provenance.exists()):
        raise FileExistsError(f"Output already exists: {output}; use --force-overwrite.")

    chapters = []
    elapsed = Fraction(0)
    expected = None
    for path, title in zip(inputs, CHAPTER_TITLES, strict=True):
        probe = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                text=True,
            )
        )
        streams = probe["streams"]
        if len(streams) != 1 or streams[0]["codec_type"] != "video":
            raise ValueError(f"Expected one silent video stream: {path}")
        stream = streams[0]
        signature = tuple(
            stream[key]
            for key in ("codec_name", "width", "height", "pix_fmt", "r_frame_rate", "time_base")
        )
        if expected is not None and signature != expected:
            raise ValueError(f"Video formats differ; re-render all stages at one quality: {path}")
        expected = signature
        duration = Fraction(stream["duration_ts"]) * Fraction(stream["time_base"])
        chapters.append(
            {
                "title": title,
                "start_ms": round(elapsed * 1000),
                "end_ms": round((elapsed + duration) * 1000),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
        elapsed += duration

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plume-video-") as temporary:
        folder = Path(temporary)
        # Safe local aliases keep arbitrary source-path characters out of the
        # concat demuxer's quoted-path syntax.
        for index, path in enumerate(inputs):
            (folder / f"clip{index}.mp4").symlink_to(path)
        playlist = folder / "inputs.ffconcat"
        playlist.write_text(
            "ffconcat version 1.0\n"
            + "".join(f"file clip{index}.mp4\n" for index in range(len(inputs))),
            encoding="utf-8",
        )
        metadata = folder / "chapters.txt"
        metadata.write_text(
            ";FFMETADATA1\ntitle=PLUME - Complete pipeline\n"
            + "".join(
                "[CHAPTER]\nTIMEBASE=1/1000\n"
                f"START={chapter['start_ms']}\nEND={chapter['end_ms']}\n"
                f"title={chapter['title']}\n"
                for chapter in chapters
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if args.force_overwrite else "-n",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-f",
                "ffmetadata",
                "-i",
                str(metadata),
                "-map",
                "0:v:0",
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    provenance.write_text(
        json.dumps(
            {
                "schema": "plume.manim-compilation.v1",
                "assembly": "stream copy; overview then stages A-F; no retiming",
                "chapters": chapters,
                "duration_seconds": float(elapsed),
                "output": {"path": str(output), "sha256": sha256_file(output)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created {output} ({float(elapsed):.1f} seconds, {len(chapters)} chapters)")


if __name__ == "__main__":
    main()
