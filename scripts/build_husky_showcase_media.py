#!/usr/bin/env python3
"""Compose a source-backed Husky route inset and a repository-ready GIF.

The paper plate uses one Isaac RTX visual reconstruction. The GIF uses a series
of Isaac RTX reconstructions at logged Gazebo poses. The plan envelope comes
from the generated network's local centerline and widths. This is visual
reconstruction, not an Isaac dynamics or navigation run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "outputs/showcase_gallery/husky_visualization"
ROBOTICS = ROOT / "outputs/showcase_robotics/husky_showcase_rockfall"
NETWORK = ROOT / "outputs/showcase_optimized/stage_b_network.json"
SOURCE = VIS / "husky_native_final.png"
FIGURE = ROOT / "paper/figures/husky_showcase_render_map.png"
GIF = ROOT / "docs/assets/husky_showcase_trajectory.gif"
PROVENANCE = ROOT / "paper/figures/husky_showcase_media_provenance.json"

WHITE = (237, 242, 240, 255)
MUTED = (174, 191, 194, 255)
CYAN = (95, 221, 218, 255)
AMBER = (241, 190, 101, 255)
GREEN = (128, 198, 140, 255)
YELLOW = (250, 217, 87, 255)
INK = (12, 23, 29, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + family, size)


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def passage_envelope(network: dict, start_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def distance(segment: dict) -> float:
        xy = np.asarray([(p["x"], p["y"]) for p in segment["centerline"]])
        return float(np.min(np.linalg.norm(xy - start_xy, axis=1)))

    selected = min(network["segments"], key=distance)
    points = selected["centerline"]
    arc = np.asarray([p["arc_length"] for p in points])
    dense = np.linspace(arc[0], arc[-1], 600)
    center = np.column_stack([
        np.interp(dense, arc, [p["x"] for p in points]),
        np.interp(dense, arc, [p["y"] for p in points]),
    ])
    widths = np.interp(dense, arc, [p["width"] for p in points])
    tangent = np.gradient(center, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left = center + normal * widths[:, None] / 2
    right = center - normal * widths[:, None] / 2
    footprint = np.concatenate([left, right[::-1]])
    return footprint, center


def pose_at(trace: list[dict], seconds: float) -> dict:
    return min(trace, key=lambda row: abs(row["sim_time_s"] - seconds))


def draw_robot(draw: ImageDraw.ImageDraw, xy: tuple[float, float], yaw: float,
               pixels_per_m: float) -> None:
    # Plan-view footprint follows the published Husky body dimensions.
    fx, fy = math.cos(yaw), -math.sin(yaw)
    rx, ry = -fy, fx
    cx, cy = xy
    length, width = .99 * pixels_per_m, .67 * pixels_per_m
    corners = []
    for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        corners.append((cx + a * fx * length / 2 + b * rx * width / 2,
                        cy + a * fy * length / 2 + b * ry * width / 2))
    draw.polygon(corners, fill=YELLOW, outline=INK, width=3)
    nose = (cx + fx * length * .35, cy + fy * length * .35)
    draw.ellipse((nose[0]-4, nose[1]-4, nose[0]+4, nose[1]+4), fill=INK)


def make_map(width: int, height: int, trace: list[dict], route: dict,
             footprint: np.ndarray, centerline: np.ndarray, seconds: float,
             show_full_path: bool) -> Image.Image:
    image = Image.new("RGBA", (width, height), (13, 25, 32, 255))
    draw = ImageDraw.Draw(image)
    bounds = (1.9, -27.7, 8.1, -22.7)
    x0, y0, x1, y1 = bounds
    scale = min((width - 62) / (x1 - x0), (height - 38) / (y1 - y0))
    ox = (width - scale * (x1 - x0)) / 2
    oy = (height - scale * (y1 - y0)) / 2

    def project(x: float, y: float) -> tuple[float, float]:
        return ox + (x - x0) * scale, height - oy - (y - y0) * scale

    for x in range(2, 9):
        u, _ = project(x, y0)
        draw.line((u, 0, u, height), fill=(37, 58, 63, 255), width=1)
    for y in range(-27, -22):
        _, v = project(x0, y)
        draw.line((0, v, width, v), fill=(37, 58, 63, 255), width=1)

    polygon = [project(*row) for row in footprint]
    draw.polygon(polygon, fill=(47, 70, 76, 255))
    draw.line(polygon + polygon[:1], fill=(107, 145, 151, 255), width=3)
    middle = [project(*row) for row in centerline]
    draw.line(middle, fill=(81, 116, 122, 255), width=2)

    start = route["start_xyz_m"][:2]
    goal = route["waypoints_xy_m"][-1]
    path = [project(*row["pose"][:2]) for row in trace]
    if show_full_path:
        draw.line(path, fill=(85, 160, 165, 255), width=6, joint="curve")
    else:
        draw.line(path, fill=(71, 107, 112, 255), width=3, joint="curve")
    completed = [project(*row["pose"][:2]) for row in trace if row["sim_time_s"] <= seconds]
    if len(completed) > 1:
        draw.line(completed, fill=CYAN, width=7, joint="curve")

    gx, gy = project(*goal)
    r = route["goal_tolerance_m"] * scale
    for angle in range(0, 360, 30):
        draw.arc((gx-r, gy-r, gx+r, gy+r), angle, angle+17, fill=AMBER, width=3)
    draw.ellipse((gx-5, gy-5, gx+5, gy+5), fill=AMBER)
    sx, sy = project(*start)
    draw.rectangle((sx-7, sy-7, sx+7, sy+7), fill=GREEN, outline=INK, width=2)

    pose = pose_at(trace, seconds)
    draw_robot(draw, project(*pose["pose"][:2]), pose["yaw"], scale)
    # The scale bar is inside the map, so it remains meaningful after resizing.
    ax, ay = 26, height - 24
    draw.line((ax, ay, ax + scale, ay), fill=WHITE, width=3)
    draw.line((ax, ay-5, ax, ay+5), fill=WHITE, width=2)
    draw.line((ax+scale, ay-5, ax+scale, ay+5), fill=WHITE, width=2)
    draw.text((ax + scale + 10, ay-12), "1 m", font=font(19), fill=WHITE)
    return image


def compose(base: Image.Image, trace: list[dict], route: dict,
            footprint: np.ndarray, centerline: np.ndarray, seconds: float,
            *, for_gif: bool, moving_scene: bool = False) -> Image.Image:
    image = base.copy().convert("RGBA")
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((58, 56, 900, 139), radius=18,
                           fill=(8, 19, 26, 224), outline=(100, 130, 133, 180), width=2)
    draw.text((87, 76), "HUSKY A200  |  LAVA-TUBE SHOWCASE",
              font=font(29, True), fill=WHITE)

    card = (1270, 60, 1870, 830)
    draw.rounded_rectangle(card, radius=24, fill=(8, 19, 26, 233),
                           outline=(102, 135, 139, 205), width=2)
    draw.text((1310, 96), "LOCAL PLAN VIEW", font=font(30, True), fill=WHITE)
    draw.text((1310, 140), "Generated passage + recorded Gazebo path",
              font=font(18), fill=MUTED)
    map_rect = (1300, 182, 1840, 681)
    draw.rounded_rectangle(map_rect, radius=11, fill=(13, 25, 32, 255),
                           outline=(81, 113, 118, 255), width=2)
    image = Image.alpha_composite(image, layer)
    map_image = make_map(map_rect[2] - map_rect[0] - 8,
                         map_rect[3] - map_rect[1] - 8,
                         trace, route, footprint, centerline, seconds,
                         show_full_path=not for_gif)
    image.alpha_composite(map_image, (map_rect[0] + 4, map_rect[1] + 4))
    draw = ImageDraw.Draw(image)
    draw.line((1310, 717, 1352, 717), fill=CYAN, width=6)
    draw.text((1368, 700), "Gazebo trajectory", font=font(19), fill=WHITE)
    draw.rectangle((1592, 707, 1611, 726), fill=GREEN)
    draw.text((1626, 700), "Start", font=font(19), fill=WHITE)
    draw.ellipse((1318, 755, 1347, 784), outline=AMBER, width=3)
    draw.text((1368, 748), "Goal tolerance: 0.5 m", font=font(19), fill=WHITE)
    draw.text((1310, 792), "Outline: nominal passage width", font=font(17), fill=MUTED)

    draw.rounded_rectangle((58, 980, 1220, 1040), radius=14,
                           fill=(8, 19, 26, 219))
    if not for_gif:
        label = "Isaac RTX visual reconstruction at Gazebo t = 16.48 s"
    elif moving_scene:
        label = (f"Gazebo pose {seconds:04.1f} / {trace[-1]['sim_time_s']:.1f} s"
                 "  |  moving cave view: Isaac RTX reconstruction")
    else:
        label = (f"Recorded Gazebo time {seconds:04.1f} / {trace[-1]['sim_time_s']:.1f} s"
                 "  |  scene render shows the 16.48 s pose")
    draw.text((82, 997), label, font=font(21), fill=WHITE)
    return image.convert("RGB")


def main() -> None:
    trace = json.loads((ROBOTICS / "trajectory.json").read_text())
    result = json.loads((ROBOTICS / "result.json").read_text())
    assert result["passed"] and result["trajectory_samples"] == len(trace)
    route = result["route"]
    network = json.loads(NETWORK.read_text())
    footprint, centerline = passage_envelope(network,
                                               np.asarray(route["start_xyz_m"][:2]))
    base = Image.open(SOURCE).convert("RGB")
    assert base.size == (1920, 1080)
    sample_time = json.loads((VIS / "render_receipt.json").read_text())["sample_time_s"]
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    compose(base, trace, route, footprint, centerline, sample_time,
            for_gif=False).save(FIGURE, optimize=True)

    # Recorded pose progression is played faster than simulated time. Both the
    # cave view and the plan inset follow the same logged Gazebo poses.
    GIF.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = VIS / "animation_frames"
    render_receipt = json.loads((frames_dir / "receipt.json").read_text())
    assert "visual reconstructions" in render_receipt["scope"]
    rendered = render_receipt["frames"]
    assert len(rendered) >= 20
    frames = []
    representative = Image.open(ROOT / rendered[len(rendered) // 2]["path"]).convert("RGB")
    palette_source = compose(representative, trace, route, footprint, centerline,
                             rendered[len(rendered) // 2]["sim_time_s"],
                             for_gif=True, moving_scene=True)
    palette_colors = palette_source.resize((960, 540), Image.Resampling.LANCZOS).quantize(
        colors=238, method=Image.Quantize.MEDIANCUT).getpalette()[:238 * 3]
    accents = [WHITE, MUTED, CYAN, AMBER, GREEN, YELLOW, INK,
               (47, 70, 76, 255), (13, 25, 32, 255), (8, 19, 26, 255)]
    for color in accents:
        palette_colors.extend(color[:3])
    palette_colors.extend([0] * (768 - len(palette_colors)))
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_colors)
    for record in rendered:
        seconds = record["sim_time_s"]
        source_frame = Image.open(ROOT / record["path"]).convert("RGB")
        assert source_frame.size == base.size
        frame = compose(source_frame, trace, route, footprint, centerline,
                        float(seconds), for_gif=True, moving_scene=True)
        small = frame.resize((960, 540), Image.Resampling.LANCZOS)
        frames.append(small.quantize(palette=palette, dither=Image.Dither.NONE))
    durations = [650] + [180] * (len(frames) - 2) + [850]
    frames[0].save(GIF, save_all=True, append_images=frames[1:],
                   duration=durations,
                   loop=0, optimize=True, disposal=2)

    provenance = {
        "scope": "Paper plate uses one visual reconstruction; GIF uses moving Isaac RTX visual reconstructions at logged Gazebo poses. No new robot dynamics or navigation trial.",
        "cave_render": str(SOURCE.relative_to(ROOT)),
        "cave_render_sha256": hash_file(SOURCE),
        "gazebo_trajectory": str((ROBOTICS / "trajectory.json").relative_to(ROOT)),
        "gazebo_trajectory_sha256": hash_file(ROBOTICS / "trajectory.json"),
        "network_plan": str(NETWORK.relative_to(ROOT)),
        "network_plan_sha256": hash_file(NETWORK),
        "animated_render_receipt": str((frames_dir / "receipt.json").relative_to(ROOT)),
        "animated_render_receipt_sha256": hash_file(frames_dir / "receipt.json"),
        "nominal_plan_segment": "nearest segment to route start; widths from generated centerline",
        "static_rendered_pose_time_s": sample_time,
        "gif": {"path": str(GIF.relative_to(ROOT)), "sha256": hash_file(GIF),
                "frames": len(frames), "dimensions_px": [960, 540],
                "playback_ms": sum(durations),
                "timing_note": "The near-stationary 0-9 s interval is condensed in the animation; displayed simulated timestamps are exact.",
                "animated_content": "recorded 2D pose, heading, and traversed path in the plan inset; native Isaac RTX cave render reconstructed at each displayed Gazebo pose"},
        "paper_figure": {"path": str(FIGURE.relative_to(ROOT)),
                         "sha256": hash_file(FIGURE),
                         "dimensions_px": [1920, 1080]},
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(FIGURE)
    print(GIF)


if __name__ == "__main__":
    main()
