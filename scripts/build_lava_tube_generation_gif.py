"""Build a GitHub-friendly animation from the accepted showcase outputs.

The network growth is reconstructed from Stage B's saved centerline ages. The
later cards are pipeline stages, and the last image is the approved Blender
render. This is not a recording of a live generation run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "outputs/showcase_optimized/stage_b_network.json"
EVENTS = ROOT / "outputs/showcase_optimized/stage_e_event_report.json"
HOST = ROOT / "outputs/showcase_optimized/stage_a_host_field.png"
RENDER = ROOT / "outputs/showcase_gallery/depth_preview/blender_lava_tube_artist_full.png"
OUTPUT = ROOT / "docs/assets/lava_tube_generation.gif"
RECEIPT = ROOT / "docs/assets/lava_tube_generation_provenance.json"

SIZE = (960, 540)
MAP = (42, 100, 690, 453)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
REGULAR = FONT_DIR / "DejaVuSans.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def world_to_map(x: float, y: float) -> tuple[int, int]:
    # Rotate the plan clockwise so the flow direction reads left to right.
    scale = 1.72  # px / m; identical in both world axes.
    return (int(round(85 + (y + 235) * scale)), int(round(278 - x * scale)))


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    iw, ih = image.size
    target_aspect = width / height
    if iw / ih > target_aspect:
        new_w = int(round(ih * target_aspect))
        left = (iw - new_w) // 2
        image = image.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(round(iw / target_aspect))
        top = (ih - new_h) // 2
        image = image.crop((0, top, iw, top + new_h))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def base_map() -> Image.Image:
    im = Image.new("RGB", SIZE, (13, 19, 23))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(MAP, radius=14, fill=(20, 31, 37), outline=(51, 68, 73), width=2)
    # Metric grid, rendered in the same rotated coordinate system as the data.
    for world_y in range(-200, 101, 50):
        u, _ = world_to_map(0, world_y)
        if MAP[0] + 10 < u < MAP[2] - 10:
            d.line((u, MAP[1] + 13, u, MAP[3] - 13), fill=(32, 49, 55), width=1)
            d.text((u - 10, MAP[1] + 34), str(world_y), font=font(10), fill=(91, 113, 118))
    for world_x in range(-50, 76, 25):
        _, v = world_to_map(world_x, 0)
        if MAP[1] + 12 < v < MAP[3] - 12:
            d.line((MAP[0] + 12, v, MAP[2] - 12, v), fill=(32, 49, 55), width=1)
    d.text((58, 113), "PLAN VIEW  •  FLOW →", font=font(12, True), fill=(145, 171, 172))
    d.text((53, 430), "World y (m)  •  plan rotated for display", font=font(10), fill=(117, 140, 143))
    return im


STAGES = [
    ("01", "Host field"),
    ("02", "Network growth"),
    ("03", "Cross-sections"),
    ("04", "Surface mesh"),
    ("05", "Rocks & events"),
    ("06", "Materials & light"),
]


def chrome(im: Image.Image, active: int, title: str, subtitle: str, progress: float) -> None:
    d = ImageDraw.Draw(im)
    d.text((42, 29), "FROM FIELD TO LAVA TUBE", font=font(24, True), fill=(241, 234, 217))
    d.text((758, 37), "PLUME  /  SHOWCASE", font=font(11, True), fill=(143, 168, 164))

    d.rounded_rectangle((711, 100, 918, 453), radius=14, fill=(25, 35, 40), outline=(53, 67, 68), width=2)
    d.text((729, 120), "GENERATION STAGES", font=font(11, True), fill=(149, 171, 168))
    for idx, (number, label) in enumerate(STAGES):
        y = 157 + idx * 47
        if idx == active:
            d.rounded_rectangle((722, y - 8, 907, y + 31), radius=8, fill=(59, 65, 55))
        color = (239, 204, 142) if idx <= active else (108, 124, 125)
        d.text((732, y), number, font=font(12, True), fill=color)
        d.text((765, y), label, font=font(12, idx == active), fill=color)

    d.text((43, 471), title, font=font(16, True), fill=(239, 227, 204))
    d.text((43, 497), subtitle, font=font(12), fill=(157, 174, 173))
    d.rounded_rectangle((715, 483, 918, 491), radius=4, fill=(46, 59, 61))
    d.rounded_rectangle((715, 483, 715 + round(203 * progress), 491), radius=4, fill=(226, 182, 108))
    d.text((716, 505), "Reconstructed from saved outputs", font=font(9), fill=(126, 145, 145))


def draw_partial_network(im: Image.Image, segments: list[dict], age: float, width_mode: bool) -> None:
    d = ImageDraw.Draw(im)
    # Draw all three source starts, then reveal the accepted centerline at each age.
    starts = [s for s in segments if float(s["age_start_s"]) == 0.0]
    for s in starts:
        p = s["centerline"][0]
        u, v = world_to_map(p["x"], p["y"])
        d.ellipse((u - 6, v - 6, u + 6, v + 6), fill=(241, 189, 107), outline=(255, 230, 174), width=2)

    fronts: list[tuple[int, int]] = []
    for segment in segments:
        pts = segment["centerline"]
        if age < pts[0]["age_s"]:
            continue
        for a, b in zip(pts, pts[1:]):
            if age < a["age_s"]:
                break
            frac = min(1.0, max(0.0, (age - a["age_s"]) / max(1e-9, b["age_s"] - a["age_s"])))
            end_x = a["x"] + frac * (b["x"] - a["x"])
            end_y = a["y"] + frac * (b["y"] - a["y"])
            start = world_to_map(a["x"], a["y"])
            end = world_to_map(end_x, end_y)
            if start != end:
                if width_mode:
                    half_width = (a["width"] + frac * (b["width"] - a["width"])) / 2
                    footprint_px = max(7, round(half_width * 2 * 1.72))
                    d.line((start, end), fill=(173, 158, 122), width=footprint_px + 2)
                    d.line((start, end), fill=(224, 212, 169), width=footprint_px)
                    d.line((start, end), fill=(247, 233, 189), width=2)
                else:
                    d.line((start, end), fill=(30, 124, 134), width=9)
                    d.line((start, end), fill=(94, 201, 199), width=4)
            if frac < 1:
                fronts.append(end)
                break
    if not width_mode:
        for u, v in fronts:
            d.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 205, 126), outline=(255, 246, 204))


def draw_events(im: Image.Image, events: list[dict], fraction: float) -> None:
    d = ImageDraw.Draw(im)
    count = round(len(events) * fraction)
    for event in events[:count]:
        u, v = world_to_map(event["x"], event["y"])
        if not (MAP[0] + 14 < u < MAP[2] - 14 and MAP[1] + 14 < v < MAP[3] - 14):
            continue
        radius = max(2, min(5, int(round(max(event["radius_x"], event["radius_y"]) * 1.7))))
        color = (164, 148, 132) if event["kind"] == "rock" else (213, 150, 94)
        d.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(73, 66, 59))


def draw_host_fields(im: Image.Image, host: Image.Image) -> None:
    """Show two real diagnostic panels saved by the accepted Stage A run."""
    d = ImageDraw.Draw(im)
    for crop, left, label in (
        ((135, 105, 373, 625), 91, "TERRAIN ELEVATION"),
        ((2065, 774, 2295, 1270), 374, "FLOW CAPACITY"),
    ):
        tile = host.crop(crop).resize((222, 268), Image.Resampling.LANCZOS)
        im.paste(tile, (left, 148))
        d.rectangle((left - 1, 147, left + 222, 416), outline=(118, 145, 146), width=1)
        d.text((left, 425), label, font=font(11, True), fill=(174, 190, 180))


def make_map_frame(base: Image.Image, host: Image.Image, segments: list[dict], events: list[dict], *,
                   active: int, age: float, progress: float, event_fraction: float = 0.0) -> Image.Image:
    if active == 0:
        im = Image.new("RGB", SIZE, (13, 19, 23))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle(MAP, radius=14, fill=(20, 31, 37), outline=(51, 68, 73), width=2)
        d.text((58, 113), "HOST ROUTING INPUTS", font=font(12, True), fill=(145, 171, 172))
    else:
        im = base.copy()
    width_mode = active >= 2
    if active == 0:
        draw_host_fields(im, host)
    else:
        draw_partial_network(im, segments, age, width_mode)
    if active >= 4:
        draw_events(im, events, event_fraction)
    if active == 0:
        title = "1 / Host field"
        subtitle = "Terrain, cover and routing conditions guide the three inlets."
    elif active == 1:
        title = "2 / Three passages grow and connect"
        subtitle = f"Saved centerline age: {age:,.0f} / {max_age:,.0f} s  •  final network: 11 segments"
    elif active == 2:
        title = "3 / Continuous passage sections"
        subtitle = "The centerlines receive varying widths and cross-section geometry."
    elif active == 3:
        title = "4 / Connected cave surface"
        subtitle = "The accepted sections are converted to a connected mesh."
    else:
        title = "5 / Grounded rocks and geological events"
        subtitle = f"{round(len(events) * event_fraction):,} / {len(events):,} accepted events shown in plan view."
    chrome(im, active, title, subtitle, progress)
    return im


def make_render_frame(render: Image.Image, opacity: float, progress: float) -> Image.Image:
    im = Image.new("RGB", SIZE, (13, 19, 23))
    fitted = fit_image(render, 960, 540)
    # The approved image keeps its warm, deliberately low-key cave lighting.
    if opacity < 1:
        im = Image.blend(im, fitted, opacity)
    else:
        im = fitted.copy()
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle((0, 0, 960, 91), fill=(8, 13, 16, 206))
    d.rectangle((0, 440, 960, 540), fill=(8, 13, 16, 209))
    d.text((39, 27), "THE FINISHED LAVA TUBE", font=font(25, True), fill=(246, 234, 211, 255))
    d.text((39, 460), "6 / Textured Blender showcase", font=font(17, True), fill=(247, 227, 190, 255))
    d.text((39, 491), "Accepted PLUME cave • grounded rockfall • final approved lighting", font=font(12), fill=(214, 218, 211, 255))
    d.rounded_rectangle((718, 481, 919, 489), radius=4, fill=(62, 67, 63, 255))
    d.rounded_rectangle((718, 481, 718 + round(201 * progress), 489), radius=4, fill=(229, 188, 113, 255))
    return im.convert("RGB")


def main() -> None:
    network = json.loads(NETWORK.read_text())
    events_file = json.loads(EVENTS.read_text())
    segments = network["segments"]
    events = events_file["events"]
    global max_age
    max_age = max(float(p["age_s"]) for s in segments for p in s["centerline"])
    base = base_map()
    host = Image.open(HOST).convert("RGB")
    render = ImageEnhance.Brightness(Image.open(RENDER).convert("RGB")).enhance(1.03)

    frames: list[Image.Image] = []
    durations: list[int] = []
    def add(im: Image.Image, duration: int) -> None:
        frames.append(im)
        durations.append(duration)

    add(make_map_frame(base, host, segments, events, active=0, age=0, progress=0), 950)
    for i in range(42):
        age = max_age * i / 41
        add(make_map_frame(base, host, segments, events, active=1, age=age,
                           progress=0.07 + 0.56 * (i / 41)), 110)
    for i in range(4):
        add(make_map_frame(base, host, segments, events, active=2, age=max_age,
                           progress=0.65 + 0.04 * i / 3), 190)
    for i in range(4):
        add(make_map_frame(base, host, segments, events, active=3, age=max_age,
                           progress=0.72 + 0.04 * i / 3), 190)
    for i in range(8):
        add(make_map_frame(base, host, segments, events, active=4, age=max_age,
                           event_fraction=(i + 1) / 8, progress=0.79 + 0.1 * i / 7), 150)
    for i in range(4):
        add(make_render_frame(render, (i + 1) / 4, 0.90 + i * 0.025), 130)
    add(make_render_frame(render, 1.0, 1.0), 1550)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=2)
    RECEIPT.write_text(json.dumps({
        "kind": "animated_reconstruction_from_accepted_showcase",
        "description": "Stage B centerline growth reconstructed from saved age_s; later stages represented in plan, ending on the approved Blender still. Not a live capture.",
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (HOST, NETWORK, EVENTS, RENDER)},
        "output": str(OUTPUT.relative_to(ROOT)),
        "frame_count": len(frames),
        "duration_ms": sum(durations),
        "network_max_age_s": max_age,
        "segment_count": len(segments),
        "event_count": len(events),
        "dimensions_px": list(SIZE),
        "plan_rotation": "clockwise; world +y points to screen right",
    }, indent=2) + "\n")
    print(f"{OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.2f} MiB, {len(frames)} frames, {sum(durations)/1000:.2f} s)")


if __name__ == "__main__":
    main()
