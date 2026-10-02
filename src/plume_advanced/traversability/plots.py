"""Separate, registered views of terrain measurements and reference constraints."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from plume_advanced.progress import report_progress

OUTSIDE = "#eeeae2"
UNAVAILABLE = "#8a80a6"
NEUTRAL = "#d7dee1"
BLOCKED = "#cb553b"
STATUS_COLORS = [OUTSIDE, "#21887b", BLOCKED, UNAVAILABLE]
STATUS_LABELS = [
    "Outside mapped cavity",
    "Passes sampled limits",
    "Blocked by sampled limits",
    "Unknown",
]
REASONS = {
    1: "Incomplete footprint support",
    2: "Slope",
    4: "Step / roughness",
    8: "Body headroom",
    16: "Placed obstacle",
    32: "Uncertain surface",
}


@dataclass(frozen=True)
class Metric:
    key: str
    field: str
    title: str
    units: str
    cmap: str
    scope: str


METRICS = (
    Metric("floor_elevation", "floor_z_m", "Floor elevation", "m", "viridis", "point geometry"),
    Metric(
        "ceiling_elevation", "ceiling_z_m", "Ceiling elevation", "m", "viridis", "point geometry"
    ),
    Metric(
        "clearance", "vertical_clearance_m", "Vertical clearance", "m", "cividis", "point geometry"
    ),
    Metric("slope", "slope_deg", "Floor slope", "degrees", "magma", "reference footprint"),
    Metric(
        "step_roughness",
        "step_m",
        "Detrended step / roughness",
        "m",
        "magma",
        "reference footprint",
    ),
    Metric(
        "body_headroom", "body_clearance_m", "Body headroom", "m", "cividis", "reference footprint"
    ),
    Metric(
        "obstacles",
        "obstacle_height_m",
        "Placed obstacle height",
        "m",
        "inferno",
        "conservative prop projection",
    ),
)


def metric_values(fields, metric):
    """Exclude padding and unsupported cells; zeros are not missing values."""
    values = np.asarray(fields[metric.field])
    valid = np.isfinite(fields["floor_z_m"]) & np.isfinite(values)
    if metric.key == "obstacles":
        valid &= fields["obstacle"]
    return np.ma.array(values, mask=~valid, copy=False)


def display_scales(output, charts, config):
    """One physical scale per quantity across every layer and ramp, no clipping."""
    extrema = {m.key: [np.inf, -np.inf, 0] for m in METRICS}
    for i, chart in enumerate(charts):
        report_progress("Traversability colour scales", i, len(charts), chart["id"])
        with np.load(output / chart["npz"], allow_pickle=False) as data:
            # Cache the shared floor/mask once per chart rather than decompressing
            # them for every metric. Only one chart is held in memory.
            fields = {
                name: data[name] for name in {"floor_z_m", "obstacle", *(m.field for m in METRICS)}
            }
        for metric in METRICS:
            values = metric_values(fields, metric)
            count = int(values.count())
            if count:
                row = extrema[metric.key]
                row[:] = [
                    min(row[0], float(values.min())),
                    max(row[1], float(values.max())),
                    row[2] + count,
                ]
    report_progress(
        "Traversability colour scales",
        len(charts),
        len(charts),
        "shared measurement ranges collected",
    )
    scales = {}
    for metric in METRICS:
        low, high, total_values = extrema[metric.key]
        measured = [low, high] if total_values else None
        # Both absolute elevation views share the same vertical datum and range.
        if metric.key in {"floor_elevation", "ceiling_elevation"}:
            low = min(extrema["floor_elevation"][0], extrema["ceiling_elevation"][0])
            high = max(extrema["floor_elevation"][1], extrema["ceiling_elevation"][1])
        else:
            low = min(0.0, low)
            threshold = {
                "slope": config.max_slope_deg,
                "step_roughness": config.max_step_m,
                "body_headroom": config.robot_height_m + 2 * config.margin_m,
            }.get(metric.key, 0.0)
            high = max(high, threshold)
        if not np.isfinite([low, high]).all():
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        scales[metric.key] = dict(
            field=metric.field,
            units=metric.units,
            scope=metric.scope,
            limits=[float(low), float(high)],
            measured_range=measured,
            finite_cells=int(total_values),
            normalization="linear; shared across all charts; no finite values clipped",
        )
    return scales


def chart_title(chart):
    return (
        f"Layer {chart['layer'] + 1}"
        if chart["kind"] == "layer"
        else f"Ramp {chart['segment_ids'][0]} · layer {chart['from_layer'] + 1} → {chart['to_layer'] + 1}"
    )


def reference_caption(config):
    return (
        f"{config.resolution_m:g} m cells · {config.robot_length_m:g} × {config.robot_width_m:g} × "
        f"{config.robot_height_m:g} m reference\nslope ≤ {config.max_slope_deg:g}° · "
        f"step/roughness ≤ {config.max_step_m:g} m · margin {config.margin_m:g} m"
    )


def extent(fields):
    y, x = fields["status"].shape
    ox, oy = fields["origin_xy_m"]
    r = float(fields["resolution_m"])
    return ox, ox + x * r, oy, oy + y * r


def _image(ax, values, fields, **kwargs):
    return ax.imshow(
        values, origin="lower", extent=extent(fields), interpolation="nearest", **kwargs
    )


def _axes(ax, title):
    ax.set(title=title, xlabel="World X (m)", ylabel="World Y (m)")


def _legend(ax, colors, labels, *, columns=2):
    ax.legend(
        handles=[Patch(color=c, label=label) for c, label in zip(colors, labels)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=columns,
        frameon=False,
        fontsize=8,
    )


def draw_status(ax, fields):
    _image(
        ax,
        np.searchsorted([0, 1, 2, 255], fields["status"]),
        fields,
        cmap=ListedColormap(STATUS_COLORS),
        vmin=0,
        vmax=3,
    )
    _axes(ax, "Reference traversability")
    _legend(ax, STATUS_COLORS, STATUS_LABELS)


def draw_metric(ax, fields, metric, scale):
    values = metric_values(fields, metric)
    # Purple marks absent measurements inside the mapped cavity. No-data must
    # never look like a flat, zero-step or clear cell.
    base = np.where((fields["status"] != 0) | np.isfinite(fields["floor_z_m"]), 1, 0)
    _image(
        ax,
        base,
        fields,
        cmap=ListedColormap([OUTSIDE, UNAVAILABLE if metric.key != "obstacles" else NEUTRAL]),
        vmin=0,
        vmax=1,
    )
    cmap = plt.get_cmap(metric.cmap).with_extremes(bad=(0, 0, 0, 0))
    image = _image(ax, values, fields, cmap=cmap, vmin=scale["limits"][0], vmax=scale["limits"][1])
    _axes(ax, metric.title)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    bar.set_label(f"{metric.title} ({metric.units})", fontsize=9)
    if metric.key == "obstacles":
        _legend(
            ax,
            [OUTSIDE, NEUTRAL],
            ["Outside mapped cavity", "No projected prop (not a mobility label)"],
        )
        message = "No placed obstacles in this chart" if values.count() == 0 else None
    else:
        _legend(ax, [OUTSIDE, UNAVAILABLE], ["Outside mapped cavity", "Measurement unavailable"])
        message = "No measurements in this chart" if values.count() == 0 else None
    if message:
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
        )


def reason_masks(fields):
    """Return every matching bit, never reduce multiple failures to one winner."""
    return {bit: (fields["reason_bits"] & bit) != 0 for bit in REASONS}


def render_reasons(path, fields, config, chart):
    fig, axes = plt.subplots(2, 3, figsize=(13, 10), layout="constrained")
    try:
        base = (fields["status"] != 0).astype(np.uint8)
        for ax, (bit, flagged) in zip(axes.flat, reason_masks(fields).items()):
            values = np.where(flagged, 2, base)
            _image(
                ax, values, fields, cmap=ListedColormap([OUTSIDE, NEUTRAL, BLOCKED]), vmin=0, vmax=2
            )
            _axes(ax, f"{REASONS[bit]} · {np.count_nonzero(flagged):,} cells")
        fig.suptitle(
            chart_title(chart)
            + " · reasons for rejection or uncertainty\n"
            + reference_caption(config),
            fontsize=12,
        )
        fig.legend(
            handles=[
                Patch(color=c, label=label)
                for c, label in zip(
                    [OUTSIDE, NEUTRAL, BLOCKED],
                    [
                        "Outside mapped cavity",
                        "Not flagged (may be unevaluated)",
                        "Reason applies; flags can overlap",
                    ],
                )
            ],
            loc="outside lower center",
            ncols=3,
            frameon=False,
            fontsize=9,
        )
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def render_views(output, charts, config):
    """Render all views from saved arrays; leave measurements and labels intact."""
    scales = display_scales(output, charts, config)
    total = len(charts) * (len(METRICS) + 3)  # status, scalar views, reasons, overview
    done = 0
    for chart in charts:
        with np.load(output / chart["npz"], allow_pickle=False) as data:
            fields = {name: data[name] for name in data.files}
        h, w = fields["status"].shape
        chart["views"] = {}
        for metric in (None, *METRICS):
            key = metric.key if metric else "traversability"
            report_progress("Traversability map views", done, total, f"{chart['id']}: {key}")
            path = output / (f"{chart['id']}_{key}.png" if metric else chart["preview"])
            fig, ax = plt.subplots(figsize=(8, 9) if h > w else (11, 6), layout="constrained")
            try:
                if metric:
                    draw_metric(ax, fields, metric, scales[key])
                else:
                    draw_status(ax, fields)
                if metric and metric.scope != "reference footprint":
                    caption = f"{config.resolution_m:g} m cells · {metric.scope} · shared physical colour scale"
                else:
                    caption = reference_caption(config)
                fig.suptitle(chart_title(chart) + "\n" + caption, fontsize=11)
                fig.savefig(path, dpi=150)
            finally:
                plt.close(fig)
            chart["views"][key] = dict(
                file=path.name,
                field=metric.field if metric else "status",
                units=metric.units if metric else "class code",
                scale=key if metric else None,
            )
            done += 1
        path = output / f"{chart['id']}_reasons.png"
        report_progress(
            "Traversability map views", done, total, f"{chart['id']}: rejection reasons"
        )
        render_reasons(path, fields, config, chart)
        chart["views"]["reasons"] = dict(
            file=path.name, field="reason_bits", units="bit mask", scale=None
        )
        done += 1
        report_progress("Traversability map views", done, total, f"{chart['id']}: overview")
        path = output / f"{chart['id']}_overview.png"
        fig, axes = plt.subplots(2, 3, figsize=(15, 13), layout="constrained")
        try:
            draw_status(axes.flat[0], fields)
            for ax, key in zip(
                list(axes.flat)[1:],
                ("floor_elevation", "clearance", "slope", "step_roughness", "obstacles"),
            ):
                metric = next(m for m in METRICS if m.key == key)
                draw_metric(ax, fields, metric, scales[key])
            fig.suptitle(
                chart_title(chart) + " · terrain map set\n" + reference_caption(config), fontsize=13
            )
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        chart["overview"] = path.name
        done += 1
    report_progress("Traversability map views", done, total, "all map sets rendered")
    return scales
