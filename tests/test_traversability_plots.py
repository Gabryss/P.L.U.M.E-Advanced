"""Measurement views must preserve units, missing data and overlapping failures."""

import json

import matplotlib.pyplot as plt
import numpy as np
from test_traversability import box, request

from plume_advanced.progress import progress_scope
from plume_advanced.traversability.config import TraversabilityConfig
from plume_advanced.traversability.export import export_traversability
from plume_advanced.traversability.plots import (
    METRICS,
    display_scales,
    draw_metric,
    metric_values,
    reason_masks,
)


def fields(z=0.0):
    return dict(
        floor_z_m=np.array([[z, z, np.nan], [z, z, np.nan]]),
        ceiling_z_m=np.full((2, 3), z + 2),
        vertical_clearance_m=np.full((2, 3), 2.0),
        slope_deg=np.array([[0.0, 30.0, np.nan], [np.nan, 15.0, np.nan]]),
        step_m=np.array([[0.0, 0.2, np.nan], [np.nan, 0.01, np.nan]]),
        body_clearance_m=np.array([[2.0, 0.4, np.nan], [np.nan, 2.0, np.nan]]),
        obstacle=np.zeros((2, 3), bool),
        obstacle_height_m=np.zeros((2, 3)),
        status=np.array([[1, 2, 0], [2, 1, 0]], np.uint8),
        reason_bits=np.array([[0, 2 | 4 | 8, 0], [1, 0, 0]], np.uint8),
        origin_xy_m=np.array([-1.0, 3.0]),
        resolution_m=np.asarray(0.25),
    )


def test_missing_metrics_are_not_zero_and_padding_is_not_terrain():
    data = fields()
    slope = metric_values(data, next(m for m in METRICS if m.key == "slope"))
    assert slope[0, 0] == 0 and not slope.mask[0, 0]
    assert slope.mask[1, 0] and slope.mask[0, 2]
    roof = metric_values(data, next(m for m in METRICS if m.key == "ceiling_elevation"))
    assert roof.mask[:, 2].all()  # padding has a finite dummy ceiling, but no floor
    obstacles = metric_values(data, next(m for m in METRICS if m.key == "obstacles"))
    assert obstacles.count() == 0  # zeros mean no props, not occupied zero-height props
    data["obstacle"][0, 1] = True
    data["obstacle_height_m"][0, 1] = 0.35
    obstacles = metric_values(data, next(m for m in METRICS if m.key == "obstacles"))
    assert obstacles.count() == 1 and obstacles[0, 1] == 0.35


def test_shared_physical_scales_cover_every_layer_without_clipping(tmp_path):
    charts = []
    for i, z in enumerate((-5.0, 20.0)):
        path = tmp_path / f"layer_{i}.npz"
        np.savez_compressed(path, **fields(z))
        charts.append(dict(id=f"layer_{i}", npz=path.name))
    scales = display_scales(tmp_path, charts, TraversabilityConfig())
    assert scales["floor_elevation"]["limits"] == [-5.0, 22.0]
    assert scales["floor_elevation"]["limits"] == scales["ceiling_elevation"]["limits"]
    assert scales["slope"]["limits"] == [0.0, 30.0]
    assert scales["step_roughness"]["limits"] == [0.0, 0.2]
    assert scales["obstacles"]["measured_range"] is None
    assert scales["obstacles"]["finite_cells"] == 0
    assert scales["floor_elevation"]["units"] == "m" and scales["slope"]["units"] == "degrees"
    json.dumps(scales, allow_nan=False)


def test_empty_chart_has_finite_display_limits_and_no_invented_measurements(tmp_path):
    data = fields()
    data["floor_z_m"][:] = np.nan
    np.savez_compressed(tmp_path / "empty.npz", **data)
    scales = display_scales(tmp_path, [dict(id="empty", npz="empty.npz")], TraversabilityConfig())
    for scale in scales.values():
        assert scale["measured_range"] is None and scale["finite_cells"] == 0
        assert np.isfinite(scale["limits"]).all()
        assert scale["limits"][1] > scale["limits"][0]
    json.dumps(scales, allow_nan=False)


def test_multiple_rejection_reasons_are_visible_in_each_panel():
    masks = reason_masks(fields())
    assert masks[2][0, 1] and masks[4][0, 1] and masks[8][0, 1]
    assert not masks[1][0, 1] and not masks[16][0, 1]
    assert masks[1][1, 0]


def test_scalar_plot_uses_world_axes_unit_label_no_interpolation_and_explicit_missing():
    data = fields()
    metric = next(m for m in METRICS if m.key == "slope")
    before = {k: v.copy() for k, v in data.items()}
    fig, ax = plt.subplots()
    try:
        draw_metric(ax, data, metric, dict(limits=[0.0, 30.0]))
        layer = ax.images[-1]
        assert layer.origin == "lower" and layer.get_interpolation() == "nearest"
        np.testing.assert_allclose(layer.get_extent(), [-1.0, -0.25, 3.0, 3.5])
        assert layer.get_clim() == (0.0, 30.0)
        assert fig.axes[-1].get_ylabel() == "Floor slope (degrees)"
        assert "Measurement unavailable" in [t.get_text() for t in ax.get_legend().get_texts()]
        assert layer.get_array().mask[1, 0]
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
    finally:
        plt.close(fig)


def test_no_props_is_labelled_explicitly_instead_of_looking_blocked():
    metric = next(m for m in METRICS if m.key == "obstacles")
    fig, ax = plt.subplots()
    try:
        draw_metric(ax, fields(), metric, dict(limits=[0.0, 1.0]))
        assert "No placed obstacles in this chart" in [t.get_text() for t in ax.texts]
    finally:
        plt.close(fig)


def test_all_view_files_are_hashed_registered_and_rendering_has_progress(tmp_path):
    from plume_advanced.identity import sha256_file

    mesh = box()
    steps = []
    with progress_scope(lambda step, n, total, detail: steps.append((step, n, total, detail))):
        files = export_traversability(request(), mesh.vertices, mesh.faces, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    chart = manifest["charts"][0]
    assert set(chart["views"]) == {"traversability", "reasons", *(m.key for m in METRICS)}
    assert chart["views"]["slope"]["field"] == "slope_deg"
    assert chart["views"]["slope"]["units"] == "degrees"
    assert chart["overview"] == "layer_0_overview.png"
    assert len(files) == len(set(files))
    hashes = {**chart["files_sha256"], **manifest["files_sha256"]}
    assert {p.name for p in files} == {*hashes, "manifest.json"}
    for name, digest in hashes.items():
        assert tmp_path / name in files
        assert sha256_file(tmp_path / name) == digest
    work = [row for row in steps if row[0] == "Traversability map views"]
    assert [row[1] for row in work] == list(range(11))
    assert work[-1][1] == work[-1][2] == 10
    assert not plt.get_fignums()
