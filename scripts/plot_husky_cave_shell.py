"""Plot the recorded showcase Husky route inside the exported cave shell.

Panel A cuts the delivered cave-wall GLB at a fixed elevation. Panel B samples
vertical intersections of the same wall mesh along the first Gazebo trajectory.
The shell excludes separate rock props; the plotted lines are robot origins,
not collision-body clearance tests.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import vtk
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
GLB = ROOT / "outputs/showcase_export/blender/plume_cave.glb"
ROBOTICS = ROOT / "outputs/showcase_robotics"
RUNS = (
    ROBOTICS / "husky_showcase_rockfall/trajectory.json",
    ROBOTICS / "husky_showcase_rockfall_repeat2/trajectory.json",
    ROBOTICS / "husky_showcase_rockfall_repeat3/trajectory.json",
)
RESULT = ROBOTICS / "husky_showcase_rockfall/result.json"
OUTPUT = ROOT / "paper/figures/husky_showcase_shell.png"
RECEIPT = ROOT / "paper/figures/husky_showcase_shell_provenance.json"

# Broad local selection from the exported, Z-up cave wall. The figure axes use
# a smaller crop well inside these limits, so selection edges do not show.
ROI = (-1.0, 11.0, -31.0, -20.0, 162.0, 174.0)
SLICE_Z = 164.8
COLORS = ("#11698b", "#ba593d", "#6650a4")
SHELL = "#416e76"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cave_wall_local() -> tuple[np.ndarray, np.ndarray]:
    """Read only the cave-wall primitive and map glTF Y-up back to Z-up."""
    with GLB.open("rb") as stream:
        magic, version, _ = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError("Expected a binary glTF 2.0 cave export")
        json_size, chunk_type = struct.unpack("<II", stream.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError("GLB JSON chunk missing")
        document = json.loads(stream.read(json_size))
        binary_start = 20 + json_size + 8
    primitive = document["meshes"][0]["primitives"][0]

    def view(accessor_index: int, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        accessor = document["accessors"][accessor_index]
        buffer_view = document["bufferViews"][accessor["bufferView"]]
        offset = binary_start + buffer_view["byteOffset"] + accessor.get("byteOffset", 0)
        return np.memmap(GLB, mode="r", dtype=dtype, offset=offset, shape=shape)

    positions_accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    indices_accessor = document["accessors"][primitive["indices"]]
    positions = view(primitive["attributes"]["POSITION"], "<f4", (positions_accessor["count"], 3))
    faces = view(primitive["indices"], "<u4", (indices_accessor["count"] // 3, 3))
    x0, x1, y0, y1, z0, z1 = ROI
    # glTF stores canonical (x, y, z) as (x, z, -y).
    inside = ((positions[:, 0] >= x0) & (positions[:, 0] <= x1)
              & (positions[:, 1] >= z0) & (positions[:, 1] <= z1)
              & (positions[:, 2] >= -y1) & (positions[:, 2] <= -y0))
    selected: list[np.ndarray] = []
    for start in range(0, len(faces), 500_000):
        block = faces[start:start + 500_000]
        selected.append(np.asarray(block[np.all(inside[block], axis=1)], dtype=np.uint32))
    local_faces = np.vstack(selected)
    used, inverse = np.unique(local_faces, return_inverse=True)
    p = positions[used]
    vertices = np.column_stack((p[:, 0], -p[:, 2], p[:, 1])).astype(np.float32)
    return vertices, inverse.reshape(-1, 3).astype(np.int64)


def vtk_mesh(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    packed = np.column_stack((np.full(len(faces), 3), faces)).ravel().astype(np.int64)
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed, deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def horizontal_section(mesh: vtk.vtkPolyData) -> tuple[np.ndarray, list[np.ndarray]]:
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, SLICE_Z)
    plane.SetNormal(0, 0, 1)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    output = cutter.GetOutput()
    points = vtk_to_numpy(output.GetPoints().GetData())
    lines = output.GetLines()
    connectivity = vtk_to_numpy(lines.GetConnectivityArray())
    offsets = vtk_to_numpy(lines.GetOffsetsArray())
    segments = [points[connectivity[a:b], :2]
                for a, b in zip(offsets[:-1], offsets[1:]) if b - a >= 2]
    return points, segments


def shell_bounds_at_y(points: np.ndarray, y_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples_y, left, right = [], [], []
    for y in y_grid:
        xs = points[np.abs(points[:, 1] - y) < 0.045, 0]
        if len(xs) >= 2:
            samples_y.append(y)
            left.append(float(xs.min()))
            right.append(float(xs.max()))
    if len(samples_y) < len(y_grid) // 2:
        raise ValueError("Too few slice points for the plan envelope")
    return np.interp(y_grid, samples_y, left), np.interp(y_grid, samples_y, right)


def vertical_shell(mesh: vtk.vtkPolyData, trace: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    locator = vtk.vtkOBBTree()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    floor, roof = [], []
    for row in trace:
        x, y, _ = row["pose"]
        hits = vtk.vtkPoints()
        ids = vtk.vtkIdList()
        locator.IntersectWithLine((x, y, 162), (x, y, 168), hits, ids)
        if hits.GetNumberOfPoints() != 2:
            raise ValueError(f"Expected floor and roof at ({x:.3f}, {y:.3f}); got {hits.GetNumberOfPoints()} intersections")
        z = sorted(hits.GetPoint(i)[2] for i in range(hits.GetNumberOfPoints()))
        floor.append(z[0])
        roof.append(z[1])
    return np.asarray(floor), np.asarray(roof)


def path_progress(trace: list[dict]) -> np.ndarray:
    xy = np.asarray([row["pose"][:2] for row in trace])
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def main() -> None:
    traces = [json.loads(path.read_text()) for path in RUNS]
    result = json.loads(RESULT.read_text())
    assert result["passed"]
    goal = result["route"]["waypoints_xy_m"][-1]
    tolerance = result["route"]["goal_tolerance_m"]
    vertices, faces = cave_wall_local()
    mesh = vtk_mesh(vertices, faces)
    slice_points, slice_segments = horizontal_section(mesh)
    y_grid = np.linspace(-28.5, -22.0, 401)
    left, right = shell_bounds_at_y(slice_points, y_grid)
    floor, roof = vertical_shell(mesh, traces[0])
    assert np.all(floor < np.asarray([r["pose"][2] for r in traces[0]]))
    assert np.all(roof > np.asarray([r["pose"][2] for r in traces[0]]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 11.5, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, (plan, profile) = plt.subplots(1, 2, figsize=(8.4, 4.8), dpi=180,
                                        gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.subplots_adjust(left=.09, right=.985, top=.89, bottom=.24, wspace=.24)

    plan.fill_betweenx(y_grid, left, right, color="#e0eeec", alpha=.94, zorder=1)
    plan.add_collection(LineCollection(slice_segments, colors=SHELL, linewidths=1.2,
                                       alpha=.92, zorder=2))
    for i, trace in enumerate(traces):
        xy = np.asarray([row["pose"][:2] for row in trace])
        plan.plot(xy[:, 0], xy[:, 1], color=COLORS[i], linewidth=2.5 - i * .18,
                  alpha=.97, zorder=5 + i)
    start = traces[0][0]["pose"]
    plan.scatter(start[0], start[1], s=62, color="#202428", edgecolor="white",
                 linewidth=.8, zorder=10)
    plan.scatter(goal[0], goal[1], marker="*", s=160, color="#202428",
                 edgecolor="white", linewidth=.5, zorder=10)
    plan.add_patch(Circle(goal, tolerance, fill=False, edgecolor="#444b50",
                          linestyle=(0, (2, 2)), linewidth=1.6, zorder=8))
    plan.set(xlim=(1.0, 8.0), ylim=(-28.1, -22.3), xlabel="World x (m)",
             ylabel="World y (m)", title=f"A  Plan section of cave shell at z = {SLICE_Z:.1f} m")
    plan.set_aspect("equal", adjustable="box")
    plan.grid(color="#cbd5d4", alpha=.5, linewidth=.7)
    plan.set_axisbelow(True)

    primary_progress = path_progress(traces[0])
    profile.fill_between(primary_progress, floor, roof, color="#e0eeec", alpha=.94,
                         zorder=1)
    profile.plot(primary_progress, floor, color=SHELL, linewidth=1.7, zorder=2)
    profile.plot(primary_progress, roof, color=SHELL, linewidth=1.7, zorder=2)
    for i, trace in enumerate(traces):
        progress = path_progress(trace)
        z = np.asarray([row["pose"][2] for row in trace])
        profile.plot(progress, z, color=COLORS[i], linewidth=2.2 - i * .15,
                     alpha=.96, zorder=4 + i)
    profile.text(primary_progress[-1] - .02, roof[-1] + .025, "Roof", color=SHELL,
                 fontsize=10, ha="right", va="bottom")
    profile.text(primary_progress[-1] - .02, floor[-1] - .025, "Floor", color=SHELL,
                 fontsize=10, ha="right", va="top")
    profile.set(xlim=(0, 2.75), ylim=(164.1, 165.56),
                xlabel="Travelled path length (m)", ylabel="World z (m)",
                title="B  Exported floor and roof along Trial 1")
    profile.grid(color="#cbd5d4", alpha=.5, linewidth=.7)
    profile.set_axisbelow(True)

    cave_handle = Patch(facecolor="#e0eeec", edgecolor=SHELL,
                        label="Cave interior / shell")
    trial_handles = [Line2D([], [], color=COLORS[i], linewidth=2.5,
                            label=f"Gazebo trial {i+1}") for i in range(3)]
    start_handle = Line2D([], [], marker="o", color="none",
                          markerfacecolor="#202428", markersize=7, label="Start")
    goal_handle = Line2D([], [], marker="*", color="none",
                         markerfacecolor="#202428", markersize=11, label="Goal")
    tolerance_handle = Line2D([], [], color="#444b50", linestyle=(0, (2, 2)),
                              linewidth=1.6, label="0.5 m tolerance")
    # Matplotlib populates a two-row legend by columns.
    handles = [cave_handle, start_handle, trial_handles[0], goal_handle,
               trial_handles[1], tolerance_handle, trial_handles[2]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .065), fontsize=8.5, columnspacing=1.2,
               handlelength=2.4)
    fig.text(.5, .025, "Cave wall: exported showcase mesh  •  robot lines: recorded Gazebo origins",
             ha="center", va="center", color="#606d70", fontsize=8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)

    RECEIPT.write_text(json.dumps({
        "description": "Local exported cave-wall horizontal shell slice and vertical ray intersections along recorded Gazebo Husky trajectories; rock props are excluded and no robot-body clearance is inferred.",
        "sources": {str(path.relative_to(ROOT)): digest(path)
                    for path in (GLB, *RUNS, RESULT)},
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": digest(OUTPUT),
        "local_selection_xyz_m": list(ROI),
        "horizontal_slice_z_m": SLICE_Z,
        "local_mesh_vertices": int(len(vertices)),
        "local_mesh_faces": int(len(faces)),
        "vertical_profile_trace": "Trial 1",
        "vertical_ray_intersections_per_sample": 2,
        "trial_1_origin_minus_floor_min_m": float(np.min(np.asarray([r["pose"][2] for r in traces[0]]) - floor)),
        "trial_1_roof_minus_origin_min_m": float(np.min(roof - np.asarray([r["pose"][2] for r in traces[0]]))),
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
