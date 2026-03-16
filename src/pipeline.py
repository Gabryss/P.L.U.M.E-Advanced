from __future__ import annotations

import time
from pathlib import Path

from dataclasses import replace

from src.config import RunConfig, write_config_snapshot
from src.geology.generator import generate_geology
from src.hydro.solver import solve_hydraulic_head
from src.meshing.extractor import export_obj, extract_mesh
from src.speleogenesis.evolution import initialize_speleogenesis, update_speleogenesis
from src.state import SimulationState
from src.utils.io import configure_logger, ensure_directory, write_csv, write_json
from src.utils.random import make_rng
from src.visualization.mesh_preview import save_mesh_preview
from src.visualization.metrics import compute_metrics, numerical_warnings
from src.visualization.plots import save_histograms, save_time_series
from src.visualization.progress import progress_iter
from src.visualization.slices import save_central_slices


def _snapshot_outputs(run_dir: Path, state: SimulationState, metrics_history: list[dict[str, float | int]], iteration: int) -> None:
    save_central_slices(
        {
            "material_id": state.geology.material_id.astype(float),
            "fracture_density": state.geology.fracture_density,
            "hydraulic_head": state.hydro.hydraulic_head,
            "water_flux": state.hydro.water_flux,
            "porosity": state.speleogenesis.porosity,
            "solid_fraction": state.speleogenesis.solid_fraction,
            "void_mask": state.speleogenesis.void_mask.astype(float),
        },
        run_dir / "debug" / f"slices_{iteration:04d}.png",
    )
    save_time_series(metrics_history, run_dir / "debug" / "time_series.png")


def run_simulation(
    config: RunConfig,
    output_dir: str | Path,
    *,
    seed: int | None = None,
    run_name: str | None = None,
) -> dict[str, object]:
    actual_seed = config.seed if seed is None else seed
    actual_name = config.name if run_name is None else run_name
    run_dir = ensure_directory(output_dir)
    logger = configure_logger(run_dir / "run.log")
    write_config_snapshot(replace(config, seed=actual_seed, name=actual_name), run_dir / "config_snapshot.yaml")

    rng = make_rng(actual_seed)
    geology = generate_geology(config.domain, config.geology, rng)
    speleo = initialize_speleogenesis(geology, config.dissolution)
    hydro = solve_hydraulic_head(config.domain, config.hydro, speleo.effective_permeability)
    state = SimulationState(domain=config.domain, geology=geology, hydro=hydro, speleogenesis=speleo)

    logger.info("run_id=%s seed=%s stage=initialize", actual_name, actual_seed)
    metrics_history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for iteration in progress_iter(
        range(1, config.dissolution.iterations + 1),
        enabled=config.monitoring.progress_bar_enabled,
        desc="simulation",
    ):
        hydro = solve_hydraulic_head(config.domain, config.hydro, state.speleogenesis.effective_permeability)
        speleo = update_speleogenesis(state.geology, hydro, state.speleogenesis, config.dissolution)
        state = SimulationState(domain=config.domain, geology=state.geology, hydro=hydro, speleogenesis=speleo)

        elapsed = time.perf_counter() - started
        metrics = compute_metrics(state, iteration, elapsed)
        metrics_history.append(metrics)

        if iteration % config.monitoring.log_interval == 0 or iteration == config.dissolution.iterations:
            logger.info(
                "run_id=%s seed=%s stage=simulate iteration=%s void_fraction=%.6f mean_porosity=%.6f max_flux=%.6f components=%s",
                actual_name,
                actual_seed,
                iteration,
                metrics["cave_volume_fraction"],
                metrics["mean_porosity"],
                metrics["max_water_flux"],
                metrics["connected_components"],
            )
            for warning in numerical_warnings(state):
                logger.warning("run_id=%s iteration=%s warning=%s", actual_name, iteration, warning)

        if iteration % config.monitoring.slice_interval == 0:
            _snapshot_outputs(run_dir, state, metrics_history, iteration)

        if iteration % config.monitoring.histogram_interval == 0:
            save_histograms(
                {
                    "solubility": state.geology.solubility,
                    "permeability": state.speleogenesis.effective_permeability,
                    "porosity": state.speleogenesis.porosity,
                    "solid_fraction": state.speleogenesis.solid_fraction,
                    "dissolution_damage": state.speleogenesis.dissolution_damage,
                },
                run_dir / "debug" / f"histograms_{iteration:04d}.png",
            )

    vertices, faces, method = extract_mesh(state.phi(config.dissolution.void_threshold), config.domain.voxel_size)
    export_obj(vertices, faces, run_dir / "mesh" / "cave.obj")
    save_mesh_preview(vertices, faces, run_dir / "debug" / "mesh_preview.png")
    write_csv(metrics_history, run_dir / "metrics.csv")
    save_time_series(metrics_history, run_dir / "debug" / "time_series.png")

    final_metrics = metrics_history[-1] if metrics_history else compute_metrics(state, 0, 0.0)
    summary = {
        "run_id": actual_name,
        "seed": actual_seed,
        "output_dir": str(run_dir),
        "mesh_method": method,
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "final_metrics": final_metrics,
    }
    write_json(summary, run_dir / "summary.json")
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"run_id: {actual_name}\n")
        handle.write(f"seed: {actual_seed}\n")
        handle.write(f"mesh_method: {method}\n")
        handle.write(f"mesh_vertex_count: {len(vertices)}\n")
        handle.write(f"mesh_face_count: {len(faces)}\n")
        handle.write(f"cave_volume_fraction: {final_metrics['cave_volume_fraction']}\n")
        handle.write(f"connected_components: {final_metrics['connected_components']}\n")

    return summary
