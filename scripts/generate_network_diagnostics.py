#!/usr/bin/env python3
"""Generate Stage A-C network diagnostics without building a cave mesh."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.artifacts import (
    export_network_artifact,
    export_section_artifact,
)
from plume_advanced.evaluation.visualization.dashboard import render_diagnostic_dashboard
from plume_advanced.evaluation.visualization.emplacement import (
    render_emplacement_phase_activity,
)
from plume_advanced.stages.host_field import HostFieldGenerator
from plume_advanced.stages.network import CaveNetworkGenerator, export_network_report
from plume_advanced.stages.section_field import SectionFieldGenerator
from plume_advanced.visualization.drained_pools import DrainedPoolPlotter
from plume_advanced.visualization.network import CaveNetworkPlotter
from plume_advanced.visualization.section_field import SectionFieldPlotter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDC = ROOT / "outputs/reference_morphology/pdc_calibration_sections.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/project.toml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/valentine_network_review",
    )
    parser.add_argument(
        "--density-sweep",
        default="0.5,1,2,3",
        help="Comma-separated network-density values; use an empty string to disable.",
    )
    parser.add_argument("--pdc-calibration", type=Path, default=DEFAULT_PDC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_project_config(args.config)
    host = HostFieldGenerator(config.host_field).generate()
    network = CaveNetworkGenerator(config.network).generate(
        host, section_config=config.section_field,
        quality_report_path=args.output / "network_quality_report.json",
        quality_progress=print,
    )
    sections = SectionFieldGenerator(config.section_field).generate(network)

    export_network_report(network, args.output / "stage_b_network_report.json")
    export_network_artifact(network, args.output / "stage_b_network.json")
    export_section_artifact(sections, args.output / "stage_c_sections")
    CaveNetworkPlotter().render(host, network, args.output / "stage_b_network.png")
    from plume_advanced.visualization.host_field import HostFieldPlotter
    HostFieldPlotter().render(host, args.output / "stage_a_host_field.png")
    if network.config.topology.style == "interconnected":
        from plume_advanced.visualization.interconnected import render_interconnected
        render_interconnected(host, network, sections, args.output / "stage_bc_interconnected.png")
    if network.config.topology.style == "trunk_dominated":
        from plume_advanced.visualization.network_topology import render_topology_footprint
        render_topology_footprint(network, sections, args.output / "stage_bc_topology_footprint.png")
    if (network.config.systems.count == 1 and network.config.topology.style == "general") or network.config.topology.generation_mode == "independent_growth":
        render_emplacement_phase_activity(
            network,
            args.output / "stage_b_emplacement_history.png",
        )
    SectionFieldPlotter().render(network, sections, args.output / "stage_c_sections.png")
    if (network.config.systems.count == 1 and network.config.topology.style == "general") or network.config.topology.generation_mode == "independent_growth":
        DrainedPoolPlotter().render(
            network,
            sections,
            args.output / "stage_c_drained_pools.png",
        )

    density_sweep = _density_sweep(args.density_sweep, config.network, host, network)
    pdc = _pdc_calibration(args.pdc_calibration)
    render_diagnostic_dashboard(
        network,
        sections,
        args.output / "stage_bc_evaluation_dashboard.png",
        density_sweep=density_sweep,
        pdc_calibration=pdc,
        provenance={
            "config_path": str(args.config.resolve()),
            "scope": "stages_A_to_C_only",
        },
    )
    summary = {
        "scope": "host, network, and section field only; no mesh generated",
        "network": network.summary(),
        "sections": sections.summary(),
        "density_sweep": {
            str(density): candidate.summary() for density, candidate in density_sweep.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _density_sweep(specification, network_config, host, primary_network):
    # The interacting-system grammar has its own controls. A density sweep
    # here would repeat an identical network and imply a nonexistent effect.
    if network_config.systems.count > 1 or network_config.topology.style != "general":
        return {}
    values = [float(value.strip()) for value in specification.split(",") if value.strip()]
    result = {}
    for density in sorted(set(values)):
        if not 0.0 <= density <= 3.0:
            raise ValueError("density sweep values must be in [0, 3]")
        if abs(density - network_config.network_density) <= 1e-12:
            result[density] = primary_network
        else:
            result[density] = CaveNetworkGenerator(
                replace(network_config, network_density=density)
            ).generate(host)
    return result


def _pdc_calibration(path: Path):
    if not path.is_file():
        return None
    records = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            record = {}
            for key, value in row.items():
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = value
            records.append(record)
    return {"partition": "calibration", "records": records}


if __name__ == "__main__":
    raise SystemExit(main())
