"""Shared generation helpers that stop before unnecessary expensive stages."""

from __future__ import annotations

from dataclasses import replace

from plume_advanced.config import ProjectConfig, project_config_manifest
from plume_advanced.evaluation.provenance import semantic_hash
from plume_advanced.stages.host_field import HostField, HostFieldGenerator, RoutingWeights
from plume_advanced.stages.network import CaveNetwork, CaveNetworkGenerator
from plume_advanced.stages.section_field import SectionField, SectionFieldGenerator
from plume_advanced.world import derive_stage_seeds


def for_seed(project: ProjectConfig, seed: int) -> ProjectConfig:
    stage_seeds = derive_stage_seeds(seed)
    return replace(
        project,
        procedural_seed=seed,
        stage_seeds=stage_seeds,
        host_field=replace(project.host_field, random_seed=stage_seeds.host),
        network=replace(project.network, random_seed=stage_seeds.network),
        section_field=replace(project.section_field, random_seed=stage_seeds.sections),
        events=replace(project.events, random_seed=stage_seeds.events),
        geometry=replace(project.geometry, random_seed=stage_seeds.geometry),
    )


def with_routing_condition(project: ProjectConfig, condition: str) -> ProjectConfig:
    weights = project.host_field.routing_weights
    if condition == "full":
        resolved = weights
    elif condition == "unconditioned":
        resolved = RoutingWeights(enabled=False)
    elif condition.startswith("no_"):
        resolved = weights.without(condition.removeprefix("no_"))
    else:
        raise ValueError(f"Unknown host-ablation condition: {condition}")
    return replace(project, host_field=replace(project.host_field, routing_weights=resolved))


def generate_host(project: ProjectConfig) -> HostField:
    return HostFieldGenerator(project.host_field).generate()


def generate_network(project: ProjectConfig) -> tuple[HostField, CaveNetwork]:
    host = generate_host(project)
    return host, CaveNetworkGenerator(project.network).generate(host)


def generate_sections(project: ProjectConfig) -> tuple[HostField, CaveNetwork, SectionField]:
    host, network = generate_network(project)
    return host, network, SectionFieldGenerator(project.section_field).generate(network)


def config_hash(project: ProjectConfig) -> str:
    return semantic_hash(project_config_manifest(project))


__all__ = [
    "config_hash",
    "for_seed",
    "generate_host",
    "generate_network",
    "generate_sections",
    "with_routing_condition",
]
