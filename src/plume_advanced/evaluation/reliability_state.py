"""Campaign identity, preflight and process locks shared by the reliability CLI."""

from __future__ import annotations

import json
import math
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from plume_advanced.identity import runtime_identity, sha256_file

if TYPE_CHECKING:
    from plume_advanced.config import ProjectConfig
    from plume_advanced.evaluation.reliability import ReliabilityCase


def project_inputs(project: ProjectConfig, config: Path) -> tuple[Path, ...]:
    """Include configured external inputs, not just the TOML that names them."""
    files = [config.resolve()]
    for name in (
        "cave_diffuse_texture",
        "cave_normal_texture",
        "cave_roughness_texture",
        "cave_displacement_texture",
    ):
        value = getattr(project.geometry, name)
        if value:
            files.append(Path(value).resolve())
    if project.events.enabled and project.events.include_rock_props:
        for name in ("rocky_source_path", "rocky_texture_dir"):
            value = getattr(project.events, name)
            if value:
                path = Path(value).resolve()
                files.extend(
                    p
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and not any(s in p.parts for s in (".git", "__pycache__"))
                )
                if not path.exists():
                    files.append(path)
    root = Path(__file__).resolve().parents[3]
    files.extend(p for p in (root / "uv.lock", root / "pyproject.toml") if p.is_file())
    assets = Path(__file__).resolve().parents[1] / "material_assets"
    files.extend(
        p for p in sorted(assets.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    )
    return tuple(sorted(set(files)))


def plan_identity(cases: list[ReliabilityCase], *, source: str, replay: bool) -> dict:
    """Missing/invalid inputs remain explicit cases, never silently omitted."""
    from plume_advanced.config import load_project_config

    inputs = {}
    for case in cases:
        config = Path(case.config).resolve()
        paths: tuple[Path, ...] = (config,)
        try:
            project = load_project_config(config, world_body=case.body, seed_override=case.seed)
            paths = project_inputs(project, config)
        except (OSError, ValueError):
            pass  # The worker records the precise configuration failure.
        for path in paths:
            if str(path) not in inputs:
                inputs[str(path)] = sha256_file(path) if path.is_file() else None
    return dict(
        schema="plume.reliability-plan.v1",
        cases=[asdict(c) for c in cases],
        source_sha256=source,
        runtime=runtime_identity(),
        inputs=inputs,
        replay=replay,
    )


def preflight(cases: list[ReliabilityCase], output: Path | None = None) -> dict:
    from plume_advanced.acceptance import require_available_acceptance
    from plume_advanced.config import load_project_config
    from plume_advanced.evaluation.reliability_reports import diagnose

    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            if case.voxel_size is not None and (
                not math.isfinite(case.voxel_size) or case.voxel_size <= 0
            ):
                raise ValueError("voxel size must be finite and positive")
            config = Path(case.config)
            project = load_project_config(config, world_body=case.body, seed_override=case.seed)
            if not project.network.quality.enabled:
                raise ValueError("Reliability checks require network.quality.enabled = true")
            for path in project_inputs(project, config):
                if not path.is_file():
                    raise FileNotFoundError(path)
            warnings = []
            if case.scope == "full":
                require_available_acceptance(project.acceptance)
                voxel = case.voxel_size or project.geometry.voxel_size
                warnings.append(
                    f"Voxel size {voxel:g} m: passage resolution is measured after sections; configuration alone cannot certify it."
                )
                if any(str(p).lower().endswith(".exr") for p in project_inputs(project, config)):
                    if not shutil.which("convert"):
                        raise FileNotFoundError(
                            "ImageMagick convert is required for configured EXR textures"
                        )
                if not project.export.max_visual_triangles or not project.export.max_asset_bytes:
                    warnings.append(
                        "Some export budgets are disabled. Set export.max_visual_triangles and max_asset_bytes to your simulation limits."
                    )
            elif project.acceptance.profile != "research":
                warnings.append("Stage-only evaluation: the full acceptance profile was not evaluated.")
            records.append(dict(case=asdict(case), status="passed", warnings=warnings,
                                acceptance_policy=asdict(project.acceptance)))
        except Exception as error:
            records.append(
                dict(
                    case=asdict(case), status="failed", diagnostic=diagnose(error, "configuration"),
                    inspection=getattr(error, "report", None),
                )
            )
    free = None
    if output:
        parent = output.resolve()
        while not parent.exists():
            parent = parent.parent
        free = shutil.disk_usage(parent).free
    return dict(
        passed=all(r["status"] == "passed" for r in records),
        cases=records,
        available_disk_bytes=free,
        runtime=runtime_identity(),
        scope="Configuration and input availability only; no geometry or native engine import checked.",
    )


@contextmanager
def campaign_lock(output: Path):
    """OS lock releases on crash; a leftover file is harmless, not a stale lock."""
    with (output / ".campaign.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another process is using this campaign") from error
        else:
            import fcntl

            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is using this campaign") from error
        try:
            yield
        finally:
            if sys.platform == "win32":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def read_plan(output: Path) -> dict:
    try:
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if plan["schema"] != "plume.reliability-plan.v1" or not plan["cases"]:
            raise ValueError("Unknown plan schema or empty campaign")
        return plan
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("No valid resumable campaign plan; use a new output directory") from error
