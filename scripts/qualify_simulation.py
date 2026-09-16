#!/usr/bin/env python3
"""Generate/check a short 4K cave and publish a native-checked simulation package.

The working generation and editor projects remain evidence. Only ready/ is a
delivery: it appears after numerical checks and all requested editor checks pass.
Requires a source checkout and the editors supported by check_native_engines.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from plume_advanced.config import load_project_config
from plume_advanced.evaluation.reliability_reports import verify_result
from plume_advanced.exporters.atomic import atomic_output_directory
from plume_advanced.identity import package_source_hash, sha256_file

REPO = Path(__file__).resolve().parents[1]


def write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def checked_inputs(run: Path) -> dict:
    result, error = verify_result(run)
    if error or result is None:
        raise ValueError(f"Generation receipt failed: {error}")
    quality_path = run / "pipeline_quality_report.json"
    quality = json.loads(quality_path.read_text())
    acceptance = quality.get("acceptance", {})
    if not quality.get("passed") or not acceptance.get("passed"):
        raise ValueError("Generation has no passing acceptance report")
    if acceptance.get("policy", {}).get("profile") != "simulation":
        raise ValueError("Qualification requires the simulation acceptance profile")
    required = ("mesh", "clearance", "collision", "resolution", "relief",
                "texture_integrity", "pbr_textures", "export_budgets")
    if acceptance.get("policy", {}).get("require_ground_routes"):
        required += ("ground_routes",)
    if any(acceptance.get("checks", {}).get(key, {}).get("status") != "passed"
           for key in required):
        raise ValueError("Simulation, relief and full PBR checks must all pass")
    package = run / "export"
    receipts = quality.get("export_inspection", {}).get("serialized", {}).get("files", [])
    if not receipts:
        raise ValueError("Export serialization receipts are missing")
    for row in receipts:
        path = package / row["path"]
        if (not path.resolve().is_relative_to(package.resolve()) or not row.get("passed")
                or sha256_file(path) != row["sha256"]):
            raise ValueError(f"Export changed since inspection: {row['path']}")
    textures = quality.get("export_inspection", {}).get("textures", {})
    attempts = textures.get("package_attempts", [])
    if not textures.get("passed") or not attempts or not attempts[-1].get("passed"):
        raise ValueError("Texture package evidence is missing")
    for row in attempts[-1].get("checked_files", []):
        path = package / row["path"]
        if (not path.resolve().is_relative_to(package.resolve())
                or sha256_file(path) != row["sha256"]):
            raise ValueError(f"Material package changed since inspection: {row['path']}")
    # Snapshot every material/helper file too, so an edit during the native check
    # cannot be carried into the delivered copy under an older receipt.
    files = {str(p.relative_to(run)): sha256_file(p)
             for p in sorted(package.rglob("*")) if p.is_file()}
    expected_files = {name for name in result["artifacts"] if name.startswith("export/")}
    if set(files) != expected_files:
        raise ValueError("Export contains files not covered by the generation receipt")
    for name in ("pipeline_quality_report.json", "resolved_config.json", "stage_b_network.json",
                 "stage_c_sections.json", "stage_c_sections.npz"):
        files[name] = sha256_file(run / name)
    for name in ("case.json", "pipeline_recovery.json"):
        if name in result["artifacts"]:
            files[name] = sha256_file(run / name)
    return files


def checked_replay(run: Path) -> dict:
    """Require the campaign's independently executed, intact cold replay."""
    campaign = run.parent.parent
    try:
        summary = json.loads((campaign / "summary.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Cold replay campaign evidence is missing or unreadable") from error
    rows = [row for row in summary.get("cases", [])
            if (campaign / row.get("directory", "")).resolve() == run.resolve()]
    if len(rows) != 1 or rows[0].get("status") != "passed" or rows[0].get("replay_passed") is not True:
        raise ValueError("A passing cold replay is required for this generation")
    relative = rows[0].get("replay_directory")
    if not isinstance(relative, str):
        raise ValueError("Cold replay directory is missing")
    replay = (campaign / relative).resolve()
    if (not run.parent.name.startswith("case_")
            or not replay.is_relative_to(campaign.resolve())
            or replay.parent.name != run.parent.name.replace("case_", "replay_", 1)
            or replay == run.resolve()):
        raise ValueError("Cold replay must be a separate attempt in the same campaign")
    original, error = verify_result(run)
    repeated, replay_error = verify_result(replay)
    if error or replay_error or original is None or repeated is None:
        raise ValueError(f"Cold replay integrity failed: {error or replay_error}")
    for name in ("identity", "case", "source_sha256", "runtime"):
        if not original.get(name) or original[name] != repeated.get(name):
            raise ValueError(f"Cold replay {name} does not match the original generation")
    if repeated.get("python_hash_seed") != 37 or original.get("python_hash_seed") != 11:
        raise ValueError("Cold replay must use the campaign's independent hash seed")
    return dict(passed=True, directory=str(replay), receipt_sha256=sha256_file(replay / "result.json"),
                identity=repeated["identity"], source_sha256=repeated["source_sha256"],
                runtime=repeated["runtime"], python_hash_seed=37)


def run_command(command: list[str], log: Path, timeout: float) -> None:
    print(f"Checking {log.stem}; details in {log}", flush=True)
    with log.open("x") as stream:
        with subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                # The native adapter catches KeyboardInterrupt and kills its
                # separately grouped editor/compiler workers before exiting.
                # SIGTERM would bypass that cleanup and could leave an editor.
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    if code:
        raise RuntimeError(f"{log.stem} exited {code}; inspect {log}")


def qualify(run: Path, output: Path, editors: dict[str, Path], timeout: float) -> dict:
    before = checked_inputs(run)
    replay = checked_replay(run)
    generation_receipt_hash = sha256_file(run / "result.json")
    generation = json.loads((run / "result.json").read_text())
    ground_required = json.loads((run / "pipeline_quality_report.json").read_text()).get(
        "acceptance", {}).get("policy", {}).get("require_ground_routes", False)
    source = package_source_hash()
    controller_hash = sha256_file(Path(__file__))
    adapter = REPO / "scripts/check_native_engines.py"
    adapter_hash = sha256_file(adapter)
    fixtures = {name: sha256_file(REPO / "tests/fixtures" / name)
                for name in ("unity/PlumeNativeCheck.cs", "unreal/native_check.py")}
    command = [sys.executable, str(adapter), str(run), "--output", str(output / "native"),
               "--timeout", str(timeout)]
    for name, editor in editors.items():
        command += [f"--{name}", str(editor)]
    run_command(command, output / "native.log", timeout * (2 * len(editors) + 1) + 60)
    summary_path = output / "native/native_summary.json"
    native = json.loads(summary_path.read_text())
    if not native.get("passed") or any(
        native.get("checks", {}).get(name, {}).get("native", {}).get("passed") is not True
        or native.get("checks", {}).get(name, {}).get("body_validation", {}).get("passed") is not True
        or (ground_required and native.get("checks", {}).get(name, {}).get("ground_validation", {}).get("passed") is not True)
        for name in editors
    ):
        raise ValueError("Every requested native editor must pass")
    receipt = json.loads((output / "native/native_input_receipt.json").read_text())
    if (receipt.get("glb_sha256") != before["export/plume_cave.glb"]
            or receipt.get("quality_sha256") != before["pipeline_quality_report.json"]
            or receipt.get("collision_obj_sha256") != before["export/plume_cave_collision.obj"]
            or receipt.get("fixture_sha256") != fixtures
            or receipt.get("adapter_source") != source):
        raise ValueError("Native receipt belongs to different simulation inputs")
    if (checked_inputs(run) != before or package_source_hash() != source
            or checked_replay(run) != replay
            or sha256_file(run / "result.json") != generation_receipt_hash
            or sha256_file(Path(__file__)) != controller_hash
            or sha256_file(adapter) != adapter_hash
            or any(sha256_file(REPO / "tests/fixtures" / name) != digest
                   for name, digest in fixtures.items())):
        raise ValueError("Inputs or validation code changed during qualification")
    with atomic_output_directory(output / "ready") as staging:
        for relative in before:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / relative, target)
            if sha256_file(target) != before[relative]:
                raise ValueError(f"Input changed while copying: {relative}")
        shutil.copyfile(summary_path, staging / "native_summary.json")
        shutil.copyfile(output / "native/native_input_receipt.json", staging / "native_input_receipt.json")
        shutil.copyfile(Path(replay["directory"]) / "result.json", staging / "replay_result.json")
        if sha256_file(staging / "replay_result.json") != replay["receipt_sha256"]:
            raise ValueError("Cold replay receipt changed while copying")
        record = dict(schema="plume.simulation-delivery.v1", passed=True,
            source_sha256=source, native_adapter_sha256=adapter_hash,
            generation=dict(source_sha256=generation.get("source_sha256"),
                            runtime=generation.get("runtime"), case=generation.get("case"),
                            identity=generation.get("identity"),
                            receipt_sha256=generation_receipt_hash),
            qualification_source_sha256=controller_hash,
            ground_robot_checked=ground_required,
            cold_replay=replay,
            editors=list(editors), files=before,
            native_summary_sha256=sha256_file(staging / "native_summary.json"),
            native_input_receipt_sha256=sha256_file(staging / "native_input_receipt.json"),
            scope="Numerical simulation profile plus imported material and cooked finite-body collision checks. "
                  "Interior cavity asset; no grounded robot dynamics, sensor calibration or frame-time guarantee.")
        write(staging / "simulation_ready.json", record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", type=Path, help="Generate and cold-replay this simulation scenario")
    inputs.add_argument("--run", type=Path, help="Existing passing plume-check attempt directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New qualification directory")
    parser.add_argument("--unity", type=Path)
    parser.add_argument("--unreal", type=Path)
    parser.add_argument("--generation-timeout", type=int, default=7200)
    parser.add_argument("--memory-limit-mib", type=int, default=24576)
    parser.add_argument("--editor-timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    editors = {key: value.resolve() for key in ("unity", "unreal")
               if (value := getattr(args, key)) is not None}
    if not editors or any(not p.is_file() for p in editors.values()):
        parser.error("Specify at least one installed Unity or Unreal editor")
    if min(args.generation_timeout, args.memory_limit_mib, args.editor_timeout) <= 0:
        parser.error("Resource limits must be positive")
    if not 0 <= args.seed <= 2**32-1:
        parser.error("Seed must be an unsigned 32-bit integer")
    if args.config:
        policy = load_project_config(args.config).acceptance
        if policy.profile != "simulation" or not policy.require_textures:
            parser.error("Use simulation acceptance with require_textures=true")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.config:
            campaign = output / "generation"
            run_command([sys.executable, "-m", "plume_advanced.evaluation.reliability",
                "--configs", str(args.config.resolve()), "--seeds", str(args.seed),
                "--scope", "full", "--timeout", str(args.generation_timeout),
                "--memory-limit-mib", str(args.memory_limit_mib), "--output", str(campaign)],
                output / "generation.log", 2 * args.generation_timeout + 120)
            run = campaign / "case_0000/attempt_0000"
        else:
            run = args.run.resolve()
        result = qualify(run, output, editors, args.editor_timeout)
    except (Exception, KeyboardInterrupt) as error:
        result = dict(passed=False, error_type=type(error).__name__, error=str(error))
    result["elapsed_s"] = time.monotonic() - started
    write(output / "qualification.json", result)
    print(f"Simulation {'ready' if result['passed'] else 'rejected'}: {output}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
