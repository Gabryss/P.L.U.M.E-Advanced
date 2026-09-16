"""Durable evidence and plain-language diagnostics for unattended campaigns.

Receipts detect accidental changes to local results; they are not signatures for
untrusted downloads. Checkpoints and generated application scripts remain trusted
local artifacts.
"""

from __future__ import annotations

import errno
import html
import json
from pathlib import Path
from urllib.parse import quote

from plume_advanced.identity import sha256_file


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _ground_routes_failed(report: object) -> bool:
    """Read an explicit failure from a surface report or final recovery attempt."""
    if not isinstance(report, dict):
        return False
    ground = report.get("ground_traversal")
    if isinstance(ground, dict) and ground.get("passed") is False:
        return True
    attempts = report.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        return False
    inspection = attempts[-1].get("inspection")
    if not isinstance(inspection, dict):
        return False
    ground = inspection.get("ground_traversal")
    return isinstance(ground, dict) and ground.get("passed") is False


def diagnose(error: BaseException, stage: str) -> dict[str, str]:
    """Classify known conditions; unknown exceptions never become seed retries."""
    from plume_advanced.acceptance import AcceptanceError
    from plume_advanced.exporters.targets import ExportBudgetError
    from plume_advanced.exporters.texture_recovery import TextureRecoveryError
    from plume_advanced.pipeline.recovery import PipelineRecoveryError
    from plume_advanced.stages.network_quality import NetworkQualityError
    from plume_advanced.stages.network_systems import GenerationDomainError
    from plume_advanced.stages.surface_topology import SurfaceTopologyError

    category = "unexpected_error"
    action = (
        "Keep worker.log and the saved configuration. Reproduce this seed with the same "
        "version, run the regression suite, and investigate the first traceback. "
        "Changing seeds or weakening checks would hide the defect."
    )
    if isinstance(error, AcceptanceError):
        category = "acceptance_requirements"
        action = (
            "Read the named checks in pipeline_quality_report.json inspection or the preflight "
            "inspection. Failed or unavailable requirements block publication. Supply the missing "
            "validation capability or correct generation inputs in a new campaign; do not mark "
            "an unchecked requirement passed or silently downgrade the profile."
        )
    elif isinstance(error, GenerationDomainError):
        category = "host_domain"
        action = (
            "Enlarge host_field.grid or reduce the requested extent/source spacing in a "
            "new configuration. The requested tubes do not fit; seed retries cannot fix this."
        )
    elif isinstance(error, NetworkQualityError):
        category = "network_rejected"
        action = (
            "Read network_quality.json for the failed constraints. The bounded network "
            "candidate sequence is exhausted. Correct host/route constraints in a new "
            "campaign; retain this seed as a regression case."
        )
    elif isinstance(error, SurfaceTopologyError) and _ground_routes_failed(
        getattr(error, "report", None)
    ):
        category = "ground_routes_rejected"
        action = (
            "Read ground_traversal in pipeline_recovery.json or pipeline_quality_report.json. "
            "Inspect failed stations, motion intervals, floor slope/step measurements, attempted "
            "detours and the query budget. Required ground-route placement did not pass; a finite "
            "search failure does not establish that no route exists. Retain this case and its "
            "robot limits. Investigate the route plan or terrain inputs before a new campaign."
        )
    elif isinstance(error, PipelineRecoveryError):
        category = "surface_rejected"
        action = (
            "Read pipeline_recovery.json for localized defects, rejected local repairs and "
            "deterministic network replacements. The complete recovery budget is exhausted. "
            "Retain this case; compare finer resolution or corrected host/section constraints "
            "in a new campaign. Do not delete passages or relax acceptance gates."
        )
    elif isinstance(error, SurfaceTopologyError):
        category = "surface_rejected"
        action = (
            "Read resolution.json and worker.log. All permitted surface repairs failed. "
            "Test finer geometry.voxel_size in a new campaign and compare local clearances. "
            "Do not delete connected passages or bypass the topology gate."
        )
    elif isinstance(error, TextureRecoveryError):
        category = "texture_rejected"
        action = (
            "Read pipeline_quality_report.json inspection for the texture failure and attempted "
            "repairs. Restore missing/corrupt source maps or correct their declared normal "
            "convention. If serialized bindings or adapters still fail after rebuilding, retain "
            "the evidence and investigate the exporter. Changing the network seed cannot fix "
            "a material failure. Start a new campaign if source assets or configuration change."
        )
    elif isinstance(error, ExportBudgetError):
        category = "export_budget"
        action = (
            "The export exceeds an explicit simulation budget. Use a smaller extent or "
            "verified resolution in a new campaign, or change the budget only if the target "
            "simulator supports it. PLUME does not silently decimate passages."
        )
    elif isinstance(error, MemoryError):
        category = "memory_budget"
        action = (
            "Check peak_rss_mib and available RAM. Resume with a larger --memory-limit-mib "
            "only if the machine can support it, or use a smaller extent in a new campaign."
        )
    elif isinstance(error, OSError) and error.errno == errno.ENOSPC:
        category = "disk_space"
        action = (
            "Free disk space outside the retained evidence, then resume the same campaign. "
            "Partial checkpoints will be verified and rebuilt if necessary."
        )
    elif isinstance(error, (FileNotFoundError, PermissionError, ImportError)):
        category = "input_or_environment"
        action = (
            "Restore the named file, permission or installed dependency. Run --preflight. "
            "If input contents or dependencies change, start a new campaign."
        )
    elif stage == "configuration" and isinstance(error, ValueError):
        category = "configuration"
        action = "Correct the configuration value named in the error and start a new campaign."
    elif stage == "portable_validation":
        category = "export_validation"
        action = (
            "Read asset_checks.json. Correct the failed geometry, material or provenance "
            "check before importing. Preserve this export as regression evidence."
        )
    return dict(
        category=category,
        stage=stage,
        error_type=type(error).__name__,
        message=str(error),
        action=action,
    )


def process_diagnostic(category: str, message: str, stage: str = "worker") -> dict[str, str]:
    actions = {
        "timeout": "Inspect progress.jsonl for the last operation. Resume with a larger --timeout; validated geometry checkpoints are reused. No seed changes are needed.",
        "worker_crash": "Inspect worker.log and system resource reports. A native crash has no safe generic seed repair. Preserve this case and reproduce it after correcting the environment or code.",
        "interrupted": "Resume the same campaign. Completed results are verified and unfinished work resumes from compatible local stage checkpoints.",
        "artifact_integrity": "Saved evidence is missing or changed. Resume to rebuild the affected case; previous attempts are preserved. Do not treat the changed asset as validated.",
        "replay_mismatch": "Compare the original and replay identities and logs. This is a reproducibility failure; do not replace the seed or mark it passed.",
    }
    return dict(
        category=category,
        stage=stage,
        message=message,
        action=actions.get(category, "Inspect worker.log and retain the failed case."),
    )


def seal_result(output: Path, result: dict) -> dict:
    """Hash all delivered evidence, excluding this result and disposable progress state."""
    if result.get("status") == "passed":
        result["artifacts"] = {
            path.relative_to(output).as_posix(): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in {"result.json", "result.sha256", "worker.log", "progress.jsonl"}
            and not path.name.endswith(".tmp")
        }
        if not result["artifacts"]:
            raise ValueError("A passing worker must retain evidence")
    write_json(output / "result.json", result)
    (output / "result.sha256.tmp").write_text(sha256_file(output / "result.json"))
    (output / "result.sha256.tmp").replace(output / "result.sha256")
    return result


def read_result(output: Path) -> dict | None:
    """Read intact result metadata, including an earlier failure's diagnosis."""
    try:
        if (output / "result.sha256").read_text() != sha256_file(output / "result.json"):
            return None
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def verify_result(output: Path) -> tuple[dict | None, str | None]:
    """Return only successful, intact results. Never follow paths outside a case."""
    try:
        result = read_result(output)
        if result is None:
            return None, "Missing, changed or unreadable result receipt"
        if result.get("status") != "passed":
            return None, "Previous attempt did not pass"
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return None, "Successful result has no integrity receipt"
        for name, digest in artifacts.items():
            path = output / name
            if not path.resolve().is_relative_to(output.resolve()):
                return None, f"Receipt path escapes case: {name}"
            if not path.is_file() or sha256_file(path) != digest:
                return None, f"Missing or changed artifact: {name}"
        if not result.get("identity"):
            return None, "Successful result has no deterministic identity"
        return result, None
    except (OSError, ValueError, TypeError, AttributeError):
        return None, "Missing or unreadable result receipt"


def write_report(output: Path, summary: dict) -> None:
    """Offline report; all labels are escaped and all case links stay local."""
    rows = []
    for row in summary["cases"]:
        case = row["case"]
        label = (
            f"{Path(case['config']).stem} · {case.get('body') or 'preset'} · seed {case['seed']}"
        )
        links = []
        directory = row.get("directory")
        if directory and (output / directory).resolve().is_relative_to(output.resolve()):
            for name in (
                "result.json",
                "worker.log",
                "progress.jsonl",
                "network_quality.json",
                "resolution.json",
                "surface_quality.json",
                "asset_checks.json",
                "export/plume_cave.glb",
                "export/continuous_material/SETUP.txt",
            ):
                if (output / directory / name).is_file():
                    links.append(
                        f'<a href="{quote(directory + "/" + name)}">{html.escape(name)}</a>'
                    )
        replay = row.get("replay_directory")
        if replay and (output / replay).resolve().is_relative_to(output.resolve()):
            links.append(f'<a href="{quote(replay + "/result.json")}">Replay result</a>')
        diagnostic = row.get("diagnostic", {})
        warnings = row.get("warnings", [])
        detail = diagnostic.get("message", "")
        action = diagnostic.get("action", "")
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(detail)}<p>{html.escape(action)}</p>"
            + "".join(f'<p class="warning">{html.escape(w)}</p>' for w in warnings)
            + f"<small>{' · '.join(links)}</small></td></tr>"
        )
    status = (
        "Checks passed"
        if summary["passed"]
        else ("Checks failed" if summary["complete"] else "Incomplete — safe to resume")
    )
    reason = summary.get("campaign_error", "")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>PLUME reliability report</title><style>
body{{font:16px system-ui;max-width:1250px;margin:3rem auto;padding:0 1rem;background:#fafafa;color:#17212b}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:1rem;text-align:left;vertical-align:top;border-bottom:1px solid #ccd}}
small{{line-height:2}}a{{color:#075e9e}}.warning{{color:#744400}}code{{overflow-wrap:anywhere}}
</style><h1>{status}</h1><p>{html.escape(reason)}</p>
<p>{len(summary["cases"])} / {summary["planned_cases"]} cases recorded.
Fresh-process replay: {"required" if summary["replay"] else "NOT CHECKED"}.</p>
<p>Passing means the recorded numerical and portable-asset checks passed. It is not a geological,
continuous walkability, self-intersection or native Unity/Unreal certification. Native Blender
renders and visual review are separate. Warnings need inspection even when checks pass.</p>
<p>Resume or refresh this report: <code>plume-check --output {html.escape(str(output))} --resume</code>.
Recheck saved evidence without generation: add <code>--report-only</code>.</p>
<table><thead><tr><th>Case</th><th>Status</th><th>Evidence and next action</th></tr></thead><tbody>
{"".join(rows)}</tbody></table><p>Previous attempts and their logs are preserved beside each case.</p>
<p>Source: <code>{html.escape(summary["source_sha256"])}</code></p></html>"""
    temporary = output / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "report.html")
