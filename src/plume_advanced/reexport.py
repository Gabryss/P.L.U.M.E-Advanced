"""Re-export an accepted local cave checkpoint without regenerating its geometry."""

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

from plume_advanced.cli import SOURCE_ROOT, _run_inputs
from plume_advanced.config import load_project_config
from plume_advanced.evaluation.local_geometry import section_resolution_report
from plume_advanced.exporters import export_target_asset
from plume_advanced.exporters.errors import ExportBudgetError
from plume_advanced.exporters.package_recovery import recover_size_rejected_package
from plume_advanced.pipeline import StageCheckpointStore, pipeline_fingerprint
from plume_advanced.progress import (
    GenerationTimeBudgetError,
    TerminalProgress,
    report_progress,
    work_budget,
)


def _checkpoint_store(root: Path, fingerprint: str) -> StageCheckpointStore:
    """Honor a local, explicitly verified compatibility receipt for export only.

    Ordinary generation/resume still requires exact source identity. A receipt
    binds one reviewed source update to specific unchanged checkpoint payloads;
    subsequent code/config edits cannot silently reuse it.
    """
    receipt_path = root / "reexport_compatibility.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") == "plume.reexport-compatibility.v1"
                and receipt.get("reexport_fingerprint") == fingerprint):
            source_fingerprint = receipt["source_fingerprint"]
            checkpoints = receipt["checkpoints"]
            if (len(checkpoints) != 2 or "accepted_base" not in checkpoints
                    or sum(name.startswith("final_geometry_") for name in checkpoints) != 1):
                raise ValueError("Invalid re-export compatibility checkpoint list")
            for stage, digest in checkpoints.items():
                if not all(character.isalnum() or character == "_" for character in stage):
                    raise ValueError("Invalid re-export compatibility checkpoint name")
                metadata = json.loads((root / f"{stage}.json").read_text())
                if (metadata.get("fingerprint") != source_fingerprint
                        or metadata.get("payload_sha256") != digest):
                    raise ValueError("No matching checkpoint for the re-export compatibility receipt")
            return StageCheckpointStore(root, source_fingerprint)
    return StageCheckpointStore(root, fingerprint)


def _time_limit(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative; 0 means no timer")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Unchanged original generation recipe")
    parser.add_argument("--source", type=Path, help="Generation output directory")
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    parser.add_argument("--recover-package", type=Path,
                        help="Publish a retained, checked size-rejected package with a revised byte limit")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--body", choices=("earth", "mars", "moon"))
    parser.add_argument("--max-visual-triangles", type=int)
    parser.add_argument("--max-asset-bytes", type=int)
    parser.add_argument("--visual-max-error-m", type=float)
    parser.add_argument("--max-seconds", type=_time_limit, default=0.0,
                        help="Optional export time limit; default 0 has no timer, independent of the generation recipe")
    parser.add_argument("--debug", action="store_true", help="Include tracebacks for budget failures")
    args = parser.parse_args(argv)
    if args.recover_package:
        if any(getattr(args, name) is not None for name in
               ("config", "source", "checkpoint_directory", "body", "max_visual_triangles", "visual_max_error_m")):
            parser.error("--recover-package changes only the byte limit; omit generation/configuration options")
        if args.max_asset_bytes is None:
            parser.error("--recover-package requires an explicit --max-asset-bytes limit")
    elif args.config is None or args.source is None:
        parser.error("Re-export requires --config and --source, or use --recover-package")
    if args.output.exists() or (args.source and args.source.resolve().is_relative_to(args.output.resolve())):
        raise ValueError("Re-export needs a new output directory separate from the source/checkpoints")
    try:
        with work_budget(args.max_seconds):
            trace_path = args.output.with_name(args.output.name + ".progress.jsonl")
            progress = TerminalProgress(total_stages=1, trace_path=trace_path)
            progress.log(f"Export progress log: {trace_path}")
            timer = f"{args.max_seconds:g}s time limit" if args.max_seconds else "no time limit"
            progress.start("Recover export" if args.recover_package else "Re-export", timer)
            if args.recover_package:
                recover_size_rejected_package(args.recover_package, args.output,
                                              max_asset_bytes=args.max_asset_bytes)
                progress.finish("checked asset bytes reused; revised file size budget passed")
            else:
                _reexport(args)
                progress.finish("accepted geometry reused; export and qualification checks repeated")
    except ExportBudgetError as error:
        if args.debug:
            raise
        print(str(error), file=sys.stderr)
        if action := error.report.get("repair_action"):
            print(action, file=sys.stderr)
        return 2
    except GenerationTimeBudgetError:
        if args.debug:
            raise
        print(f"Export stopped at the explicit --max-seconds limit ({args.max_seconds:g}s). "
              "The source geometry checkpoints are unchanged. Omit --max-seconds to retry without a timer.",
              file=sys.stderr)
        return 2
    finally:
        TerminalProgress.close_active()
    return 0


def _reexport(args):
    manifest = json.loads((args.source / "run_manifest.json").read_text())
    seed = manifest["resolved_config"]["procedural_seed"]
    project = load_project_config(args.config, world_body=args.body, seed_override=seed)
    fingerprint = pipeline_fingerprint(project, inputs=_run_inputs(args.config, project), source_root=SOURCE_ROOT)
    checkpoint_root = args.checkpoint_directory or args.source / ".plume-checkpoints"
    if checkpoint_root.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Re-export must not replace the source checkpoint directory")
    store = _checkpoint_store(checkpoint_root, fingerprint)
    overrides = {key: getattr(args, key) for key in ("max_visual_triangles", "max_asset_bytes", "visual_max_error_m")
                 if getattr(args, key) is not None}
    export = replace(project.export, **overrides)
    report_progress("Load base checkpoint", detail="verifying hash and loading accepted geometry; no internal counter")
    accepted = store.load("accepted_base")
    if accepted is None:
        raise ValueError("No matching accepted base checkpoint; keep the original recipe, assets, code and runtime for re-export")
    final_stage = "final_geometry_"+accepted.context_sha256[:16]
    sections = accepted.sections
    accepted_identity = accepted.report["accepted_identity"]
    # Retain only small section/identity records before loading the final mesh.
    del accepted
    report_progress("Load final checkpoint", detail=f"{final_stage}: verifying hash and loading geometry; no internal counter")
    geometry = store.load(final_stage)
    if geometry is None:
        raise ValueError("No matching final geometry checkpoint; generation must finish geometry before re-export")
    provenance = dict(schema="plume.reexport.v1",
        source=str(args.source.resolve()), source_fingerprint=store.fingerprint,
        reexport_fingerprint=fingerprint, max_seconds=args.max_seconds,
        root_seed=seed, accepted_identity=accepted_identity,
        original_export=manifest["resolved_config"]["export"], effective_export=asdict(export),
        overrides=overrides)
    export_target_asset(geometry, export, args.output, acceptance=project.acceptance,
        resolution=section_resolution_report(sections, geometry.config.voxel_size),
        reexport_provenance=provenance)


if __name__ == "__main__":
    raise SystemExit(main())
