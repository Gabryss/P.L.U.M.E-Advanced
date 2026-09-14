"""Archive compact final campaign evidence; leave meshes, projects and checkpoints in outputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = Path(__file__).resolve().parent
    manifest = []

    def copy(source, destination):
        if not source.is_file():
            return
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(
            dict(
                source=str(source.relative_to(root)),
                archived=str(destination),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )

    for name in [
        "audit.json",
        "native_output_receipts.json",
        "native_campaign.json",
        "native_campaign_v1.json",
        "run_native_campaign.py",
        "tests.log",
        "material_tests.log",
        "blender_import_tests.log",
        "native_runner_tests_v2.log",
        "lint.log",
        "final_changed_files_lint.log",
        "repository_lint.log",
        "types.log",
        "texture_test_inventory.log",
    ]:
        copy(root / name, Path(name))
    protocol = json.loads((root / "protocol_v2.json").read_text())
    for group in ["single", *protocol["groups"]]:
        for name in ["summary.json", "plan.json"]:
            copy(root / group / name, Path(group) / name)
        for attempt in sorted((root / group).glob("*/attempt_0000")):
            for name in [
                "result.json",
                "result.sha256",
                "pipeline_quality_report.json",
                "pipeline_recovery.json",
                "network_quality.json",
                "texture_recovery.json",
                "export_topology.json",
                "asset_checks.json",
                "timings.json",
                "resolved_config.json",
            ]:
                copy(attempt / name, attempt.relative_to(root) / name)
    native = json.loads((root / "native_campaign.json").read_text())
    for row in native["cases"]:
        if "output" not in row:
            continue
        folder = Path(row["output"])
        for name in [
            "native_summary.json",
            "native_input_receipt.json",
            "unity_project/native_result.json",
            "unreal_run_01/native_result.json",
            "unity_project/bootstrap.json",
        ]:
            copy(folder / name, folder.relative_to(root) / name)
    (out / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived {len(manifest)} compact evidence files")


if __name__ == "__main__":
    main()
