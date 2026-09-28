"""Retain fully checked, over-budget packages and publish them after a size-only review."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from plume_advanced.identity import sha256_file
from plume_advanced.progress import report_progress

from .atomic import atomic_output_directory
from .errors import ExportBudgetError

RECEIPT = "size_recovery.json"
SCHEMA = "plume.size-recovery.v1"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _files(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Recovery packages must not contain symbolic links")
    return {path.relative_to(root).as_posix(): path for path in paths if path.is_file()}


def _require_checked_package(root: Path) -> dict:
    inspection = json.loads((root / "pipeline_inspection.json").read_text())
    if (inspection.get("passed") is not True
            or any(inspection.get(key, {}).get("passed") is not True
                   for key in ("raw", "visual", "textures", "serialized", "acceptance"))):
        raise ValueError("Size recovery requires completed geometry, texture and acceptance checks")
    return inspection


def preserve_size_rejected_package(staging: Path, destination: Path, *, primary_asset: Path,
                                   target: str, max_asset_bytes: int) -> Path:
    """Detach from transaction cleanup only after every non-size check passed.

    Move first so an interrupted checksum operation still leaves the files.
    An incomplete receipt cannot be recovered automatically.
    """
    _require_checked_package(staging)
    primary = primary_asset.relative_to(staging).as_posix()
    retained = Path(tempfile.mkdtemp(prefix=f"{destination.name}.size-rejected-",
                                    dir=staging.parent))
    retained.rmdir()
    staging.replace(retained)
    inventory = {}
    files = _files(retained)
    for index, (name, path) in enumerate(sorted(files.items())):
        report_progress("Preserve checked export", index, len(files),
                        f"{name}: hashing {path.stat().st_size / 1048576:,.1f} MiB")
        inventory[name] = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
    receipt = dict(schema=SCHEMA, status="size_rejected", primary_asset=primary,
                   target=target, original_max_asset_bytes=max_asset_bytes, files=inventory)
    _write_json(retained / RECEIPT, receipt)
    report_progress("Checked export preserved", len(files), len(files), str(retained))
    return retained


def recover_size_rejected_package(package: Path, destination: Path, *, max_asset_bytes: int) -> Path:
    """Change only the byte limit; verify all saved bytes and retain the original copy."""
    if type(max_asset_bytes) is not int or max_asset_bytes < 0:
        raise ValueError("max_asset_bytes must be a nonnegative integer")
    package, destination = package.resolve(), destination.resolve()
    if (destination.exists() or destination.is_relative_to(package)
            or package.is_relative_to(destination)):
        raise ValueError("Recovery needs a new output directory separate from the retained package")
    files = _files(package)
    receipt = json.loads((package / RECEIPT).read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "size_rejected":
        raise ValueError("No complete size-only recovery receipt")
    inventory = receipt.get("files", {})
    if not inventory or set(files) != {*inventory, RECEIPT}:
        raise ValueError("Retained package file inventory changed")
    for name in inventory:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Invalid recovery package path")
    if receipt.get("primary_asset") not in inventory:
        raise ValueError("Recovery receipt has no primary asset")
    oversized = {name: path.stat().st_size for name, path in files.items()
                 if name != RECEIPT and max_asset_bytes and path.stat().st_size > max_asset_bytes}
    if oversized:
        raise ExportBudgetError(f"Export file size budget still exceeded ({max_asset_bytes:,} bytes): {oversized}",
            report=dict(passed=False, failures=[f"Size budget still exceeded: {oversized}"],
                        preserved_package=str(package)))
    for index, (name, expected) in enumerate(sorted(inventory.items())):
        report_progress("Verify retained export", index, len(inventory), name)
        path = files[name]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Retained package file changed: {name}")
    inspection = _require_checked_package(package)
    if inspection["acceptance"]["policy"].get("require_export_budgets") and not max_asset_bytes:
        raise ValueError("The saved acceptance policy requires a finite file size budget")
    with atomic_output_directory(destination) as staging:
        for index, name in enumerate(sorted(inventory)):
            report_progress("Copy checked export", index, len(inventory), name)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(files[name], target)
            # Check the copied bytes, too: a source changed during recovery must
            # never be published using an earlier checksum.
            if sha256_file(target) != inventory[name]["sha256"]:
                raise ValueError(f"Retained package changed while copying: {name}")
        limits = inspection["acceptance"]["checks"]["export_budgets"].get("limits")
        if limits is not None:
            limits["asset_bytes"] = max_asset_bytes
        _write_json(staging / "pipeline_inspection.json", inspection)
        provenance_path = staging / "reexport.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            provenance["effective_export"]["max_asset_bytes"] = max_asset_bytes
            provenance["overrides"]["max_asset_bytes"] = max_asset_bytes
            _write_json(provenance_path, provenance)
        recovery = dict(schema="plume.export-recovery.v1", source=str(package),
                        original_max_asset_bytes=receipt["original_max_asset_bytes"],
                        max_asset_bytes=max_asset_bytes, verified_files=inventory,
                        scope="Unchanged checked assets; only the byte limit and package metadata updated. No new native-engine qualification.")
        _write_json(staging / "export_recovery.json", recovery)
        if receipt["target"] == "all":
            manifest_path = staging / receipt["primary_asset"]
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault("shared_files", []).append("export_recovery.json")
            _write_json(manifest_path, manifest)
        sizes = {name: path.stat().st_size for name, path in _files(staging).items()
                 if name != "export_size_report.json"}
        report_path = staging / "export_size_report.json"
        size_report = json.loads(report_path.read_text())
        size_report.update(passed=True, oversized_files={}, files=sizes, total_file_bytes=sum(sizes.values()))
        size_report["limits"]["asset_bytes"] = max_asset_bytes
        _write_json(report_path, size_report)
        if max_asset_bytes and any(path.stat().st_size > max_asset_bytes for path in _files(staging).values()):
            raise ExportBudgetError("Updated package metadata exceeds the new size budget; retained source unchanged")
        report_progress("Publish recovered export", detail=str(destination))
    return destination / receipt["primary_asset"]
