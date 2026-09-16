"""Small user recipes over shared, versioned generation presets.

Recipes expand in memory into the same strict schema used by scientific
experiments. They never edit a second configuration file. Relative asset paths
always belong to the user's recipe, not to the installed package.
"""

from __future__ import annotations

import copy
import json
from importlib.resources import files
from typing import Any


def _catalog() -> dict[str, Any]:
    return json.loads(files("plume_advanced").joinpath("presets.json").read_text())


def available_presets() -> tuple[str, ...]:
    """Return stable preset names without importing generation stages."""
    return tuple(sorted(_catalog()["presets"]))


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand_recipe(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand a recipe, or return an ordinary advanced configuration unchanged.

    Tables merge recursively; scalar values and arrays replace whole values.
    All stage keys are subsequently checked by the schema-4 configuration loader.
    """
    if "recipe_version" not in raw:
        return copy.deepcopy(raw)
    if type(raw["recipe_version"]) is not int or raw["recipe_version"] != 1:
        raise ValueError("recipe_version must be the integer 1")
    if "schema_version" in raw:
        raise ValueError("Use recipe_version or schema_version, not both")
    catalog = _catalog()
    name = raw.get("preset")
    if not isinstance(name, str) or name not in catalog["presets"]:
        raise ValueError("Unknown recipe preset; choose: " + ", ".join(sorted(catalog["presets"])))
    preset = catalog["presets"][name]
    settings = _merge(catalog["common"], catalog["families"][preset["family"]])
    settings = _merge(settings, preset["settings"])
    overrides = {
        key: value for key, value in raw.items() if key not in {"recipe_version", "preset"}
    }
    # Changing the body must also select its material unless explicitly supplied.
    world = overrides.get("world")
    if isinstance(world, dict) and "body" in world and "material" not in world:
        settings.get("world", {}).pop("material", None)
    export = overrides.get("export")
    if isinstance(export, dict):
        if "target" in export and not isinstance(export["target"], str):
            raise ValueError("export.target must be a string")
        base_export = settings.setdefault("export", {})
        if "format" in export:
            base_export.pop("file_format", None)
        if "file_format" in export:
            base_export.pop("format", None)
        if "target" in export and not {"format", "file_format"} & export.keys():
            from plume_advanced.world import APPLICATION_EXPORT_FORMATS

            target = export["target"].strip().lower()
            base_export.pop("file_format", None)
            base_export["format"] = (
                "auto" if target == "all" else APPLICATION_EXPORT_FORMATS.get(target, "glb")
            )
    return _merge(settings, overrides)
