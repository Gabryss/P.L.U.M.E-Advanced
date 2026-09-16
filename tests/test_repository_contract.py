"""Keep the entry README, linked guides and installable resources usable."""

import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image

from plume_advanced.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]


GUIDES = {"installation", "usage", "configuration", "architecture", "evaluation", "simulators"}
DOCUMENTS = [ROOT / "README.md", *(ROOT / "docs" / f"{name}.md" for name in sorted(GUIDES))]


def markdown_anchors(text):
    headings = re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)
    return {re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-") for heading in headings}


def test_readme_has_one_ordered_quickstart():
    text = (ROOT / "README.md").read_text()
    assert re.findall(r"^## (.+)$", text, re.MULTILINE) == [
        "Installation", "Usage", "Simulators", "Documentation", "Limits",
    ]
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    assert len(blocks) == 2
    assert "uv sync --locked" in blocks[0] and "plume-generate" not in blocks[0]
    assert blocks[1].strip() == "uv run plume-generate --output outputs/first_cave/network.png"
    assert "--no-sync" not in text


def test_documentation_local_links_and_anchors():
    for document in DOCUMENTS:
        text = document.read_text()
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            url = urlsplit(target)
            if url.scheme:
                continue
            linked = (document.parent / unquote(url.path)).resolve() if url.path else document
            assert linked.exists(), (document, target)
            assert linked.is_relative_to(ROOT), (document, target)
            if url.fragment and linked.suffix == ".md":
                assert url.fragment in markdown_anchors(linked.read_text()), (document, target)


def test_documentation_toml_examples_resolve(tmp_path):
    examples = []
    for document in DOCUMENTS:
        examples.extend(re.findall(r"```toml\n(.*?)```", document.read_text(), re.DOTALL))
    assert examples
    for index, block in enumerate(examples):
        raw = tomllib.loads(block)
        if not {"recipe_version", "schema_version"} & raw.keys():
            block = 'recipe_version = 1\npreset = "preview"\n' + block
        path = tmp_path / f"example-{index}.toml"
        path.write_text(block)
        load_project_config(path)


def test_technical_guides_are_centralized_and_linked():
    for directory in ("src", "scripts", "config", "tests"):
        assert not list((ROOT / directory).rglob("*.md")), directory
    guides = set((ROOT / "docs").rglob("*.md"))
    assert guides == set(DOCUMENTS[1:])
    readme = (ROOT / "README.md").read_text()
    for guide in guides:
        assert f"]({guide.relative_to(ROOT).as_posix()}" in readme
        assert "](../README.md)" in guide.read_text()


def test_repository_excludes_retired_evaluation_archives():
    for directory in ("docs/reviews", "docs/maintenance", "paper/campaigns", "paper/overleaf"):
        assert not (ROOT / directory).exists(), directory


def test_removed_animation_workflow_has_no_advertised_entrypoints():
    for name in ("assemble_manim_video.py", "prepare_manim_assets.py", "render_manim_video.py"):
        assert not (ROOT / "scripts" / name).exists()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "video" not in project["project"]["optional-dependencies"]
    assert not any(p.name.startswith("manim") for p in (ROOT / "src").rglob("*.py"))
    assert "--extra video" not in (ROOT / "README.md").read_text()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert "manim" not in {package["name"] for package in lock["package"]}


def test_packaged_recipe_resources_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    data = project["tool"]["setuptools"]["package-data"]["plume_advanced"]
    assert "presets.json" in data and "default_project.toml" in data
    assert "material_assets/*.txt" in data
    assert "evaluation/resources/*.toml" in data
    assert "evaluation/resources/seeds/*.txt" in data
    assert "evaluation/resources/splits/*.txt" in data
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["plume_advanced*"]


def test_simulator_gallery_is_archived_and_matches_receipts():
    """Clearing generation outputs must not remove or silently replace README captures."""
    directory = ROOT / "docs/simulators"
    gallery = json.loads((directory / "gallery.json").read_text())
    assert {row["application"] for row in gallery["images"]} == {
        "Blender", "Unity", "Unreal Engine", "Gazebo", "Isaac Sim",
    }
    documentation = "\n".join(document.read_text() for document in DOCUMENTS)
    for row in gallery["images"]:
        for role in ("image", "receipt", "source_receipt", "generation_receipt"):
            if role not in row:
                continue
            path = (ROOT / row[role]["path"]).resolve()
            assert path.is_relative_to(directory.resolve())
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[role]["sha256"]
        assert Path(row["image"]["path"]).name in documentation
        assert row["captured_on"] and row["application_version"] and row["scope"]


def test_simulator_ui_gallery_assets_and_links():
    directory = ROOT / "docs/simulators/ui"
    gallery = json.loads((directory / "gallery.json").read_text())
    assert {row["application"] for row in gallery["images"]} == {
        "Blender", "Unity", "Unreal Engine", "Gazebo", "Isaac Sim",
    }
    guide = (ROOT / "docs/simulators.md").read_text()
    for row in gallery["images"]:
        path = (directory / row["image"]).resolve()
        assert path.is_relative_to(directory)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        with Image.open(path) as screenshot:
            assert list(screenshot.size) == row["dimensions"]
            screenshot.verify()
        assert f"](simulators/ui/{row['image']})" in guide
        assert row["application_version"] and row["captured_on"]
