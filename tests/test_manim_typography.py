"""Optional renderer checks for stable, resolution-independent text layout."""

import pytest

manim = pytest.importorskip("manim")

from plume_advanced.media.manim_typography import PresentationText  # noqa: E402


@pytest.fixture(autouse=True)
def text_cache(tmp_path):
    with manim.tempconfig({"text_dir": str(tmp_path / "text")}):
        yield


@pytest.mark.parametrize(
    "label",
    ["D · Geometry", "Same branch · sampled profiles", "positive density = carved void"],
)
def test_small_labels_keep_proportional_glyph_positions(label):
    small = PresentationText(label, font="DejaVu Sans", font_size=15)
    large = PresentationText(label, font="DejaVu Sans", font_size=30).scale(0.5)
    assert len(small) == len(large)
    assert small.width == pytest.approx(large.width, rel=0.005)
    # Compare every glyph's position, not just the overall bounding box.
    for a, b in zip(small, large, strict=True):
        assert a.get_center()[0] == pytest.approx(b.get_center()[0], abs=0.006)


def test_font_size_and_video_config_are_preserved():
    resolution = (manim.config.pixel_width, manim.config.pixel_height)
    text = PresentationText("Geometry", font_size=15)
    assert text.font_size == pytest.approx(15)
    text.font_size = 30
    assert text.font_size == pytest.approx(30)
    assert (manim.config.pixel_width, manim.config.pixel_height) == resolution


def test_explicit_dimensions_and_long_headings():
    label = "From host fields to continuous cave geometry"
    text = PresentationText(label, font_size=38, width=12)
    assert text.width == pytest.approx(12)
    # Pango may combine letters such as "fi" into one ligature glyph.
    assert len(text) == len(manim.Text(label, font_size=38))
