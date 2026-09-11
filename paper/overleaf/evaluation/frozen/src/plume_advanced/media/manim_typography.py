"""High-precision vector typography for the optional Manim video renderer."""

from manim import Text as ManimText
from manim import config, tempconfig


class PresentationText(ManimText):
    """Shape small labels without Pango's visible integer-advance rounding.

    Manim divides font sizes before sending them to Pango. At our annotation
    sizes, glyph advances are then rounded to only a few SVG units. Shape on a
    32x larger SVG canvas and shrink the vector paths, not a raster image.
    The larger font size also gives these SVGs distinct cache keys.
    """

    LAYOUT_SCALE = 32

    def __init__(self, text: str, *, font_size: float = 48, **kwargs) -> None:
        scale = self.LAYOUT_SCALE
        explicit_size = kwargs.get("width") is not None or kwargs.get("height") is not None
        # The canvas must grow too, so long headings are not clipped during
        # Pango's SVG generation. tempconfig restores the video resolution.
        with tempconfig(
            {
                "pixel_width": config.pixel_width * scale,
                "pixel_height": config.pixel_height * scale,
            }
        ):
            super().__init__(text, font_size=font_size * scale, **kwargs)
        if not explicit_size:
            self.scale(1 / scale)
        self._font_size = float(font_size)
        self.initial_height = self.height
