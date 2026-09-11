"""Value contracts for semantic network metadata."""

from collections.abc import Mapping

type SegmentMetadataValue = (
    str | int | float | bool | None | list[SegmentMetadataValue] | dict[str, SegmentMetadataValue]
)


def metadata_float(metadata: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Read a numeric scalar, using the default for absent or null values."""

    value = metadata.get(key)
    if value is None:
        return default
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Network metadata {key!r} must be numeric, got {type(value).__name__}")
    return float(value)
