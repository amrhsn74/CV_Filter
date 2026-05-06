"""Phase 4: Region segmentation from intensity and edge strength."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .edge_detection import compute_intensity_map, sobel_gradients

Number = float
GrayImage = List[List[Number]]
RegionMap = List[List[str]]

REGION_EDGE = "edge"
REGION_DARK = "dark"
REGION_MID = "mid"
REGION_BRIGHT = "bright"

REGION_LABELS = (REGION_EDGE, REGION_DARK, REGION_MID, REGION_BRIGHT)


def _as_number(value: object, name: str) -> Number:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got bool.")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc


def _validate_gray_map(edge_magnitude: Sequence[Sequence[object]]) -> GrayImage:
    if len(edge_magnitude) == 0:
        raise ValueError("edge_magnitude must have at least one row.")

    width = len(edge_magnitude[0])
    if width == 0:
        raise ValueError("edge_magnitude must have at least one column.")

    normalized: GrayImage = []
    for row_index, row in enumerate(edge_magnitude):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular edge_magnitude: row 0 has width {width}, row {row_index} has width {len(row)}."
            )

        norm_row: List[Number] = []
        for col_index, value in enumerate(row):
            norm_row.append(_as_number(value, f"edge_magnitude[{row_index}][{col_index}]"))
        normalized.append(norm_row)

    return normalized


def classify_pixel_region(
    intensity: Number,
    edge_strength: Number,
    edge_threshold: Number,
    dark_threshold: Number,
    bright_threshold: Number,
) -> str:
    """
    Classify one pixel into one of:
    edge, dark, mid, bright.

    Rule priority:
    1) edge (if edge_strength >= edge_threshold)
    2) dark (if intensity < dark_threshold)
    3) bright (if intensity >= bright_threshold)
    4) mid (otherwise)
    """
    if edge_strength >= edge_threshold:
        return REGION_EDGE
    if intensity < dark_threshold:
        return REGION_DARK
    if intensity >= bright_threshold:
        return REGION_BRIGHT
    return REGION_MID


def segment_regions(
    image: Sequence[Sequence[Sequence[object]]],
    edge_magnitude: Sequence[Sequence[object]] | None = None,
    edge_threshold: Number = 200.0,
    dark_threshold: Number = 85.0,
    bright_threshold: Number = 170.0,
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
) -> Tuple[RegionMap, GrayImage, GrayImage]:
    """
    Segment image into regions using intensity and edge magnitude.

    Args:
        image: RGB nested-list image [height][width][3].
        edge_magnitude:
            Optional precomputed 2D edge magnitude map.
            If None, Sobel magnitude is computed from image.
        edge_threshold: Edge region threshold.
        dark_threshold: Intensity threshold below which region is dark.
        bright_threshold: Intensity threshold at/above which region is bright.
        border_mode: Passed to Sobel when edge_magnitude is not provided.
        pad_value: Passed to Sobel when edge_magnitude is not provided.

    Returns:
        (region_map, intensity_map, edge_magnitude_map)
    """
    edge_t = _as_number(edge_threshold, "edge_threshold")
    dark_t = _as_number(dark_threshold, "dark_threshold")
    bright_t = _as_number(bright_threshold, "bright_threshold")

    if dark_t > bright_t:
        raise ValueError("dark_threshold must be <= bright_threshold.")

    intensity_map = compute_intensity_map(image)
    height = len(intensity_map)
    width = len(intensity_map[0])

    if edge_magnitude is None:
        _, _, computed_edge = sobel_gradients(
            image=image,
            border_mode=border_mode,
            pad_value=pad_value,
        )
    else:
        computed_edge = _validate_gray_map(edge_magnitude)

    if len(computed_edge) != height or len(computed_edge[0]) != width:
        raise ValueError(
            f"edge_magnitude shape mismatch: expected {height}x{width}, got {len(computed_edge)}x{len(computed_edge[0])}."
        )

    region_map: RegionMap = []
    for y in range(height):
        region_row: List[str] = []
        for x in range(width):
            region_row.append(
                classify_pixel_region(
                    intensity=intensity_map[y][x],
                    edge_strength=computed_edge[y][x],
                    edge_threshold=edge_t,
                    dark_threshold=dark_t,
                    bright_threshold=bright_t,
                )
            )
        region_map.append(region_row)

    return region_map, intensity_map, computed_edge


def count_regions(region_map: Sequence[Sequence[str]]) -> Dict[str, int]:
    """Count how many pixels belong to each region label."""
    counts: Dict[str, int] = {
        REGION_EDGE: 0,
        REGION_DARK: 0,
        REGION_MID: 0,
        REGION_BRIGHT: 0,
    }

    if len(region_map) == 0:
        return counts

    width = len(region_map[0])
    for row_index, row in enumerate(region_map):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular region_map: row 0 has width {width}, row {row_index} has width {len(row)}."
            )

        for col_index, label in enumerate(row):
            if label not in counts:
                raise ValueError(
                    f"Invalid region label at ({row_index}, {col_index}): {label!r}. "
                    f"Expected one of {REGION_LABELS}."
                )
            counts[label] += 1

    return counts
