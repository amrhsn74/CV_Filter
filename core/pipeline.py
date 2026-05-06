"""Phase 6: Region-based filter processing pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .edge_detection import compute_intensity_map, sobel_gradients
from .filters import adjust_contrast, blur_mean, increase_brightness, sharpen
from .segmentation import (
    REGION_BRIGHT,
    REGION_DARK,
    REGION_EDGE,
    REGION_LABELS,
    REGION_MID,
    segment_regions,
)

Number = float
Pixel = List[Number]
ImageData = List[List[Pixel]]
GrayImage = List[List[Number]]
RegionMap = List[List[str]]

FILTER_BLUR = "blur"
FILTER_SHARPEN = "sharpen"
FILTER_CONTRAST = "contrast"
FILTER_BRIGHTNESS = "brightness"


def _validate_region_map_shape(region_map: Sequence[Sequence[str]], height: int, width: int) -> None:
    if len(region_map) != height:
        raise ValueError(f"region_map height mismatch: expected {height}, got {len(region_map)}.")

    for row_index, row in enumerate(region_map):
        if len(row) != width:
            raise ValueError(
                f"region_map width mismatch at row {row_index}: expected {width}, got {len(row)}."
            )
        for col_index, label in enumerate(row):
            if label not in REGION_LABELS:
                raise ValueError(
                    f"Invalid region label at ({row_index}, {col_index}): {label!r}. "
                    f"Expected one of {REGION_LABELS}."
                )


def _validate_rgb_image_shape(image: Sequence[Sequence[Sequence[object]]], height: int, width: int, name: str) -> None:
    if len(image) != height:
        raise ValueError(f"{name} height mismatch: expected {height}, got {len(image)}.")

    for row_index, row in enumerate(image):
        if len(row) != width:
            raise ValueError(
                f"{name} width mismatch at row {row_index}: expected {width}, got {len(row)}."
            )
        for col_index, pixel in enumerate(row):
            if len(pixel) != 3:
                raise ValueError(
                    f"Invalid pixel in {name} at ({row_index}, {col_index}): expected 3 channels, got {len(pixel)}."
                )


def _normalize_gray_map(
    gray_map: Sequence[Sequence[object]],
    height: int,
    width: int,
    name: str,
) -> GrayImage:
    if len(gray_map) != height:
        raise ValueError(f"{name} height mismatch: expected {height}, got {len(gray_map)}.")

    normalized: GrayImage = []
    for row_index, row in enumerate(gray_map):
        if len(row) != width:
            raise ValueError(
                f"{name} width mismatch at row {row_index}: expected {width}, got {len(row)}."
            )
        normalized_row: List[Number] = []
        for col_index, value in enumerate(row):
            if isinstance(value, bool):
                raise ValueError(f"{name}[{row_index}][{col_index}] must be numeric, got bool.")
            try:
                normalized_row.append(float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name}[{row_index}][{col_index}] must be numeric, got {value!r}."
                ) from exc
        normalized.append(normalized_row)
    return normalized


def generate_filtered_images(
    image: Sequence[Sequence[Sequence[object]]],
    blur_kernel_size: int = 3,
    sharpen_amount: Number = 1.0,
    contrast_factor: Number = 1.2,
    contrast_midpoint: Number = 127.5,
    brightness_delta: Number = 30.0,
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> Dict[str, ImageData]:
    """
    Generate the filtered image set required by Phase 6.

    Mapping intent from workplan:
    - edge    -> sharpen output
    - bright  -> brightness output
    - dark    -> contrast output
    - mid     -> blur output
    """
    blurred = blur_mean(
        image=image,
        kernel_size=blur_kernel_size,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )
    sharpened = sharpen(
        image=image,
        amount=sharpen_amount,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )
    contrast = adjust_contrast(
        image=image,
        factor=contrast_factor,
        midpoint=contrast_midpoint,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )
    brightness = increase_brightness(
        image=image,
        delta=brightness_delta,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )

    return {
        FILTER_BLUR: blurred,
        FILTER_SHARPEN: sharpened,
        FILTER_CONTRAST: contrast,
        FILTER_BRIGHTNESS: brightness,
    }


def combine_by_region_map(
    region_map: Sequence[Sequence[str]],
    *,
    edge_image: Sequence[Sequence[Sequence[object]]],
    dark_image: Sequence[Sequence[Sequence[object]]],
    mid_image: Sequence[Sequence[Sequence[object]]],
    bright_image: Sequence[Sequence[Sequence[object]]],
) -> ImageData:
    """
    Combine filtered images into one output using the region map labels.
    """
    if len(region_map) == 0 or len(region_map[0]) == 0:
        raise ValueError("region_map must have at least one row and one column.")

    height = len(region_map)
    width = len(region_map[0])
    _validate_region_map_shape(region_map, height, width)
    _validate_rgb_image_shape(edge_image, height, width, "edge_image")
    _validate_rgb_image_shape(dark_image, height, width, "dark_image")
    _validate_rgb_image_shape(mid_image, height, width, "mid_image")
    _validate_rgb_image_shape(bright_image, height, width, "bright_image")

    combined: ImageData = []
    for y in range(height):
        out_row: List[Pixel] = []
        for x in range(width):
            label = region_map[y][x]
            if label == REGION_EDGE:
                source = edge_image[y][x]
            elif label == REGION_DARK:
                source = dark_image[y][x]
            elif label == REGION_MID:
                source = mid_image[y][x]
            else:  # REGION_BRIGHT
                source = bright_image[y][x]
            out_row.append([source[0], source[1], source[2]])
        combined.append(out_row)

    return combined


def process_region_based_filters(
    image: Sequence[Sequence[Sequence[object]]],
    region_map: Sequence[Sequence[str]] | None = None,
    edge_magnitude: Sequence[Sequence[object]] | None = None,
    edge_threshold: Number = 200.0,
    dark_threshold: Number = 85.0,
    bright_threshold: Number = 170.0,
    segmentation_border_mode: str = "ignore",
    segmentation_pad_value: Number = 0.0,
    blur_kernel_size: int = 3,
    sharpen_amount: Number = 1.0,
    contrast_factor: Number = 1.2,
    contrast_midpoint: Number = 127.5,
    brightness_delta: Number = 30.0,
    filter_border_mode: str = "ignore",
    filter_pad_value: Number = 0.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> Dict[str, object]:
    """
    Full Phase 6 orchestrator.

    Returns a dict containing:
    - region_map
    - intensity_map
    - edge_magnitude
    - filtered_images
    - final_image
    """
    if region_map is None:
        computed_region_map, intensity_map, computed_edge = segment_regions(
            image=image,
            edge_magnitude=edge_magnitude,
            edge_threshold=edge_threshold,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
            border_mode=segmentation_border_mode,
            pad_value=segmentation_pad_value,
        )
    else:
        intensity_map = compute_intensity_map(image)
        height = len(intensity_map)
        width = len(intensity_map[0])
        _validate_region_map_shape(region_map, height, width)
        computed_region_map = [list(row) for row in region_map]
        if edge_magnitude is None:
            _, _, computed_edge = sobel_gradients(
                image=image,
                border_mode=segmentation_border_mode,
                pad_value=segmentation_pad_value,
            )
        else:
            computed_edge = _normalize_gray_map(edge_magnitude, height, width, "edge_magnitude")

    filtered_images = generate_filtered_images(
        image=image,
        blur_kernel_size=blur_kernel_size,
        sharpen_amount=sharpen_amount,
        contrast_factor=contrast_factor,
        contrast_midpoint=contrast_midpoint,
        brightness_delta=brightness_delta,
        border_mode=filter_border_mode,
        pad_value=filter_pad_value,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )

    final_image = combine_by_region_map(
        computed_region_map,
        edge_image=filtered_images[FILTER_SHARPEN],
        dark_image=filtered_images[FILTER_CONTRAST],
        mid_image=filtered_images[FILTER_BLUR],
        bright_image=filtered_images[FILTER_BRIGHTNESS],
    )

    return {
        "region_map": computed_region_map,
        "intensity_map": intensity_map,
        "edge_magnitude": computed_edge,
        "filtered_images": filtered_images,
        "final_image": final_image,
    }
