"""Phase 3: Sobel edge detection built on the convolution engine."""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence, Tuple

from .convolution import apply_kernel

Number = float
GrayImage = List[List[Number]]
Pixel = List[Number]
ImageData = List[List[Pixel]]

# These kernels are provided in the common image-processing form.
SOBEL_X: List[List[Number]] = [
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
]
SOBEL_Y: List[List[Number]] = [
    [-1.0, -2.0, -1.0],
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 1.0],
]


def _validate_rgb_image(image: Sequence[Sequence[Sequence[object]]]) -> ImageData:
    """Validate RGB nested-list shape and normalize values to floats."""
    if len(image) == 0:
        raise ValueError("Image must have at least one row.")

    width = len(image[0])
    if width == 0:
        raise ValueError("Image must have at least one column.")

    normalized: ImageData = []
    for row_index, row in enumerate(image):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular image data: row 0 has width {width}, row {row_index} has width {len(row)}."
            )

        normalized_row: List[Pixel] = []
        for col_index, pixel in enumerate(row):
            if len(pixel) != 3:
                raise ValueError(
                    f"Invalid pixel at ({row_index}, {col_index}): expected 3 channels, got {len(pixel)}."
                )

            channels: List[Number] = []
            for channel_index, channel in enumerate(pixel):
                if isinstance(channel, bool):
                    raise ValueError(
                        f"Invalid channel at ({row_index}, {col_index}, {channel_index}): bool is not allowed."
                    )
                try:
                    channels.append(float(channel))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid channel at ({row_index}, {col_index}, {channel_index}): {channel!r}."
                    ) from exc

            normalized_row.append(channels)
        normalized.append(normalized_row)

    return normalized


def compute_intensity_map(image: Sequence[Sequence[Sequence[object]]]) -> GrayImage:
    """
    Compute grayscale intensity using the project rule:
    intensity = (R + G + B) / 3.
    """
    rgb_image = _validate_rgb_image(image)
    intensity: GrayImage = []
    for row in rgb_image:
        intensity.append([(pixel[0] + pixel[1] + pixel[2]) / 3.0 for pixel in row])
    return intensity


def _gray_to_rgb(gray: GrayImage) -> ImageData:
    """Convert a 2D gray map into [height][width][3] with repeated channels."""
    return [[[value, value, value] for value in row] for row in gray]


def sobel_gradients(
    image: Sequence[Sequence[Sequence[object]]],
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
) -> Tuple[GrayImage, GrayImage, GrayImage]:
    """
    Compute Sobel X, Sobel Y, and gradient magnitude maps.

    Returns:
        gx_map, gy_map, magnitude_map (all 2D float lists)
    """
    intensity = compute_intensity_map(image)
    gray_rgb = _gray_to_rgb(intensity)

    gx_rgb = apply_kernel(
        gray_rgb,
        SOBEL_X,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=False,
        round_output=False,
    )
    gy_rgb = apply_kernel(
        gray_rgb,
        SOBEL_Y,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=False,
        round_output=False,
    )

    height = len(intensity)
    width = len(intensity[0])
    gx_map: GrayImage = []
    gy_map: GrayImage = []
    magnitude_map: GrayImage = []

    for y in range(height):
        gx_row: List[Number] = []
        gy_row: List[Number] = []
        mag_row: List[Number] = []
        for x in range(width):
            gx = gx_rgb[y][x][0]
            gy = gy_rgb[y][x][0]
            gx_row.append(gx)
            gy_row.append(gy)
            mag_row.append(sqrt((gx * gx) + (gy * gy)))
        gx_map.append(gx_row)
        gy_map.append(gy_row)
        magnitude_map.append(mag_row)

    return gx_map, gy_map, magnitude_map


def sobel_edge_image(
    image: Sequence[Sequence[Sequence[object]]],
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    """
    Generate an RGB edge image from Sobel gradient magnitude.

    Returns an image in the same nested-list RGB format used by Phase 1 I/O.
    """
    _, _, magnitude = sobel_gradients(
        image=image,
        border_mode=border_mode,
        pad_value=pad_value,
    )

    if isinstance(output_min, bool) or isinstance(output_max, bool):
        raise ValueError("output_min and output_max must be numeric, got bool.")
    try:
        min_value = float(output_min)
        max_value = float(output_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_min and output_max must be numeric.") from exc

    if min_value > max_value:
        raise ValueError("output_min must be <= output_max.")

    edge_image: ImageData = []
    for row in magnitude:
        edge_row: List[Pixel] = []
        for value in row:
            out_value = value
            if clamp_output:
                if out_value < min_value:
                    out_value = min_value
                elif out_value > max_value:
                    out_value = max_value
            if round_output:
                out_value = int(round(out_value))

            edge_row.append([out_value, out_value, out_value])
        edge_image.append(edge_row)

    return edge_image
