"""Phase 2: Pure-Python 2D convolution for RGB nested-list images."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Number = float
Pixel = List[Number]
ImageData = List[List[Pixel]]
KernelData = List[List[Number]]


def _as_number(value: object, context: str) -> Number:
    """Convert value to float and raise a clear error when it is not numeric."""
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric, got bool.")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, got {value!r}.") from exc


def _validate_image(image: Sequence[Sequence[Sequence[object]]]) -> Tuple[ImageData, int, int]:
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

            normalized_row.append(
                [
                    _as_number(pixel[0], f"Pixel channel at ({row_index}, {col_index}, 0)"),
                    _as_number(pixel[1], f"Pixel channel at ({row_index}, {col_index}, 1)"),
                    _as_number(pixel[2], f"Pixel channel at ({row_index}, {col_index}, 2)"),
                ]
            )

        normalized.append(normalized_row)

    return normalized, len(image), width


def _validate_kernel(kernel: Sequence[Sequence[object]]) -> KernelData:
    if len(kernel) == 0:
        raise ValueError("Kernel must have at least one row.")

    width = len(kernel[0])
    if width == 0:
        raise ValueError("Kernel must have at least one column.")

    if width % 2 == 0 or len(kernel) % 2 == 0:
        raise ValueError("Kernel dimensions must be odd so it has a center pixel.")

    normalized: KernelData = []
    for row_index, row in enumerate(kernel):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular kernel: row 0 has width {width}, row {row_index} has width {len(row)}."
            )
        normalized.append(
            [
                _as_number(value, f"Kernel value at ({row_index}, {col_index})")
                for col_index, value in enumerate(row)
            ]
        )

    return normalized


def _clamp(value: Number, lower: Number, upper: Number) -> Number:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def apply_kernel(
    image: Sequence[Sequence[Sequence[object]]],
    kernel: Sequence[Sequence[object]],
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
    clamp_output: bool = False,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = False,
) -> ImageData:
    """
    Apply a 2D convolution kernel to an RGB image represented as [height][width][3].

    Args:
        image: Nested-list RGB image.
        kernel: Odd-sized 2D kernel.
        border_mode:
            - "ignore": out-of-bounds samples are skipped.
            - "pad": out-of-bounds samples use pad_value.
        pad_value: Constant sample value for "pad" border mode.
        clamp_output: Clamp each output channel to [output_min, output_max].
        output_min: Lower clamp bound (used when clamp_output=True).
        output_max: Upper clamp bound (used when clamp_output=True).
        round_output: Round each channel to nearest integer at the end.

    Returns:
        Convolved image with same dimensions as input.
    """
    image_data, height, width = _validate_image(image)
    kernel_data = _validate_kernel(kernel)

    if border_mode not in {"ignore", "pad"}:
        raise ValueError("border_mode must be either 'ignore' or 'pad'.")

    pad = _as_number(pad_value, "pad_value")
    lower = _as_number(output_min, "output_min")
    upper = _as_number(output_max, "output_max")
    if lower > upper:
        raise ValueError("output_min must be <= output_max.")

    kernel_height = len(kernel_data)
    kernel_width = len(kernel_data[0])
    center_y = kernel_height // 2
    center_x = kernel_width // 2
    # Flip once to perform true convolution instead of cross-correlation.
    flipped_kernel = [row[::-1] for row in kernel_data[::-1]]

    output: ImageData = []
    for y in range(height):
        out_row: List[Pixel] = []
        for x in range(width):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0

            for ky in range(kernel_height):
                src_y = y + ky - center_y
                for kx in range(kernel_width):
                    src_x = x + kx - center_x
                    weight = flipped_kernel[ky][kx]

                    if 0 <= src_y < height and 0 <= src_x < width:
                        source_pixel = image_data[src_y][src_x]
                        acc_r += source_pixel[0] * weight
                        acc_g += source_pixel[1] * weight
                        acc_b += source_pixel[2] * weight
                    elif border_mode == "pad":
                        acc_r += pad * weight
                        acc_g += pad * weight
                        acc_b += pad * weight

            if clamp_output:
                acc_r = _clamp(acc_r, lower, upper)
                acc_g = _clamp(acc_g, lower, upper)
                acc_b = _clamp(acc_b, lower, upper)

            if round_output:
                out_row.append([int(round(acc_r)), int(round(acc_g)), int(round(acc_b))])
            else:
                out_row.append([acc_r, acc_g, acc_b])

        output.append(out_row)

    return output
