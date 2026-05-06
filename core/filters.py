"""Phase 5: Basic image filters in pure Python."""

from __future__ import annotations

from typing import List, Sequence

from .convolution import apply_kernel

Number = float
Pixel = List[Number]
ImageData = List[List[Pixel]]


def _as_number(value: object, context: str) -> Number:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric, got bool.")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, got {value!r}.") from exc


def _validate_rgb_image(image: Sequence[Sequence[Sequence[object]]]) -> ImageData:
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

    return normalized


def _apply_point_operation(
    image: Sequence[Sequence[Sequence[object]]],
    op,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    data = _validate_rgb_image(image)
    min_value = _as_number(output_min, "output_min")
    max_value = _as_number(output_max, "output_max")
    if min_value > max_value:
        raise ValueError("output_min must be <= output_max.")

    out: ImageData = []
    for row in data:
        out_row: List[Pixel] = []
        for pixel in row:
            transformed: List[Number] = []
            for channel in pixel:
                value = op(channel)
                if clamp_output:
                    if value < min_value:
                        value = min_value
                    elif value > max_value:
                        value = max_value
                if round_output:
                    value = int(round(value))
                transformed.append(value)
            out_row.append(transformed)
        out.append(out_row)
    return out


def blur_mean(
    image: Sequence[Sequence[Sequence[object]]],
    kernel_size: int = 3,
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    """Apply mean blur using an odd-sized square kernel."""
    if not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be an integer.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    weight = 1.0 / float(kernel_size * kernel_size)
    kernel = [[weight for _ in range(kernel_size)] for _ in range(kernel_size)]

    return apply_kernel(
        image=image,
        kernel=kernel,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )


def sharpen(
    image: Sequence[Sequence[Sequence[object]]],
    amount: Number = 1.0,
    border_mode: str = "ignore",
    pad_value: Number = 0.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    """Apply a classic 3x3 sharpen kernel."""
    strength = _as_number(amount, "amount")
    if strength < 0:
        raise ValueError("amount must be >= 0.")

    kernel = [
        [0.0, -strength, 0.0],
        [-strength, 1.0 + (4.0 * strength), -strength],
        [0.0, -strength, 0.0],
    ]

    return apply_kernel(
        image=image,
        kernel=kernel,
        border_mode=border_mode,
        pad_value=pad_value,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )


def adjust_contrast(
    image: Sequence[Sequence[Sequence[object]]],
    factor: Number = 1.2,
    midpoint: Number = 127.5,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    """Adjust contrast around a midpoint (default 127.5 for 8-bit style values)."""
    contrast_factor = _as_number(factor, "factor")
    center = _as_number(midpoint, "midpoint")

    return _apply_point_operation(
        image=image,
        op=lambda channel: ((channel - center) * contrast_factor) + center,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )


def increase_brightness(
    image: Sequence[Sequence[Sequence[object]]],
    delta: Number = 30.0,
    clamp_output: bool = True,
    output_min: Number = 0.0,
    output_max: Number = 255.0,
    round_output: bool = True,
) -> ImageData:
    """Increase brightness by adding a constant delta to each channel."""
    offset = _as_number(delta, "delta")
    return _apply_point_operation(
        image=image,
        op=lambda channel: channel + offset,
        clamp_output=clamp_output,
        output_min=output_min,
        output_max=output_max,
        round_output=round_output,
    )
