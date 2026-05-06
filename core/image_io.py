"""Phase 1: Pure-Python P3 PPM image I/O."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Pixel = List[int]
ImageData = List[List[Pixel]]


def _tokenize_ppm(path: str) -> List[str]:
    """Read file and return whitespace-separated tokens with comments removed."""
    tokens: List[str] = []
    with open(path, "r", encoding="ascii") as ppm_file:
        for raw_line in ppm_file:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
    return tokens


def read_ppm_p3(path: str) -> Tuple[ImageData, int, int, int]:
    """
    Read a P3 PPM file.

    Returns:
        (pixels, width, height, max_value)
        pixels is a 3D list in shape [height][width][3].
    """
    tokens = _tokenize_ppm(path)
    if len(tokens) < 4:
        raise ValueError("Invalid PPM file: missing header values.")

    if tokens[0] != "P3":
        raise ValueError("Invalid PPM file: expected P3 magic number.")

    try:
        width = int(tokens[1])
        height = int(tokens[2])
        max_value = int(tokens[3])
    except ValueError as exc:
        raise ValueError("Invalid PPM file: width, height, and max value must be integers.") from exc

    if width <= 0 or height <= 0:
        raise ValueError("Invalid PPM file: width and height must be positive.")
    if max_value <= 0:
        raise ValueError("Invalid PPM file: max value must be positive.")
    if max_value > 65535:
        raise ValueError("Invalid PPM file: max value must be <= 65535 for PPM.")

    channel_tokens = tokens[4:]
    expected_values = width * height * 3
    if len(channel_tokens) != expected_values:
        raise ValueError(
            f"Invalid PPM file: expected {expected_values} channel values, found {len(channel_tokens)}."
        )

    channel_values: List[int] = []
    for token in channel_tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid PPM file: non-integer channel value '{token}'.") from exc
        if value < 0 or value > max_value:
            raise ValueError(
                f"Invalid PPM file: channel value {value} outside range 0..{max_value}."
            )
        channel_values.append(value)

    pixels: ImageData = []
    index = 0
    for _ in range(height):
        row: List[Pixel] = []
        for _ in range(width):
            row.append([channel_values[index], channel_values[index + 1], channel_values[index + 2]])
            index += 3
        pixels.append(row)

    return pixels, width, height, max_value


def write_ppm_p3(path: str, pixels: Sequence[Sequence[Sequence[int]]], max_value: int = 255) -> None:
    """Write image data to a P3 PPM file."""
    if not isinstance(max_value, int):
        raise ValueError("max_value must be an integer.")
    if max_value <= 0:
        raise ValueError("max_value must be positive.")
    if max_value > 65535:
        raise ValueError("max_value must be <= 65535 for PPM.")

    height = len(pixels)
    if height == 0:
        raise ValueError("Image must have at least one row.")

    width = len(pixels[0])
    if width == 0:
        raise ValueError("Image must have at least one column.")

    for row_index, row in enumerate(pixels):
        if len(row) != width:
            raise ValueError(
                f"Non-rectangular image data: row 0 has width {width}, row {row_index} has width {len(row)}."
            )

    lines: List[str] = [
        "P3",
        f"{width} {height}",
        str(max_value),
    ]

    for row_index, row in enumerate(pixels):
        for col_index, pixel in enumerate(row):
            if len(pixel) != 3:
                raise ValueError(
                    f"Invalid pixel at ({row_index}, {col_index}): expected 3 channels, got {len(pixel)}."
                )

            clamped = []
            for channel in pixel:
                try:
                    value = int(channel)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid channel value at ({row_index}, {col_index}): {channel!r}."
                    ) from exc
                if value < 0:
                    value = 0
                elif value > max_value:
                    value = max_value
                clamped.append(value)

            lines.append(f"{clamped[0]} {clamped[1]} {clamped[2]}")

    with open(path, "w", encoding="ascii", newline="\n") as ppm_file:
        ppm_file.write("\n".join(lines))
        ppm_file.write("\n")
