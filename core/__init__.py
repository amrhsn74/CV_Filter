"""Core package for CV project phases."""

from .convolution import apply_kernel
from .edge_detection import (
    SOBEL_X,
    SOBEL_Y,
    compute_intensity_map,
    sobel_edge_image,
    sobel_gradients,
)
from .filters import (
    adjust_contrast,
    blur_mean,
    increase_brightness,
    sharpen,
)
from .image_io import read_ppm_p3, write_ppm_p3
from .pipeline import (
    FILTER_BLUR,
    FILTER_BRIGHTNESS,
    FILTER_CONTRAST,
    FILTER_SHARPEN,
    combine_by_region_map,
    generate_filtered_images,
    process_region_based_filters,
)
from .segmentation import (
    REGION_BRIGHT,
    REGION_DARK,
    REGION_EDGE,
    REGION_LABELS,
    REGION_MID,
    classify_pixel_region,
    count_regions,
    segment_regions,
)

__all__ = [
    "read_ppm_p3",
    "write_ppm_p3",
    "apply_kernel",
    "SOBEL_X",
    "SOBEL_Y",
    "compute_intensity_map",
    "sobel_gradients",
    "sobel_edge_image",
    "blur_mean",
    "sharpen",
    "adjust_contrast",
    "increase_brightness",
    "FILTER_BLUR",
    "FILTER_SHARPEN",
    "FILTER_CONTRAST",
    "FILTER_BRIGHTNESS",
    "generate_filtered_images",
    "combine_by_region_map",
    "process_region_based_filters",
    "REGION_EDGE",
    "REGION_DARK",
    "REGION_MID",
    "REGION_BRIGHT",
    "REGION_LABELS",
    "classify_pixel_region",
    "segment_regions",
    "count_regions",
]
