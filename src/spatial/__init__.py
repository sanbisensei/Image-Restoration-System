"""
Spatial domain processing package: Custom Convolution and Spatial Filters.
"""

from src.spatial.convolution import convolve2d, convolve2d_grayscale
from src.spatial.mean_filter import create_mean_kernel, apply_mean_filter
from src.spatial.gaussian_filter import create_gaussian_kernel, apply_gaussian_filter
from src.spatial.median_filter import apply_median_filter, apply_median_filter_grayscale

__all__ = [
    "convolve2d",
    "convolve2d_grayscale",
    "create_mean_kernel",
    "apply_mean_filter",
    "create_gaussian_kernel",
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_median_filter_grayscale",
]
