"""
Image Input/Output and Display Utilities.

Provides reliable loading, saving, and plotting of images across formats (PNG, JPG, BMP, TIFF),
ensuring explicit channel order (RGB vs OpenCV BGR) and strict data type/range control.

OpenCV Usage Policy:
OpenCV is used here strictly for file system I/O (imread, imwrite) and color conversions
(BGR <-> RGB), complying with the academic policy that library functions assist I/O
without replacing custom restoration algorithms.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.image_helpers import to_float, to_uint8
from src.utils.validation import validate_image, is_valid_grayscale, is_valid_rgb


def load_image(
    path: Union[str, Path],
    as_gray: bool = False,
    to_normalized_float: bool = False,
) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the target image file.
    as_gray : bool, default=False
        If True, loads image as a 2D grayscale array.
        If False, loads image as an RGB array (H, W, 3).
    to_normalized_float : bool, default=False
        If True, converts pixel values to float64 in range [0.0, 1.0].
        If False, keeps native uint8 representation in range [0, 255].

    Returns
    -------
    np.ndarray
        Loaded image array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist on disk.
    ValueError
        If the file could not be decoded by OpenCV.
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {path_obj}")

    # Read using OpenCV
    # Note: cv2.imread loads color images in BGR format
    if as_gray:
        img = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read/decode image: {path_obj}. File may be corrupted or unsupported.")

    # Convert BGR to RGB for 3D color images
    if not as_gray and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    validate_image(img, allow_color=not as_gray, require_2d=as_gray)

    if to_normalized_float:
        img = to_float(img)

    return img


def save_image(
    path: Union[str, Path],
    image: np.ndarray,
) -> Path:
    """
    Save an image array to disk.

    Handles floating-point images by safely converting them to uint8 [0, 255]
    and converts RGB back to OpenCV's BGR order before writing.

    Parameters
    ----------
    path : str or Path
        Destination path where image will be saved.
    image : np.ndarray
        Image array to write (uint8 or normalized float, grayscale or RGB).

    Returns
    -------
    Path
        Resolved path of the saved file.

    Raises
    ------
    ValueError
        If image array is invalid or write operation fails.
    """
    validate_image(image)

    path_obj = Path(path).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Ensure image is in uint8 format for standard file storage
    uint8_img = to_uint8(image, clip=True)

    # Convert RGB -> BGR for OpenCV imwrite
    if uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
        save_array = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)
    else:
        save_array = uint8_img

    success = cv2.imwrite(str(path_obj), save_array)
    if not success:
        raise ValueError(f"Failed to write image to {path_obj}")

    return path_obj


def display_image(
    image: np.ndarray,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Display a single image using Matplotlib.

    Parameters
    ----------
    image : np.ndarray
        Image array to display.
    title : str, optional
        Figure title.
    cmap : str, optional
        Colormap name. Defaults to 'gray' for 2D images.
    figsize : tuple, default=(6, 6)
        Matplotlib figure size.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str or Path, optional
        If provided, saves figure to this path.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object.
    """
    validate_image(image)

    fig, ax = plt.subplots(figsize=figsize)

    if image.ndim == 2:
        effective_cmap = cmap if cmap is not None else "gray"
        ax.imshow(image, cmap=effective_cmap, vmin=0 if image.dtype == np.uint8 else 0.0,
                  vmax=255 if image.dtype == np.uint8 else 1.0)
    else:
        # Display RGB
        disp_img = to_uint8(image) if np.issubdtype(image.dtype, np.floating) else image
        ax.imshow(disp_img)

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig


def display_comparison(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Display multiple images in a comparative grid.

    Parameters
    ----------
    images : list of np.ndarray
        List of image arrays to compare.
    titles : list of str, optional
        Titles for each individual image subplot.
    cols : int, default=2
        Number of grid columns.
    figsize : tuple, optional
        Matplotlib figure dimensions.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str or Path, optional
        If provided, saves figure to disk.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images provided for comparison.")

    for img in images:
        validate_image(img)

    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 4, rows * 4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < n:
            img = images[idx]
            if img.ndim == 2:
                vmax = 255 if img.dtype == np.uint8 else 1.0
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            else:
                disp_img = to_uint8(img) if np.issubdtype(img.dtype, np.floating) else img
                ax.imshow(disp_img)

            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=11)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig
