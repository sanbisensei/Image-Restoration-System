"""
Phase 1 Foundation Demonstration Script.

Runs through the foundational components:
1. Loading various image formats (PNG grayscale, JPG RGB).
2. Diagnostic inspection using get_image_info.
3. Safe conversion between uint8 and normalized float representations.
4. Converting RGB to grayscale using standard luminance weights.
5. Saving images to disk and generating comparison plots.
"""

from pathlib import Path
from src.image_io import load_image, save_image, display_comparison
from src.utils.image_helpers import to_float, to_uint8, rgb_to_gray
from src.utils.validation import get_image_info


def main():
    print("=" * 60)
    print("CSE 438 — Image Restoration System: Phase 1 Demonstration")
    print("=" * 60)

    data_dir = Path("data/original")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load grayscale PNG
    gray_path = data_dir / "checkerboard.png"
    img_gray = load_image(gray_path, as_gray=True)
    info_gray = get_image_info(img_gray)
    print(f"\n[1] Loaded Grayscale Image: {gray_path.name}")
    print(f"    Shape: {info_gray['shape']}, Dtype: {info_gray['dtype']}, Range: [{info_gray['min']}, {info_gray['max']}]")

    # 2. Load RGB JPG
    rgb_path = data_dir / "color_test.jpg"
    img_rgb = load_image(rgb_path, as_gray=False)
    info_rgb = get_image_info(img_rgb)
    print(f"\n[2] Loaded RGB Image: {rgb_path.name}")
    print(f"    Shape: {info_rgb['shape']}, Dtype: {info_rgb['dtype']}, Range: [{info_rgb['min']}, {info_rgb['max']}]")

    # 3. Float normalization check
    img_flt = to_float(img_gray)
    info_flt = get_image_info(img_flt)
    print(f"\n[3] Normalized Float Conversion:")
    print(f"    Dtype: {info_flt['dtype']}, Range: [{info_flt['min']:.4f}, {info_flt['max']:.4f}], Mean: {info_flt['mean']:.4f}")

    # Reconvert back to uint8
    img_u8_back = to_uint8(img_flt)
    diff = abs(img_gray.astype(int) - img_u8_back.astype(int)).max()
    print(f"    Round-trip (uint8 -> float -> uint8) Maximum Pixel Difference: {diff}")

    # 4. Color conversion: RGB to Grayscale
    gray_converted = rgb_to_gray(img_rgb)
    print(f"\n[4] RGB to Grayscale Conversion:")
    print(f"    Original RGB Shape: {img_rgb.shape} -> Converted Grayscale Shape: {gray_converted.shape}")

    # 5. Save outputs and comparison figure
    saved_gray_path = out_dir / "phase1_checkerboard_saved.png"
    save_image(saved_gray_path, img_flt)
    print(f"\n[5] Saved Normalized Float as PNG: {saved_gray_path}")

    comparison_fig_path = out_dir / "phase1_comparison.png"
    display_comparison(
        images=[img_gray, img_rgb, gray_converted, img_u8_back],
        titles=[
            "1. Checkerboard (uint8)",
            "2. Color Test (RGB)",
            "3. Converted Grayscale",
            "4. Round-trip Restored",
        ],
        cols=2,
        show=False,
        save_path=comparison_fig_path,
    )
    print(f"    Saved Comparison Figure: {comparison_fig_path}")

    print("\n" + "=" * 60)
    print("Phase 1 verification completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
