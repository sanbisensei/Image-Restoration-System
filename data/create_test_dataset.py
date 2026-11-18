"""
Script to generate standard synthetic test images for the DIP restoration system.

Generates:
1. checkerboard.png (256x256, grayscale) - Perfect square wave pattern, high-contrast edges.
2. geometric_shapes.png (256x256, grayscale) - Circles, lines, rects for motion blur and edge tests.
3. gradient_ramp.png (256x256, grayscale) - Continuous ramp and impulse test pattern.
4. synthetic_scene.png (256x256, grayscale) - Multi-scale textures and structures.
5. color_test.jpg (256x256, RGB) - 3-channel color image for testing color handling & JPEG format.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_checkerboard(size: int = 256, block_size: int = 32) -> np.ndarray:
    """Generate high-contrast binary checkerboard."""
    coords = np.indices((size, size)) // block_size
    board = ((coords[0] + coords[1]) % 2) * 255
    return board.astype(np.uint8)


def generate_geometric_shapes(size: int = 256) -> np.ndarray:
    """Generate geometric shapes (circle, rotated ellipse, rectangle, diagonal lines)."""
    img = np.zeros((size, size), dtype=np.uint8)

    # Rectangle
    cv2.rectangle(img, (20, 20), (100, 100), 200, -1)

    # Circle
    cv2.circle(img, (180, 70), 45, 240, -1)

    # Diagonal lines (good for directional motion blur test)
    for offset in range(-40, 50, 15):
        cv2.line(img, (120 + offset, 140), (220 + offset, 240), 180, 2)

    # Dark filled circle inside bright rect
    cv2.rectangle(img, (25, 140), (105, 230), 160, -1)
    cv2.circle(img, (65, 185), 25, 40, -1)

    return img


def generate_gradient_ramp(size: int = 256) -> np.ndarray:
    """Generate horizontal gradient ramp with center impulse and step edges."""
    x = np.linspace(0, 255, size, dtype=np.float64)
    ramp = np.tile(x, (size, 1)).astype(np.uint8)

    # Add vertical step bands in top half
    ramp[:64, 64:128] = 220
    ramp[:64, 128:192] = 50

    # Add impulse point (single white pixel at center)
    ramp[size // 2, size // 2] = 255

    return ramp


def generate_synthetic_scene(size: int = 256) -> np.ndarray:
    """Generate a synthetic landscape scene with sky gradient, ground, house, and sun."""
    img = np.zeros((size, size), dtype=np.uint8)

    # Sky gradient
    for y in range(160):
        val = int(80 + (y / 160.0) * 120)
        img[y, :] = val

    # Sun
    cv2.circle(img, (200, 50), 22, 255, -1)

    # Ground
    img[160:, :] = 60

    # House base
    cv2.rectangle(img, (60, 130), (140, 210), 190, -1)

    # House roof (triangle)
    pts = np.array([[50, 130], [150, 130], [100, 80]], np.int32)
    cv2.fillPoly(img, [pts], 230)

    # Door and window
    cv2.rectangle(img, (85, 165), (115, 210), 40, -1)
    cv2.rectangle(img, (70, 140), (85, 155), 255, -1)
    cv2.rectangle(img, (115, 140), (130, 155), 255, -1)

    return img


def generate_color_test(size: int = 256) -> np.ndarray:
    """Generate an RGB color image with primary and secondary color regions."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)

    # Red quadrant
    rgb[:128, :128, 0] = 220
    rgb[:128, :128, 1] = 40
    rgb[:128, :128, 2] = 40

    # Green quadrant
    rgb[:128, 128:, 0] = 40
    rgb[:128, 128:, 1] = 200
    rgb[:128, 128:, 2] = 40

    # Blue quadrant
    rgb[128:, :128, 0] = 40
    rgb[128:, :128, 1] = 80
    rgb[128:, :128, 2] = 220

    # Yellow quadrant
    rgb[128:, 128:, 0] = 230
    rgb[128:, 128:, 1] = 210
    rgb[128:, 128:, 2] = 40

    # Center white circle
    cv2.circle(rgb, (128, 128), 35, (255, 255, 255), -1)

    return rgb


def create_all_test_images(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "checkerboard.png"), generate_checkerboard())
    cv2.imwrite(str(output_dir / "geometric_shapes.png"), generate_geometric_shapes())
    cv2.imwrite(str(output_dir / "gradient_ramp.png"), generate_gradient_ramp())
    cv2.imwrite(str(output_dir / "synthetic_scene.png"), generate_synthetic_scene())

    # Save color image as JPEG to verify JPG format reading
    color_bgr = cv2.cvtColor(generate_color_test(), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "color_test.jpg"), color_bgr)

    print(f"Test dataset created successfully in {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "original"
    create_all_test_images(base_dir)
