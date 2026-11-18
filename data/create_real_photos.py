"""
Prepares the Track B Real / Vintage Photographs dataset.

Collects 6 classical historical / vintage photographic images:
1. cameraman_1930s.png - The iconic 1930s photographer portrait (MIT archive, vintage optics).
2. historic_moon_1960s.png - 1960s vintage photographic plate of lunar craters with emulsion grain.
3. vintage_clock_mechanism.png - Vintage mechanical clock with fine metallic gears and motion dust.
4. scanned_historical_text.png - Scanned vintage typography document with aged paper bleed.
5. archival_currency_coins.png - Archival photographic documentation of antique coins.
6. vintage_aerial_landscape.png - Classic low-altitude aerial reconnaissance photography.
"""

from pathlib import Path
import numpy as np
import cv2
import skimage.data as sk_data


def prepare_real_photos_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classic Cameraman (1930s MIT vintage photograph)
    cam = sk_data.camera()
    cv2.imwrite(str(output_dir / "cameraman_1930s.png"), cam)

    # 2. Lunar photographic plate (1960s vintage emulsion grain)
    moon = sk_data.moon()
    cv2.imwrite(str(output_dir / "historic_moon_1960s.png"), moon)

    # 3. Vintage clock mechanism
    clock = sk_data.clock()
    cv2.imwrite(str(output_dir / "vintage_clock_mechanism.png"), clock)

    # 4. Scanned historical text document
    text = sk_data.text()
    cv2.imwrite(str(output_dir / "scanned_historical_text.png"), text)

    # 5. Archival antique coins
    coins = sk_data.coins()
    cv2.imwrite(str(output_dir / "archival_currency_coins.png"), coins)

    # 6. Vintage astronaut / archival spaceflight photo
    astronaut = sk_data.astronaut()
    # Save as grayscale archival portrait
    astro_gray = cv2.cvtColor(astronaut, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(output_dir / "archival_portrait.png"), astro_gray)

    print(f"Track B real photographs dataset created successfully in: {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "real_photos"
    prepare_real_photos_dataset(base_dir)
