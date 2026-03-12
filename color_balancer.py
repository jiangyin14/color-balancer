#!/usr/bin/env python3
"""
color_balancer.py - Color calibration tool using the colour-science library.

Given N rectangular regions in an image with known reference colors, the
tool extracts the measured colors from those regions, computes a color
correction transform, and applies it to the entire image.
"""

import argparse
import json
import sys
import warnings

import numpy as np
from PIL import Image

import colour

# Suppress non-critical colour-science warnings at the module level
warnings.filterwarnings("ignore", category=colour.utilities.ColourWarning)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_region_mean(
    image_array: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Return the mean normalised RGB value (0-1) of a rectangular region.

    Parameters
    ----------
    image_array:
        Float64 H×W×3 array with values in [0, 1].
    x1, y1, x2, y2:
        Pixel coordinates of two opposite corners of the rectangle
        (order does not matter).

    Returns
    -------
    numpy.ndarray
        Shape (3,) mean RGB colour.
    """
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    # Guard against out-of-bounds coordinates
    h, w = image_array.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    region = image_array[y_min : y_max + 1, x_min : x_max + 1]
    return region.mean(axis=(0, 1))


def calibrate_image(
    image_path: str,
    patches: list,
    output_path: str,
    method: str = "Cheung 2004",
) -> None:
    """Calibrate an image's colours using known reference patches and save it.

    Parameters
    ----------
    image_path:
        Path to the input image.
    patches:
        List of patch definitions.  Each item is a dict with:

        * ``"region"``: ``[[x1, y1], [x2, y2]]`` – two corners of a
          rectangle in pixel coordinates.
        * ``"color"``: ``[R, G, B]`` – the *reference* (target) colour of
          that region, expressed as integers in the range 0-255.

    output_path:
        Path where the calibrated image will be saved.
    method:
        Colour correction method passed to
        :func:`colour.colour_correction`.  One of:

        * ``"Cheung 2004"`` (default)
        * ``"Finlayson 2015"``
        * ``"Vandermonde"``

    Raises
    ------
    ValueError
        If fewer than one calibration patch is supplied, or if a patch
        dict is missing required keys.
    FileNotFoundError
        If *image_path* does not exist.
    """
    if not patches:
        raise ValueError("At least one calibration patch must be provided.")

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float64) / 255.0

    measured_colors = []
    reference_colors = []

    for i, patch in enumerate(patches):
        if "region" not in patch or "color" not in patch:
            raise ValueError(
                f"Patch {i} is missing required keys 'region' and/or 'color'."
            )

        (x1, y1), (x2, y2) = patch["region"]
        measured = extract_region_mean(img_array, int(x1), int(y1), int(x2), int(y2))
        measured_colors.append(measured)

        ref = np.array(patch["color"], dtype=np.float64) / 255.0
        reference_colors.append(ref)

    M_T = np.array(measured_colors)   # test (measured) colours
    M_R = np.array(reference_colors)  # reference (target) colours

    # Reshape to (N_pixels, 3), correct, then reshape back
    h, w, _ = img_array.shape
    flat = img_array.reshape(-1, 3)
    corrected_flat = colour.colour_correction(flat, M_T, M_R, method=method)
    corrected = corrected_flat.reshape(h, w, 3)

    # Clip to valid range before converting to uint8
    corrected = np.clip(corrected, 0.0, 1.0)
    output_array = (corrected * 255).round().astype(np.uint8)

    output_img = Image.fromarray(output_array, mode="RGB")
    output_img.save(output_path)
    print(f"Calibrated image saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color_balancer",
        description=(
            "Calibrate the colours of an image using N reference patches.\n\n"
            "Each patch specifies a rectangular region in the image and the\n"
            "known reference colour for that region.  A colour correction\n"
            "transform is fitted to those patches and applied to the whole image."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input image.")
    parser.add_argument("output", help="Path for the calibrated output image.")
    parser.add_argument(
        "-p",
        "--patches",
        metavar="PATCHES_JSON",
        required=True,
        help=(
            "Path to a JSON file describing the calibration patches.  "
            "The file must contain a list of objects, each with:\n"
            '  "region": [[x1,y1],[x2,y2]]  (pixel coordinates)\n'
            '  "color":  [R,G,B]             (reference colour, 0-255)'
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        default="Cheung 2004",
        choices=["Cheung 2004", "Finlayson 2015", "Vandermonde"],
        help="Colour correction method (default: 'Cheung 2004').",
    )
    return parser


def main(argv: list | None = None) -> int:
    """Entry point for the command-line interface.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        with open(args.patches, encoding="utf-8") as fh:
            patches = json.load(fh)
    except FileNotFoundError:
        print(f"Error: patches file not found: {args.patches}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse patches JSON: {exc}", file=sys.stderr)
        return 1

    try:
        calibrate_image(
            image_path=args.input,
            patches=patches,
            output_path=args.output,
            method=args.method,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
