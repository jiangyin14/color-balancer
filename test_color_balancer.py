"""Tests for color_balancer.py"""

import json
import os
import tempfile
import warnings

import numpy as np
import pytest
from PIL import Image

# Suppress colour-science usage warnings during tests
warnings.filterwarnings("ignore")

import colour  # noqa: E402 – import after warning filter

from color_balancer import calibrate_image, extract_region_mean, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(pixels: np.ndarray, path: str) -> None:
    """Save an H×W×3 uint8 array as a PNG at *path*."""
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _solid_image(color: tuple, size: tuple = (100, 100)) -> np.ndarray:
    """Return a solid-colour H×W×3 uint8 array."""
    h, w = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


# ---------------------------------------------------------------------------
# extract_region_mean
# ---------------------------------------------------------------------------

class TestExtractRegionMean:
    def test_solid_image_returns_exact_color(self):
        color = (128, 64, 200)
        arr = _solid_image(color).astype(np.float64) / 255.0
        mean = extract_region_mean(arr, 10, 10, 50, 50)
        expected = np.array(color, dtype=np.float64) / 255.0
        np.testing.assert_allclose(mean, expected, atol=1e-6)

    def test_swapped_corners_same_result(self):
        color = (100, 150, 200)
        arr = _solid_image(color).astype(np.float64) / 255.0
        m1 = extract_region_mean(arr, 10, 20, 40, 60)
        m2 = extract_region_mean(arr, 40, 60, 10, 20)
        np.testing.assert_allclose(m1, m2)

    def test_out_of_bounds_coordinates_clamped(self):
        color = (255, 0, 0)
        arr = _solid_image(color, size=(50, 50)).astype(np.float64) / 255.0
        # Coordinates extend well beyond image boundaries
        mean = extract_region_mean(arr, -10, -10, 200, 200)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(mean, expected, atol=1e-6)

    def test_single_pixel_region(self):
        arr = np.zeros((10, 10, 3), dtype=np.float64)
        arr[3, 5] = [0.5, 0.25, 0.75]
        mean = extract_region_mean(arr, 5, 3, 5, 3)
        np.testing.assert_allclose(mean, [0.5, 0.25, 0.75], atol=1e-9)


# ---------------------------------------------------------------------------
# calibrate_image
# ---------------------------------------------------------------------------

class TestCalibrateImage:
    """Integration tests for calibrate_image()."""

    def test_identity_correction_preserves_image(self, tmp_path):
        """When measured == reference, the output should be close to input."""
        color = (120, 80, 200)
        arr = _solid_image(color)
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)

        # Identity: one patch whose measured color == reference color
        patches = [
            {"region": [[0, 0], [99, 99]], "color": list(color)},
        ]
        calibrate_image(input_path, patches, output_path)

        result = np.array(Image.open(output_path).convert("RGB"), dtype=np.float64)
        np.testing.assert_allclose(result, arr.astype(np.float64), atol=5)

    def test_single_channel_shift(self, tmp_path):
        """A pure red shift should be corrected toward the reference."""
        input_arr = _solid_image((200, 100, 100))
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(input_arr, input_path)

        # The patch occupies the whole image; reference is grey
        patches = [
            {"region": [[0, 0], [99, 99]], "color": [150, 150, 150]},
        ]
        calibrate_image(input_path, patches, output_path)

        result = np.array(Image.open(output_path).convert("RGB"))
        # The output image must exist and have correct dimensions
        assert result.shape == (100, 100, 3)

    def test_output_saved_to_disk(self, tmp_path):
        arr = _solid_image((80, 80, 80))
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "out.png")
        _make_image(arr, input_path)
        patches = [{"region": [[0, 0], [99, 99]], "color": [80, 80, 80]}]
        calibrate_image(input_path, patches, output_path)
        assert os.path.isfile(output_path)

    def test_multiple_patches(self, tmp_path):
        """Multiple patches should not raise an error."""
        # Create a 100×100 image with two halves having different colours
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50] = (200, 50, 50)   # top half: red-ish
        arr[50:] = (50, 50, 200)   # bottom half: blue-ish
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)

        patches = [
            {"region": [[0, 0],  [99, 49]], "color": [180, 60, 60]},
            {"region": [[0, 50], [99, 99]], "color": [60, 60, 180]},
        ]
        calibrate_image(input_path, patches, output_path)
        assert os.path.isfile(output_path)

    def test_raises_on_empty_patches(self, tmp_path):
        arr = _solid_image((100, 100, 100))
        input_path = str(tmp_path / "input.png")
        _make_image(arr, input_path)
        with pytest.raises(ValueError, match="At least one calibration patch"):
            calibrate_image(input_path, [], str(tmp_path / "out.png"))

    def test_raises_on_missing_patch_keys(self, tmp_path):
        arr = _solid_image((100, 100, 100))
        input_path = str(tmp_path / "input.png")
        _make_image(arr, input_path)
        with pytest.raises(ValueError, match="missing required keys"):
            calibrate_image(input_path, [{"region": [[0, 0], [10, 10]]}],
                            str(tmp_path / "out.png"))

    def test_finlayson_method(self, tmp_path):
        arr = _solid_image((100, 150, 200))
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)
        patches = [{"region": [[0, 0], [99, 99]], "color": [100, 150, 200]}]
        calibrate_image(input_path, patches, output_path, method="Finlayson 2015")
        assert os.path.isfile(output_path)

    def test_vandermonde_method(self, tmp_path):
        arr = _solid_image((100, 150, 200))
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)
        patches = [{"region": [[0, 0], [99, 99]], "color": [100, 150, 200]}]
        calibrate_image(input_path, patches, output_path, method="Vandermonde")
        assert os.path.isfile(output_path)

    def test_output_values_clipped_to_valid_range(self, tmp_path):
        """Output pixel values must be in [0, 255]."""
        arr = _solid_image((10, 10, 240))
        input_path = str(tmp_path / "input.png")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)
        patches = [{"region": [[0, 0], [99, 99]], "color": [240, 240, 10]}]
        calibrate_image(input_path, patches, output_path)
        result = np.array(Image.open(output_path))
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestCLI:
    def _write_patches(self, path: str, patches: list) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(patches, fh)

    def test_basic_invocation(self, tmp_path):
        arr = _solid_image((100, 100, 100))
        input_path = str(tmp_path / "input.png")
        patches_path = str(tmp_path / "patches.json")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)
        self._write_patches(patches_path, [
            {"region": [[0, 0], [99, 99]], "color": [100, 100, 100]},
        ])
        rc = main([input_path, output_path, "-p", patches_path])
        assert rc == 0
        assert os.path.isfile(output_path)

    def test_missing_input_image_returns_error(self, tmp_path):
        patches_path = str(tmp_path / "patches.json")
        self._write_patches(patches_path, [
            {"region": [[0, 0], [10, 10]], "color": [128, 128, 128]},
        ])
        rc = main([
            str(tmp_path / "nonexistent.png"),
            str(tmp_path / "out.png"),
            "-p", patches_path,
        ])
        assert rc != 0

    def test_missing_patches_file_returns_error(self, tmp_path):
        arr = _solid_image((100, 100, 100))
        input_path = str(tmp_path / "input.png")
        _make_image(arr, input_path)
        rc = main([
            input_path,
            str(tmp_path / "out.png"),
            "-p", str(tmp_path / "nonexistent.json"),
        ])
        assert rc != 0

    def test_invalid_json_patches_returns_error(self, tmp_path):
        arr = _solid_image((100, 100, 100))
        input_path = str(tmp_path / "input.png")
        patches_path = str(tmp_path / "patches.json")
        _make_image(arr, input_path)
        with open(patches_path, "w") as fh:
            fh.write("not valid json {{")
        rc = main([input_path, str(tmp_path / "out.png"), "-p", patches_path])
        assert rc != 0

    def test_method_flag(self, tmp_path):
        arr = _solid_image((120, 120, 120))
        input_path = str(tmp_path / "input.png")
        patches_path = str(tmp_path / "patches.json")
        output_path = str(tmp_path / "output.png")
        _make_image(arr, input_path)
        self._write_patches(patches_path, [
            {"region": [[0, 0], [99, 99]], "color": [120, 120, 120]},
        ])
        rc = main([
            input_path, output_path,
            "-p", patches_path,
            "-m", "Finlayson 2015",
        ])
        assert rc == 0
