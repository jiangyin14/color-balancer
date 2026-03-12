# color-balancer

Balance and recover colours in a digital image using a set of known reference
colour patches.  The tool is powered by the
[colour-science](https://www.colour-science.org/) library.

---

## How it works

1. You identify **N rectangular regions** in the image that contain colours
   you already know (e.g. a printed colour chart).
2. You supply those known *reference* colours alongside the pixel coordinates
   of each region.
3. The tool extracts the *measured* mean colour from every region, fits a
   colour-correction transform from measured → reference, and applies it to
   the **entire image**.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```
python color_balancer.py INPUT OUTPUT -p PATCHES_JSON [-m METHOD]
```

| Argument | Description |
|---|---|
| `INPUT` | Path to the input image |
| `OUTPUT` | Path for the calibrated output image |
| `-p PATCHES_JSON` | JSON file describing the calibration patches (required) |
| `-m METHOD` | Colour correction method: `Cheung 2004` (default), `Finlayson 2015`, or `Vandermonde` |

### Patches JSON format

The patches file is a JSON array of objects.  Each object must have:

* `"region"` – `[[x1, y1], [x2, y2]]`: pixel coordinates of two opposite
  corners of a rectangle (order does not matter).
* `"color"` – `[R, G, B]`: the known reference colour for that region,
  expressed as integers in the range 0–255.

**Example `patches.json`:**

```json
[
  {"region": [[10,  10], [60,  60]],  "color": [115, 82, 68]},
  {"region": [[70,  10], [120, 60]],  "color": [194, 150, 130]},
  {"region": [[10,  70], [60, 120]],  "color": [98, 122, 157]}
]
```

### Example

```bash
python color_balancer.py photo.jpg calibrated.jpg -p patches.json
```

```bash
# Using the Finlayson 2015 method
python color_balancer.py photo.jpg calibrated.jpg -p patches.json -m "Finlayson 2015"
```

## Running tests

```bash
pip install pytest
pytest test_color_balancer.py -v
```
