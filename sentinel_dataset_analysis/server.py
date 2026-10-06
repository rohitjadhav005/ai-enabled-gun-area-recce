#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py
=========
READ-ONLY Flask backend for the SEN12MS-CR-TS dataset viewer.

Serves the HTML UI and API endpoints to browse ROIs, acquisitions,
patches, and render GeoTIFF bands as PNG images on-the-fly.

What this server NEVER does
----------------------------
* Never modifies, renames, moves, deletes or copies any original TIFF.
* Never writes inside the dataset root.
* Pixel data is only read to render a display image; nothing is persisted.

Usage (PowerShell, from the repository root)
--------------------------------------------
    python sentinel_dataset_analysis\\server.py
    python sentinel_dataset_analysis\\server.py --port 5001
    python sentinel_dataset_analysis\\server.py --dataset-root "C:\\path\\to\\Dataset"
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow importing helpers from the scripts/ sibling folder
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # sentinel_dataset_analysis/
_WORKSPACE = _HERE.parent                         # sentineldataset/
sys.path.insert(0, str(_HERE / "scripts"))

from visualize_dataset import (  # type: ignore[import]
    DEFAULT_WORKSPACE_ROOT,
    FALSE_COLOR_BANDS,
    S1_BAND_NAMES,
    S1_STRETCH,
    S2_BAND_NAMES,
    S2_STRETCH,
    TRUE_COLOR_BANDS,
    band_stats,
    find_dataset_root,
    list_roi_keys,
    read_bands,
    resolve_patch,
    stretch,
    to_rgb,
    RASTERIO_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
try:
    from flask import Flask, jsonify, request, send_file, render_template
except ImportError:
    print("Flask is not installed. Run:  pip install flask", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image  # type: ignore[import-untyped]
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)
log = logging.getLogger("werkzeug")

# Will be set by main() / CLI
DATASET_ROOT: Optional[Path] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROIS_DIR_RE = re.compile(r"^ROIs(?P<num>\d+)$", re.IGNORECASE)
PATCH_ID_RE = re.compile(r"_patch_(?P<pid>\d+)\.tiff?$", re.IGNORECASE)


def _require_dataset() -> Tuple[Optional[dict], Optional[Path]]:
    """Return (error_json, dataset_root) – error_json is set when root is missing."""
    if DATASET_ROOT is None or not DATASET_ROOT.is_dir():
        return {"error": "Dataset root not found or not set."}, None
    return None, DATASET_ROOT


def _roi_parts(roi_key: str):
    """Split 'asiaWest_n/1868/127' -> (partition, roi_num, roi_id)."""
    parts = [p for p in roi_key.replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def _array_to_png_bytes(arr_uint8: np.ndarray) -> bytes:
    """Convert an (H, W) or (H, W, 3) uint8 array to PNG bytes via Pillow."""
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf.read()


def _stretch_to_uint8(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Contrast-stretch a band to [0, 255] uint8."""
    return (stretch(band, lo, hi) * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Routes – UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    err, root = _require_dataset()
    if err:
        return jsonify({"dataset_root": None, "rasterio": RASTERIO_AVAILABLE,
                        "pil": PIL_AVAILABLE, "error": err["error"]}), 503
    keys = list_roi_keys(root)
    return jsonify({
        "dataset_root": str(root),
        "rasterio": RASTERIO_AVAILABLE,
        "pil": PIL_AVAILABLE,
        "roi_count": len(keys),
    })


@app.route("/api/rois")
def api_rois():
    err, root = _require_dataset()
    if err:
        return jsonify(err), 503
    keys = list_roi_keys(root)
    rois = []
    for key in keys:
        partition, roi_num, roi_id = _roi_parts(key)
        rois.append({
            "key": key,
            "partition": partition,
            "roi_collection": roi_num,
            "roi_id": roi_id,
        })
    return jsonify({"rois": rois, "count": len(rois)})


@app.route("/api/rois/<path:roi_key>/acquisitions")
def api_acquisitions(roi_key: str):
    err, root = _require_dataset()
    if err:
        return jsonify(err), 503

    partition, roi_num, roi_id = _roi_parts(roi_key)
    if not partition:
        return jsonify({"error": f"Invalid roi_key: {roi_key}"}), 400

    roi_dir = root / partition / f"ROIs{roi_num}" / roi_id
    if not roi_dir.is_dir():
        return jsonify({"error": f"ROI directory not found: {roi_dir}"}), 404

    acq_set: Dict[int, Dict[str, bool]] = {}
    for sensor in ("S1", "S2"):
        sensor_dir = roi_dir / sensor
        if not sensor_dir.is_dir():
            continue
        for acq_dir in sensor_dir.iterdir():
            if acq_dir.is_dir() and acq_dir.name.isdigit():
                idx = int(acq_dir.name)
                if idx not in acq_set:
                    acq_set[idx] = {"s1": False, "s2": False}
                acq_set[idx][sensor.lower()] = any(acq_dir.glob("*.tif*"))

    # Extract one representative date per acquisition from a filename
    dates: Dict[int, Dict[str, str]] = {}
    date_re = re.compile(r"_(\d{4}-\d{2}-\d{2})_patch_")
    for sensor in ("S1", "S2"):
        sensor_dir = roi_dir / sensor
        if not sensor_dir.is_dir():
            continue
        for acq_dir in sorted(sensor_dir.iterdir()):
            if not (acq_dir.is_dir() and acq_dir.name.isdigit()):
                continue
            idx = int(acq_dir.name)
            for f in acq_dir.iterdir():
                m = date_re.search(f.name)
                if m:
                    dates.setdefault(idx, {})[sensor.lower()] = m.group(1)
                    break

    acquisitions = []
    for idx in sorted(acq_set.keys()):
        info = acq_set[idx]
        acquisitions.append({
            "index": idx,
            "has_s1": info["s1"],
            "has_s2": info["s2"],
            "s1_date": dates.get(idx, {}).get("s1", ""),
            "s2_date": dates.get(idx, {}).get("s2", ""),
        })

    return jsonify({"roi_key": roi_key, "acquisitions": acquisitions})


@app.route("/api/rois/<path:roi_key>/acquisitions/<int:acq_idx>/patches")
def api_patches(roi_key: str, acq_idx: int):
    err, root = _require_dataset()
    if err:
        return jsonify(err), 503

    partition, roi_num, roi_id = _roi_parts(roi_key)
    if not partition:
        return jsonify({"error": f"Invalid roi_key: {roi_key}"}), 400

    patch_ids: List[int] = []
    for sensor in ("S1", "S2"):
        acq_dir = root / partition / f"ROIs{roi_num}" / roi_id / sensor / str(acq_idx)
        if not acq_dir.is_dir():
            continue
        for f in acq_dir.glob("*.tif*"):
            m = PATCH_ID_RE.search(f.name)
            if m:
                patch_ids.append(int(m.group("pid")))

    patch_ids = sorted(set(patch_ids))
    return jsonify({
        "roi_key": roi_key,
        "acquisition_index": acq_idx,
        "patch_ids": patch_ids,
        "count": len(patch_ids),
    })


@app.route("/api/image")
def api_image():
    """
    Serve a single-band or RGB image as PNG.

    Query params:
        roi_key  : e.g. asiaWest_n/1868/127
        acq      : acquisition index (int)
        patch    : patch id (int)
        view     : one of  s1_vv | s1_vh | s2_true | s2_false
    """
    err, root = _require_dataset()
    if err:
        return jsonify(err), 503
    if not RASTERIO_AVAILABLE:
        return jsonify({"error": "rasterio not installed"}), 503
    if not PIL_AVAILABLE:
        return jsonify({"error": "Pillow not installed"}), 503

    roi_key = request.args.get("roi_key", "")
    try:
        acq = int(request.args.get("acq", 0))
        patch = int(request.args.get("patch", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "acq and patch must be integers"}), 400

    view = request.args.get("view", "s2_true").lower()
    if view not in ("s1_vv", "s1_vh", "s2_true", "s2_false"):
        return jsonify({"error": "view must be s1_vv, s1_vh, s2_true, or s2_false"}), 400

    sensor = "S1" if view.startswith("s1") else "S2"
    path = resolve_patch(root, roi_key, acq, sensor, patch)
    if path is None:
        return jsonify({"error": f"Patch not found: roi={roi_key} acq={acq} patch={patch} sensor={sensor}"}), 404

    try:
        if view == "s1_vv":
            data = read_bands(path)  # (2, H, W)
            arr = _stretch_to_uint8(data[0], *S1_STRETCH)

        elif view == "s1_vh":
            data = read_bands(path)
            band = data[1] if data.shape[0] > 1 else data[0]
            arr = _stretch_to_uint8(band, *S1_STRETCH)

        elif view == "s2_true":
            data = read_bands(path, indices=list(TRUE_COLOR_BANDS))  # (3, H, W)
            rgb = (to_rgb(data, *S2_STRETCH) * 255).clip(0, 255).astype(np.uint8)
            arr = rgb

        else:  # s2_false
            data = read_bands(path, indices=list(FALSE_COLOR_BANDS))
            rgb = (to_rgb(data, *S2_STRETCH) * 255).clip(0, 255).astype(np.uint8)
            arr = rgb

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    png_bytes = _array_to_png_bytes(arr)
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        max_age=300,
    )


@app.route("/api/image/metadata")
def api_image_metadata():
    """Return pixel statistics for a patch (no pixel data sent to client)."""
    err, root = _require_dataset()
    if err:
        return jsonify(err), 503
    if not RASTERIO_AVAILABLE:
        return jsonify({"error": "rasterio not installed"}), 503

    roi_key = request.args.get("roi_key", "")
    try:
        acq = int(request.args.get("acq", 0))
        patch = int(request.args.get("patch", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "acq and patch must be integers"}), 400

    result = {}
    for sensor, label in (("S1", "s1"), ("S2", "s2")):
        path = resolve_patch(root, roi_key, acq, sensor, patch)
        if path is None:
            result[label] = None
            continue
        try:
            import rasterio  # type: ignore[import-untyped]
            with rasterio.open(path) as src:
                meta = {
                    "file": path.name,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtype": str(src.dtypes[0]),
                    "crs": src.crs.to_string() if src.crs else "",
                    "date": "",
                }
                # extract date from filename
                m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
                if m:
                    meta["date"] = m.group(1)
            result[label] = meta
        except Exception as exc:
            result[label] = {"error": str(exc)}

    return jsonify({"roi_key": roi_key, "acq": acq, "patch": patch, **result})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="READ-ONLY Flask viewer for the SEN12MS-CR-TS dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="Dataset root (auto-detected if omitted).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (use 0.0.0.0 for LAN access).")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode (auto-reload).")
    return parser.parse_args(argv)


def main(argv=None):
    global DATASET_ROOT

    args = parse_args(argv)

    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root.resolve()
    else:
        DATASET_ROOT = find_dataset_root(DEFAULT_WORKSPACE_ROOT)

    if DATASET_ROOT is None or not DATASET_ROOT.is_dir():
        print(
            f"ERROR: Could not find dataset root under '{DEFAULT_WORKSPACE_ROOT}'.\n"
            "Use --dataset-root to specify it explicitly.",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print("  SEN12MS-CR-TS Dataset Viewer  –  READ ONLY")
    print("=" * 60)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  ROIs found   : {len(list_roi_keys(DATASET_ROOT))}")
    print(f"  Rasterio     : {'available' if RASTERIO_AVAILABLE else 'NOT installed'}")
    print(f"  Pillow       : {'available' if PIL_AVAILABLE else 'NOT installed'}")
    print(f"  Server       : http://{args.host}:{args.port}/")
    print("=" * 60)
    print("  Open the URL above in your browser.")
    print("  Press CTRL+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
