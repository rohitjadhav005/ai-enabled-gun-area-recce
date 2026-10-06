#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_dataset.py
====================

Simple, READ-ONLY visualization of the SEN12MS-CR-TS Sentinel-1 (SAR) /
Sentinel-2 (optical) reconnaissance dataset.

What this script does
---------------------
1. Auto-detects the dataset root by searching for the 'Dataset' subfolder
   relative to the project root, so no hardcoded absolute paths are required.
2. Resolves one SAR/optical patch triplet for a chosen ROI, acquisition index
   and patch id (same filename convention as inspect_dataset.py).
3. Reads the GeoTIFF *pixel* data (unlike the Phase 1 header-only inspector).
4. Contrast-stretches each band and renders one figure:
       - S1 VV  (SAR backscatter, dB, grayscale)
       - S1 VH  (SAR backscatter, dB, grayscale)
       - S2 true-colour RGB (B4/B3/B2) or false-colour (B8/B4/B3)
5. Saves the figure as PNG inside sentinel_dataset_analysis/visualizations/.

`--overview` instead renders a small metadata summary (S1 vs S2 file counts per
ROI and unique acquisition dates) from metadata/*.csv written by
inspect_dataset.py.

What this script NEVER does
---------------------------
* It never moves, renames, edits, deletes or copies any original TIFF.
* It never writes inside the dataset root (outputs live in this folder only).
* It performs no scientific correction: bands are only contrast-stretched for
  display, and S1 values are shown as delivered (dB).

Real dataset structure (SEN12MS-CR-TS)
---------------------------------------
  <dataset_root>/
  └── <partition>/            e.g. asiaWest_n
      └── ROIs<number>/       e.g. ROIs1868, ROIs1970, ROIs2017
          └── <roi_id>/       e.g. 127, 112, 57
              ├── S1/
              │   └── <acq_idx>/   e.g. 0, 1, 2, 3
              │       └── s1_ROIs<number>_<roi_id>_ImgNo_<acq_idx>_<date>_patch_<id>.tif
              └── S2/
                  └── <acq_idx>/
                      └── s2_ROIs<number>_<roi_id>_ImgNo_<acq_idx>_<date>_patch_<id>.tif

ROI key format: "<partition>/<roi_number>/<roi_id>"
  e.g.  asiaWest_n/1868/127

Usage (PowerShell, from the repository root)
--------------------------------------------
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py --patch-id 5 --show
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py --false-color
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py --list
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py --overview
    python sentinel_dataset_analysis\\scripts\\visualize_dataset.py --dataset-root "C:\\path\\to\\Dataset"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency: rasterio (GeoTIFF reader)
# ---------------------------------------------------------------------------
try:
    # pyrefly: ignore [missing-import]
    import rasterio

    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    rasterio = None
    RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Path layout (resolved from this file's location):
#   <project_root>/scripts/visualize_dataset.py
#   => <project_root> == sentinel_dataset_analysis/
#   => <workspace_root> == sentineldataset/   (one level up)
# The actual image data lives in <workspace_root>/Dataset/
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sentinel_dataset_analysis/
DEFAULT_WORKSPACE_ROOT = DEFAULT_PROJECT_ROOT.parent         # sentineldataset/

# Candidate sub-folder names (case-insensitive) that hold the actual .tif data.
# The script tries them in order and uses the first one that exists.
DATASET_SUBFOLDER_CANDIDATES = ["Dataset", "dataset", "data", "Data"]

# Directories to always skip when searching for data.
SKIP_DIRS = {"sentinel_dataset_analysis", "__pycache__", ".git", ".venv", "venv", "node_modules"}

# SEN12MS band ordering (0-based indices).
S1_BAND_NAMES = ("VV", "VH")
S2_BAND_NAMES = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
)
TRUE_COLOR_BANDS = (3, 2, 1)   # R=B4, G=B3, B=B2
FALSE_COLOR_BANDS = (7, 3, 2)  # R=B8 (NIR), G=B4, B=B3

# Filename patterns used in SEN12MS-CR-TS:
#   s1_ROIs1868_127_ImgNo_3_2018-02-12_patch_49.tif
#   s2_ROIs1868_127_ImgNo_3_2018-02-12_patch_49.tif
S1_FILENAME_RE = re.compile(r"^s1_", re.IGNORECASE)
S2_FILENAME_RE = re.compile(r"^s2_", re.IGNORECASE)
DATE_RE = re.compile(r"_(?P<date>\d{4}-\d{2}-\d{2})_patch_\d+\.tiff?$", re.IGNORECASE)
PATCH_ID_RE = re.compile(r"_patch_(?P<pid>\d+)\.tiff?$", re.IGNORECASE)
IMGNO_RE = re.compile(r"_ImgNo_(?P<idx>\d+)_", re.IGNORECASE)
ROIS_DIR_RE = re.compile(r"^ROIs(?P<num>\d+)$", re.IGNORECASE)

# Display stretch defaults (percentiles of the finite pixel values).
S1_STRETCH = (2.0, 98.0)   # SAR dB
S2_STRETCH = (2.0, 98.0)   # optical reflectance DN


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure a console logger (idempotent)."""
    logger = logging.getLogger("visualize_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(levelname)-7s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Dataset root auto-detection
# ---------------------------------------------------------------------------
def find_dataset_root(workspace_root: Path) -> Optional[Path]:
    """
    Auto-detect the dataset root directory.

    Search strategy (in order):
    1. Check if any of the DATASET_SUBFOLDER_CANDIDATES exists directly under
       workspace_root and contains at least one partition directory with ROIs*.
    2. Fall back to workspace_root itself if it directly contains ROIs-style
       partition directories.

    Returns the detected dataset root, or None if nothing is found.
    """
    # Strategy 1: look for a Dataset/ subfolder
    for candidate in DATASET_SUBFOLDER_CANDIDATES:
        candidate_path = workspace_root / candidate
        if candidate_path.is_dir():
            return candidate_path

    # Strategy 2: workspace_root itself contains partitions with ROIs
    for child in workspace_root.iterdir():
        if child.is_dir() and child.name not in SKIP_DIRS:
            for grandchild in child.iterdir():
                if grandchild.is_dir() and ROIS_DIR_RE.match(grandchild.name):
                    return workspace_root

    return None


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------
def stretch(
    band: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Contrast-stretch a 2-D band to the float range [0, 1].

    Non-finite pixels (NaN/Inf) are mapped to 0. If the band is constant, a
    black image is returned rather than dividing by zero.
    """
    data = np.asarray(band, dtype=np.float64)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.float64)

    values = data[finite]
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float64)

    out = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)
    out[~finite] = 0.0
    return out


def band_stats(band: np.ndarray) -> Tuple[float, float, float]:
    """Return (min, max, mean) over finite pixels (NaN when there are none)."""
    data = np.asarray(band, dtype=np.float64)
    finite = np.isfinite(data)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")
    v = data[finite]
    return float(v.min()), float(v.max()), float(v.mean())


def read_bands(path: Path, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read selected bands (0-based); returns an array shaped (bands, H, W)."""
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(
            "rasterio is not installed. Install with: pip install rasterio"
        )
    with rasterio.open(path) as src:
        # rasterio band indexes are 1-based; the public API here is 0-based.
        if indices is None:
            wanted = list(range(1, src.count + 1))
        else:
            invalid = [i for i in indices if i < 0 or i >= src.count]
            if invalid:
                raise ValueError(
                    f"{path.name}: band index(es) {invalid} out of range for a "
                    f"{src.count}-band raster (valid: 0..{src.count - 1})"
                )
            wanted = [i + 1 for i in indices]
        return src.read(wanted)


def extract_date(path: Path) -> str:
    """Pull the acquisition date out of a SEN12MS patch filename."""
    match = DATE_RE.search(path.name)
    return match.group("date") if match else "unknown"


def to_rgb(stack: np.ndarray, lo_pct: float, hi_pct: float, gamma: float = 1.0) -> np.ndarray:
    """Stack of (3, H, W) DN bands -> display-ready (H, W, 3) float RGB in [0, 1]."""
    channels = [stretch(stack[i], lo_pct, hi_pct, gamma) for i in range(stack.shape[0])]
    return np.dstack(channels)


# ---------------------------------------------------------------------------
# Dataset navigation (read-only)
# ---------------------------------------------------------------------------
def list_roi_keys(dataset_root: Path) -> List[str]:
    """
    Return available ROI keys as ``<partition>/<roi_number>/<roi_id>``.

    Real SEN12MS-CR-TS layout under dataset_root:
        <partition>/ROIs<number>/<roi_id>/S1|S2/...

    Examples of returned keys:
        asiaWest_n/1868/127
        asiaWest_n/1970/112
    """
    keys: List[str] = []
    if not dataset_root.is_dir():
        return keys

    for partition in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if partition.name in SKIP_DIRS:
            continue
        collections = sorted(
            p for p in partition.iterdir()
            if p.is_dir() and ROIS_DIR_RE.match(p.name)
        )
        for collection in collections:
            roi_num = ROIS_DIR_RE.match(collection.name).group("num")  # type: ignore[union-attr]
            for roi_dir in sorted(p for p in collection.iterdir() if p.is_dir()):
                keys.append(f"{partition.name}/{roi_num}/{roi_dir.name}")
    return keys


def scan_tif_counts(
    dataset_root: Path,
    roi_keys: List[str],
    limit: int = 50,
) -> Dict[str, Dict[str, int]]:
    """
    Return a dict mapping each roi_key -> {"S1": count, "S2": count}.

    Scans only the first `limit` tif files per sensor folder to keep this
    lightweight (lazy metadata inspection – no pixel data loaded).
    """
    counts: Dict[str, Dict[str, int]] = {}
    for key in roi_keys:
        parts = key.split("/")
        if len(parts) < 3:
            continue
        partition, roi_num, roi_id = parts[0], parts[1], parts[2]
        roi_dir = dataset_root / partition / f"ROIs{roi_num}" / roi_id
        s1_count = 0
        s2_count = 0
        for sensor, counter in (("S1", "s1_count"), ("S2", "s2_count")):
            sensor_dir = roi_dir / sensor
            if sensor_dir.is_dir():
                n = sum(
                    1 for acq in sensor_dir.iterdir()
                    if acq.is_dir()
                    for f in acq.glob("*.tif*")
                    if f.is_file()
                )
                if sensor == "S1":
                    s1_count = n
                else:
                    s2_count = n
        counts[key] = {"S1": s1_count, "S2": s2_count}
    return counts


def resolve_patch(
    dataset_root: Path,
    roi_key: str,
    acquisition_index: int,
    sensor: str,
    patch_id: int,
) -> Optional[Path]:
    """Locate the GeoTIFF for a (roi_key, acquisition_index, sensor, patch_id)."""
    parts = [p for p in roi_key.replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        return None
    partition, roi_num, roi_id = parts[0], parts[1], parts[2]
    folder = (
        dataset_root / partition / f"ROIs{roi_num}" / roi_id
        / sensor.upper() / str(acquisition_index)
    )
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(f"*_patch_{patch_id}.tif*"))
    return matches[0] if matches else None


def find_any_patch(
    dataset_root: Path,
    roi_key: str,
) -> Tuple[Optional[Path], Optional[Path], int, int]:
    """
    Find the first available S1+S2 patch pair for *roi_key*.

    Returns (s1_path, s2_path, acquisition_index, patch_id).
    At least one of s1_path/s2_path will be non-None.
    """
    parts = [p for p in roi_key.replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        return None, None, 0, 0
    partition, roi_num, roi_id = parts[0], parts[1], parts[2]
    roi_dir = dataset_root / partition / f"ROIs{roi_num}" / roi_id

    # Collect acquisition indices that exist for S1 or S2
    acq_indices = set()
    for sensor in ("S1", "S2"):
        sensor_dir = roi_dir / sensor
        if sensor_dir.is_dir():
            for acq_dir in sensor_dir.iterdir():
                if acq_dir.is_dir() and acq_dir.name.isdigit():
                    acq_indices.add(int(acq_dir.name))

    for acq_idx in sorted(acq_indices):
        for sensor_dir_name in ("S1", "S2"):
            acq_dir = roi_dir / sensor_dir_name / str(acq_idx)
            if not acq_dir.is_dir():
                continue
            tifs = sorted(acq_dir.glob("*.tif*"))
            if not tifs:
                continue
            # Extract patch_id from the first tif found
            m = PATCH_ID_RE.search(tifs[0].name)
            pid = int(m.group("pid")) if m else 0
            s1 = resolve_patch(dataset_root, roi_key, acq_idx, "S1", pid)
            s2 = resolve_patch(dataset_root, roi_key, acq_idx, "S2", pid)
            if s1 is not None or s2 is not None:
                return s1, s2, acq_idx, pid

    return None, None, 0, 0


# ---------------------------------------------------------------------------
# Patch figure
# ---------------------------------------------------------------------------
def visualize_patch(
    dataset_root: Path,
    out_dir: Path,
    roi_key: str,
    acquisition_index: int,
    patch_id: int,
    color_bands: Tuple[int, int, int],
    color_label: str,
    show: bool,
    dpi: int,
    logger: logging.Logger,
    auto_find: bool = False,
) -> Optional[Path]:
    """Render S1 VV / S1 VH / S2 colour-composite for one patch; return the PNG path."""
    s1_path = resolve_patch(dataset_root, roi_key, acquisition_index, "S1", patch_id)
    s2_path = resolve_patch(dataset_root, roi_key, acquisition_index, "S2", patch_id)

    # If exact acq/patch combo not found and auto_find is enabled, search for any valid patch
    if s1_path is None and s2_path is None and auto_find:
        logger.info(
            "Patch (acq=%d, patch=%d) not found for ROI %s; "
            "auto-searching for any available patch ...",
            acquisition_index, patch_id, roi_key,
        )
        s1_path, s2_path, acquisition_index, patch_id = find_any_patch(dataset_root, roi_key)

    if s1_path is None and s2_path is None:
        logger.error(
            "No S1/S2 patch found for ROI=%s acq=%d patch=%d",
            roi_key, acquisition_index, patch_id,
        )
        return None

    import matplotlib.pyplot as plt
    if not show:
        plt.switch_backend("Agg")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    # --- S1 (SAR) VV / VH, dB, grayscale ---
    if s1_path is not None:
        s1 = read_bands(s1_path)  # (2, H, W): VV, VH
        if s1.shape[0] < 2:
            logger.warning(
                "S1 patch has %d band(s); expected VV + VH. Falling back to band 1.",
                s1.shape[0],
            )
        vv = s1[0]
        vh = s1[1] if s1.shape[0] > 1 else s1[0]
        vv_min, vv_max, _ = band_stats(vv)
        vh_min, vh_max, _ = band_stats(vh)
        axes[0].imshow(stretch(vv, *S1_STRETCH), cmap="gray")
        axes[0].set_title(f"S1 {S1_BAND_NAMES[0]} (dB)  [{vv_min:.1f}, {vv_max:.1f}]", fontsize=10)
        axes[1].imshow(stretch(vh, *S1_STRETCH), cmap="gray")
        axes[1].set_title(f"S1 {S1_BAND_NAMES[1]} (dB)  [{vh_min:.1f}, {vh_max:.1f}]", fontsize=10)
    else:
        for ax in (axes[0], axes[1]):
            ax.text(0.5, 0.5, "S1 missing", ha="center", va="center", fontsize=11)

    # --- S2 (optical) colour composite ---
    if s2_path is not None:
        s2 = read_bands(s2_path, indices=list(color_bands))  # (3, H, W)
        axes[2].imshow(to_rgb(s2, *S2_STRETCH))
        band_names = "/".join(S2_BAND_NAMES[i] for i in color_bands)
        axes[2].set_title(f"S2 {color_label} ({band_names})", fontsize=10)
    else:
        axes[2].text(0.5, 0.5, "S2 missing", ha="center", va="center", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    s1_date = extract_date(s1_path) if s1_path else "n/a"
    s2_date = extract_date(s2_path) if s2_path else "n/a"
    fig.suptitle(
        f"SEN12MS-CR-TS patch - ROI {roi_key} | "
        f"acquisition #{acquisition_index} | patch {patch_id}\n"
        f"S1 date {s1_date}  |  S2 date {s2_date}  "
        f"(percentile stretch {int(S2_STRETCH[0])}-{int(S2_STRETCH[1])}%)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_key = roi_key.replace("/", "-").replace("\\", "-")
    out_path = out_dir / f"patch_{safe_key}_acq{acquisition_index}_patch{patch_id}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Metadata overview figure
# ---------------------------------------------------------------------------
def visualize_overview(
    project_root: Path,
    out_dir: Path,
    show: bool,
    dpi: int,
    logger: logging.Logger,
) -> Optional[Path]:
    """Render a metadata summary (counts per ROI + unique dates) from inspect_csv output."""
    import pandas as pd

    inv_path = project_root / "metadata" / "dataset_inventory.csv"
    if not inv_path.exists():
        logger.error("Missing %s - run inspect_dataset.py first.", inv_path)
        return None

    inv = pd.read_csv(inv_path)

    required_columns = {"roi_key", "sensor", "acquisition_date"}
    missing = sorted(required_columns - set(inv.columns))
    if missing:
        logger.error(
            "%s is missing column(s): %s - rerun inspect_dataset.py.",
            inv_path.name,
            ", ".join(missing),
        )
        return None
    if inv.empty:
        logger.error(
            "%s has 0 rows - rerun inspect_dataset.py without --limit.",
            inv_path.name,
        )
        return None

    import matplotlib.pyplot as plt
    if not show:
        plt.switch_backend("Agg")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = inv.groupby(["roi_key", "sensor"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["S1", "S2"], fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[0], color=["#2b6cb0", "#c05621"])
    axes[0].set_title("GeoTIFF count per ROI (S1 vs S2)")
    axes[0].set_xlabel("ROI key")
    axes[0].set_ylabel("files")
    axes[0].tick_params(axis="x", labelrotation=45)

    dates = inv.groupby("roi_key")["acquisition_date"].nunique()
    dates.plot(kind="bar", ax=axes[1], color="#2f855a")
    axes[1].set_title("Unique acquisition dates per ROI")
    axes[1].set_xlabel("ROI key")
    axes[1].set_ylabel("dates")
    axes[1].tick_params(axis="x", labelrotation=45)

    fig.suptitle(
        f"SEN12MS-CR-TS dataset overview - {inv['roi_key'].nunique()} ROIs, "
        f"{len(inv)} files",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_overview.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="READ-ONLY visualization of the SEN12MS-CR-TS "
        "Sentinel-1 (SAR) / Sentinel-2 (optical) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help=(
            "Explicit path to the dataset root (the folder containing partition "
            "directories such as 'asiaWest_n'). "
            "When omitted the script auto-detects it by looking for a 'Dataset' "
            "subfolder next to the project."
        ),
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="Analysis folder where PNGs are written.")
    parser.add_argument("--roi-key", default=None,
                        help="ROI key like asiaWest_n/1868/127 (default: first found).")
    parser.add_argument("--acquisition-index", type=int, default=0,
                        help="Acquisition / time-step index (0-3).")
    parser.add_argument("--patch-id", type=int, default=0,
                        help="Patch id within the acquisition.")
    parser.add_argument("--false-color", action="store_true",
                        help="Use false colour (B8/B4/B3) instead of true colour (B4/B3/B2).")
    parser.add_argument("--overview", action="store_true",
                        help="Render a metadata summary instead of a single patch.")
    parser.add_argument("--list", action="store_true",
                        help="List available ROI keys and exit.")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window in addition to saving the PNG.")
    parser.add_argument("--dpi", type=int, default=150, help="Output PNG resolution.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    out_dir = project_root / "visualizations"
    logger = setup_logging(verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("SEN12MS-CR-TS dataset visualization - READ ONLY")
    logger.info("Project root : %s", project_root)
    logger.info("Output folder: %s", out_dir)

    # ------------------------------------------------------------------
    # Resolve dataset root
    # ------------------------------------------------------------------
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.resolve()
        logger.info("Dataset root : %s  (explicit --dataset-root)", dataset_root)
    else:
        # Auto-detect: look for a 'Dataset' subfolder next to the project root.
        detected = find_dataset_root(DEFAULT_WORKSPACE_ROOT)
        if detected is None:
            logger.error(
                "Could not auto-detect the dataset root under '%s'. "
                "Expected a subfolder named 'Dataset' (or similar) containing "
                "partition directories with ROIs* sub-directories. "
                "Use --dataset-root to specify the path explicitly.",
                DEFAULT_WORKSPACE_ROOT,
            )
            return 2
        dataset_root = detected.resolve()
        logger.info("Dataset root : %s  (auto-detected)", dataset_root)

    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    if not dataset_root.exists():
        logger.error("Dataset root does not exist: %s", dataset_root)
        return 2
    if not dataset_root.is_dir():
        logger.error("Dataset root is not a directory: %s", dataset_root)
        return 2
    if args.dpi < 1:
        logger.error("--dpi must be a positive integer (got %d).", args.dpi)
        return 2
    if args.acquisition_index < 0:
        logger.error("--acquisition-index must be zero or greater (got %d).", args.acquisition_index)
        return 2
    if args.patch_id < 0:
        logger.error("--patch-id must be zero or greater (got %d).", args.patch_id)
        return 2

    # ------------------------------------------------------------------
    # Discover ROIs and log a brief inventory summary
    # ------------------------------------------------------------------
    keys = list_roi_keys(dataset_root)
    logger.info("Detected dataset structure:")
    logger.info("  Partitions   : %s",
                ", ".join(sorted({k.split("/")[0] for k in keys})) or "none")
    logger.info("  ROI folders  : %d", len(keys))

    if keys:
        # Lightweight count of a sample ROI to confirm files are present
        sample_counts = scan_tif_counts(dataset_root, keys[:1])
        sample_key = keys[0]
        sc = sample_counts.get(sample_key, {})
        logger.info(
            "  Sample ROI   : %s  ->  S1 files: %d  |  S2 files: %d",
            sample_key, sc.get("S1", 0), sc.get("S2", 0),
        )

    # ------------------------------------------------------------------
    # --list mode
    # ------------------------------------------------------------------
    if args.list:
        logger.info("All available ROI keys (%d):", len(keys))
        for key in keys:
            logger.info("  %s", key)
        return 0

    # ------------------------------------------------------------------
    # --overview mode
    # ------------------------------------------------------------------
    if args.overview:
        result = visualize_overview(project_root, out_dir, args.show, args.dpi, logger)
        return 0 if result else 4

    # ------------------------------------------------------------------
    # Patch visualization mode
    # ------------------------------------------------------------------
    if not RASTERIO_AVAILABLE:
        logger.error(
            "rasterio is required to read GeoTIFF pixels. Install it with: pip install rasterio"
        )
        return 3

    if not keys:
        searched_dirs = ", ".join(
            str(dataset_root / candidate)
            for candidate in DATASET_SUBFOLDER_CANDIDATES
        )
        logger.error(
            "No ROI folders found under: %s\n"
            "  Expected structure: <partition>/ROIs<number>/<roi_id>/S1|S2/\n"
            "  Searched in      : %s\n"
            "  File extensions  : .tif, .tiff\n"
            "  Use --dataset-root to point directly at the folder that contains "
            "the partition directories.",
            dataset_root,
            searched_dirs,
        )
        return 5

    roi_key = args.roi_key
    if not roi_key:
        roi_key = keys[0]
        logger.info("No --roi-key given; defaulting to %s", roi_key)
    else:
        if roi_key not in keys:
            logger.error("Unknown --roi-key: %s", roi_key)
            logger.error("Available ROI keys:")
            for key in keys:
                logger.error("  %s", key)
            return 6

    color_bands = FALSE_COLOR_BANDS if args.false_color else TRUE_COLOR_BANDS
    color_label = "false colour" if args.false_color else "true colour"

    try:
        result = visualize_patch(
            dataset_root=dataset_root,
            out_dir=out_dir,
            roi_key=roi_key,
            acquisition_index=args.acquisition_index,
            patch_id=args.patch_id,
            color_bands=color_bands,
            color_label=color_label,
            show=args.show,
            dpi=args.dpi,
            logger=logger,
            auto_find=True,   # auto-find if the requested acq/patch combo is absent
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Could not render the requested patch: %s", exc)
        return 7
    if result is None:
        return 6

    logger.info("-" * 70)
    logger.info("Original dataset was NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
