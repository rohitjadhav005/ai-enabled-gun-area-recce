#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_dataset.py
==================

Phase 1 (READ-ONLY) inspection + metadata generation for the SEN12MS-CR-TS
based Sentinel-1 (SAR) / Sentinel-2 (optical) reconnaissance dataset.

What this script does
---------------------
1. Recursively scans the dataset root (default:
   C:\\Users\\rohii\\OneDrive\\Desktop\\sentineldataset).
2. Identifies geographic partitions, ROI collections, ROI ids, sensors,
   acquisition sequence folders, acquisition dates and patch identifiers.
3. Reads GeoTIFF *headers only* (never pixel data) to capture width, height,
   band count and data type.
4. Detects empty / unreadable TIFFs and duplicate filenames.
5. Proposes potential SAR <-> optical patch relationships and marks the ones
   whose dates/dimensions do not line up as "unverified" instead of assuming
   they are valid pairs.
6. Writes reports + CSV metadata into the sibling analysis folder.

What this script NEVER does
---------------------------
* It never moves, renames, deletes, edits or copies any original TIFF.
* It never reads pixel payloads into memory (header metadata only).
* It never writes anything inside the dataset root (outputs live separately).
* It never overwrites files it did not generate in this run.

Outputs (all under sentinel_dataset_analysis/)
----------------------------------------------
reports/   dataset_summary.txt, folder_structure.txt, validation_report.txt
metadata/  dataset_inventory.csv, roi_metadata.csv, image_relationships.csv
logs/      inspect_dataset_<timestamp>.log

Usage (PowerShell, from the repository root)
--------------------------------------------
    python sentinel_dataset_analysis\\scripts\\inspect_dataset.py
    python sentinel_dataset_analysis\\scripts\\inspect_dataset.py `
        --dataset-root "C:\\path\\to\\sentineldataset"
    python sentinel_dataset_analysis\\scripts\\inspect_dataset.py --limit 200   # smoke test
    python sentinel_dataset_analysis\\scripts\\inspect_dataset.py --skip-raster  # headers off
    python sentinel_dataset_analysis\\scripts\\inspect_dataset.py --no-relationships
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency: rasterio (GeoTIFF header reader)
# ---------------------------------------------------------------------------
try:
    import rasterio

    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    rasterio = None
    RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT.parent

IMAGE_EXTENSIONS = (".tif", ".tiff")

SENSOR_DIR_RE = re.compile(r"^(?P<sensor>s[12])$", re.IGNORECASE)
NUMERIC_DIR_RE = re.compile(r"^\d+$")

# Sentinel-1/2 SEN12MS-CR-TS patch filename pattern.
FILENAME_RE = re.compile(
    r"^(?P<sensor>s[12])"
    r"_ROIs(?P<roi_collection>\d+)"
    r"_(?P<roi_id>\d+)"
    r"_ImgNo_(?P<acquisition_index>\d+)"
    r"_(?P<acquisition_date>\d{4}-\d{2}-\d{2})"
    r"_patch_(?P<patch_id>\d+)\.tiff?$",
    re.IGNORECASE,
)

# Folders that must never be scanned / reported as dataset content.
IGNORED_DIR_NAMES = {"sentinel_dataset_analysis", ".git", "__pycache__", ".ipynb_checkpoints"}

INVENTORY_COLUMNS = [
    "file_name",
    "original_path",
    "geographic_partition",
    "roi_collection",
    "roi_id",
    "sensor",
    "acquisition_index",
    "acquisition_date",
    "patch_id",
    "width",
    "height",
    "band_count",
    "data_type",
    "file_size_mb",
    "validation_status",
]

ROI_COLUMNS = [
    "geographic_partition",
    "roi_collection",
    "roi_id",
    "roi_key",
    "sensors_present",
    "total_image_count",
    "s1_image_count",
    "s2_image_count",
    "acquisition_count",
    "acquisition_indices",
    "unique_date_count",
    "s1_date_min",
    "s1_date_max",
    "s2_date_min",
    "s2_date_max",
    "patch_id_min",
    "patch_id_max",
    "patches_per_acquisition_median",
    "width",
    "height",
    "s1_band_count",
    "s2_band_count",
    "s1_data_type",
    "s2_data_type",
    "crs",
    "unreadable_count",
    "empty_count",
    "duplicate_filename_count",
    "notes",
]

RELATIONSHIP_COLUMNS = [
    "relationship_id",
    "geographic_partition",
    "roi_collection",
    "roi_id",
    "roi_key",
    "acquisition_index",
    "patch_id",
    "s1_file_name",
    "s2_file_name",
    "s1_acquisition_date",
    "s2_acquisition_date",
    "date_delta_days",
    "match_criteria",
    "relationship_type",
    "validation_status",
    "notes",
]

# A SAR/optical pair is treated as a *verified* match only when the
# acquisition dates are identical. Anything else is reported as unverified.
VERIFIED_DATE_DELTA_DAYS = 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure a console + timestamped file logger. Idempotent."""
    logger = logging.getLogger("inspect_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"inspect_dataset_{stamp}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Filesystem helpers (read-only)
# ---------------------------------------------------------------------------
def ensure_output_dirs(project_root: Path) -> Dict[str, Path]:
    """Create the analysis output folders if missing and return them."""
    dirs = {
        "root": project_root,
        "scripts": project_root / "scripts",
        "metadata": project_root / "metadata",
        "reports": project_root / "reports",
        "logs": project_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def is_within(child: Path, parent: Path) -> bool:
    """True when `child` is inside `parent` (path-safe on Windows)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def os_walk(root: Path, ignored_names: Iterable[str]) -> Iterable[Tuple[str, List[str], List[str]]]:
    """Thin wrapper around os.walk that tolerates permission errors."""
    ignored = set(ignored_names)
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d not in ignored]
        yield dirpath, dirnames, filenames


def find_tiff_files(
    dataset_root: Path,
    excluded_dirs: Sequence[Path],
    logger: logging.Logger,
) -> List[Path]:
    """Recursively collect every GeoTIFF under `dataset_root`, skipping excluded dirs."""
    logger.info("Scanning for GeoTIFF files under: %s", dataset_root)
    started = time.perf_counter()
    found: List[Path] = []

    for dirpath, dirnames, filenames in os_walk(dataset_root, ignored_names=IGNORED_DIR_NAMES):
        current = Path(dirpath)
        if any(is_within(current, ex) for ex in excluded_dirs):
            dirnames[:] = []
            continue

        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(current / name)

    elapsed = time.perf_counter() - started
    logger.info("Discovered %d GeoTIFF file(s) in %.2f s", len(found), elapsed)
    found.sort()
    return found


# ---------------------------------------------------------------------------
# Filename / path parsing
# ---------------------------------------------------------------------------
def parse_record(file_path: Path, dataset_root: Path) -> Optional[Dict[str, Any]]:
    """Derive structured attributes from a patch path + filename.

    Returns None when the file is not a recognisable SEN12MS-CR-TS patch.
    Path layout (relative to the dataset root):
        <geographic_partition>/<roi_collection>/<roi_id>/<S1|S2>/<acquisition_index>/<file>
    """
    match = FILENAME_RE.match(file_path.name)
    if not match:
        return None

    try:
        rel_parts = file_path.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = file_path.parts

    parts = rel_parts[:-1]  # drop the filename
    geographic_partition = parts[0] if len(parts) >= 1 else ""
    roi_collection = parts[1] if len(parts) >= 2 else ""
    roi_id = parts[2] if len(parts) >= 3 else ""
    sensor_dir = parts[3] if len(parts) >= 4 else ""
    acquisition_dir = parts[4] if len(parts) >= 5 else ""

    groups = match.groupdict()

    # Filename is authoritative for ids/dates; directory names fill any gaps.
    sensor = (groups["sensor"] or "").upper()
    if not sensor and SENSOR_DIR_RE.match(sensor_dir):
        sensor = sensor_dir.upper()

    acquisition_index = groups["acquisition_index"] or (
        acquisition_dir if NUMERIC_DIR_RE.match(acquisition_dir) else ""
    )

    return {
        "file_name": file_path.name,
        "original_path": str(file_path),
        "geographic_partition": geographic_partition,
        "roi_collection": groups["roi_collection"] or roi_collection,
        "roi_id": groups["roi_id"] or roi_id,
        "sensor": sensor,
        "acquisition_index": acquisition_index,
        "acquisition_date": groups["acquisition_date"] or "",
        "patch_id": groups["patch_id"] or "",
        "sensor_dir_name": sensor_dir,
        "acquisition_dir_name": acquisition_dir,
        "hidden_in_scan_root": len(rel_parts) < 5,
    }


def roi_key(partition: str, roi_collection: str, roi_id: str) -> str:
    """Stable, human-readable identifier for a ROI."""
    if partition:
        return f"{partition}/{roi_collection}/{roi_id}"
    return f"{roi_collection}/{roi_id}"


# ---------------------------------------------------------------------------
# GeoTIFF header inspection
# ---------------------------------------------------------------------------
def read_raster_header(file_path: Path) -> Dict[str, Any]:
    """Read GeoTIFF header metadata only (never pixel data).

    Returns width, height, band_count, data_type, crs, status, error.
    """
    info: Dict[str, Any] = {
        "width": None,
        "height": None,
        "band_count": None,
        "data_type": None,
        "crs": None,
        "status": "ok",
        "error": "",
    }

    try:
        size_bytes = file_path.stat().st_size
    except OSError as exc:
        info["status"] = "unreadable"
        info["error"] = f"stat failed: {exc}"
        return info

    if size_bytes == 0:
        info["status"] = "empty"
        info["error"] = "zero-byte file"
        return info

    if not RASTERIO_AVAILABLE:
        info["status"] = "unknown"
        info["error"] = "rasterio not installed - header not read"
        return info

    try:
        with rasterio.open(file_path) as dataset:
            info["width"] = int(dataset.width)
            info["height"] = int(dataset.height)
            info["band_count"] = int(dataset.count)
            dtypes = sorted({str(d) for d in dataset.dtypes})
            info["data_type"] = "|".join(dtypes) if dtypes else ""
            info["crs"] = dataset.crs.to_string() if dataset.crs is not None else ""
    except Exception as exc:  # rasterio.errors.RasterioIOError and friends
        info["status"] = "unreadable"
        info["error"] = f"{type(exc).__name__}: {exc}"

    return info


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------
def scan_dataset(
    dataset_root: Path,
    project_root: Path,
    logger: logging.Logger,
    skip_raster: bool = False,
    limit: Optional[int] = None,
    progress_every: int = 500,
) -> Tuple[List[Dict[str, Any]], List[Path], Counter]:
    """Walk the dataset once and build one inventory record per TIFF.

    Returns (records, unrecognised_files, scan_stats).
    """
    excluded_dirs = [project_root]
    files = find_tiff_files(dataset_root, excluded_dirs=excluded_dirs, logger=logger)

    if limit is not None and limit > 0:
        logger.warning("Limiting scan to the first %d file(s) (smoke-test mode).", limit)
        files = files[:limit]

    records: List[Dict[str, Any]] = []
    unrecognised: List[Path] = []
    stats: Counter = Counter()

    started = time.perf_counter()
    total = len(files)

    for index, file_path in enumerate(files, start=1):
        parsed = parse_record(file_path, dataset_root)
        if parsed is None:
            unrecognised.append(file_path)
            stats["unrecognised"] += 1
            continue

        # File size (MB).
        try:
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 6)
        except OSError:
            size_mb = None

        if skip_raster:
            header = {
                "width": None,
                "height": None,
                "band_count": None,
                "data_type": None,
                "crs": None,
                "status": "not_checked",
                "error": "raster header read skipped (--skip-raster)",
            }
        else:
            header = read_raster_header(file_path)

        stats[f"status::{header['status']}"] += 1
        stats[f"sensor::{(parsed['sensor'] or 'UNKNOWN').upper()}"] += 1

        record = {
            "file_name": parsed["file_name"],
            "original_path": parsed["original_path"],
            "geographic_partition": parsed["geographic_partition"],
            "roi_collection": parsed["roi_collection"],
            "roi_id": parsed["roi_id"],
            "sensor": parsed["sensor"],
            "acquisition_index": parsed["acquisition_index"],
            "acquisition_date": parsed["acquisition_date"],
            "patch_id": parsed["patch_id"],
            "width": header["width"],
            "height": header["height"],
            "band_count": header["band_count"],
            "data_type": header["data_type"],
            "file_size_mb": size_mb,
            "validation_status": header["status"],
            "crs": header["crs"],
            "roi_key": roi_key(
                parsed["geographic_partition"], parsed["roi_collection"], parsed["roi_id"]
            ),
            "error_detail": header["error"],
            "hidden_in_scan_root": parsed["hidden_in_scan_root"],
        }
        records.append(record)

        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            logger.info(
                "  ... %d/%d files processed (%.1f files/s)", index, total, rate
            )

    elapsed = time.perf_counter() - started
    logger.info(
        "Scanned %d file(s) in %.2f s (%.1f files/s)",
        total,
        elapsed,
        (total / elapsed) if elapsed else 0.0,
    )
    stats["total_scanned"] = total
    return records, unrecognised, stats


# ---------------------------------------------------------------------------
# Metadata table builders
# ---------------------------------------------------------------------------
INVENTORY_COLUMNS_FULL = INVENTORY_COLUMNS + ["roi_key", "crs", "error_detail"]


def build_inventory_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per TIFF file, using the requested inventory schema."""
    if not records:
        return pd.DataFrame(columns=INVENTORY_COLUMNS_FULL)
    df = pd.DataFrame(records)
    return df.reindex(columns=INVENTORY_COLUMNS_FULL)


def find_duplicate_filenames(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Map a filename -> list of paths when the same filename appears >1 time."""
    by_name: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        by_name[record["file_name"]].append(record["original_path"])
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def _mode_or_blank(series: pd.Series) -> Any:
    """Return the most common non-null value, or blank when unavailable."""
    values = series.dropna()
    values = values[values.astype(str).str.len() > 0]
    if values.empty:
        return ""
    return values.mode().iloc[0]


def build_roi_df(records: List[Dict[str, Any]], duplicates: Dict[str, List[str]]) -> pd.DataFrame:
    """Summarise availability + properties for each ROI."""
    if not records:
        return pd.DataFrame(columns=ROI_COLUMNS)

    df = pd.DataFrame(records)
    dup_paths = {p for paths in duplicates.values() for p in paths}

    rows: List[Dict[str, Any]] = []
    for key, group in df.groupby("roi_key", dropna=False):
        partition = str(group["geographic_partition"].iloc[0])
        roi_collection = str(group["roi_collection"].iloc[0])
        roi_id = str(group["roi_id"].iloc[0])

        sensors = sorted({s for s in group["sensor"].dropna().unique() if s})
        s1 = group[group["sensor"] == "S1"]
        s2 = group[group["sensor"] == "S2"]

        acq_indices = sorted(
            {str(v) for v in group["acquisition_index"].dropna().unique() if str(v)}
        )
        dates = sorted({d for d in group["acquisition_date"].dropna().unique() if d})
        s1_dates = sorted({d for d in s1["acquisition_date"].dropna().unique() if d})
        s2_dates = sorted({d for d in s2["acquisition_date"].dropna().unique() if d})

        patch_ids: List[int] = []
        for value in group["patch_id"].dropna().unique():
            try:
                patch_ids.append(int(value))
            except (TypeError, ValueError):
                continue

        patches_per_acq = (
            group.groupby("acquisition_index").size().median() if not group.empty else None
        )

        width = _mode_or_blank(group["width"])
        height = _mode_or_blank(group["height"])
        has_crs = "crs" in group.columns and group["crs"].notna().any()
        crs = _mode_or_blank(group["crs"]) if has_crs else ""

        notes_parts: List[str] = []
        if group["hidden_in_scan_root"].any():
            notes_parts.append("Some files sit outside the expected 5-level hierarchy.")
        if "S1" in sensors and "S2" in sensors:
            overlap = set(s1_dates) & set(s2_dates)
            if not overlap:
                notes_parts.append("SAR and optical acquisition dates do not overlap exactly.")
            else:
                notes_parts.append(f"{len(overlap)} shared acquisition date(s) found.")
        if not notes_parts:
            notes_parts.append("No structural anomalies detected.")

        rows.append(
            {
                "geographic_partition": partition,
                "roi_collection": roi_collection,
                "roi_id": roi_id,
                "roi_key": key,
                "sensors_present": "|".join(sensors),
                "total_image_count": int(len(group)),
                "s1_image_count": int(len(s1)),
                "s2_image_count": int(len(s2)),
                "acquisition_count": len(acq_indices),
                "acquisition_indices": "|".join(acq_indices),
                "unique_date_count": len(dates),
                "s1_date_min": s1_dates[0] if s1_dates else "",
                "s1_date_max": s1_dates[-1] if s1_dates else "",
                "s2_date_min": s2_dates[0] if s2_dates else "",
                "s2_date_max": s2_dates[-1] if s2_dates else "",
                "patch_id_min": min(patch_ids) if patch_ids else None,
                "patch_id_max": max(patch_ids) if patch_ids else None,
                "patches_per_acquisition_median": patches_per_acq,
                "width": width,
                "height": height,
                "s1_band_count": _mode_or_blank(s1["band_count"]) if not s1.empty else "",
                "s2_band_count": _mode_or_blank(s2["band_count"]) if not s2.empty else "",
                "s1_data_type": _mode_or_blank(s1["data_type"]) if not s1.empty else "",
                "s2_data_type": _mode_or_blank(s2["data_type"]) if not s2.empty else "",
                "crs": crs,
                "unreadable_count": int((group["validation_status"] == "unreadable").sum()),
                "empty_count": int((group["validation_status"] == "empty").sum()),
                "duplicate_filename_count": int(group["original_path"].isin(dup_paths).sum()),
                "notes": " ".join(notes_parts),
            }
        )

    roi_df = pd.DataFrame(rows).sort_values("roi_key").reset_index(drop=True)
    return roi_df.reindex(columns=ROI_COLUMNS)


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def build_relationships_df(
    records: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Propose SAR <-> optical patch relationships.

    A relationship is only marked *verified* when the SAR and optical patches
    share the same ROI, acquisition index, patch id AND an identical
    acquisition date. Everything else is marked *unverified*.
    """
    if not records:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    # Group by the identifiers we can actually trust: roi + acquisition + patch.
    grouped: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"S1": [], "S2": []}
    )
    for record in records:
        sensor = (record.get("sensor") or "").upper()
        if sensor not in ("S1", "S2"):
            continue
        key = (
            record.get("roi_key", ""),
            str(record.get("acquisition_index", "")),
            str(record.get("patch_id", "")),
        )
        grouped[key][sensor].append(record)

    rows: List[Dict[str, Any]] = []
    counter = 0
    for (key, acq_index, patch_id), sensors in grouped.items():
        s1_records = sensors["S1"]
        s2_records = sensors["S2"]
        if not s1_records or not s2_records:
            # Singleton (only one sensor) - not a pair, skip here.
            continue

        if len(s1_records) > 1 and logger:
            logger.warning(
                "Multiple S1 records for roi=%s acq=%s patch=%s; using first.",
                key,
                acq_index,
                patch_id,
            )
        if len(s2_records) > 1 and logger:
            logger.warning(
                "Multiple S2 records for roi=%s acq=%s patch=%s; using first.",
                key,
                acq_index,
                patch_id,
            )
        s1 = s1_records[0]
        s2 = s2_records[0]

        s1_date = s1.get("acquisition_date") or ""
        s2_date = s2.get("acquisition_date") or ""
        d1, d2 = _parse_date(s1_date), _parse_date(s2_date)
        delta_days = abs((d1 - d2).days) if (d1 and d2) else None

        dims_match = (
            s1.get("width") is not None
            and s2.get("width") is not None
            and s1.get("width") == s2.get("width")
            and s1.get("height") == s2.get("height")
        )

        if delta_days is not None and delta_days <= VERIFIED_DATE_DELTA_DAYS and dims_match:
            status = "verified"
            notes = "Same ROI, acquisition index, patch id and identical date; dimensions match."
        else:
            status = "unverified"
            reasons = []
            if delta_days is None:
                reasons.append("acquisition date(s) missing")
            elif delta_days > VERIFIED_DATE_DELTA_DAYS:
                reasons.append(f"acquisition dates differ by {delta_days} day(s)")
            if not dims_match:
                reasons.append("patch dimensions differ or are unknown")
            notes = "Potential pair - " + "; ".join(reasons) + "."

        counter += 1
        rows.append(
            {
                "relationship_id": f"rel_{counter:06d}",
                "geographic_partition": s1.get("geographic_partition", ""),
                "roi_collection": s1.get("roi_collection", ""),
                "roi_id": s1.get("roi_id", ""),
                "roi_key": key,
                "acquisition_index": acq_index,
                "patch_id": patch_id,
                "s1_file_name": s1.get("file_name", ""),
                "s2_file_name": s2.get("file_name", ""),
                "s1_acquisition_date": s1_date,
                "s2_acquisition_date": s2_date,
                "date_delta_days": delta_days,
                "match_criteria": "roi_key + acquisition_index + patch_id",
                "relationship_type": "sar_optical_pair",
                "validation_status": status,
                "notes": notes,
            }
        )

    rel_df = pd.DataFrame(rows)
    if rel_df.empty:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)
    rel_df = rel_df.sort_values(
        ["roi_key", "acquisition_index", "patch_id"]
    ).reset_index(drop=True)
    return rel_df.reindex(columns=RELATIONSHIP_COLUMNS)


# ---------------------------------------------------------------------------
# Folder-structure tree
# ---------------------------------------------------------------------------
def collect_dir_counts(
    dataset_root: Path, excluded_dirs: Sequence[Path]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """Return (tif count directly inside each dir, child dir paths per dir)."""
    counts: Dict[str, int] = {}
    children: Dict[str, List[str]] = defaultdict(list)

    for dirpath, dirnames, filenames in os_walk(dataset_root, ignored_names=IGNORED_DIR_NAMES):
        current = Path(dirpath)
        if any(is_within(current, ex) for ex in excluded_dirs):
            dirnames[:] = []
            continue
        tif_count = sum(1 for name in filenames if name.lower().endswith(IMAGE_EXTENSIONS))
        counts[str(current)] = tif_count
        for name in dirnames:
            children[str(current)].append(str(current / name))

    return counts, children


def render_tree(
    root: Path,
    counts: Dict[str, int],
    children: Dict[str, List[str]],
    max_depth: int = 6,
) -> List[str]:
    """Render an ASCII tree of directories with per-folder TIFF counts."""
    lines: List[str] = [f"{root.name}/"]

    def walk(directory: str, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        kids = sorted(children.get(directory, []))
        for position, kid in enumerate(kids):
            last = position == len(kids) - 1
            # ASCII-only connectors so the tree renders correctly in any
            # Windows console / default editor codepage.
            connector = "`-- " if last else "|-- "
            label = Path(kid).name + "/"
            count = counts.get(kid, 0)
            if count:
                label += f"  [{count} tif]"
            lines.append(prefix + connector + label)
            walk(kid, prefix + ("    " if last else "|   "), depth + 1)

    walk(str(root), "", 1)
    return lines


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def build_summary_text(
    dataset_root: Path,
    analysis_root: Path,
    inventory_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    unrecognised: List[Path],
    duplicates: Dict[str, List[str]],
    generated_at: str,
) -> str:
    """Compose dataset_summary.txt."""
    total_files = len(inventory_df)
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("SEN12MS-CR-TS SATELLITE DATASET INSPECTION - SUMMARY REPORT (Phase 1)")
    add("=" * 78)
    add(f"Generated at        : {generated_at}")
    add(f"Dataset root        : {dataset_root}")
    add(f"Analysis output root: {analysis_root}")
    add("Mode                : READ-ONLY inspection (no files modified)")
    add("")

    # --- 1. Totals ---------------------------------------------------------
    add("-" * 78)
    add("1. TOTALS")
    add("-" * 78)
    add(f"Total GeoTIFF files recognised : {_fmt_int(total_files)}")
    add(f"Unrecognised .tif files        : {_fmt_int(len(unrecognised))}")

    def unique_count(column: str) -> str:
        if not total_files:
            return "0"
        return _fmt_int(inventory_df[column].nunique())

    add(f"Unique geographic partitions   : {unique_count('geographic_partition')}")
    add(f"Unique ROI collections         : {unique_count('roi_collection')}")
    add(f"Unique ROIs (roi_key)          : {unique_count('roi_key')}")
    add(f"Unique acquisition indices     : {unique_count('acquisition_index')}")
    add(f"Unique acquisition dates       : {unique_count('acquisition_date')}")
    add(f"Unique patch ids               : {unique_count('patch_id')}")
    add("")

    # --- 2. Files per geographic partition ---------------------------------
    add("-" * 78)
    add("2. FILES PER GEOGRAPHIC PARTITION")
    add("-" * 78)
    if total_files:
        for part, count in inventory_df["geographic_partition"].value_counts().items():
            add(f"  {part:<24} {_fmt_int(count)}")
    else:
        add("  (no data)")
    add("")

    # --- 3. Files per sensor ----------------------------------------------
    add("-" * 78)
    add("3. FILES PER SENSOR / MODALITY")
    add("-" * 78)
    if total_files:
        for sensor, count in inventory_df["sensor"].value_counts().items():
            modality = {
                "S1": "SAR (C-band, VV+VH)",
                "S2": "Optical (13 bands)",
            }.get(sensor, "unknown")
            add(f"  {sensor:<6} {_fmt_int(count):>10}   {modality}")
    else:
        add("  (no data)")
    add("")

    # --- 4. Files per ROI --------------------------------------------------
    add("-" * 78)
    add("4. FILES PER ROI")
    add("-" * 78)
    if not roi_df.empty:
        add(f"  {'roi_key':<26}{'S1':>8}{'S2':>8}{'total':>8}   {'acq':>4}  sensors")
        for _, row in roi_df.iterrows():
            add(
                f"  {str(row['roi_key']):<26}{_fmt_int(row['s1_image_count']):>8}"
                f"{_fmt_int(row['s2_image_count']):>8}{_fmt_int(row['total_image_count']):>8}"
                f"   {row['acquisition_count']:>4}  {row['sensors_present']}"
            )
    else:
        add("  (no data)")
    add("")

    # --- 5. Acquisition dates ---------------------------------------------
    add("-" * 78)
    add("5. ACQUISITION DATES")
    add("-" * 78)
    if total_files:
        for roi_key_value, group in inventory_df.groupby("roi_key"):
            s1_rows = group.loc[group["sensor"] == "S1", "acquisition_date"]
            s2_rows = group.loc[group["sensor"] == "S2", "acquisition_date"]
            s1_dates = sorted({d for d in s1_rows.dropna().unique() if d})
            s2_dates = sorted({d for d in s2_rows.dropna().unique() if d})
            add(f"  {roi_key_value}")
            add(f"      S1: {', '.join(s1_dates) if s1_dates else '(none)'}")
            add(f"      S2: {', '.join(s2_dates) if s2_dates else '(none)'}")
    else:
        add("  (no data)")
    add("")

    # --- 6. Image properties ----------------------------------------------
    add("-" * 78)
    add("6. IMAGE PROPERTIES (GeoTIFF header metadata)")
    add("-" * 78)
    if total_files:
        add(f"  {'sensor':<7}{'width':>8}{'height':>8}{'bands':>7}   {'dtype':<14} crs")
        for sensor, group in inventory_df.groupby("sensor"):
            widths = group["width"].dropna().unique()
            heights = group["height"].dropna().unique()
            bands = group["band_count"].dropna().unique()
            dtypes = sorted({d for d in group["data_type"].dropna().unique() if d})
            crs_values = sorted({c for c in group["crs"].dropna().unique() if c})
            add(
                f"  {sensor:<7}{','.join(map(str, widths)):>8}{','.join(map(str, heights)):>8}"
                f"{','.join(map(str, bands)):>7}   {','.join(dtypes):<14} {','.join(crs_values)}"
            )
    else:
        add("  (no data)")
    add("")

    # --- 7. SAR <-> optical relationships ----------------------------------
    add("-" * 78)
    add("7. SAR <-> OPTICAL PATCH RELATIONSHIPS")
    add("-" * 78)
    if not rel_df.empty:
        status_counts = rel_df["validation_status"].value_counts().to_dict()
        add(f"  Candidate SAR/optical pairs  : {_fmt_int(len(rel_df))}")
        add(f"  Verified pairs (same date)   : {_fmt_int(status_counts.get('verified', 0))}")
        add(f"  Unverified pairs             : {_fmt_int(status_counts.get('unverified', 0))}")
        deltas = rel_df["date_delta_days"].dropna()
        if not deltas.empty:
            add(
                "  Date delta (days) min/median/max: "
                f"{int(deltas.min())} / {deltas.median()} / {int(deltas.max())}"
            )
            add("  NOTE: pairs are matched on roi_key + acquisition_index + patch_id.")
            add("        Non-zero date deltas mean the SAR and optical acquisitions were")
            add("        NOT captured on the same day and are therefore UNVERIFIED pairs.")
    else:
        add("  No candidate SAR/optical pairs were found.")
    add("")

    # --- 8. Missing / inconsistent files -----------------------------------
    add("-" * 78)
    add("8. MISSING OR INCONSISTENT FILES")
    add("-" * 78)
    if total_files:
        unreadable = int((inventory_df["validation_status"] == "unreadable").sum())
        empty = int((inventory_df["validation_status"] == "empty").sum())
        skipped = int((inventory_df["validation_status"] == "not_checked").sum())
        unknown = int((inventory_df["validation_status"] == "unknown").sum())
        add(f"  Unreadable TIFFs            : {_fmt_int(unreadable)}")
        add(f"  Empty (zero-byte) TIFFs     : {_fmt_int(empty)}")
        add(f"  Header read skipped         : {_fmt_int(skipped)}")
        add(f"  Header not read (no rasterio): {_fmt_int(unknown)}")
        add(f"  Unrecognised .tif files     : {_fmt_int(len(unrecognised))}")
        add(f"  Duplicate filenames         : {_fmt_int(len(duplicates))}")
        inconsistent = inventory_df[
            (inventory_df["width"].notna())
            & (inventory_df["band_count"].notna())
        ]
        if not inconsistent.empty:
            expected = inconsistent.groupby("sensor")["band_count"].agg(lambda s: sorted(set(s)))
            add("  Band counts observed per sensor:")
            for sensor, values in expected.items():
                add(f"      {sensor}: {values}")
    else:
        add("  (no data)")
    add("")

    # --- 9. Limitations ----------------------------------------------------
    add("-" * 78)
    add("9. KNOWN LIMITATIONS OF THIS METADATA")
    add("-" * 78)
    add("  * Everything here is derived from filenames + GeoTIFF headers only.")
    add("    No pixel values were read, so radiometric content is not validated.")
    add("  * Acquisition dates come from the filename; they are not cross-checked")
    add("    against sensor metadata inside the GeoTIFF.")
    add("  * SAR and optical acquisitions frequently occur on DIFFERENT days, so")
    add("    same-index pairs must be treated as UNVERIFIED until confirmed.")
    add("  * Geographic coordinates / tile extents are only recorded via the CRS;")
    add("    bounding boxes were intentionally not derived in this phase.")
    add("  * No target labels or terrain classes exist in the source data.")
    add("")

    # --- 10. Suitability ---------------------------------------------------
    add("-" * 78)
    add("10. SUITABILITY FOR THE NEXT STAGE")
    add("-" * 78)
    n_bad = int((inventory_df["validation_status"].isin(["unreadable", "empty"])).sum())
    if total_files and n_bad == 0 and not duplicates and not unrecognised:
        add("  The dataset is structurally consistent and fully readable. It is")
        add("  SUITABLE for reconnaissance-oriented preprocessing (patch pairing,")
        add("  normalisation, index computation, split generation) once the")
        add("  unverified SAR/optical pairing policy is agreed.")
    else:
        add("  Structural issues or unreadable files were detected. Review section 8")
        add("  before proceeding to preprocessing.")
    add("")

    # --- 11. Recommended structure (proposal only) -------------------------
    add("-" * 78)
    add("11. RECOMMENDED CLASSIFICATION STRUCTURE (PROPOSAL - NOT CREATED)")
    add("-" * 78)
    add("  The source data only contains SAR (S1) and optical (S2) imagery.")
    add("  It has NO terrain, feature or annotation data, so those folders are")
    add("  marked optional and must NOT be created until such data exists.")
    add("")
    add("  sentinel_recce_dataset/")
    add("  |-- raw_data/                 # (read-only link/pointer to originals)")
    add("  |-- processed_data/")
    add("  |     |-- SAR/                # Sentinel-1 derived products")
    add("  |     |-- OPTICAL/            # Sentinel-2 derived products")
    add("  |     |-- TERRAIN/            # OPTIONAL - create only if DEM data obtained")
    add("  |     \\-- FEATURES/           # OPTIONAL - create only if features derived")
    add("  |-- metadata/                 # inventory + relationship CSVs")
    add("  |-- annotations/              # empty until labels exist")
    add("  |-- splits/")
    add("  |     |-- train/")
    add("  |     |-- validation/")
    add("  |     \\-- test/")
    add("  \\-- scripts/")
    add("")
    add("  Suggested leaf naming inside SAR/ and OPTICAL/:")
    add("      <geographic_partition>/<roi_collection>/<roi_id>/acq_<index>/patch_<id>.tif")
    add("")
    add("=" * 78)
    add("END OF SUMMARY - see metadata/ for machine-readable tables.")
    add("=" * 78)

    return "\n".join(lines) + "\n"


def build_validation_text(
    dataset_root: Path,
    inventory_df: pd.DataFrame,
    duplicates: Dict[str, List[str]],
    unrecognised: List[Path],
    generated_at: str,
) -> str:
    """Compose validation_report.txt covering quality + integrity checks."""
    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add("DATASET VALIDATION REPORT (Phase 1 - quality & integrity checks)")
    add("=" * 78)
    add(f"Generated at : {generated_at}")
    add(f"Dataset root : {dataset_root}")
    add("")

    total = len(inventory_df)
    add("-" * 78)
    add("A. FILE INTEGRITY")
    add("-" * 78)
    if total:
        for status, count in inventory_df["validation_status"].value_counts().items():
            add(f"  {status:<14} {_fmt_int(count)}")
    else:
        add("  (no files inspected)")
    add("")

    add("-" * 78)
    add("B. UNREADABLE / EMPTY FILES")
    add("-" * 78)
    if total:
        bad = inventory_df[inventory_df["validation_status"].isin(["unreadable", "empty"])]
        if bad.empty:
            add("  None. All inspected TIFFs opened successfully.")
        else:
            for _, row in bad.iterrows():
                add(f"  [{row['validation_status']}] {row['original_path']}")
                if row.get("error_detail"):
                    add(f"        reason: {row['error_detail']}")
    else:
        add("  (no files inspected)")
    add("")

    add("-" * 78)
    add("C. UNRECOGNISED .tif FILES (naming convention mismatch)")
    add("-" * 78)
    if unrecognised:
        for path in unrecognised:
            add(f"  {path}")
    else:
        add("  None. Every .tif matched the expected naming convention.")
    add("")

    add("-" * 78)
    add("D. DUPLICATE FILENAMES")
    add("-" * 78)
    if duplicates:
        for name, paths in sorted(duplicates.items()):
            add(f"  {name}  ({len(paths)} copies)")
            for path in paths:
                add(f"        {path}")
    else:
        add("  None. All filenames are unique.")
    add("")

    add("-" * 78)
    add("E. DIMENSION / BAND CONSISTENCY")
    add("-" * 78)
    if total:
        checked = inventory_df[inventory_df["validation_status"] == "ok"]
        if checked.empty:
            add("  No header-validated files to check (run without --skip-raster).")
        else:
            for sensor, group in checked.groupby("sensor"):
                widths = sorted({int(w) for w in group["width"].dropna().unique()})
                heights = sorted({int(h) for h in group["height"].dropna().unique()})
                bands = sorted({int(b) for b in group["band_count"].dropna().unique()})
                consistent = len(widths) == 1 and len(heights) == 1 and len(bands) == 1
                flag = "OK" if consistent else "INCONSISTENT"
                add(f"  {sensor}: width={widths} height={heights} bands={bands}  -> {flag}")
    else:
        add("  (no files inspected)")
    add("")

    add("-" * 78)
    add("F. OVERALL VERDICT")
    add("-" * 78)
    problems = 0
    if total:
        problems += int((inventory_df["validation_status"].isin(["unreadable", "empty"])).sum())
    problems += len(unrecognised) + len(duplicates)
    if problems == 0 and total:
        add("  PASS - dataset is readable, complete and consistently named.")
    elif total:
        add(f"  REVIEW - {problems} issue(s) detected (see sections B-D above).")
    else:
        add("  NO DATA - nothing was inspected.")
    add("")
    add("=" * 78)
    return "\n".join(lines) + "\n"


def build_folder_structure_text(root: Path, tree_lines: List[str], generated_at: str) -> str:
    """Compose folder_structure.txt (directory tree with TIFF counts)."""
    header = [
        "=" * 78,
        "DATASET FOLDER STRUCTURE (Phase 1 - read-only snapshot)",
        "=" * 78,
        f"Generated at : {generated_at}",
        f"Root         : {root}",
        "Counts shown as [N tif] indicate GeoTIFF files directly inside a folder.",
        "",
    ]
    footer = [
        "",
        "=" * 78,
        "END OF FOLDER STRUCTURE",
        "=" * 78,
    ]
    return "\n".join(header + tree_lines + footer) + "\n"


def write_text(path: Path, content: str, logger: logging.Logger) -> None:
    """Write a UTF-8 text report, overwriting only this tool's own outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    logger.info("Wrote report: %s", path)


def write_csv(path: Path, df: pd.DataFrame, logger: logging.Logger) -> None:
    """Write a DataFrame to CSV (UTF-8), overwriting only this tool's own outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    logger.info("Wrote table : %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 READ-ONLY inspection + metadata generation for a "
        "Sentinel-1/Sentinel-2 (SEN12MS-CR-TS) reconnaissance dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder that contains the dataset partitions (never modified).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Analysis folder where reports/metadata/logs are written.",
    )
    parser.add_argument(
        "--skip-raster",
        action="store_true",
        help="Skip GeoTIFF header reads (fastest; width/height/bands left blank).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect the first N files (smoke test). Default: all files.",
    )
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="Skip building image_relationships.csv.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth for the folder-structure tree.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level console logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    project_root = args.project_root.resolve()

    if not dataset_root.is_dir():
        print(f"Dataset root is not a directory: {dataset_root}", file=sys.stderr)
        return 2
    if dataset_root == project_root:
        print(
            "Analysis output folder cannot be the dataset root; choose a separate "
            "--project-root so dataset files remain untouched.",
            file=sys.stderr,
        )
        return 2
    if args.max_depth < 0:
        print("--max-depth must be zero or greater.", file=sys.stderr)
        return 2

    dirs = ensure_output_dirs(project_root)
    logger = setup_logging(dirs["logs"], verbose=args.verbose)

    if args.limit is not None and args.limit < 1:
        logger.error(
            "--limit must be a positive integer (got %d). "
            "Omit --limit to inspect every file.", args.limit,
        )
        return 2
    if args.limit:
        logger.warning(
            "--limit %d: only the first %d file(s) will be inspected and the "
            "previous full inventory/reports will be OVERWRITTEN with partial data.",
            args.limit, args.limit,
        )

    logger.info("=" * 70)
    logger.info("SEN12MS-CR-TS dataset inspection (Phase 1) - READ ONLY")
    logger.info("Dataset root : %s", dataset_root)
    logger.info("Analysis root: %s", project_root)
    logger.info("=" * 70)

    if not RASTERIO_AVAILABLE and not args.skip_raster:
        logger.warning(
            "rasterio is not installed - TIFF headers cannot be read. "
            "Install it with:  pip install rasterio  (or run with --skip-raster)."
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Scan --------------------------------------------------------------
    records, unrecognised, _scan_stats = scan_dataset(
        dataset_root=dataset_root,
        project_root=project_root,
        logger=logger,
        skip_raster=args.skip_raster,
        limit=args.limit,
    )

    if not records:
        logger.error("No recognisable patch files were found under %s", dataset_root)
        return 3

    # --- Build metadata tables --------------------------------------------
    inventory_df = build_inventory_df(records)
    duplicates = find_duplicate_filenames(records)
    roi_df = build_roi_df(records, duplicates)
    rel_df = (
        pd.DataFrame(columns=RELATIONSHIP_COLUMNS)
        if args.no_relationships
        else build_relationships_df(records, logger=logger)
    )

    write_csv(dirs["metadata"] / "dataset_inventory.csv", inventory_df, logger)
    write_csv(dirs["metadata"] / "roi_metadata.csv", roi_df, logger)
    if not args.no_relationships:
        write_csv(dirs["metadata"] / "image_relationships.csv", rel_df, logger)

    # --- Build reports -----------------------------------------------------
    counts, children = collect_dir_counts(dataset_root, excluded_dirs=[project_root])
    tree_lines = render_tree(dataset_root, counts, children, max_depth=args.max_depth)

    summary_text = build_summary_text(
        dataset_root=dataset_root,
        analysis_root=project_root,
        inventory_df=inventory_df,
        roi_df=roi_df,
        rel_df=rel_df,
        unrecognised=unrecognised,
        duplicates=duplicates,
        generated_at=generated_at,
    )
    validation_text = build_validation_text(
        dataset_root=dataset_root,
        inventory_df=inventory_df,
        duplicates=duplicates,
        unrecognised=unrecognised,
        generated_at=generated_at,
    )
    structure_text = build_folder_structure_text(dataset_root, tree_lines, generated_at)

    write_text(dirs["reports"] / "dataset_summary.txt", summary_text, logger)
    write_text(dirs["reports"] / "validation_report.txt", validation_text, logger)
    write_text(dirs["reports"] / "folder_structure.txt", structure_text, logger)

    # --- Console recap -----------------------------------------------------
    logger.info("-" * 70)
    logger.info("INSPECTION COMPLETE")
    logger.info("-" * 70)
    logger.info("Files inspected        : %d", len(inventory_df))
    logger.info("ROIs identified        : %d", inventory_df["roi_key"].nunique())
    logger.info("Candidate SAR/optical pairs : %d", len(rel_df))
    logger.info("Unreadable/empty files : %d",
                int(inventory_df["validation_status"].isin(["unreadable", "empty"]).sum()))
    logger.info("Outputs written to:")
    logger.info("  %s", dirs["metadata"])
    logger.info("  %s", dirs["reports"])
    logger.info("  %s", dirs["logs"])
    logger.info("-" * 70)
    logger.info("Original dataset was NOT modified in any way.")
    logger.info("Next: review reports/dataset_summary.txt before requesting Phase 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
