# sentinel_dataset_analysis

Comprehensive, **strictly read-only** analysis, metadata inspection, and interactive visualization suite for the **SEN12MS-CR-TS** multi-modal satellite dataset (Sentinel-1 SAR + Sentinel-2 Optical).

> **Safety & Integrity Guarantee:**
> This toolkit operates under strict **read-only** constraints. It never moves, renames, deletes, overwrites, or modifies any original dataset GeoTIFF files. All outputs (reports, CSV metadata, logs, and generated PNG figures) are stored exclusively within `sentinel_dataset_analysis/`.

---

## 1. System Overview & Architecture

The system is designed to explore, validate, and visualize high-resolution multi-sensor satellite imagery across space and time without loading large arrays into memory unnecessarily.

```
                           +----------------------------------------+
                           |  Raw Dataset: Dataset/ (15,360 TIFFs)  |
                           |  - Sentinel-1 SAR (VV, VH)             |
                           |  - Sentinel-2 Optical (13 Bands)       |
                           +-------------------+--------------------+
                                               |
                     +-------------------------+-------------------------+
                     | (Read-Only)                                       | (Read-Only)
                     v                                                   v
     +-------------------------------+                   +-------------------------------+
     |      inspect_dataset.py       |                   |     visualize_dataset.py      |
     |  - Header-only scanning       |                   |  - Single patch CLI rendering |
     |  - CRS & dimension validation |                   |  - 2-98% percentile stretch   |
     |  - Temporal relationship pairs|                   |  - Static PNG export          |
     +---------------+---------------+                   +-------------------------------+
                     |                                                   |
                     v                                                   v
         [ metadata/ & reports/ ]                            [ visualizations/*.png ]
                     |                                                   |
                     +-------------------------+-------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |             server.py             |
                             |   Flask REST API & Web Backend    |
                             +-----------------+-----------------+
                                               | (HTTP / JSON / PNG)
                                               v
                             +-----------------------------------+
                             |       templates/index.html        |
                             |  Interactive Satellite Dashboard  |
                             +-----------------------------------+
```

---

## 2. Dataset Organization & Analysis Breakdown

### Directory Hierarchy (SEN12MS-CR-TS Layout)
The actual dataset structure auto-detected under `Dataset/` follows this organization:

```
Dataset/
└── asiaWest_n/                                # Geographic partition
    ├── ROIs1868/
    │   └── 127/                               # ROI ID
    │       ├── S1/                            # Sentinel-1 SAR
    │       │   ├── 0/                         # Acquisition Index 0 (ImgNo 0)
    │       │   ├── 1/                         # Acquisition Index 1 (ImgNo 1)
    │       │   ├── 2/                         # Acquisition Index 2 (ImgNo 2)
    │       │   └── 3/                         # Acquisition Index 3 (ImgNo 3)
    │       └── S2/                            # Sentinel-2 Multispectral
    │           ├── 0/ ... 3/
    ├── ROIs1970/ (ROIs: 57, 83, 112)
    └── ROIs2017/ (ROIs: 69, 115, 130)
```

### Sensor Modalities
1. **Sentinel-1 (SAR - Synthetic Aperture Radar):**
   - **Bands:** 2 channels — `VV` (vertical transmit/receive) and `VH` (cross-polarization).
   - **Characteristics:** Microwave active sensor (C-band) penetrating clouds, rain, and darkness. Stored as backscatter coefficients in decibels (dB).
2. **Sentinel-2 (Optical / Multispectral):**
   - **Bands:** 13 spectral bands (`B1` through `B12` + `B8A`).
   - **True Color RGB:** Red = `B4`, Green = `B3`, Blue = `B2` (natural human-vision composite).
   - **False Color NIR:** Red = `B8` (Near Infrared), Green = `B4`, Blue = `B3`. Emphasizes vegetation vigor, biomass, and separates land from water.

### Dataset Metrics Discovered
- **Total GeoTIFFs:** 15,360 files.
- **Regions of Interest (ROIs):** 7 unique spatial areas across `asiaWest_n`.
- **Temporal Sequence:** 4 seasonal acquisition cycles (`0`, `1`, `2`, `3`) per ROI.
- **Spatial Coverage:** 270 spatial patch tiles per acquisition (Patches `0` to `269`).
- **Data Integrity:** 0 zero-byte files, 0 corrupted files, 100% readable GeoTIFF headers.

---

## 3. Implemented Components & Scripts

### 1. `inspect_dataset.py` (Dataset Profiler & Auditor)
- **Lazy Header Inspection:** Reads raster tags, CRS projections, spatial dimensions, data types, and band counts using `rasterio` without pulling heavy pixel data into RAM.
- **Temporal Alignment Analysis:** Maps corresponding S1 and S2 acquisitions by matching `roi_key + acquisition_index + patch_id` and records the exact date delta (`s1_date` vs `s2_date`).
- **Structured CSV Outputs:**
  - `metadata/dataset_inventory.csv`: Full file catalog with size, resolution, and band details.
  - `metadata/roi_metadata.csv`: Aggregated summary per ROI.
  - `metadata/image_relationships.csv`: Paired multi-modal records sorted numerically.
- **Reports:**
  - `reports/dataset_summary.txt`: Executive statistical digest.
  - `reports/validation_report.txt`: File integrity and health audits.
  - `reports/folder_structure.txt`: Visual ASCII tree with tile counts.

### 2. `visualize_dataset.py` (CLI Visualizer)
- **Pixel-Level Processing:** Pulls requested SAR and optical bands on demand.
- **Percentile Contrast Stretch:** Computes dynamic range across finite pixels (2nd to 98th percentile) to reveal features in SAR backscatter and optical reflectance.
- **Figure Generation:** Generates a 3-panel figure (S1 VV, S1 VH, S2 Color) saved directly to `visualizations/`.

### 3. `server.py` & `templates/index.html` (Interactive Web Viewer)
- **Flask REST API:**
  - `/api/status`: Reports dataset root, availability of `rasterio`/`Pillow`, and ROI totals.
  - `/api/rois`: Lists all detected ROIs.
  - `/api/rois/<key>/acquisitions`: Returns available time-steps with acquisition dates.
  - `/api/rois/<key>/acquisitions/<idx>/patches`: Returns patch ID list.
  - `/api/image`: Renders S1 VV, S1 VH, or S2 True/False color into PNG streams in-memory.
  - `/api/image/metadata`: Provides quick tile dimensions and date information.
- **Modern Dark UI:**
  - Dropdown ROI navigation.
  - Acquisition time-step chips.
  - Patch selector with previous (`‹`) and next (`›`) stepper controls.
  - Side-by-side tri-panel view with real-time toggle between True Color and NIR False Color.

---

## 4. Setup & Installation

### Requirements
- Python 3.9+
- Dependencies: `rasterio`, `pillow`, `flask`, `pandas`, `numpy`, `matplotlib`

```powershell
pip install rasterio pillow flask pandas numpy matplotlib
```

---

## 5. Usage Guide

### A. Run the Interactive Web Viewer (Recommended)
Launch the server from the repository root:
```powershell
python ".\sentinel_dataset_analysis\server.py"
```
Open your browser at:
👉 **http://127.0.0.1:5000/**

#### Key UI Controls:
- **ROI:** Choose from any of the 7 regions (e.g. `asiaWest_n/1868/127`).
- **Acquisition:** Click `Acq 0`, `Acq 1`, `Acq 2`, or `Acq 3`.
- **Patch ID:** Enter an integer from `0` to `269` (or use `‹`/`›`).
- **Color Mode:** Select *True Color* or *False Color*.
- Click **"Load Patch"** to render.

---

### B. Run the Dataset Inspector (Profiling)
```powershell
# Quick test (first 500 files, skips pixel reading)
python ".\sentinel_dataset_analysis\scripts\inspect_dataset.py" --limit 500

# Full comprehensive profiling of all 15,360 files
python ".\sentinel_dataset_analysis\scripts\inspect_dataset.py"
```

---

### C. Run the Command-Line Visualizer (Static PNG Export)
```powershell
# Visualize default patch (saved to sentinel_dataset_analysis/visualizations/)
python ".\sentinel_dataset_analysis\scripts\visualize_dataset.py"

# List all available ROI identifiers
python ".\sentinel_dataset_analysis\scripts\visualize_dataset.py" --list

# Render specific patch with False Color NIR
python ".\sentinel_dataset_analysis\scripts\visualize_dataset.py" --roi-key "asiaWest_n/1970/83" --acquisition-index 0 --patch-id 10 --false-color
```

---

## 6. Directory Layout

```
sentinel_dataset_analysis/
├── server.py                     # Flask web viewer backend (read-only)
├── templates/
│   └── index.html                # Interactive single-page web dashboard
├── scripts/
│   ├── inspect_dataset.py        # Metadata profiler & consistency checker
│   └── visualize_dataset.py      # CLI patch renderer & PNG exporter
├── metadata/                     # Machine-readable tables
│   ├── dataset_inventory.csv     # Full catalog of all 15,360 TIFFs
│   ├── roi_metadata.csv          # Per-ROI summary statistics
│   └── image_relationships.csv   # Paired S1/S2 patch relationships
├── reports/                      # Human-readable summary audits
│   ├── dataset_summary.txt       # Executive analysis report
│   ├── folder_structure.txt      # ASCII folder hierarchy with file counts
│   └── validation_report.txt     # Data integrity verification
├── visualizations/               # Saved high-resolution PNG patch figures
├── logs/                         # Execution logs with timestamps
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
