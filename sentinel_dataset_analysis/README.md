# sentinel_dataset_analysis

Phase 1 **read-only** inspection + metadata generation for the SEN12MS-CR-TS
Sentinel-1 (SAR) / Sentinel-2 (optical) dataset used by the AI reconnaissance project.

> Nothing in the original dataset is moved, renamed, edited or deleted.
> The tools only **read** filenames, GeoTIFF headers, and (for visualization)
> pixel data, and write their own reports/CSVs/PNGs into this folder.

## Folder layout

```
sentinel_dataset_analysis/
├── scripts/
│   ├── inspect_dataset.py        # the inspection script
│   └── visualize_dataset.py      # READ-ONLY patch / overview plots (PNG)
├── metadata/                     # generated CSV tables
│   ├── dataset_inventory.csv
│   ├── roi_metadata.csv
│   └── image_relationships.csv
├── reports/                      # generated human-readable reports
│   ├── dataset_summary.txt
│   ├── folder_structure.txt
│   └── validation_report.txt
├── visualizations/               # generated PNG figures
├── logs/                         # timestamped run logs
├── requirements.txt
└── README.md
```

## 1. Install dependencies

```powershell
pip install -r "C:\Users\rohii\OneDrive\Desktop\sentineldataset\sentinel_dataset_analysis\requirements.txt"
```

Or individually:

```powershell
pip install pandas numpy rasterio matplotlib
```

`rasterio` ships its own GDAL bindings, so you do **not** need a separate
`GDAL`/`osgeo` install. `matplotlib` is only needed for the visualization
script. `geopandas`/`shapely` are **not** used.

Verify:

```powershell
python -c "import rasterio, pandas, numpy, matplotlib; print('deps OK')"
```

## 2. Run the inspection

Defaults are already correct for this machine (dataset root =
`...\sentineldataset`, outputs = this folder), so:

```powershell
cd "C:\Users\rohii\OneDrive\Desktop\sentineldataset"
python "sentinel_dataset_analysis\scripts\inspect_dataset.py"
```

A full run reads GeoTIFF headers for all ~15,360 files and takes a few minutes.

### Useful options

| Option | Purpose |
| --- | --- |
| `--limit 500` | Only inspect the first N files (quick smoke test). |
| `--skip-raster` | Skip TIFF header reads — fastest inventory, width/height/bands left blank. |
| `--no-relationships` | Skip `image_relationships.csv`. |
| `--max-depth 6` | Depth of the folder-structure tree. |
| `--verbose` | Debug-level console logging. |
| `--dataset-root "PATH"` | Inspect a different dataset root. |
| `--project-root "PATH"` | Write outputs somewhere else. |

Example smoke test:

```powershell
python "sentinel_dataset_analysis\scripts\inspect_dataset.py" --limit 1000
```

> **Note on partial runs:** `--limit` and `--skip-raster` write *partial*
> metadata (only the files/headers that were inspected). They are intended for
> quick checks. Always finish with a normal full run so the CSVs describe the
> whole dataset.

## 3. Where the outputs go

All generated files are written **inside this folder only**:

* `metadata\dataset_inventory.csv` – one row per GeoTIFF patch.
* `metadata\roi_metadata.csv` – one row per ROI summary.
* `metadata\image_relationships.csv` – proposed SAR↔optical patch pairs.
* `reports\dataset_summary.txt` – full narrative summary + recommendation.
* `reports\folder_structure.txt` – directory tree with per-folder TIFF counts.
* `reports\validation_report.txt` – integrity/consistency checks.
* `logs\inspect_dataset_<timestamp>.log` – full run log.
* `visualizations\*.png` – figures written by `visualize_dataset.py` (section 4).

## 4. Visualize a patch (S-1 SAR + S-2 optical)

`scripts\visualize_dataset.py` reads the **pixel** data of a single patch and
writes one PNG (S-1 VV, S-1 VH, S-2 colour composite) into `visualizations\`.
Like the inspector it is strictly READ-ONLY.

```powershell
cd "C:\Users\rohii\OneDrive\Desktop\sentineldataset"

# default: first ROI, acquisition 0, patch 0 (true colour)
python "sentinel_dataset_analysis\scripts\visualize_dataset.py"

# a specific patch in false colour (B8/B4/B3), and open a window
python "sentinel_dataset_analysis\scripts\visualize_dataset.py" `
    --roi-key asiaWest_n/1868/127 --acquisition-index 1 --patch-id 5 --false-color --show

# list every ROI key, then render a whole-dataset metadata overview
python "sentinel_dataset_analysis\scripts\visualize_dataset.py" --list
python "sentinel_dataset_analysis\scripts\visualize_dataset.py" --overview
```

| Option | Purpose |
| --- | --- |
| `--roi-key asiaWest_n/1868/127` | ROI to plot (default: first found; use `--list`). |
| `--acquisition-index N` | Acquisition / time-step index within the ROI (0-3). |
| `--patch-id N` | Patch id within that acquisition. |
| `--false-color` | False colour (B8/B4/B3) instead of true colour (B4/B3/B2). |
| `--overview` | Metadata summary (S1/S2 counts + unique dates per ROI). |
| `--list` | Print all available ROI keys and exit. |
| `--show` | Also open the figure in an interactive window. |
| `--dpi N` | Output PNG resolution (default 150). |

Display note: S-1 bands are shown **as delivered (dB)** and S-2 bands as DN,
each with a 2-98 % percentile contrast stretch (non-finite pixels -> black).
No scientific correction or rescaling is applied - this is for quick visual QA.

## 5. Safety guarantees

* `inspect_dataset.py` never opens pixel data — header metadata only.
* `visualize_dataset.py` reads pixel data **read-only** for display only.
* Neither script writes inside the dataset tree (only inside this folder).
* Re-running is safe: it overwrites only the files listed above.
* Errors on individual files are recorded as `unreadable`/`empty`, never fatal.
