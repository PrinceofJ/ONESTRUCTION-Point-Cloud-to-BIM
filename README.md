# ONESTRUCTION — Point Cloud to BIM

An end-to-end pipeline that converts a 3D LiDAR scan of a building into an IFC4 BIM model.
Rather than attempting segmentation directly in 3D, the pipeline **flattens the scan into 2D
images**, uses classical CV and SAM (Segment Anything Model) for segmentation, then projects
results back into 3D.

**Authors:** Jackson Matsumura, Ian Mendoza, Finn Wood, Marvin Recio

There are two ways to run the pipeline:

- **[Local (VS Code) pipeline](#local-vs-code-pipeline)** — the current, recommended path.
  Three segmentation methods (geometric / SAM / geometric+SAM) run on your machine; only the
  SAM stage needs a GPU (Colab).
- **[Original Colab notebooks](#original-colab-notebooks)** — the self-contained notebooks that
  run entirely in Google Colab with Google Drive.

Both implement the same core idea: rasterize the scan top-down, watershed it into rooms,
refine with SAM, fit walls, then detect openings and emit IFC4.

---

## Pipeline Overview

```
 .ply / .xyz point cloud
       │
       ▼
 ┌─────────────────────┐
 │  Room segmentation   │  Watershed + SAM → room labels
 │                      │  Exports per-room wall point clouds
 └─────────┬───────────┘
           │  room_XX_walls.ply
           ▼
 ┌─────────────────────┐
 │  Wall segmentation   │  RANSAC wall fitting → 2D wall images
 │                      │  Black = wall, white = void
 └─────────┬───────────┘
           │  wall_XX.png + metadata
           ▼
 ┌─────────────────────┐
 │  Wall image proc.    │  SAM-refined void detection
 │                      │  Door/window classification
 └─────────┬───────────┘
           │  openings.json
           ▼
 ┌─────────────────────┐
 │  JSON → IFC4         │  IfcOpenShell translator
 │                      │  Outputs model.ifc
 └─────────────────────┘
```

---

## Local (VS Code) pipeline

The notebooks run on **your machine** against the local `data/` folder and write outputs to the
local `scan2bim_out/` folder — no upload/download step. The notebooks are thin drivers; all
logic lives in the shared `scan2bim/` package.

**Three room-segmentation methods, one shared front-end.** Everyone starts with the same
`preprocessing/notebook_1_occupancy_raster.ipynb` (point cloud → 2-D rasters, `stage1`), then
picks a method under `notebooks/methods/`:

| Method | Folder | How rooms are segmented | GPU? |
|--------|--------|-------------------------|------|
| **Geometric** | `methods/geometric/` | deterministic distance-transform watershed | no (CPU) |
| **SAM** | `methods/SAM/` | SAM automatic "segment everything", no geometric prior | yes (SAM stage in Colab) |
| **Geometric + SAM** | `methods/geometric_SAM/` | watershed prior, then **prompted** SAM refinement | yes (SAM stage in Colab) |

Each method writes to its **own** stage directories under `scan2bim_out/`, so the three never
clobber each other and the **same** `evaluation/` notebooks can score them against one ground
truth. The SAM stage is the only GPU step (run it in Google Colab — copy `scan2bim/` and the
`scan2bim_out/` ZIPs to Drive and run there); everything else is CPU.

```
onestruction/                 ← open THIS folder in VS Code  (File ▸ Open Folder…)
├── params.yaml               ← ★ the ONLY file you edit (input cloud, output root, params)
├── pyproject.toml            ← makes `import scan2bim` work everywhere
├── requirements.txt
├── scan2bim/                 ← the shared package
│   ├── runconfig.py          ← load_config() + cross-stage validation (one shared loader)
│   └── ARCHITECTURE.md       ← design, data flow, behaviour notes
├── notebooks/                ← thin drivers; "Run All" top-to-bottom (logic lives in scan2bim/)
│   ├── preprocessing/        ← shared stage 1 (run once, feeds every method)
│   │   └── notebook_1_occupancy_raster.ipynb            ← point cloud → 2-D rasters
│   ├── methods/              ← three room-segmentation methods, compared head-to-head
│   │   ├── geometric/        ← METHOD 1 — watershed only (CPU)
│   │   │   ├── notebook_1_watershed.ipynb
│   │   │   └── notebook_2_wall_assignment.ipynb
│   │   ├── SAM/              ← METHOD 2 — SAM auto-segmentation, no geometric prior  (Colab/GPU)
│   │   │   ├── notebook_1_sam_auto_segmentation.ipynb
│   │   │   └── notebook_2_wall_assignment.ipynb
│   │   └── geometric_SAM/   ← METHOD 3 — watershed prior + prompted-SAM refinement
│   │       ├── notebook_1_watershed.ipynb               ← own stage dir (no clobber)
│   │       ├── notebook_2_sam_refinement.ipynb          ← SAM refinement (Colab/GPU)
│   │       └── notebook_3_wall_assignment.ipynb         ← walls on SAM-refined masks
│   ├── converters/           ← input data-prep (not part of any method's run order)
│   │   └── s3dis_loader.ipynb                            ← S3DIS Area_3 → structural .ply
│   └── evaluation/           ← S3DIS scoring (shared; scores every method vs. one GT)
│       ├── gt_raster.ipynb                              ← ground-truth room raster
│       └── pq_eval.ipynb                                ← Panoptic Quality metrics
├── data/                     ← drop your area1.xyz here (see below)
└── scan2bim_out/             ← stage outputs + ZIPs appear here automatically
```

**The workflow is: edit `params.yaml`, then Run All.** No notebook cell is ever edited —
`file_path`, `out_root` and every geometry parameter live only in `params.yaml`, and each
notebook reads them via `scan2bim.load_config()`.

### Setup (once)

1. **Open the folder** in VS Code (`File ▸ Open Folder…` → `onestruction`). Install the
   Microsoft **Python** and **Jupyter** extensions if prompted.
2. **Create and activate a virtual environment**, then install the package editable:
   ```bash
   python -m venv .venv
   # macOS/Linux:
   source .venv/bin/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   pip install -e .
   ```
   `pip install -e .` registers `scan2bim`, so `import scan2bim` works from any notebook with
   no path juggling. (The bootstrap cell still works even if you skip this — it adds the
   project root to `sys.path` as a fallback.)
3. **Select the interpreter:** open a notebook, click the kernel picker (top-right), and choose
   the `.venv` interpreter.
4. **Add your data and point `params.yaml` at it:** put your segmented cloud at `data/area1.xyz`
   (the default), or set `input.file_path` in `params.yaml` to wherever it lives. This is the
   only place the input path is set.

### The `data/` folder

Drop your **segmented point cloud** here (walls / windows / doors), named `area1.xyz`.

- Default expected path: `data/area1.xyz` (set in `params.yaml` as `input.file_path`). To use a
  different name or location, edit that one line in `params.yaml` — never a notebook cell.
- Same format and units as the original pipeline (`.xyz`, etc.). `input.units_per_meter` in
  `params.yaml` controls the scale conversion on load.
- This folder is git-ignored (except its placeholder) so large clouds aren't committed.

### Run

**1. Preprocess once.** Open `preprocessing/notebook_1_occupancy_raster.ipynb` and **Run All**
(no cell edits). It writes `stage1_occupancy/` — the rasters every method consumes.

**2. Pick a method** under `notebooks/methods/` and **Run All** its notebooks in order:

- **Geometric** (`methods/geometric/`, CPU): `notebook_1_watershed` → `notebook_2_wall_assignment`.
- **Geometric + SAM** (`methods/geometric_SAM/`): `notebook_1_watershed` →
  `notebook_2_sam_refinement` (Colab/GPU) → `notebook_3_wall_assignment`.
- **SAM** (`methods/SAM/`): `notebook_1_sam_auto_segmentation` (Colab/GPU) →
  `notebook_2_wall_assignment`. Pure-SAM needs a real SAM backend — `notebook_1` fails loudly
  on no GPU/checkpoint rather than emitting an empty map.

The watershed is each geometric-flavoured method's first stage because wall assignment needs its
room masks; see `scan2bim/ARCHITECTURE.md`. Each notebook's last cell prints
`packaged -> …/<stage>.zip`. Every method writes its **own** stage dirs, so they coexist under
`scan2bim_out/` (each `<stage>/` also gets a `<stage>.zip` beside it):

```
scan2bim_out/
├── stage1_occupancy/                 ← shared (preprocessing)
│
├── stage2_watershed/                 ┐ geometric
├── stage3_walls/                     ┘   (room_XX_walls.ply, …)
│
├── stage_geometric_sam_watershed/    ┐ geometric + SAM
├── stage4_sam_refined/               │   (SAM refinement — Colab/GPU)
├── stage5_walls_sam_refined/         ┘   (walls on SAM-refined masks)
│
├── stage_sam_auto/                   ┐ SAM  (automatic-mask room labels — Colab/GPU)
└── stage_sam_walls/                  ┘       (walls on the SAM auto masks)
```

The geometric + SAM watershed writes `stage_geometric_sam_watershed/` (not `stage2_watershed/`)
precisely so it never overwrites the pure-geometric run's masks. `notebook_2_sam_refinement`
fails with a clear message if its upstream masks are missing — there is no silent fallback.

### S3DIS evaluation (optional, shared across methods)

The `converters/` and `evaluation/` notebooks are **not** part of any method's run order — they
score the room segmentation against Stanford 3D Indoor Scenes (S3DIS) Area 3 ground truth, so
they live in their own folders.

- **`converters/s3dis_loader.ipynb`** — input data prep, run **once**. Walks the raw
  `data/Area_3/<room>/Annotations/` folders, keeps only structural classes (wall, column, beam,
  door, window), and writes `data/area3_structural.ply`. Pure data prep — it doesn't call
  `load_config()`.
- **`evaluation/gt_raster.ipynb`** — projects each room's full S3DIS cloud onto the Stage 1 grid
  to build the ground-truth room labels (`scan2bim_out/stage_gt/gt_room_labels.npy`).
- **`evaluation/pq_eval.ipynb`** — computes **Panoptic Quality** (IoU matrix + greedy match at
  IoU > 0.5) of **each method's** rooms vs. the same ground truth →
  `scan2bim_out/stage_gt/pq_results.json`.

**Run order for evaluation:** run `converters/s3dis_loader.ipynb` once, set
`input.file_path: data/area3_structural.ply` in `params.yaml`, run `preprocessing/` and whichever
method(s) you want to score, then `evaluation/gt_raster.ipynb` → `evaluation/pq_eval.ipynb`.
Because every method writes a `room_labels.npy` on the same Stage-1 grid, one GT raster scores
them all.

### Where files go (vs Colab)

- **Inputs:** read directly from `data/` on your disk — nothing to mount or upload.
- **Outputs:** written to `scan2bim_out/` on your disk — open them in the file explorer, no
  download needed.
- Paths are anchored to the project root (`scan2bim.project_root()` walks up to find the
  package), so they resolve no matter how deep the notebook sits — e.g. when VS Code starts the
  kernel in `notebooks/methods/geometric/`.

### Optional dependencies

- The **geometric** method (preprocessing + `methods/geometric/`) is **CPU-only** and needs just
  the core deps above.
- The **SAM stages** are the GPU steps, meant for Google Colab: `methods/geometric_SAM/notebook_2_sam_refinement.ipynb`
  (and the `methods/SAM/` notebooks once implemented). They run SAM 2 (verified against
  `github.com/facebookresearch/sam2`):
  ```bash
  pip install "git+https://github.com/facebookresearch/sam2.git"   # needs torch>=2.5.1
  ```
  then download a SAM 2.1 checkpoint (e.g. `sam2.1_hiera_large.pt`) and set `CFG.sam_ckpt` /
  `CFG.sam_model_cfg` / `CFG.sam_backend` (`'sam2'` default, or `'sam3'` / `'sam1'`). The SAM
  refinement notebook handles the install + checkpoint download in its own cells. With no
  backend/checkpoint, it simply passes the watershed labels through unchanged.
- open3d's 3-D viewers open native windows locally, so you can inspect the exported `.ply` walls
  interactively (not possible in Colab).

---

## Original Colab notebooks

The self-contained notebooks at the repository root run the full pipeline in **Google Colab**
with Google Drive integration. Each installs its own dependencies in the first cell.

### 1. `RoomSegmentation.ipynb`

Segments a full-building point cloud into individual rooms.

- **Pass 1 (Geometry):** Horizontal slab extraction → top-down rasterization → distance-transform
  watershed. Rooms split naturally at doorway pinch-points. Supports local ceiling estimation for
  multi-height buildings (dropped ceilings, soffits).
- **Pass 2 (SAM Recall):** SAM automatic mask generation on residual unclaimed space recovers
  corridors and occluded rooms that geometry misses. Masks are snapped back to wall boundaries
  for crisp edges.
- **Output:** Per-room wall point clouds (`room_XX_walls.ply`) with RANSAC-fitted vertical wall
  planes.

**Key parameters:** `pixel_m` (raster resolution), `marker_h_m` (room seed depth),
`merge_ridge_m` (over-segmentation control), `slab_relative_to` (ceiling/floor reference).

### 2. `WallSegmentation.ipynb`

Takes per-room wall point clouds and generates 2D binary wall images for each wall segment.

- Fits wall planes via RANSAC, then flattens each wall's points onto a 2D grid.
- Applies morphological cleanup (close → open) and optional statistical outlier removal.
- Exports wall images as PNG with metadata JSON (pixel scale, wall normal, origin).
- Includes preview cells for visual inspection and a zip-download cell for Colab.

**Key parameters:** `flat_pixel_m` (wall image resolution, default 0.04 m/px), `morph_close_px`,
`morph_open_px`, SOR settings.

### 3. `WallImageProcessing.ipynb`

Detects and classifies doors and windows from binary wall images.

- **Void detection:** Connected-component analysis finds candidate openings.
- **SAM refinement:** Each void is used as a point prompt for SAM (upscaled 8x) to produce clean
  masks that bridge scan gaps. Nearby fragments are merged via union-find.
- **Classification heuristics:**
  - **Door:** Must touch the floor, have dimensions within range (0.5–1.6 m wide, 1.5–2.8 m tall),
    and pass a rectangularity check (≥ 0.55).
  - **Window:** Must not touch the floor, have sufficient sill height (≥ 0.3 m), and fit window
    dimensions (0.3–2.5 m wide, 0.3–2.0 m tall).
  - **Unknown:** Everything else — irregular shapes, scan-edge artifacts, or voids outside typical
    dimension ranges.
- **Output:** Annotated images, per-room `openings.json`, and a combined summary.

**Key parameters:** `min_rectangularity` (shape filter), door/window dimension ranges,
`door_floor_margin_px`.

### 4. `JSON_ifc4/json_to_ifc4.ipynb`

Translates a structured JSON description of walls, doors, windows, and rooms into a valid IFC4
file using IfcOpenShell.

- Walls defined as centerlines with thickness, extruded to height.
- Doors/windows placed by offset along their host wall.
- Rooms as arbitrary polygons (including L-shapes) extruded into IfcSpace volumes.
- Optional floor slabs and an in-notebook 3D preview.

See [`JSON_ifc4/json_schema_guide.md`](JSON_ifc4/json_schema_guide.md) for the input JSON format.

### Utility: `textToXYZ_Converter.py`

Converts Stanford 3D Indoor Scenes (S3DIS) `.txt` format (XYZRGB per line) to Open3D-compatible
point clouds (`.ply` or `.xyz`). Reads per-room text files from an `Area_N` directory, stacks
them, and optionally voxel-downsamples.

### Colab dependencies

| Notebook | Key packages |
|----------|-------------|
| RoomSegmentation | `open3d`, `supervision`, `scikit-image`, `scipy`, `opencv-python-headless`, `segment-anything` (optional) |
| WallSegmentation | `open3d`, `opencv-python-headless`, `matplotlib` |
| WallImageProcessing | `segment-anything`, `torch`, `opencv-python-headless` |
| JSON → IFC4 | `ifcopenshell` |

### Colab quick start

1. Upload your `.ply` scan to Google Drive.
2. Open `RoomSegmentation.ipynb` in Colab. Set `CFG.file_path` to your file. Run all cells.
3. Open `WallSegmentation.ipynb`. Point it at the room clouds from step 2. Run all cells.
4. Open `WallImageProcessing.ipynb`. Point `wall_image_dir` at the wall images from step 3. Run
   all cells.
5. Feed the resulting `openings.json` (along with wall geometry) into `json_to_ifc4.ipynb` to
   produce `model.ifc`.

---

## License

Contact the authors for licensing information.
