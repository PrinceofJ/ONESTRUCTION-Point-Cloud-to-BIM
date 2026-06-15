# ONESTRUCTION — Point Cloud to BIM

An end-to-end pipeline that converts a 3D LiDAR scan of a building into an IFC4 BIM model. Rather than attempting segmentation directly in 3D, the pipeline **flattens the scan into 2D images**, uses classical CV and SAM (Segment Anything Model) for segmentation, then projects results back into 3D.

Designed to run on **Google Colab** with Google Drive integration.

**Authors:** Jackson Matsumura, Ian Mendoza, Finn Wood, Marvin Recio

---

## Pipeline Overview

```
 .ply point cloud
       │
       ▼
 ┌─────────────────────┐
 │  RoomSegmentation    │  Watershed + SAM → room labels
 │  (.ipynb)            │  Exports per-room wall point clouds
 └─────────┬───────────┘
           │  room_XX_walls.ply
           ▼
 ┌─────────────────────┐
 │  WallSegmentation    │  RANSAC wall fitting → 2D wall images
 │  (.ipynb)            │  Black = wall, white = void
 └─────────┬───────────┘
           │  wall_XX.png + metadata
           ▼
 ┌─────────────────────┐
 │  WallImageProcessing │  SAM-refined void detection
 │  (.ipynb)            │  Door/window classification
 └─────────┬───────────┘
           │  openings.json
           ▼
 ┌─────────────────────┐
 │  JSON → IFC4         │  IfcOpenShell translator
 │  (.ipynb)            │  Outputs model.ifc
 └─────────────────────┘
```

---

## Notebooks

### 1. `RoomSegmentation.ipynb`

Segments a full-building point cloud into individual rooms.

- **Pass 1 (Geometry):** Horizontal slab extraction → top-down rasterization → distance-transform watershed. Rooms split naturally at doorway pinch-points. Supports local ceiling estimation for multi-height buildings (dropped ceilings, soffits).
- **Pass 2 (SAM Recall):** SAM automatic mask generation on residual unclaimed space recovers corridors and occluded rooms that geometry misses. Masks are snapped back to wall boundaries for crisp edges.
- **Output:** Per-room wall point clouds (`room_XX_walls.ply`) with RANSAC-fitted vertical wall planes.

**Key parameters:** `pixel_m` (raster resolution), `marker_h_m` (room seed depth), `merge_ridge_m` (over-segmentation control), `slab_relative_to` (ceiling/floor reference).

### 2. `WallSegmentation.ipynb`

Takes per-room wall point clouds and generates 2D binary wall images for each wall segment.

- Fits wall planes via RANSAC, then flattens each wall's points onto a 2D grid.
- Applies morphological cleanup (close → open) and optional statistical outlier removal.
- Exports wall images as PNG with metadata JSON (pixel scale, wall normal, origin).
- Includes preview cells for visual inspection and a zip-download cell for Colab.

**Key parameters:** `flat_pixel_m` (wall image resolution, default 0.04 m/px), `morph_close_px`, `morph_open_px`, SOR settings.

### 3. `WallImageProcessing.ipynb`

Detects and classifies doors and windows from binary wall images.

- **Void detection:** Connected-component analysis finds candidate openings.
- **SAM refinement:** Each void is used as a point prompt for SAM (upscaled 8x) to produce clean masks that bridge scan gaps. Nearby fragments are merged via union-find.
- **Classification heuristics:**
  - **Door:** Must touch the floor, have dimensions within range (0.5–1.6 m wide, 1.5–2.8 m tall), and pass a rectangularity check (≥ 0.55).
  - **Window:** Must not touch the floor, have sufficient sill height (≥ 0.3 m), and fit window dimensions (0.3–2.5 m wide, 0.3–2.0 m tall).
  - **Unknown:** Everything else — irregular shapes, scan-edge artifacts, or voids outside typical dimension ranges.
- **Output:** Annotated images, per-room `openings.json`, and a combined summary.

**Key parameters:** `min_rectangularity` (shape filter), door/window dimension ranges, `door_floor_margin_px`.

### 4. `JSON_ifc4/json_to_ifc4.ipynb`

Translates a structured JSON description of walls, doors, windows, and rooms into a valid IFC4 file using IfcOpenShell.

- Walls defined as centerlines with thickness, extruded to height.
- Doors/windows placed by offset along their host wall.
- Rooms as arbitrary polygons (including L-shapes) extruded into IfcSpace volumes.
- Optional floor slabs and an in-notebook 3D preview.

See [`JSON_ifc4/json_schema_guide.md`](JSON_ifc4/json_schema_guide.md) for the input JSON format.

---

## Utilities

### `textToXYZ_Converter.py`

Converts Stanford 3D Indoor Scenes (S3DIS) `.txt` format (XYZRGB per line) to Open3D-compatible point clouds (`.ply` or `.xyz`). Reads per-room text files from an `Area_N` directory, stacks them, and optionally voxel-downsamples.

---

## Setup

All notebooks are designed for **Google Colab**. Each installs its own dependencies in the first cell.

### Dependencies

| Notebook | Key packages |
|----------|-------------|
| RoomSegmentation | `open3d`, `supervision`, `scikit-image`, `scipy`, `opencv-python-headless`, `segment-anything` (optional) |
| WallSegmentation | `open3d`, `opencv-python-headless`, `matplotlib` |
| WallImageProcessing | `segment-anything`, `torch`, `opencv-python-headless` |
| JSON → IFC4 | `ifcopenshell` |

### Quick start

1. Upload your `.ply` scan to Google Drive.
2. Open `RoomSegmentation.ipynb` in Colab. Set `CFG.file_path` to your file. Run all cells.
3. Open `WallSegmentation.ipynb`. Point it at the room clouds from step 2. Run all cells.
4. Open `WallImageProcessing.ipynb`. Point `wall_image_dir` at the wall images from step 3. Run all cells.
5. Feed the resulting `openings.json` (along with wall geometry) into `json_to_ifc4.ipynb` to produce `model.ifc`.

---

## Project Structure

```
.
├── RoomSegmentation.ipynb        # Step 1: Point cloud → room segmentation
├── WallSegmentation.ipynb        # Step 2: Room walls → 2D wall images
├── WallImageProcessing.ipynb     # Step 3: Wall images → door/window detection
├── JSON_ifc4/
│   ├── json_to_ifc4.ipynb        # Step 4: JSON → IFC4 BIM model
│   └── json_schema_guide.md      # Input JSON format documentation
├── textToXYZ_Converter.py        # S3DIS .txt → .ply converter
└── README.md
```

---

## License

Contact the authors for licensing information.
