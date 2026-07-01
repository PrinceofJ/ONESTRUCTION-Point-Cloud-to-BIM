# ONESTRUCTION — Point Cloud to BIM

Converts a 3D LiDAR scan of a building into an IFC4 BIM model. Instead of segmenting
directly in 3D, the pipeline flattens the scan into 2D images, segments rooms and walls
with classical CV and SAM, then projects results back into 3D.

**Authors:** Jackson Matsumura, Ian Mendoza, Finn Wood, Marvin Recio

---

## How it works

```
 .ply / .xyz point cloud
      │
      ▼
  Room segmentation    Watershed + SAM → room labels
                       Exports per-room wall point clouds
      │  room_XX_walls.ply
      ▼

  Wall segmentation    RANSAC wall fitting → 2D wall images

      │  wall images + metadata
      ▼
 
  Wall processing      Door/window classification

      │  openings.json
      ▼

  IFC4 export          IfcOpenShell → model.ifc

```

---

## Setup

Requires **Python 3.11** (Open3D doesn't support 3.12+).

```bash
# create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# install the package
pip install -e .
```

This registers `scan2bim` so all notebooks can import it. If you skip this step, the
notebooks will fail with `ModuleNotFoundError`.

Put your point cloud in `data/` (default: `data/area1.xyz`) and point `params.yaml` at it.
`params.yaml` controls the input path, output directory, and all geometry parameters.

---

## Three methods

All three methods start from the same preprocessing step, then diverge. Each writes to its
own output directories so they don't interfere with each other.

**Step 0 (shared):** Run `preprocessing/notebook_1_occupancy_raster.ipynb` first. This
rasterizes the point cloud into 2D images that every method consumes.

### Geometric (CPU only)

Deterministic watershed segmentation — no ML, no GPU.

1. `methods/geometric/notebook_1_watershed.ipynb`
2. `methods/geometric/notebook_2_wall_assignment.ipynb`

### SAM (GPU required)

SAM in "segment everything" mode, no geometric prior. Needs a CUDA GPU — run in Google
Colab on a T4 if you don't have one locally. Will not run without `torch` and a SAM
checkpoint.

1. `methods/SAM/notebook_1_sam_auto_segmentation.ipynb` — GPU
2. `methods/SAM/notebook_2_wall_assignment.ipynb`

### Geometric + SAM (hybrid)

Watershed first, then SAM refines the room boundaries. Best of both.

1. `methods/geometric_SAM/notebook_1_watershed.ipynb`
2. `methods/geometric_SAM/notebook_2_sam_refinement.ipynb` — GPU
3. `methods/geometric_SAM/notebook_3_wall_assignment.ipynb`

### Postprocessing (shared)

After any method finishes, run these in order:

1. `postprocessing/notebook_1_wall_segmentation.ipynb` — RANSAC wall fitting
2. `postprocessing/notebook_2_wall_image_processing.ipynb` — door/window detection
3. `postprocessing/notebook_3_ifc_export.ipynb` — IFC4 output

The first postprocessing notebook defaults to the geometric method's output (`STAGE3`).
If you ran a different method, change `WALL_STAGE` at the top of notebooks 1 and 3:
- Geometric: `A.STAGE3`
- SAM: `A.STAGE_SAM_WALLS`
- Geometric + SAM: `A.STAGE5`

---

## Stage outputs

Each notebook writes a stage directory (and a `.zip`) under `scan2bim_out/`:

```
scan2bim_out/
├── stage1_occupancy/              ← shared preprocessing
│
├── stage2_watershed/              ┐
├── stage3_walls/                  ┘ geometric
│
├── stage_sam_auto/                ┐
├── stage_sam_walls/               ┘ SAM
│
├── stage2_watershed/              ┐
├── stage4_sam_refined/            │ geometric + SAM
├── stage5_walls_sam_refined/      ┘
│
├── stage_wall_seg/                ┐
├── stage_wall_proc/               │ postprocessing
└── stage_ifc/                     ┘
```

---

## Evaluation (optional)

For scoring against S3DIS ground truth:

1. Run `converters/s3dis_loader.ipynb` once to prepare the ground truth data
2. Set `input.file_path: data/area3_structural.ply` in `params.yaml`
3. Run preprocessing + whichever method(s) you want to score
4. Run `evaluation/gt_raster.ipynb` → `evaluation/pq_eval.ipynb`

All methods write `room_labels.npy` on the same grid, so one ground truth raster scores
them all.

---

## GPU / SAM setup

The geometric method is CPU-only. The SAM stages need PyTorch and a SAM checkpoint:

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"   # needs torch>=2.5.1
```

Download a SAM 2.1 checkpoint (e.g. `sam2.1_hiera_large.pt`) and configure the path in
`params.yaml`. If you don't have a local GPU, run the SAM notebooks in Google Colab —
copy `scan2bim/` and the stage ZIPs to Google Drive.

---

## Project structure

```
├── params.yaml               ← edit this (input path, output root, parameters)
├── pyproject.toml
├── scan2bim/                 ← shared package (all notebook logic lives here)
├── notebooks/
│   ├── preprocessing/        ← stage 1 (run once)
│   ├── methods/
│   │   ├── geometric/        ← CPU only
│   │   ├── SAM/              ← GPU required
│   │   └── geometric_SAM/    ← hybrid
│   ├── postprocessing/       ← wall fitting, openings, IFC export
│   ├── converters/           ← data prep (S3DIS loader)
│   └── evaluation/           ← panoptic quality scoring
├── data/                     ← your point clouds go here (git-ignored)
└── scan2bim_out/             ← outputs appear here (git-ignored)
```

---

## License

Contact authors for licensing information.
