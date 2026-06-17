# Scan-to-BIM Room Segmentation — local (VS Code) project

Stages 1–3 run on **your machine** against the local `data/` folder and write outputs to
the local `scan2bim_out/` folder — no upload/download step. **Notebook 4 (SAM) is the GPU
stage:** run it in Google Colab — copy `scan2bim/` and the `scan2bim_out/` ZIPs to Drive
and run there.

```
onestruction/                 ← open THIS folder in VS Code  (File ▸ Open Folder…)
├── params.yaml               ← ★ the ONLY file you edit (input cloud, output root, params)
├── pyproject.toml            ← makes `import scan2bim` work everywhere
├── requirements.txt
├── scan2bim/                 ← the shared package
│   ├── runconfig.py          ← load_config() + cross-stage validation (one shared loader)
│   └── ARCHITECTURE.md       ← design, data flow, behaviour notes
├── notebooks/                ← "Run All" top-to-bottom, in order 1 → 2 → 3  (4 → 5 optional)
│   ├── notebook_1_occupancy_raster.ipynb
│   ├── notebook_2_watershed_segmentation.ipynb
│   ├── notebook_3_room_masks_and_wall_assignment.ipynb   ← walls on watershed masks
│   ├── notebook_4_sam_refinement.ipynb                   ← SAM refinement (Colab/GPU)
│   └── notebook_5_walls_on_sam_refined.ipynb             ← walls on SAM-refined masks
├── data/                     ← drop your area1.xyz here
└── scan2bim_out/             ← stage outputs + ZIPs appear here automatically
```

**The workflow is: edit `params.yaml`, then Run All.** No notebook cell is ever edited —
`file_path`, `out_root` and every geometry parameter live only in `params.yaml`, and each
notebook reads them via `scan2bim.load_config()`.

## Setup (once)

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
   `pip install -e .` registers `scan2bim`, so `import scan2bim` works from any notebook
   with no path juggling. (The bootstrap cell still works even if you skip this — it adds
   the project root to `sys.path` as a fallback.)
3. **Select the interpreter:** open a notebook, click the kernel picker (top-right), and
   choose the `.venv` interpreter.
4. **Add your data and point `params.yaml` at it:** put your segmented cloud at
   `data/area1.xyz` (the default), or set `input.file_path` in `params.yaml` to wherever it
   lives. This is the only place the input path is set.

## Run

Run the notebooks in order **1 → 2 → 3** — open each and **Run All** (no cell edits). The
watershed is stage 2 because wall assignment needs its room masks; see
`scan2bim/ARCHITECTURE.md`. Notebooks 1–3 run locally; the optional Notebook 4 runs in Colab
on a GPU and Notebook 5 turns its output into SAM-refined wall clouds. Each notebook's last
cell prints `packaged -> …/<stage>.zip`. Outputs accumulate under `scan2bim_out/`:

```
scan2bim_out/
├── stage1_occupancy/      + stage1_occupancy.zip
├── stage2_watershed/      + stage2_watershed.zip
├── stage3_walls/          + stage3_walls.zip   (room_XX_walls.ply, …)
├── stage4_sam_refined/    + stage4_sam_refined.zip
└── stage5_walls_sam_refined/  + stage5_walls_sam_refined.zip   (optional; N3 re-run on SAM masks)
```

**Optional (SAM-refined wall clouds):** after Notebook 4, download `stage4_sam_refined.zip`
into `scan2bim_out/` and **run Notebook 5** — it assigns walls on the SAM-refined masks and
writes `stage5_walls_sam_refined/`, leaving your `stage3_walls/` untouched. Notebook 5 fails
with a clear message if Notebook 4 has not run yet (there is no switch to edit).

## Where files go (vs Colab)

- **Inputs:** read directly from `data/` on your disk — nothing to mount or upload.
- **Outputs:** written to `scan2bim_out/` on your disk — open them in the file explorer,
  no download needed.
- Paths are anchored to the project root inside the bootstrap cell, so they resolve even
  though VS Code starts the kernel in `notebooks/`.

## Optional dependencies

- Stages 1–3 are **CPU-only** and need just the core deps above.
- **Notebook 4 (SAM)** is the GPU stage and is meant for Google Colab. It runs SAM 2
  (verified against `github.com/facebookresearch/sam2`):
  ```bash
  pip install "git+https://github.com/facebookresearch/sam2.git"   # needs torch>=2.5.1
  ```
  then downloads a SAM 2.1 checkpoint (e.g. `sam2.1_hiera_large.pt`) and sets
  `CFG.sam_ckpt` / `CFG.sam_model_cfg` / `CFG.sam_backend` (`'sam2'` default, or `'sam3'` /
  `'sam1'`). Notebook 4 handles the install + checkpoint download in its own cells. With no
  backend/checkpoint, it simply passes the watershed labels through unchanged.
- open3d's 3-D viewers open native windows locally, so you can inspect the exported
  `.ply` walls interactively (not possible in Colab).
