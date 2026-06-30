# Scan-to-BIM  - Full Pipeline Runbook

Step-by-step setup and run instructions for the room-segmentation pipeline, from a clean
machine through the optional SAM-refined wall clouds. 

**Big picture:** five notebooks, five output stages. Notebooks 1–3 + 5 run **locally (CPU)**;
Notebook 4 runs in **Google Colab (GPU)**. The watershed walls (stage 3) come from Notebook 3;
the optional SAM-refined walls (stage 5) come from a dedicated **Notebook 5**.

```
N1 occupancy ─▶ N2 watershed ─▶ N3 wall assignment ─▶ stage3_walls/        (local, always)
                      │
                      └─▶ N4 SAM refinement (Colab) ─▶ download ─▶ N5 walls on SAM masks ─▶ stage5_walls_sam_refined/  (optional)
```

> **You edit exactly one file: `params.yaml`.** It holds the input cloud path, the output
> root, and any geometry overrides. Every notebook reads it via `scan2bim.load_config()`, so
> running a notebook is always just "Run All"  - never edit a cell.

---

## 0. Prerequisites

- **Windows** with [winget](https://learn.microsoft.com/windows/package-manager/winget/)
  (ships with Windows 11). On macOS/Linux use your package manager / pyenv instead.
- **Git** and **VS Code** with the Microsoft **Python** and **Jupyter** extensions.
- A **Google account** with Google Drive (only for Notebook 4  - the GPU stage).

> ### ⚠️ Python version matters
> `open3d` does **not** work on the newest Python releases  - **pin to Python 3.12**
> (do **not** use 3.13+). The commands below install and use 3.12 explicitly.

---

## 1. Get the code

```bash
git clone <your-repo-url> onestruction
cd onestruction
```

Or open the existing `onestruction` folder in VS Code (`File ▸ Open Folder…`).

---

## 2. Install Python 3.12

```powershell
winget install Python.Python.3.12
```

Close and reopen your terminal afterward so `py -3.12` is on the PATH. Verify:

```powershell
py -3.12 --version      # should print Python 3.12.x
```

---

## 3. Create a clean virtual environment

If a `.venv` already exists, remove it first so you start from 3.12:

**Git Bash / WSL**
```bash
rm -rf .venv
py -3.12 -m venv .venv
```

**PowerShell**
```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.12 -m venv .venv
```

---

## 4. Activate the environment

Use the line that matches your shell (run from the project root):

| Shell             | Command                              |
|-------------------|--------------------------------------|
| Git Bash / WSL    | `source .venv/Scripts/activate`      |
| PowerShell        | `.venv\Scripts\Activate.ps1`         |
| cmd.exe           | `.venv\Scripts\activate.bat`         |
| macOS / Linux     | `source .venv/bin/activate`          |

> PowerShell may block the activation script. If so, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and re-activate.

Your prompt should now be prefixed with `(.venv)`.

---

## 5. Install the package

```bash
pip install -e .
```

This installs `scan2bim` plus its core deps (numpy, scipy, scikit-image,
opencv-python-headless, pillow, matplotlib, **open3d**), so `import scan2bim` works from any
notebook. This may take a FEW minutes to get all the packages. Let it run.

Optional extras:

```bash
pip install -e ".[dev]"   # adds pytest (to run the test suite)
pip install -e ".[sam]"   # adds torch/torchvision (only needed in Colab for Notebook 4)
```

Sanity check:

```bash
python -c "import scan2bim, open3d; print('scan2bim', scan2bim.__version__, '| open3d', open3d.__version__)"
pytest tests/ -q          # (requires the [dev] extra)  - should show 20+ passing
```

---

## 6. Add your data and set `params.yaml`

Drop your point cloud at `data/area1.xyz` (the default). To use a different file or location,
set `input.file_path` in **`params.yaml`**  - that is the single place the input path lives:

```yaml
input:
  file_path: data/area1.xyz      # resolved relative to the project root
  units_per_meter: 1.0
output:
  out_root: scan2bim_out
```

Anything you don't list falls back to the `Config` defaults in `scan2bim/config.py`. You
never edit a notebook cell to change a path or a parameter.

---

## 7. Select the kernel in VS Code

Open any notebook, click the kernel picker (top-right), and choose the **`.venv`**
interpreter. Do this for each notebook the first time you open it.

---

## 8. Run the main pipeline (local, CPU)  - `1 → 2 → 3`

Run each notebook top to bottom. Each writes a numbered folder **and** a `.zip` under
`scan2bim_out/`, and its last cell prints `packaged -> …/<stage>.zip`.

1. **Notebook 1  - occupancy raster** → `stage1_occupancy/`
   (`occupancy.png`, `wallness.npy`, `coverage.npy`, `transform.json`, `config.json`)
2. **Notebook 2  - watershed segmentation** → `stage2_watershed/`
   (`room_labels.npy`, `walls.npy`, `footprint.npy`, …)
3. **Notebook 3  - wall assignment** → `stage3_walls/`
   Watershed-only  - just Run All. It writes per-room `room_XX_walls.ply`,
   `room_wall_masks.npz`, … There is no source switch to set.

Each notebook validates that the cloud and grid it loads match the upstream stage it consumes
(same `file_path`, `pixel_m`, slab params, and an in-bounds sanity check). If `params.yaml`
changed between stages, you get a clear, named error instead of silently wrong output.

✅ You now have usable **watershed-based** wall clouds. Stop here unless you want
SAM-refined room shapes (steps 9–11).

---

## 9. Notebook 4  - SAM refinement (Google Colab, GPU)

Notebook 4 is the only GPU stage and runs in Colab.

1. Copy these to Google Drive (e.g. `MyDrive/onestruction/`):
   - the `scan2bim/` **package** folder
   - **`params.yaml`** (so Colab's `load_config()` sees the same geometry you ran locally)
   - `scan2bim_out/stage1_occupancy.zip`
   - `scan2bim_out/stage2_watershed.zip`

   > You do **not** copy `stage3_walls`  - Notebook 4 only reads stages 1 and 2.
2. Open `notebooks/notebook_4_sam_refinement.ipynb` in Colab
   (set **Runtime ▸ Change runtime type ▸ GPU**).
3. In the **single paths cell**, set `PROJECT_DIR` to your Drive folder (the one that contains
   `scan2bim/`, `params.yaml`, and `scan2bim_out/`). That's the only edit.
4. Run all cells. The install cell installs SAM 2 + downloads the checkpoint **itself** (it
   skips either step if already present  - no uncommenting). Notebook 4 loads `CFG` via
   `load_config()` and validates it against the stage-2 `config.json`, so a geometry mismatch
   fails loudly. It writes `stage4_sam_refined/` to Drive, packages `stage4_sam_refined.zip`,
   and its last cell triggers a browser download of that ZIP.

> With no GPU/checkpoint, Notebook 4 passes the watershed labels through unchanged  - it
> never fabricates masks.

---

## 10. Run Notebook 5 for SAM-refined wall clouds (local)  - the "stage 5" output

1. Place the downloaded **`stage4_sam_refined.zip`** into your local `scan2bim_out/`.
2. Open **Notebook 5** (`notebook_5_walls_on_sam_refined.ipynb`) and **Run All**  - no edits.
   It assigns walls on the SAM-refined masks (the same boundary-ring assignment as Notebook 3)
   and writes a **separate** folder `stage5_walls_sam_refined/`, leaving `stage3_walls/`
   untouched, so both versions coexist for comparison.

   > If Notebook 4 hasn't run (no `stage4_sam_refined/`), Notebook 5 stops immediately with a
   > clear "run Notebook 4 first" error  - it never silently falls back to the watershed masks.

---

## 11. Final output layout

```
scan2bim_out/
├── stage1_occupancy/            (N1)
├── stage2_watershed/            (N2)
├── stage3_walls/                (N3, watershed masks)      ← always
├── stage4_sam_refined/          (N4, Colab)                ← optional
└── stage5_walls_sam_refined/    (N5, SAM masks)            ← optional
```

Each folder also has a matching `<stage>.zip`. The per-room 3-D wall clouds
(`room_XX_walls.ply`) live in `stage3_walls/` and/or `stage5_walls_sam_refined/`; open them
with open3d locally to inspect.

---

## Troubleshooting

- **`pip install -e .` fails on open3d / no matching distribution**  - you're almost
  certainly on Python 3.13+. Recreate the venv with `py -3.12` (steps 3–5).
- **`py` not found**  - reopen the terminal after `winget install`, or call the full path /
  use `python3.12`.
- **PowerShell "running scripts is disabled"**  - run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate.
- **Notebook can't `import scan2bim`**  - make sure the `.venv` kernel is selected (step 7);
  the bootstrap cell also adds the project root to `sys.path` as a fallback.
- **`FileNotFoundError: Stage 'stageN' not found`** / **"run Notebook 4 first"**  - run the
  producing notebook first, and confirm `output.out_root` in `params.yaml` points at the same
  `scan2bim_out/` (for stage 4, that the ZIP was downloaded into the local `scan2bim_out/`).
- **`Config mismatch on '<field>'`** or **"only N% of the cloud falls inside the upstream
  grid"**  - a stage is being run against an upstream stage that used a different cloud or
  `pixel_m`. Fix `input.file_path` / the geometry in `params.yaml` and **re-run from the
  stage that changed** (the error names the offending field).
