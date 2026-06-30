# Running the evaluation notebooks

The two notebooks in this folder score the three segmentation methods against a clean,
paper-faithful ground truth:

| Notebook | Role | Produces |
|---|---|---|
| `gt_raster.ipynb` (NB7) | Build clean **GT room labels** (interior area only) on the Stage-1 grid | `stage_gt/gt_room_labels.npy` (+ QA PNG) |
| `pq_eval.ipynb` (NB8) | Score every method: **mean room IoU** (paper Eq. 6/7) + over/under-seg, and PQ | `stage_gt/room_results.json`, `stage_gt/pq_results.json` |

They are **thin drivers**  - all logic lives in `scan2bim.eval` (pure + unit-tested). You only
edit `params.yaml`, never the notebook cells.

---

## What must already exist (upstream stages)

The evaluation notebooks consume artifacts other notebooks wrote into `{out_root}/`. Each method
reads its **own** stage (no aliasing). Run the producers first:

| Stage dir | Produced by | Needed for | Compute |
|---|---|---|---|
| `stage1_occupancy/` | `preprocessing/notebook_1_occupancy_raster.ipynb` | **Always**  - the grid + the shared wall scaffold; both eval notebooks need it | CPU (local) |
| `stage2_watershed/` | `methods/geometric/notebook_1_watershed.ipynb` | the **Geometry** row | CPU (local) |
| `stage_sam_auto/` | `methods/SAM/notebook_1_sam_auto_segmentation.ipynb` | the **SAM** row | **GPU + SAM checkpoint** (Colab) |
| `stage4_sam_refined/` | `methods/geometric_SAM/notebook_2_sam_refinement.ipynb` (needs `stage2` first) | the **Geometry + SAM** row | **GPU + SAM checkpoint** (Colab) |

> The wall-assignment notebooks (`geometric/notebook_2`, `SAM/notebook_2`,
> `geometric_SAM/notebook_3`) are **not** needed here  - they feed the wall-accuracy evaluation
> (research-fixes Task 04), not room IoU.

**Missing or stale stages are handled gracefully.** `pq_eval` skips any method whose stage is
absent (`SKIP  - stage not run`) or whose grid doesn't match the GT (`SKIP  - grid … != GT …`,
which happens if a stage was produced on an older scan/resolution). You still get a valid table
for whatever ran  - you don't need all three methods to get a result.

---

## Step-by-step

### 1. Point `params.yaml` at a matched scan + GT pair
The scan (`input.file_path`) and the GT (`groundtruth.gt_dir`) must be the **same scene** in the
**same world frame**. Bundled matched pairs:

```yaml
input:
  file_path: data/area1.xyz     # full cloud (interior points present)
groundtruth:
  gt_dir: data/Area_1           # per-room <room>/Annotations/<class>_*.txt
```

(For Area_3 instead, set `file_path: data/area3.xyz` **and** `gt_dir: data/Area_3` together.)
Use a **full** cloud as the scan, not `data/area3_structural.ply`  - a structural-only cloud loses
every room to void-rejection unless you also set `watershed.min_coverage_frac: 0.0`.

### 2. Run the upstream stages (once per scan)
At minimum, for a result you can use:
- `preprocessing/notebook_1_occupancy_raster.ipynb`  → `stage1_occupancy/`  *(required)*
- `methods/geometric/notebook_1_watershed.ipynb`     → `stage2_watershed/`  *(Geometry; local)*

Optionally, for the other two rows (need GPU + a SAM checkpoint, typically on Colab):
- `methods/SAM/notebook_1_sam_auto_segmentation.ipynb`         → `stage_sam_auto/`
- `methods/geometric_SAM/notebook_2_sam_refinement.ipynb`      → `stage4_sam_refined/`

### 3. Build the ground truth  - `gt_raster.ipynb` (NB7)
Reads each room's `Annotations/`, keeps the **interior** classes and drops the structural +
clutter classes (`wall`/`beam`/`column`/`door`/`window`/`clutter`), rasterises onto the Stage-1
grid. **Hard-fails** if under 98% of GT points back-project into the grid (the scan/GT frame
gate). Verify the printed `OK  - GT and scan share the coordinate frame` line and eyeball the QA
PNG (`stage_gt/gt_room_labels_color.png`)  - room blobs should have **no wall outlines**.

### 4. Score the methods  - `pq_eval.ipynb` (NB8)
Loads the GT, the shared Stage-1 wall scaffold, and each method's own labels; prints the primary
**mean room IoU** table (+ over/under-seg) and the secondary **PQ** table; writes
`room_results.json` and `pq_results.json` into `stage_gt/`.

---

## How to run a notebook

Either open it in Jupyter/VS Code and **Run All** top-to-bottom, or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/evaluation/gt_raster.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/evaluation/pq_eval.ipynb
```

Quick sanity check of the result without opening a notebook:

```bash
python -c "import json; d=json.load(open('scan2bim_out/stage_gt/room_results.json')); [print('%-16s meanIoU=%.3f over=%d under=%d'%(k,v['mean_iou'],v['over_seg'],v['under_seg'])) for k,v in d.items()]"
```

---

## Outputs (`{out_root}/stage_gt/`)

- `gt_room_labels.npy`  - int32 `(H,W)`: `0` exterior, `>=1` room id.
- `gt_room_labels_color.png`  - QA image.
- `room_results.json`  - **primary**, per method: `mean_iou`, `per_room` (per-room IoU),
  `over_seg`, `under_seg`, `matched_pairs`, room counts. (Task 07 aggregates this across areas.)
- `pq_results.json`  - **secondary**, per method: `SQ`, `RQ`, `PQ`, `TP`, `FP`, `FN`.

## Troubleshooting

- **`Frame mismatch … need >= 98%`** in NB7 → the scan and GT are different scenes/frames. Fix the
  pair in `params.yaml` (and re-run preprocessing NB1 if you changed the scan).
- **A method prints `SKIP  - grid … != GT …`** → that stage was produced on an older grid. Re-run
  its producing notebook against the current Stage-1 grid.
- **`No method produced labels on the GT grid`** → you haven't run any method's stage yet; run at
  least `geometric/notebook_1_watershed.ipynb`.
- **`Stage 'stageX' not found`** → run the producing notebook (see the table above), or it's an
  optional method you can ignore.
