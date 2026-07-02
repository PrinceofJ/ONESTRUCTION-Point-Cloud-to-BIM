# Scan-to-BIM Room Segmentation — Refactored Pipeline

This refactor splits the original monolithic `RoomSegmentation` notebook into a shared
Python package (`scan2bim/`) plus a set of thin driver notebooks. All reusable logic lives in
the package so nothing is duplicated across notebooks; each notebook loads the previous
stage's outputs from disk (or its ZIP) and writes a structured, zipped output directory.

**Three-method comparison, one shared front-end.** The notebooks are organised under
`notebooks/` by stage and method:

```
notebooks/
├── preprocessing/   one shared stage 1 (occupancy raster) feeding every method
├── methods/
│   ├── geometric/      METHOD 1 — deterministic watershed only
│   ├── SAM/            METHOD 2 — SAM auto-segmentation, no geometric prior  (Colab/GPU)
│   └── geometric_SAM/  METHOD 3 — watershed prior + prompted-SAM refinement
├── converters/      input data-prep (S3DIS → structural .ply)
└── evaluation/      Panoptic-Quality scoring (shared; scores every method vs. one GT)
```

The three methods are alternative ways to turn the shared Stage-1 rasters into room labels;
each writes its own stage directories so they coexist, and each ends in the **same**
boundary-ring wall assignment (the logic in `room_wall_masks_boundary_ring`, called on whichever
room masks the method produced).

**One edit surface.** The only file a user edits is `params.yaml` at the project root. Every
notebook's first cell is `CFG = scan2bim.load_config()` — the shared loader
(`scan2bim/runconfig.py`) reads `params.yaml` over the `Config` defaults, resolves
`file_path` **and** `out_root` to absolute paths under the project root, and returns the one
`CFG`. There is no per-notebook bootstrap or `Config(...)` literal anymore. `project_root()`
walks up from the kernel's CWD to find the package, so notebooks resolve the root no matter how
deep they sit under `notebooks/`.

---

## Inputs & coordinate-frame integrity (scan + GT must share a frame)

The process is **scan in → GT in → segment → compare → evaluate**, and is dataset-agnostic.
Two **independently sourced** inputs are set in `params.yaml` and must be a matched pair:

- **The scan** — `input.file_path`, the real-life cloud the three methods *segment*. It must
  be a **full** cloud (interior points present) so the watershed's void-rejection and coverage
  gating behave — **not** a structural-only cloud such as `data/area3_structural.ply`
  (wall/column/beam/door/window annotations only), on which the watershed loses every room to
  void-rejection unless `min_coverage_frac == 0`. Stage 1 rasterizes the scan and writes the
  `transform.json` that is the single coordinate contract for every later stage.
- **The ground-truth model** — `groundtruth.gt_dir`, a directory of per-room point files
  `<room>/<room>.txt` (wall accuracy additionally uses `Annotations/wall_N.txt`).
  `evaluation/gt_raster.ipynb` rasterizes these onto the Stage-1 grid to build
  `gt_room_labels.npy`; `pq_eval` scores each method's labels against it. The GT does **not**
  come from the scan.

Because the two are sourced separately, nothing structurally guarantees they share an XY frame
+ units. `gt_raster.ipynb` enforces it: it back-projects the GT room points through the scan's
Stage-1 transform and **hard-fails if under 98 %** land in-grid, printing the in-bounds fraction
and the GT-vs-grid bounding boxes (`scan2bim.grid_world_bbox`). The QA overlay (GT vs. watershed
rooms) is a required visual gate before any `pq_eval` number is trusted. Notebook 1 additionally
warns, via `scan2bim.interior_coverage_fraction`, if a structural-only cloud is loaded while
`min_coverage_frac > 0`. This frame guarantee also covers the wall GT (`wall_N.txt`), which lives
in the same Area frame as the room files.

For the bundled data, the scan `data/area1.xyz` pairs with `data/gt_dir = data/Area_1` (~100 %
in-grid). `data/Area_3` is a **different** scene/frame (~44 % in-grid) and the gate rejects it —
the kind of silent scan/GT mismatch this check exists to catch. Swap both `file_path` and
`gt_dir` to any other matched pair to run a new scene.

---

## Execution order — shared preprocessing, then one method

The wall-assignment algorithm's **first step is "obtain room mask `M_i`"**, and room masks
are produced by a segmentation stage. So segmentation must run **before** wall assignment.
Every method shares the Stage-1 raster, then runs its own segmentation → wall-assignment chain:

```
preprocessing/notebook_1_occupancy_raster   →  stage1_occupancy   (shared by all methods)
      │
      ├─ METHOD 1  geometric/
      │     notebook_1_watershed              →  stage2_watershed
      │     notebook_2_wall_assignment        →  stage3_walls
      │
      ├─ METHOD 2  SAM/                                              (pure SAM; GPU)
      │     notebook_1_sam_auto_segmentation  →  stage_sam_auto
      │     notebook_2_wall_assignment        →  stage_sam_walls
      │
      └─ METHOD 3  geometric_SAM/
            notebook_1_watershed              →  stage_geometric_sam_watershed
            notebook_2_sam_refinement (GPU)   →  stage4_sam_refined
            notebook_3_wall_assignment        →  stage5_walls_sam_refined
```

Each method's wall-assignment stage reads only **its own** masks and writes its own stage dir,
so the methods never clobber each other and a fresh run of any one method completes with no
switch to set. The geometric and geometric+SAM methods both start from a watershed; the
geometric+SAM copy writes `stage_geometric_sam_watershed` (not `stage2_watershed`) precisely so
the two coexist.

> **Wiring note (follow-up):** `geometric_SAM/notebook_1_watershed` writes its own
> `stage_geometric_sam_watershed`, but `geometric_SAM/notebook_2_sam_refinement` (the original
> Colab SAM notebook) still reads the watershed masks from `stage2_watershed` (`A.STAGE2`).
> Repointing the SAM-refinement stage at `stage_geometric_sam_watershed` is a small follow-up;
> it was left untouched here so the SAM-refinement logic is byte-for-byte the original.

Every notebook also loads what it needs from the upstream ZIP, so any stage can be re-run
independently once its dependencies have been produced once. Each consuming stage validates
that the cloud + grid it loads match the upstream `config.json` (`assert_upstream_config`) and
that the reloaded cloud lands inside the upstream raster (`assert_points_in_grid`).

---

## Per-notebook responsibility, inputs, outputs

### Preprocessing (shared)

#### `preprocessing/notebook_1_occupancy_raster`  (`stage1_occupancy`)
- **Does:** load cloud → slab crop → rasterise. Behaviour identical to the original.
- **In:** point cloud at `CFG.file_path`.
- **Out:** `occupancy.png`, `wall_mask.npy`, `coverage.npy`,
  `transform.json`, `config.json`.

### Method 1 — geometric (`methods/geometric/`)

#### `notebook_1_watershed`  (`stage2_watershed`)
- **Does:** the original deterministic distance-transform watershed, unchanged, relocated
  into `segment_rooms_watershed`.
- **In:** `wall_mask.npy`, `coverage.npy`, `transform.json` from stage 1.
- **Out:** `room_labels.npy` (`-1` wall / `0` exterior / `≥1` rooms),
  `room_labels_color.png`, `walls.npy`, `footprint.npy`, `transform.json`, `config.json`.

#### `notebook_2_wall_assignment`  (`stage3_walls`)
- **Does:** the **boundary-ring** wall assignment on the **watershed** masks, then
  back-projection to 3-D.
- **In:** `wall_mask.npy` + `transform.json` (stage 1), `room_labels.npy` (stage 2), and the
  point cloud (reloaded with the original loader).
- **Out:** `room_XX_walls.ply` per room, `room_wall_masks.npz`, `room_labels.npy`,
  `transform.json`, `config.json`.

### Method 2 — SAM (`methods/SAM/`)

The paper's pure-SAM room pipeline (Albadri et al., ISPRS 2025, §3.1). Logic lives in
`scan2bim.sam_auto`; the model (SAM's automatic mask generator) is isolated behind one
`AutoMaskGenerator` adapter, and everything else is a pure, unit-tested function over plain
arrays (`tests/test_sam_auto.py`) — the same model/math split as `sam_refine`.

#### `notebook_1_sam_auto_segmentation`  (`stage_sam_auto`, GPU/Colab)
- **Does:** runs SAM in automatic "segment everything" mode on the Stage-1 rasters (no
  watershed prior, no prompts) via `segment_rooms_sam_auto`, then deterministically:
  drops masks below the px noise floor, drops the exterior background mask via `coverage`,
  resolves overlaps (higher predicted-IoU wins; order-independent), re-imposes `-1` on every
  wall pixel, classifies room/not-room by area (`A = sam_auto_min_room_area_m2`, paper 1.5 m²),
  and — when enabled in `params.yaml` — does corridor reprocessing (`sam_reprocess_residual`,
  paper §4.2) and/or the outward boundary buffer (`sam_auto_buffer_rooms`, paper `do`). SAM's
  Table-1 params are `Config` fields (`sam_points_per_side`, `sam_pred_iou_thresh`, …).
- **In:** `occupancy.png`, `wall_mask.npy`, `coverage.npy`, `transform.json` (stage 1); a SAM
  checkpoint. **Requires a real SAM backend** — fails loudly on no GPU/checkpoint
  (`debug['ran']` is False) rather than passing through; pure-SAM with no SAM is meaningless.
- **Out:** `room_labels.npy` (Stage-1 grid), `room_labels_color.png`, `transform.json`,
  `config.json`.

#### `notebook_2_wall_assignment`  (`stage_sam_walls`)
- **Does:** the identical boundary-ring assignment (`room_wall_masks_boundary_ring`), on the
  SAM-method masks from `stage_sam_auto`. Same call as the geometric method — only the mask
  source differs — then back-projects within the floor↔ceiling band.
- **In:** `wall_mask.npy` + `transform.json` (stage 1), `room_labels.npy` (`stage_sam_auto`),
  the point cloud.
- **Out:** same artifact set as the geometric wall-assignment, to `stage_sam_walls/`.

### Method 3 — geometric + SAM (`methods/geometric_SAM/`)

#### `notebook_1_watershed`  (`stage_geometric_sam_watershed`)
- **Does:** functionally identical to `geometric/notebook_1_watershed` — the same
  `segment_rooms_watershed` — but writes a **distinct** stage dir so it never clobbers the
  pure-geometric run's `stage2_watershed`.
- **In/Out:** same as the geometric watershed, except the output dir is
  `stage_geometric_sam_watershed`.

#### `notebook_2_sam_refinement`  (`stage4_sam_refined`, GPU/Colab)
- **Does:** prompts SAM per watershed room (points from the eroded interior + room box),
  then a **single-pass region-adjacency-graph** relabel merges/splits rooms behind the
  model-agnostic `MaskGenerator` (default backend SAM2). Confidence × geometry gated and
  snapped to the wall scaffold. Passes the watershed labels through unchanged if no SAM
  backend/checkpoint is available (never fabricates masks).
- **In:** `occupancy.png`, `wall_mask.npy`, `coverage.npy` (stage 1); `room_labels.npy`,
  `walls.npy`, `footprint.npy`, `config.json` (`stage2_watershed` — see the wiring note above);
  a SAM 2.1 checkpoint.
- **Out:** `room_labels_refined.npy`, `room_labels_refined_color.png`, `transform.json`,
  `config.json`.

#### `notebook_3_wall_assignment`  (`stage5_walls_sam_refined`)
- **Does:** the identical boundary-ring assignment, but on the **SAM-refined** masks
  (`room_labels_refined.npy`) from stage 4. **Fails loudly** if stage 4 is absent ("run the
  SAM-refinement notebook first") — never falls back to the watershed masks.
- **In:** `wall_mask.npy` + `transform.json` (stage 1), `room_labels_refined.npy` (stage 4),
  the point cloud.
- **Out:** same artifact set as the geometric wall-assignment, to `stage5_walls_sam_refined/`.

---

## Data flow / artifact contract

Short labels: **Pre** = `preprocessing/notebook_1_occupancy_raster`; **G·ws / G·wa** =
`geometric/` watershed / wall-assignment; **GS·ws / GS·ref / GS·wa** = `geometric_SAM/`
watershed / sam-refinement / wall-assignment; **S·auto / S·wa** = `SAM/` auto-segmentation /
wall-assignment (`stage_sam_auto → stage_sam_walls`, mirroring G's two-stage shape).

| Artifact (stage)                                  | Produced by | Consumed by                         |
|---------------------------------------------------|-------------|-------------------------------------|
| `transform.json` (stage1)                         | Pre         | every later stage                   |
| `wall_mask.npy` (stage1)                          | Pre         | G·ws, GS·ws, G·wa, GS·ref, GS·wa, S·* |
| `coverage.npy` (stage1)                           | Pre         | G·ws, GS·ws, GS·ref                 |
| `occupancy.png` (stage1)                          | Pre         | GS·ref                              |
| `room_labels.npy` (stage2_watershed)              | G·ws        | G·wa, GS·ref                        |
| `walls.npy`, `footprint.npy` (stage2_watershed)   | G·ws        | GS·ref                              |
| `room_labels.npy` (stage_geometric_sam_watershed) | GS·ws       | — *(intended GS·ref; see wiring note)* |
| `room_labels_refined.npy` (stage4_sam_refined)    | GS·ref      | GS·wa                               |
| `stage3_walls/`, `stage5_walls_sam_refined/`      | G·wa, GS·wa | downstream 3-D consumers            |
| point cloud (external)                            | (external)  | Pre, G·wa, GS·wa, (GS·ref opt.)     |

Stage directories live under `CFG.out_root`; each is zipped to `{stage}.zip`. The point
cloud is treated as an external Drive input (every stage that needs it reloads it
deterministically), so it is **not** bundled into the ZIPs.

---

## Geometric vs. geometric+SAM wall clouds

The **geometric** method's `notebook_2_wall_assignment` assigns walls on the **watershed** masks
and writes `stage3_walls`. The **geometric+SAM** method produces SAM-refined wall clouds with its
own three notebooks (never by flipping a switch on the geometric ones):

1. Run the **geometric+SAM** watershed (`geometric_SAM/notebook_1_watershed`) →
   `stage_geometric_sam_watershed`.
2. Run **`geometric_SAM/notebook_2_sam_refinement`** in Colab to produce `stage4_sam_refined.zip`;
   download it into your local `scan2bim_out/`.
3. **Run `geometric_SAM/notebook_3_wall_assignment`.** It loads the refined labels
   (`room_labels_refined.npy`) from stage 4, runs the identical boundary-ring assignment, and
   writes **`stage5_walls_sam_refined`** without touching `stage3_walls`. If stage 4 is missing it
   stops with a clear error.

So the geometric wall clouds (`stage3_walls`) and the geometric+SAM wall clouds
(`stage5_walls_sam_refined`) coexist side by side; both hold the same artifact set
(`room_XX_walls.ply`, `room_wall_masks.npz`, `room_labels.npy`, `transform.json`,
`config.json`) and feed the same downstream 3-D consumers. The `SAM/` method, once implemented,
adds a third parallel pair (`stage_sam_auto` → `stage_sam_walls`).

---

## The one behavioural change — wall assignment

**Original (`room_footprints` / `split_rooms_to_clouds`):** for each room, dilate its
interior and intersect with the **occupancy** wall pixels (`labels == -1`); back-project
the points there.

**New (`room_wall_masks_boundary_ring`):** for each room mask `M_i`:
`I_i = erode(M_i)` → `B_i = M_i \ I_i` → `B_i' = dilate(B_i, r_w)` →
`walls_i = B_i' ∩ wall_mask`; back-project those pixels.

Two differences that matter:
1. **Wall source is the slab `wall_mask`** (binary slab-occupancy wall pixels) — the same
   deterministic, method-agnostic raster every method shares (research-fixes Task 03/05). It
   replaced an earlier span-based `wallness` raster that saturated room interiors.
2. **Geometry is a boundary ring**, not the whole dilated interior — tighter, less prone to
   grabbing a neighbouring room's wall.

Radii: `CFG.room_erode_m` and `CFG.wall_dilate_m` (`r_w`). If either is `None` it is
auto-derived from the estimated wall thickness using the **same** distance-transform
median heuristic the original used inside `room_footprints`.

---

## SAM refinement — prompted, graph-based (geometric+SAM, `notebook_2_sam_refinement`)

This is the **geometric+SAM** method's refinement stage. SAM is **not** run in automatic
"segment everything" mode here (that is the separate **SAM** method); the watershed is a strong
prior and SAM only adjusts topology where it is confident and the geometry is weak.

**Pipeline (`scan2bim.sam_refine.refine_with_sam`):**
1. **SAM input image** (`build_sam_image`): stack the three Stage-1 rasters as channels —
   occupancy (free space) / slab wall mask (structure) / coverage (scanned data). A realistic
   top-down built from data already computed; no point cloud needed in this stage. (The
   colourised label map is QA-only and never fed to SAM.)
2. **Prompt per watershed room:** positive points from the room's *eroded* interior (the
   same erosion `walls.resolve_ring_radii_px` uses), the room bounding box, and optional
   negative points from neighbours. Every returned mask is labelled by construction — no
   IoU matching.
3. **Single-pass region-adjacency graph** (`relabel_by_sam`): nodes = watershed rooms. A
   confident SAM mask **spanning** an edge votes to merge it; a confident SAM mask **cutting**
   a room votes to split it (the split line is placed on the DT ridge by a tiny local
   watershed, not on SAM's raw outline). The whole graph is resolved once (union-find →
   connected components), so the result is order-independent.
4. **Confidence × geometry gating:** SAM may override the watershed only where its predicted
   IoU ≥ `sam_conf_thresh` **and** the shared boundary is an open DT ridge (`wall_frac` low
   **and** ridge ≥ `sam_open_ridge_m`). A wall-backed boundary is never overridden.
5. **Snap to the scaffold:** output is intersected with free space and `-1` is re-imposed on
   every wall pixel; final labels keep the watershed's exact shape and `-1/0/≥1` convention.
6. **Safety rail:** a merge/split is accepted only if it does not drop a cheap sanity score
   (coverage + compactness − wall straddle) by more than `sam_min_sanity_margin`.

**Model abstraction.** `MaskGenerator` is a *prompted* segmenter (`set_image` + `predict`).
`build_mask_generator(CFG)` picks the backend from `CFG.sam_backend` (`'sam2'` default |
`'sam3'` | `'sam1'`); the refinement code only calls `set_image`/`predict`, so swapping the
model = one adapter + the config string. SAM2 uses `build_sam2` + `SAM2ImagePredictor`;
SAM1 uses `SamPredictor`. `_build_sam3` is a temporary stub targeting the same shape — point
its two imports at your SAM3 build (SAM3 also adds text/concept prompts, not needed here).

**Default thresholds** (all in `config.py`): `sam_conf_thresh=0.88`, `sam_wall_frac_max=0.20`,
`sam_open_ridge_m=0.40`, `sam_merge_cover_frac=0.60`, `sam_split_min_frac=0.25`,
`sam_min_sanity_margin=-0.02`, `sam_pos_points=8`, `sam_neg_points=4`, `sam_image_mode='stack'`.
They are conservative (SAM rarely overrides the watershed); loosen `sam_conf_thresh` /
`sam_merge_cover_frac` to let SAM act more often.

**Point at a different model:** set `CFG.sam_backend` (`'sam2'|'sam3'|'sam1'`), `CFG.sam_ckpt`
(checkpoint path) and `CFG.sam_model_cfg` (the Hydra config name, e.g.
`configs/sam2.1/sam2.1_hiera_l.yaml`). For SAM3, set `sam_backend='sam3'` and adapt the two
imports in `_build_sam3`; nothing else changes.

---

## Assumptions

- Stages share one `CFG`, built once from `params.yaml` by `load_config()`; the transform
  from the preprocessing raster is the single coordinate contract. Cross-stage validation
  (`assert_upstream_config` + `assert_points_in_grid`) enforces that every stage saw the same
  cloud + grid.
- The cloud loads to metres and voxel-downsamples deterministically, so reloading it in any
  wall-assignment stage reproduces the same points and stays aligned to the Stage-1 transform.
- The point cloud is an external input, not a produced artifact, so it is not zipped.
- Each wall-assignment stage consumes its method's `room_labels.npy`, since its step 1 ("obtain
  room mask `M_i`") needs the masks — `geometric/notebook_2` reads the watershed masks,
  `geometric_SAM/notebook_3` reads the SAM-refined masks, `SAM/notebook_2` (when implemented)
  reads the SAM auto-segmentation masks.
- SAM is optional: with no backend/checkpoint, `geometric_SAM/notebook_2_sam_refinement` returns
  the watershed labels unchanged.

---

## Future cleanup (out of scope here)

- The slab/raster path recomputes floor/ceiling histograms in several functions; compute
  once and thread the values through.
- The legacy `room_footprints` / `split_rooms_to_clouds` and the legacy door/endpoint
  bridging are retained only for the legacy debug overlay — drop once the new method is
  accepted.
- The prompted-SAM gating thresholds (`sam_conf_thresh`, `sam_merge_cover_frac`, …) are
  conservative defaults; calibrate them once a labelled set exists.

---

## Running

Open the project folder in VS Code, create a venv and `pip install -e .`, put your cloud
at `data/area1.xyz` (or set `input.file_path` in `params.yaml`), then **Run All** on
`preprocessing/notebook_1_occupancy_raster` (CPU) — no cell edits — followed by one method's
notebooks under `notebooks/methods/`. The pure-**geometric** method
(`geometric/notebook_1_watershed` → `notebook_2_wall_assignment`) is CPU-only. The
**SAM-refinement** stage of the geometric+SAM method
(`geometric_SAM/notebook_2_sam_refinement`) is the GPU step: copy `scan2bim/`, `params.yaml`,
and the `scan2bim_out/` ZIPs to Google Drive and run it in Colab on a GPU (it loads CFG via
`load_config()`, validated against the upstream `config.json`, so its output aligns
pixel-for-pixel with the watershed). `geometric_SAM/notebook_3_wall_assignment` then turns the
SAM-refined masks into wall clouds (`stage5_walls_sam_refined`). See the top-level `README.md`
for step-by-step setup.
