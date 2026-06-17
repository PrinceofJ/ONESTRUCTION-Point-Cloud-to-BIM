# Scan-to-BIM Room Segmentation — Refactored Pipeline

This refactor splits the original monolithic `RoomSegmentation` notebook into a shared
Python package (`scan2bim/`) plus five thin driver notebooks. All reusable logic lives in
the package so nothing is duplicated across notebooks; each notebook loads the previous
stage's outputs from disk (or its ZIP) and writes a structured, zipped output directory.

**One edit surface.** The only file a user edits is `params.yaml` at the project root. Every
notebook's first cell is `CFG = scan2bim.load_config()` — the shared loader
(`scan2bim/runconfig.py`) reads `params.yaml` over the `Config` defaults, resolves
`file_path` **and** `out_root` to absolute paths under the project root, and returns the one
`CFG`. There is no per-notebook bootstrap or `Config(...)` literal anymore.

---

## Execution order — `1 → 2 → 3 → 4`

The wall-assignment algorithm's **first step is "obtain room mask `M_i`"**, and room masks
are produced only by the watershed. So the watershed must run **before** wall assignment.
The notebooks (and their stage folders) are numbered to match that dependency order:

```
Notebook 1  (occupancy rasters)
      │
      ▼
Notebook 2  (watershed → room masks)
      │
      ▼
Notebook 3  (boundary-ring wall assignment on WATERSHED masks → stage3_walls)   ← local, always
      │
      ▼
Notebook 4  (prompted SAM refinement of the masks; GPU/Colab)                   ┐ optional
      │                                                                         │
      ▼                                                                         │
Notebook 5  (the SAME boundary-ring assignment on SAM-refined masks → stage5)   ┘
```

Notebook 3 is **watershed-only** — it never reads SAM output. SAM-refined wall clouds are a
dedicated **Notebook 5**, so a fresh `1 → 2 → 3` always completes with no switch to set.

Every notebook also loads what it needs from the upstream ZIP, so any stage can be re-run
independently once its dependencies have been produced once. Each consuming stage validates
that the cloud + grid it loads match the upstream `config.json` (`assert_upstream_config`) and
that the reloaded cloud lands inside the upstream raster (`assert_points_in_grid`).

---

## Per-notebook responsibility, inputs, outputs

### Notebook 1 — Occupancy Raster Generation  (`stage1_occupancy`)
- **Does:** load cloud → slab crop → rasterise. Behaviour identical to the original.
- **In:** point cloud at `CFG.file_path`.
- **Out:** `occupancy.png`, `wall_mask.npy`, `wallness.npy`, `coverage.npy`,
  `transform.json`, `config.json`.

### Notebook 2 — Watershed Segmentation  (`stage2_watershed`)
- **Does:** the original deterministic distance-transform watershed, unchanged, relocated
  into `segment_rooms_watershed`.
- **In:** `wall_mask.npy` (or `wallness.npy` if `use_wallness`), `coverage.npy`,
  `transform.json` from stage 1.
- **Out:** `room_labels.npy` (`-1` wall / `0` exterior / `≥1` rooms),
  `room_labels_color.png`, `walls.npy`, `footprint.npy`, `transform.json`, `config.json`.

### Notebook 3 — Room Masks & Wall Assignment  (`stage3_walls`)
- **Does:** the **new boundary-ring** wall assignment on the **watershed** masks, then
  back-projection to 3-D. Watershed-only — no `mask_source` switch.
- **In:** `wallness.npy` + `transform.json` (stage 1), `room_labels.npy` (stage 2), and the
  point cloud (reloaded with the original loader).
- **Out:** `room_XX_walls.ply` per room, `room_wall_masks.npz`, `room_labels.npy`,
  `transform.json`, `config.json`.

### Notebook 5 — Walls on SAM-Refined Masks  (`stage5_walls_sam_refined`)
- **Does:** the identical boundary-ring assignment as Notebook 3, but on the **SAM-refined**
  masks (`room_labels_refined.npy`) from stage 4. **Fails loudly** if stage 4 is absent
  ("run Notebook 4 first") — never falls back to the watershed masks.
- **In:** `wallness.npy` + `transform.json` (stage 1), `room_labels_refined.npy` (stage 4),
  and the point cloud.
- **Out:** same artifact set as Notebook 3, written to `stage5_walls_sam_refined/`.

### Notebook 4 — Prompted SAM Refinement  (`stage4_sam_refined`, GPU/Colab)
- **Does:** prompts SAM per watershed room (points from the eroded interior + room box),
  then a **single-pass region-adjacency-graph** relabel merges/splits rooms behind the
  model-agnostic `MaskGenerator` (default backend SAM2). Confidence × geometry gated and
  snapped to the wall scaffold. Passes the watershed labels through unchanged if no SAM
  backend/checkpoint is available (never fabricates masks).
- **In:** `occupancy.png`, `wallness.npy`, `coverage.npy` (stage 1); `room_labels.npy`,
  `walls.npy`, `footprint.npy`, `config.json` (stage 2); a SAM 2.1 checkpoint.
- **Out:** `room_labels_refined.npy`, `room_labels_refined_color.png`, `transform.json`,
  `config.json`.

---

## Data flow / artifact contract

| Artifact            | Produced by | Consumed by         |
|---------------------|-------------|---------------------|
| `transform.json`    | N1          | N2, N3, N4          |
| `wall_mask.npy`     | N1          | N2                  |
| `wallness.npy`      | N1          | N3, N4              |
| `coverage.npy`      | N1          | N2, N4              |
| `occupancy.png`     | N1          | N4                  |
| `room_labels.npy`   | N2          | N3, N4              |
| `walls.npy`         | N2          | N4                  |
| `footprint.npy`     | N2          | N4                  |
| `room_labels_refined.npy` | N4    | N5 (opt.)           |
| `stage5_walls_sam_refined/` | N5    | downstream 3-D consumers |
| point cloud         | (external)  | N1, N3, N5, (N4 opt.) |

Stage directories live under `CFG.out_root`; each is zipped to `{stage}.zip`. The point
cloud is treated as an external Drive input (every stage that needs it reloads it
deterministically), so it is **not** bundled into the ZIPs.

---

## Optional: SAM-refined wall clouds (`stage5_walls_sam_refined`)

Notebook 3 assigns walls on the **watershed** masks and writes `stage3_walls`. SAM-refined
wall clouds are produced by the dedicated **Notebook 5** (not by re-running N3 with a switch):

1. Run `1 → 2 → 3` locally as usual (`stage3_walls` from the watershed masks).
2. Run **Notebook 4** in Colab to produce `stage4_sam_refined.zip`; download it into your
   local `scan2bim_out/`.
3. **Run Notebook 5.** It loads the refined labels (`room_labels_refined.npy`) from stage 4,
   runs the identical boundary-ring assignment, and writes **`stage5_walls_sam_refined`**
   without touching `stage3_walls`. If stage 4 is missing it stops with a clear error.

So the watershed-based wall clouds (`stage3_walls`) and the SAM-refined wall clouds
(`stage5_walls_sam_refined`) coexist side by side; both hold the same artifact set
(`room_XX_walls.ply`, `room_wall_masks.npz`, `room_labels.npy`, `transform.json`,
`config.json`) and feed the same downstream 3-D consumers.

---

## The one behavioural change — wall assignment

**Original (`room_footprints` / `split_rooms_to_clouds`):** for each room, dilate its
interior and intersect with the **occupancy** wall pixels (`labels == -1`); back-project
the points there.

**New (`room_wall_masks_boundary_ring`):** for each room mask `M_i`:
`I_i = erode(M_i)` → `B_i = M_i \ I_i` → `B_i' = dilate(B_i, r_w)` →
`walls_i = B_i' ∩ wallness`; back-project those pixels.

Two differences that matter:
1. **Wall source is the `wallness` raster** (vertical-extent, furniture-suppressed) rather
   than the post-segmentation occupancy wall pixels.
2. **Geometry is a boundary ring**, not the whole dilated interior — tighter, less prone to
   grabbing a neighbouring room's wall.

Radii: `CFG.room_erode_m` and `CFG.wall_dilate_m` (`r_w`). If either is `None` it is
auto-derived from the estimated wall thickness using the **same** distance-transform
median heuristic the original used inside `room_footprints`.

---

## SAM refinement — prompted, graph-based (Notebook 4)

SAM is **not** run in automatic "segment everything" mode. The watershed is a strong prior;
SAM only adjusts topology where it is confident and the geometry is weak.

**Pipeline (`scan2bim.sam_refine.refine_with_sam`):**
1. **SAM input image** (`build_sam_image`): stack the three N1 rasters as channels —
   occupancy (free space) / wallness (structure) / coverage (scanned data). A realistic
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
  from Notebook 1 is the single coordinate contract. Cross-stage validation
  (`assert_upstream_config` + `assert_points_in_grid`) enforces that every stage saw the same
  cloud + grid.
- The cloud loads to metres and voxel-downsamples deterministically, so reloading it in
  Notebook 3/5 reproduces the same points and stays aligned to Notebook 1's transform.
- The point cloud is an external input, not a produced artifact, so it is not zipped.
- Notebook 3 (wall assignment) also consumes the watershed `room_labels.npy`, since its
  step 1 ("obtain room mask `M_i`") needs the masks.
- Notebook 3 is watershed-only; SAM-refined wall clouds come from **Notebook 5**, which reads
  `room_labels_refined.npy` from stage 4.
- SAM is optional: with no backend/checkpoint, Notebook 4 returns the watershed labels.

---

## Future cleanup (out of scope here)

- The slab/raster path recomputes floor/ceiling histograms in several functions; compute
  once and thread the values through.
- The legacy `room_footprints` / `split_rooms_to_clouds` and the legacy door/endpoint
  bridging are retained only for the legacy debug overlay — drop once the new method is
  accepted.
- `rasterize_wallness` over-flags cells (it measures vertical span over *all* points in a
  cell, so floor+ceiling returns flag nearly everywhere); this is a separate algorithm issue
  left for a follow-up — the cleanup refactor changed plumbing only.
- The prompted-SAM gating thresholds (`sam_conf_thresh`, `sam_merge_cover_frac`, …) are
  conservative defaults; calibrate them once a labelled set exists.

---

## Running

Open the project folder in VS Code, create a venv and `pip install -e .`, put your cloud
at `data/area1.xyz` (or set `input.file_path` in `params.yaml`), then **Run All** on notebooks
**1 → 2 → 3** locally (CPU) — no cell edits. Each writes `{stage}.zip` under `scan2bim_out/`.
**Notebook 4 is the GPU stage:** copy `scan2bim/`, `params.yaml`, and the `scan2bim_out/` ZIPs
to Google Drive and run it in Colab on a GPU (it loads CFG via `load_config()`, validated
against the stage-2 `config.json`, so its output aligns pixel-for-pixel with the watershed).
**Notebook 5** turns the SAM-refined masks into wall clouds (`stage5_walls_sam_refined`). See
the top-level `README.md` / `RUNBOOK.md` for step-by-step setup.
