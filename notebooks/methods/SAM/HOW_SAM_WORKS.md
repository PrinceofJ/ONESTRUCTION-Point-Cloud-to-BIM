# How the pure-SAM room segmentation works

This is the "SAM (no geometric prior)" branch of the three-method comparison. It reproduces
the room pipeline from Albadri et al. (ISPRS Archives XLVIII-G-2025-131, §3.1): run SAM in
**automatic "segment everything" mode** on a top-down slab image, then turn the raw masks
into clean room labels. No watershed, no prompts.

The notebook is [`notebook_1_sam_auto_segmentation.ipynb`](notebook_1_sam_auto_segmentation.ipynb);
the logic lives in [`scan2bim/sam_auto.py`](../../../scan2bim/sam_auto.py) and the image
builder in [`scan2bim/sam_refine.py`](../../../scan2bim/sam_refine.py).

---

## The big picture

SAM (Segment Anything Model) is a general image-segmentation network. It does **not** know
what a "room" is — it just finds coherent regions in an image. Our job is:

1. **Give SAM the right picture** — a top-down binary image where walls are black lines and
   room interiors are empty space.
2. **Let SAM segment everything** — its automatic generator drops a grid of points over the
   image and produces one candidate mask per point ("what region is this pixel part of?").
3. **Turn masks into rooms** — resolve overlaps, drop walls, drop things that are too small
   or aren't over scanned area, and hand back an integer label map.

The output is a `room_labels.npy` on the **same grid** the other two methods use, so
`evaluation/pq_eval.ipynb` scores all three against the same ground truth with zero
resampling. Label convention (identical to the watershed method):

| value | meaning |
|-------|---------|
| `-1`  | wall    |
| `0`   | exterior / unclaimed |
| `>=1` | a room  |

---

## Step 1 — Build the image SAM sees (`build_sam_image`)

Everything starts from the **Stage-1 occupancy raster** (`occupancy.png`): a top-down
projection of a thin horizontal **slab** of the point cloud, where `255 = free space` and
`0 = wall/occupied`.

`build_sam_image(occ, wall_mask, coverage, mode=...)` turns that into the 3-channel uint8
image SAM ingests. **The mode matters a lot:**

- **`mode='occupancy'` (what this notebook uses).** Just the binary slab occupancy `Ms`,
  copied into all 3 channels. Walls are black lines, room interiors are empty. This is the
  paper-faithful input — SAM segments the empty pockets *between* walls, which are the rooms.
- **`mode='stack'` (the library default).** Packs three different channels: occupancy + the
  slab wall mask + the **coverage** raster (where the scan actually has data). The coverage
  channel *fills in* scanned interiors — which makes SAM tend to grab the whole building as
  one blob. Good for the *prompted* refinement method (Notebook 4), wrong for pure-SAM.

> That's why the notebook sets `CFG.sam_image_mode = 'occupancy'` locally instead of in
> `params.yaml`: the refinement method legitimately wants `'stack'`, so the choice is scoped
> per-method. Flip it to `'stack'` in that cell only if you want to ablate.

The colourised label PNG is **never** fed to SAM — only the binary/greyscale rasters.

---

## Step 2 — Run SAM's automatic mask generator (`build_auto_mask_generator`)

**Where this happens in the notebook** ([`notebook_1_sam_auto_segmentation.ipynb`](notebook_1_sam_auto_segmentation.ipynb)):

- **Section 2, cell `sa1_05`** — installs SAM 2 and downloads the checkpoint
  (`SAM_CKPT`, `SAM_CFG`). Setup only; nothing runs yet.
- **Section 3, cell `sa1_07`** — picks the backend/checkpoint:
  `CFG.sam_backend = 'sam2'`, `CFG.sam_ckpt = SAM_CKPT`, `CFG.sam_model_cfg = SAM_CFG`.
- **Section 4, cell `sa1_09`** — the generator actually **runs** here, inside
  `scan2bim.segment_rooms_sam_auto(image, walls=slab_walls, coverage=coverage, cfg=CFG)`.

**Where this happens in the library:**

- [`scan2bim/sam_auto.py:71`](../../../scan2bim/sam_auto.py#L71) — `build_auto_mask_generator`
  builds the backend for `cfg.sam_backend` (dispatch table at
  [`sam_auto.py:68`](../../../scan2bim/sam_auto.py#L68); per-backend builders `_build_auto_sam2`
  etc. at [`sam_auto.py:52`](../../../scan2bim/sam_auto.py#L52)).
- [`scan2bim/sam_auto.py:33`](../../../scan2bim/sam_auto.py#L33) — `_amg_kwargs` maps the
  Table-1 config knobs onto the generator.
- [`scan2bim/sam_auto.py:223`](../../../scan2bim/sam_auto.py#L223) — the actual
  `generator.generate(image)` call, inside
  [`segment_rooms_sam_auto`](../../../scan2bim/sam_auto.py#L192).

`segment_rooms_sam_auto` builds a backend generator for `cfg.sam_backend`
(`'sam2'` default, also `'sam1'` / `'sam3'`) and calls `generator.generate(image)`. That
returns a list of records, each with a boolean `segmentation` mask and a `predicted_iou`
confidence score.

The knobs (paper Table 1, defined in [`config.py`](../../../scan2bim/config.py#L142)) are:

| config field | default | what it does |
|--------------|---------|--------------|
| `sam_points_per_side` | `15` | grid of query points; **more = finer/more masks** (paper used 11/15/30 per case study) |
| `sam_pred_iou_thresh` | `0.85` | drop masks SAM isn't confident are accurate |
| `sam_stability_score_thresh` | `0.95` | drop masks that change a lot under threshold jitter |
| `sam_min_mask_region_area` | — | drop tiny mask fragments |
| `sam_crop_n_layers` / `..._downscale_factor` | — | multi-crop passes for big images |

**`points_per_side` is the main dial.** Too low → SAM misses small rooms; too high → it
over-segments and you get one room split into several masks (later resolved by overlap
painting, but still noisier).

> **This step needs a CUDA GPU** — run it in Colab (Runtime → Change runtime type → GPU).
> Pure-SAM is meaningless without a backend, so if none builds, the function returns an
> all-exterior map with `debug['ran'] = False` and the notebook **raises loudly** rather than
> fabricating rooms.

---

## Step 3 — Masks → room labels (`masks_to_room_labels`)

Raw SAM masks overlap and include junk. This function cleans them into one integer label map:

1. **Clip walls out** — `mask & ~walls`; rooms never sit on wall pixels.
2. **Reject voids** — if a mask barely overlaps the **coverage** raster (mostly unscanned
   space / exterior), drop it. Controlled by `sam_auto_min_coverage_frac`.
   *(Gotcha: on a structural-only cloud with no coverage, keep this at `0.0` or every mask
   gets rejected and you get 0 rooms.)*
3. **Resolve overlaps deterministically** — sort candidates ascending by
   `(score, area, -index)` and paint them in order, so the **highest-confidence / largest /
   lowest-index** mask is painted *last* and wins any overlap. Order-independent → same
   result every run.
4. `-1` is stamped back onto wall pixels; rooms are relabelled `1..N`.

Then **`classify_rooms_by_area`** drops any room smaller than
`sam_auto_min_room_area_m2` (paper's `A = 1.5 m²`), converted to pixels via `pixel_m`. This
is what removes closets, mask slivers, and noise.

---

## Step 4 — Optional refinement passes

Both **off** by default; enable in `params.yaml`.

- **`sam_reprocess_residual`** (`reprocess_residual`) — SAM often misses **corridors**
  (long thin spaces). This masks the image down to *only* the unclaimed free space and runs
  SAM again with a sparser grid (`sam_residual_points_per_side`, paper = 5), merging any new
  rooms in with fresh ids. This is the "corridor-reprocessing trick" from the paper.
- **`sam_auto_buffer_rooms`** (`buffer_room_labels`) — grows each room outward by
  `do_buffer_px` into adjacent unclaimed free space via a watershed, respecting walls as
  barriers. Cleans up the thin unlabeled gap SAM leaves around room edges.

Finally `labels[walls] = -1` ("walls are sacred") and a last relabel to `1..N`.

The returned `debug` dict reports the whole funnel:
`n_masks → n_kept (n_void_dropped) → n_rooms_pass1 → n_rooms_out`, plus `reprocess.n_added`.
**Read this every run** — it tells you where rooms were lost.

---

## How to use it properly — a checklist

1. **Match the upstream stage.** SAM consumes the Stage-1 raster. `CFG.file_path` (the cloud)
   and every grid field (`pixel_m`, slab, voxel, up-axis…) **must** equal what produced that
   raster — `assert_upstream_config` enforces this and will stop you otherwise. If you change
   the cloud in `params.yaml`, **re-run the occupancy-raster stage** so the raster matches.
2. **Use a GPU.** Colab T4 is enough. Without a backend the notebook raises on purpose.
3. **Keep `sam_image_mode = 'occupancy'`** for pure-SAM (paper-faithful). `'stack'` is for the
   prompted-refinement method.
4. **Tune in this order when results look wrong:**
   - Missing small rooms → raise `sam_points_per_side`, or lower `sam_auto_min_room_area_m2`.
   - Rooms merged into one blob → check you're on `'occupancy'` mode, not `'stack'`.
   - Missing corridors → turn on `sam_reprocess_residual`.
   - Ragged room edges / thin gaps → turn on `sam_auto_buffer_rooms`.
   - **0 rooms** → check `sam_auto_min_coverage_frac` (set `0.0` for structural-only clouds)
     and confirm the backend actually built (`debug['ran'] == True`).
5. **Read the `debug` dict** to see exactly which stage dropped your rooms.
6. **Output** lands in `stage_sam_auto/room_labels.npy` on the Stage-1 grid, ready for
   `notebook_2_wall_assignment.ipynb` and `evaluation/pq_eval.ipynb`.

---

## One-line mental model

> **SAM finds regions in a wall-line drawing of the floor; the pipeline decides which of those
> regions are actually rooms** (over scanned area, big enough, walls removed, overlaps
> resolved), optionally recovering corridors and cleaning edges.
