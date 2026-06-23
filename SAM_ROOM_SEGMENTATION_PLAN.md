# Plan — Fitting the Paper's SAM Room Segmentation into the Pipeline

**Paper:** Albadri et al., *A SAM-Based Approach for Automatic Indoor Point Cloud
Segmentation*, ISPRS Archives XLVIII-G-2025-131 (Geospatial Week 2025).

**Goal:** implement the paper's room-segmentation method as the project's **pure-SAM
method (Method 2)** — currently a stub — so the three room-segmentation methods
(**geometric**, **geometric+SAM**, **pure SAM**) can be run and scored head-to-head, with the
new logic *modular* (model behind one adapter) and *easily testable* (deterministic core
exercised with hand-built fixtures, no GPU).

This is a design/plan document only. No code is changed by reading it.

---

## 1. Key finding — the paper *is* our pure-SAM method

The paper's pipeline and our stubbed `notebooks/methods/SAM/` chain describe the same thing:
run SAM in **automatic "segment everything" mode** on a top-down occupancy image (no
watershed prior, no prompts), turn the masks into room labels, classify room/not-room by
area, and buffer each room outward by ~half a wall thickness to recover wall points.

The stub's own TODO already names the target module (`scan2bim.sam_auto`) and the output
contract (`room_labels.npy` on the Stage-1 grid → `stage_sam_auto` → `notebook_2_wall_assignment`
→ `stage_sam_walls`). So this is mostly *filling in a slot the architecture already cut*, not
new architecture.

### Paper stage → project component

| Paper stage (§3.1) | Paper detail | Where it lands in our project |
|---|---|---|
| Occupancy image creation | Section `S` below ceiling (`ds`, `t`); project `S`→`Ms` (for masks) and full PCD→`Mf` (for retrieval); pixel size `l` | `preprocessing/notebook_1_occupancy_raster` already does slab-crop + raster; `sam_refine.build_sam_image()` already stacks occupancy/wallness/coverage → reuse (see §6) |
| 2D mask generation | `SamAutomaticMaskGenerator` with `points_`, `iou_`, `stability_`, `n_layers`, `down_factor`, `min_mask` | **NEW** `AutoMaskGenerator` adapter + `build_auto_mask_generator(cfg)` (the only GPU/model piece) |
| Point-cloud retrieval | group `Mf` pixels by mask color, pull points per group | Our methods stay in label-raster space; retrieval is done once, downstream, by the shared `notebook_2_wall_assignment` (back-projection). Pure-SAM only needs to emit `room_labels.npy` |
| Classification (room / not-room) | keep masks with cross-sectional area ≥ `A` (1.5 m²) | **NEW** `classify_rooms_by_area()` — deterministic, testable |
| Boundary buffer | expand room outward by `do` = ½ wall thickness, re-retrieve | **NEW** `buffer_room_labels()` — deterministic; `cfg.do_buffer_m` already exists |
| Corridor reprocessing | re-run SAM on residual (`rest`) with `points_=5` | **NEW** optional `reprocess_residual()` second pass — deterministic plumbing around one more model call |

What the paper additionally does (doors/windows, §3.2) is **out of scope** here — that maps to
the existing `WallImageProcessing` / wall-assignment stages and the user scoped this request to
room segmentation.

---

## 2. Design principle — split the model from the math

The single most important decision for "easily testable" is to **isolate the one
non-deterministic, GPU-bound step** (SAM's automatic mask generation) behind a thin adapter,
and make **everything else a pure function over plain arrays**. This mirrors how
`sam_refine.py` already isolates `relabel_by_sam` (model-free, unit-tested in
`test_sam_refine.py`) from `build_mask_generator` (the model).

```
                 ┌─────────────────────────────────────────────────────┐
   GPU / model   │  AutoMaskGenerator.generate(image) -> [ {segmentation,│
   (mockable)    │     area, predicted_iou, stability_score}, ... ]      │
                 └───────────────────────────┬─────────────────────────┘
                                             │  list of bool masks + scores
                 ┌───────────────────────────▼─────────────────────────┐
   CPU / pure    │  masks_to_room_labels(masks, walls, coverage, cfg)    │
   (unit-tested) │  classify_rooms_by_area(labels, cfg)                  │
                 │  buffer_room_labels(labels, walls, cfg)               │
                 │  (optional) reprocess_residual(...)                   │
                 └───────────────────────────┬─────────────────────────┘
                                             │  int32 room_labels (-1/0/>=1)
                 ┌───────────────────────────▼─────────────────────────┐
   orchestrator  │  segment_rooms_sam_auto(image, walls, coverage, cfg, │
                 │     generator=None) -> (labels, debug)               │
                 └─────────────────────────────────────────────────────┘
```

The orchestrator accepts an injected `generator`, so tests pass a **fake** generator that
returns hand-built masks — no torch, no checkpoint, no CUDA. Same trick the prompted-SAM tests
use, and the same safe pass-through behaviour when no backend is available.

---

## 3. New module: `scan2bim/sam_auto.py`

Proposed surface (signatures + responsibilities; bodies to be written in implementation phase):

### 3.1 Model abstraction (the only GPU piece)

```python
class AutoMaskGenerator:
    """Automatic 'segment everything' segmenter: generate(image) -> list of masks.
    Distinct from sam_refine.MaskGenerator, which is a *prompted* (set_image+predict)
    segmenter. This one takes NO prompts — it is the paper's SamAutomaticMaskGenerator."""
    name = 'sam-auto'
    def generate(self, image):                      # -> list[dict]  (segmentation, predicted_iou, ...)
        raise NotImplementedError

class _AutoAdapter(AutoMaskGenerator):
    """Wraps SAM1 SamAutomaticMaskGenerator / SAM2 SAM2AutomaticMaskGenerator, both of
    which expose .generate(rgb) returning the standard list-of-dict record."""

def _build_auto_sam1(cfg, device): ...              # SamAutomaticMaskGenerator(sam, points_per_side=...)
def _build_auto_sam2(cfg, device): ...              # SAM2AutomaticMaskGenerator(model, ...)
def _build_auto_sam3(cfg, device): ...              # temporary stub, same shape (mirror _build_sam3)

def build_auto_mask_generator(cfg) -> AutoMaskGenerator:
    """Factory keyed by cfg.sam_backend ('sam2'|'sam3'|'sam1'); maps the paper's SAM
    params (cfg.sam_points_per_side, sam_pred_iou_thresh, ...) onto the chosen backend's
    automatic-mask-generator kwargs. Raises if the backend can't be built."""
```

Reuse `sam_refine.build_sam_image()` for the input image — no new image builder needed if we
go with the shared-grid approach (§6).

### 3.2 Deterministic core (fully unit-testable, no model)

```python
def masks_to_room_labels(masks, scores, walls, coverage, cfg, transform=None):
    """SAM's mask set -> int32 room labels on the given grid.
      - drop masks below cfg.sam_min_mask_region_area (px) and below area A,
      - drop masks that mostly cover exterior / unscanned space (via coverage),
      - resolve overlaps deterministically (e.g. larger predicted_iou wins, ties by area),
      - snap to scaffold: re-impose -1 on every wall pixel, 0 on exterior,
        compact room ids to 1..k (reuse watershed._relabel_rooms).
    Returns (labels, debug)."""

def classify_rooms_by_area(labels, cfg):
    """Paper's room/not-room step: relabel-to-0 any region with cross-sectional area
    < cfg.sam_auto_min_room_area_m2 (the paper's A). Pure; returns new labels."""

def buffer_room_labels(labels, walls, cfg):
    """Paper's boundary buffer: dilate each room outward by cfg.do_buffer_m (~½ wall
    thickness) to reclaim wall-adjacent pixels, without crossing into another room.
    Pure morphological op on the label raster. (Note: in our pipeline the *3-D* wall
    recovery is done later by the shared boundary-ring wall-assignment; this buffer keeps
    the room rasters faithful to the paper and improves pq_eval coverage.)"""

def reprocess_residual(labels, image, walls, coverage, cfg, generator):
    """Optional paper 'corridor reprocessing': build the residual (unclaimed free space),
    re-run the generator on it with cfg.sam_residual_points_per_side, merge any new
    qualifying rooms into labels. Deterministic given the generator's masks."""
```

### 3.3 Orchestrator (mirrors `refine_with_sam`)

```python
def segment_rooms_sam_auto(image, walls, coverage, cfg, generator=None, transform=None):
    """Pure-SAM room segmentation. Returns (labels, debug).
      - if generator is None: try build_auto_mask_generator(cfg); on failure return a
        clear 'no SAM backend' debug and an all-exterior/-wall label map (no fabrication),
        OR raise — decide per §9.Q3.
      - generator.generate(image) -> masks/scores
      - masks_to_room_labels -> classify_rooms_by_area -> buffer_room_labels
      - if cfg.sam_reprocess_residual: reprocess_residual(...)
    Label convention identical to the watershed: -1 wall / 0 exterior / >=1 rooms."""
```

Export the public names from `scan2bim/__init__.py` (alongside the existing
`relabel_by_sam`, `refine_with_sam`, `build_sam_image`).

---

## 4. Config additions (`scan2bim/config.py`)

Add a `# NEW — pure-SAM auto-segmentation (Method 2)` block. Paper defaults in comments.

```python
# SAM automatic-mask-generator params (paper Table 1)
sam_points_per_side: int = 15           # paper 'points_': 11 (CS1) / 15 (CS2) / 30 (CS3)
sam_pred_iou_thresh: float = 0.85       # paper 'iou_'
sam_stability_score_thresh: float = 0.95  # paper 'stability_'
sam_crop_n_layers: int = 1              # paper 'n_layers'
sam_crop_n_points_downscale_factor: int = 2   # paper 'down_factor'
sam_min_mask_region_area: int = 100     # paper 'min_mask' (px)

# room/not-room + buffer (paper §3.1)
sam_auto_min_room_area_m2: float = 1.5  # paper 'A' (note: watershed uses min_room_area_m2=1.0)
# (boundary buffer reuses existing do_buffer_m = 0.1; paper used 0.07/0.10/0.12 per case)

# corridor reprocessing (paper §4.2)
sam_reprocess_residual: bool = False
sam_residual_points_per_side: int = 5   # paper points_=5 on the residual
```

Reuse existing fields: `sam_backend`, `sam_arch`, `sam_ckpt`, `sam_model_cfg`, `do_buffer_m`,
`pixel_m`, `coverage_*`. No breaking changes — these are all additive with defaults, so the
saved `config.json` snapshots and `assert_upstream_config` keep working.

---

## 5. Notebook drivers (thin)

Both files already exist as stubs; fill them as thin drivers over `scan2bim.sam_auto`.

**`notebooks/methods/SAM/notebook_1_sam_auto_segmentation.ipynb` → `stage_sam_auto`** (GPU/Colab):
1. `CFG = scan2bim.load_config()`
2. load Stage-1 rasters (`A.STAGE1`): `wallness`, `coverage`, `occupancy.png`, `transform`;
   `assert_upstream_config`.
3. `image = build_sam_image(occ, wallness, coverage, mode=CFG.sam_image_mode)`
4. `labels, dbg = segment_rooms_sam_auto(image, walls=wallness, coverage=coverage, cfg=CFG)`
   *(walls source = wallness, consistent with the boundary-ring assignment's `wall_source`)*
5. write `room_labels.npy`, `room_labels_color.png`, `transform.json`, `config.json` to
   `stage_sam_auto`; `package_stage`.

**`notebooks/methods/SAM/notebook_2_wall_assignment.ipynb` → `stage_sam_walls`** (CPU):
- Identical to `geometric/notebook_2_wall_assignment`, only the mask source differs
  (`room_labels.npy` from `stage_sam_auto`). The boundary-ring assignment is already shared in
  `scan2bim.walls`; this notebook just points at the SAM stage dir. (Add stage constants
  `STAGE_SAM_AUTO`, `STAGE_SAM_WALLS` to `artifacts.py`.)

This keeps the run order and "edit only `params.yaml`" contract intact.

---

## 6. The SAM input image — the one decision to make

Two faithful-to-different-things options (your second question). Both are viable; they differ
in *which grid* SAM and the labels live on.

**Option A — reuse the Stage-1 rasters (recommended).**
Feed SAM the occupancy/wallness/coverage images the preprocessing notebook already built, via
`build_sam_image()`, on the single shared grid (`pixel_m=0.03`, one `transform.json`).
- *Pros:* all three methods share one grid → every `room_labels.npy` is pixel-aligned →
  `evaluation/gt_raster` + `pq_eval` score them apples-to-apples with **zero** resampling.
  Nothing new to build; maximal code reuse; the comparison is clean and honest.
- *Cons:* 0.03 m, not the paper's 0.01 m; uses our slab definition rather than the paper's
  exact `ds`/`t`. SAM sees a coarser image, which can miss the thinnest doorways.

**Option B — replicate the paper's `Ms`/`Mf` at 0.01 m.**
Build a dedicated section image `Ms` (below ceiling by `ds`, thickness `t`) for mask
generation, and a full-projection `Mf` for retrieval, both at `l = 0.01 m`.
- *Pros:* faithful to the paper; reproduces their setup; finer grid helps SAM resolve thin
  openings and small rooms.
- *Cons:* it's a **second, finer grid** distinct from Stage-1. To compare against
  geometric/geometric+SAM in `pq_eval`, you must resample the pure-SAM labels back onto the
  Stage-1 grid (or rebuild GT at 0.01 m too) — extra plumbing, and the three methods are no
  longer run on identical inputs, which muddies the comparison.

**Recommendation:** start with **Option A** for a clean three-way comparison, and add a
`cfg.sam_auto_image = 'stage1' | 'paper'` switch later if you want to A/B the paper-faithful
0.01 m image. A cheap middle ground: keep the shared grid but lower `pixel_m` for *all* methods
if 0.03 m proves too coarse for SAM (preserves alignment, sharpens the image).

---

## 7. Testability plan — `tests/test_sam_auto.py`

Mirror `test_sam_refine.py` exactly: tiny hand-built arrays, a **fake generator**, no model,
no randomness.

```python
class FakeAutoGen(AutoMaskGenerator):
    def __init__(self, masks, scores): ...
    def generate(self, image): return [dict(segmentation=m, predicted_iou=s, ...)
                                       for m, s in zip(self.masks, self.scores)]
```

Cases to cover (deterministic core):
1. **two clean masks → two rooms**, correctly labelled, walls stay `-1`, no room pixel on a
   wall, ids compacted to 1..k.
2. **overlapping masks resolved deterministically** (higher predicted_iou / larger area wins;
   result independent of input order).
3. **small mask dropped by area** (`classify_rooms_by_area`, below `A`) → becomes exterior.
4. **mask over unscanned void dropped** via `coverage`.
5. **buffer reclaims wall-adjacent pixels** without bleeding into a neighbour
   (`buffer_room_labels`).
6. **residual reprocessing** adds a corridor room the first pass missed (feed a fake generator
   whose second `generate` returns the corridor).
7. **orchestrator pass-through / clear error** when no backend is available (no fabricated
   masks) — mirrors `test_pass_through_when_no_backend_builds`.
8. **same grid invariant:** output `labels.shape == walls.shape` and label convention holds.

Also extend `tests/test_notebooks.py` so the SAM notebooks are no longer skipped-as-stub (they
should import and run their non-GPU cells, with the model call guarded/mocked or skipped when
no CUDA), and confirm `stage_sam_auto` emits a `room_labels.npy` the evaluation can read.

Run locally with `pytest -q` — the whole `sam_auto` core runs CPU-only in milliseconds.

---

## 8. The three-way comparison (already mostly free)

Because every method emits `room_labels.npy` on the **same Stage-1 grid**, the comparison
harness already exists:

```
preprocessing/notebook_1_occupancy_raster        -> stage1_occupancy        (shared)
methods/geometric/      n1_watershed -> n2        -> stage2 / stage3_walls
methods/geometric_SAM/  n1 -> n2(GPU) -> n3       -> stage4 / stage5_walls_sam_refined
methods/SAM/            n1_sam_auto(GPU) -> n2     -> stage_sam_auto / stage_sam_walls   [NEW]
                                  │
converters/s3dis_loader  ──┐                       (one GT, scores all three)
evaluation/gt_raster   ────┴-> stage_gt/gt_room_labels.npy
evaluation/pq_eval     ───────> pq_results.json   (point pq_eval at each method's labels)
```

Confirm `pq_eval` can take the stage dir / labels path as a parameter (or loops over the three
method stage dirs). If it currently hard-codes the watershed stage, generalise it to score a
list of `(method_name, room_labels_path)` and emit per-method PQ/SQ/RQ into `pq_results.json` —
that, plus the paper's mean-IoU (Eq. 6–7) as an optional second metric, gives you the
comparison table for the write-up.

---

## 9. Open decisions to confirm before implementing

- **Q1 — SAM image grid:** Option A (shared Stage-1 grid, recommended) vs Option B (paper's
  0.01 m `Ms`/`Mf`) vs add the switch. (§6)
- **Q2 — `A` threshold:** keep the paper's 1.5 m² for pure-SAM (`sam_auto_min_room_area_m2`)
  while the watershed keeps 1.0 m²? Using the paper value is more faithful; using a shared
  value makes the methods more comparable. Suggest: keep them separate, default pure-SAM to 1.5.
- **Q3 — no-backend behaviour:** when no SAM is available, should `segment_rooms_sam_auto`
  *raise* (pure-SAM is meaningless without SAM) or *pass through* like the refinement stage?
  Suggest **raise with a clear message** in the notebook, but keep the core function returning a
  debug flag so tests stay model-free.
- **Q4 — `pq_eval` generalisation:** is it already multi-method, or does it need the small
  refactor in §8?
- **Q5 — overlap resolution rule:** confirm the deterministic tie-break
  (predicted_iou, then area, then lowest id) — it must be stable for test #2.

---

## 10. Suggested implementation order (for the next session)

1. `config.py` — add the `# NEW pure-SAM` fields (§4). *(safe, additive)*
2. `artifacts.py` — add `STAGE_SAM_AUTO`, `STAGE_SAM_WALLS` constants.
3. `scan2bim/sam_auto.py` — deterministic core first (`masks_to_room_labels`,
   `classify_rooms_by_area`, `buffer_room_labels`), then the orchestrator, then the
   `AutoMaskGenerator` adapters. Export from `__init__.py`.
4. `tests/test_sam_auto.py` — write alongside the core (TDD); all CPU.
5. `notebooks/methods/SAM/notebook_1_sam_auto_segmentation.ipynb` — fill the thin driver.
6. `notebooks/methods/SAM/notebook_2_wall_assignment.ipynb` — clone the geometric driver,
   point at `stage_sam_auto`.
7. `evaluation/pq_eval.ipynb` — generalise to score all three method stages (§8).
8. Update `README.md` / `scan2bim/ARCHITECTURE.md` to drop the "[STUBS]" markers for the SAM
   method.

Each step is independently runnable and testable; steps 1–4 need no GPU at all.
``

---

## 11. Next steps (action plan, with rationale)

Ordered by dependency. Reasons are framed around defensibility, since the comparison has to
hold up in the paper.

1. **Lock the input representation first.** Standardise on one shared raster — a clean
   horizontal section (walls-as-occupied, sliced above furniture / below beams), at **0.03 m**,
   consumed by all three methods — and run a **0.01 / 0.02 / 0.03 m pixel-size sweep** to produce
   one figure.
   *Why:* everything downstream consumes this raster, so deciding it now avoids regenerating all
   results later; and the sweep turns "why 0.03?" into a density-matched, defended choice (the
   cloud is voxel-downsampled at 0.02 m, so rasterising finer than that fragments walls) before a
   reviewer can raise it.

2. **Implement the pure-SAM method** (`scan2bim.sam_auto` + fill the two `methods/SAM/`
   notebooks) per §§3–5.
   *Why:* it's the missing third arm — the comparison currently can't distinguish three methods
   because this one never ran, so without it there's no paper.

3. **Get a real SAM 2.1 checkpoint and run geometric+SAM for real** (GPU/Colab), not in
   pass-through.
   *Why:* the hybrid's labels are currently byte-identical to the watershed; its entire claim is
   unfalsifiable until SAM actually executes.

4. **Generalise `pq_eval`** to score each method's `room_labels.npy` separately, and add the
   paper's mean-IoU (Eq. 6–7) alongside PQ/SQ/RQ.
   *Why:* the current `pq_results.json` scores the same labels three times; you need genuine
   per-method numbers, and a metric comparable to the published method strengthens the writeup.

5. **Run the full three-way comparison** on S3DIS Area-3 ground truth and build the results table.
   *Why:* this is the actual evidence — only now can you state which method wins, rather than
   asserting it from theory.

6. **Add `tests/test_sam_auto.py`** (fake generator, hand-built masks) and re-run the suite.
   *Why:* deterministic, CPU-only tests make the new method reproducible and catch regressions —
   research-code credibility, and it lets you claim the model-free core is verified.

7. **Write the methods + results sections with the defensibility framing:** shared grid isolates
   method from input, density-matched pixel size, hybrid as the contribution with geometric /
   pure-SAM as ablations — and report whatever the data actually shows.
   *Why:* this pre-empts the two most likely reviewer attacks (confounded comparison, arbitrary
   resolution) and keeps you honest — if the hybrid doesn't beat geometric on your data, "SAM
   refinement adds cost without gain here" is still a defensible, publishable finding.

**The rule threaded through all of it:** let the *design argument* decide which method you feature
(the hybrid), but let the *experiment* decide the verdict you report. Don't write the conclusion
before step 5.
