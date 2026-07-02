# ONESTRUCTION — Research caveats & limitations

> Honest ledger of what is *not* yet rigorous in the method, the metric, and the comparison to
> Albadri et al. (ISPRS XLVIII-G-2025-131). Written to seed the paper's **Limitations** section and
> to keep us from over-claiming. Verified against the code on 2026-06-29 (not just the older
> research-fixes notes). Severity = impact on a defensible research claim.

## Summary

| # | Caveat | Area | Severity | Status |
|---|--------|------|----------|--------|
| 1 | Different datasets than the paper — no head-to-head parity | Comparison | High | Inherent |
| 2 | Room IoU sits on a depressed ceiling (0.03 m raster prediction) | Metric | High | Open |
| 3 | `n_seg` / over-seg under-counts spurious predicted rooms | Metric | Med | Open (cosmetic) |
| 4 | GT-point universe is lenient on prediction over-extension | Metric | Low | Documented |
| 5 | Our SAM lacks the paper's corridor-reprocessing pass | Method (SAM) | High | Open |
| 6 | Methods use **asymmetric** room filters → measures post-filter | Fairness | High | Open (Task 05) |
| 7 | `rasterize_wallness` over-flags → corrupt walls & SAM channel | Method (walls) | High | Open (Task 03) |
| 8 | Geometry+SAM is edit-only, rarely fires ≈ Geometry; not evaluated | Method (hybrid) | High | Open (Task 06) |
| 9 | n = 1 area, single SAM run, no variance | Statistics | High | Open (Task 07) |
| 10 | End-to-end point-cloud → IFC4 not demonstrable in *this* pipeline | Downstream | Med | Separate engineer |
| 11 | Wall-accuracy metric (the stated headline) not built yet | Metric | High | Open (Task 04) |
| 12 | Working tree / packaging inconsistencies hurt reproducibility | Repro | Med | Partly fixed |

---

## 1. Comparison to the paper is protocol-level, not parity
The paper evaluates on **3 private case studies** (CS1 17, CS2 9, CS3 41 rooms); we use **S3DIS**
(public). We can legitimately say *"evaluated under Albadri et al.'s protocol"* but **not** claim
number-for-number parity. Different buildings, scan quality, clutter, and room mix. The fairest
single pairing is their **CS3 (41 rooms) ↔ S3DIS Area_1 (44 rooms)**; even that is a ballpark.
**Claim to make:** same metric, different data → compare *ranges*, not points.

## 2. Room IoU sits on a depressed ceiling (raster prediction at 0.03 m)
The paper assigns **points** to rooms; our prediction is a **cell label map at 0.03 m**, and each
GT point inherits its cell's room id. A *perfect* segmentation therefore scores **0.928, not 1.0**
on Area_1 (measured) — adjacent rooms share boundary cells. The paper effectively tops out at 1.0
(actual point assignment, 0.01 m grid). So our IoU is **systematically ~3–7% lower for equal
quality**, and "our 0.85 vs their 0.79" is not clean. Mitigation: re-raster the comparison run at
**0.01 m** (≈9× memory) to shrink the gap; it won't vanish. Implemented in
[`score_rooms_paper`](scan2bim/eval.py) (research-fixes [Task 02b](Claude%20Prompts%20Markdowns/research-fixes/02b_room_metric_paper_faithful.md)).

## 3. `n_seg` / spurious-room counting
`score_rooms_paper`'s `n_seg` counts predicted rooms **that contain ≥1 GT interior point**, so a
spurious predicted room over exterior/clutter (a false positive) is invisible to the `#seg` column
— flattering us vs the paper's raw "# segmented rooms". **Does not affect** `detection_rate`,
`mean_iou_matched`, `over_seg`, or `under_seg` (a non-overlapping pred can't be "corresponding").
Cosmetic for the table only; the paper also applies a room/not-room area filter we don't, so a raw
count would over-report instead. Decide per the writeup; not a headline-number bug.

## 4. GT-point universe is lenient on over-extension
Our IoU universe is the GT interior points, so a prediction that spills into areas with **no GT
points** is not penalized (no points there to count). The paper's `Pᵢ` is the prediction's own
retrieved cloud, which *does* pay for over-extension. Minor on S3DIS (dense interior coverage), but
a directional leniency in our favor. Documented deviation in
[Task 02b](Claude%20Prompts%20Markdowns/research-fixes/02b_room_metric_paper_faithful.md).

## 5. Our SAM lacks the paper's corridor-reprocessing pass
The paper runs a **second SAM pass** (`points_=5`) on residual points to recover corridors that
SAM misses on the first pass (their CS2/CS3 fix). Our SAM has no such pass, so it **under-detects
corridors** → lower `detection_rate` — a *method* gap, not a metric gap. "Our SAM vs their SAM" is
unfair until this is added. (Consistent with the known corridor weakness in the brief.)

## 6. The three methods are not filtered identically (fairness)
Each method drops rooms by **different** rules before any metric, so the head-to-head partly
measures the *post-filter*, not the *segmenter*:
- min room area: watershed `min_room_area_m2 = 1.0` ([config.py:57](scan2bim/config.py#L57)) vs
  pure-SAM `sam_auto_min_room_area_m2 = 1.5` ([config.py:157](scan2bim/config.py#L157)).
- void/coverage gate: watershed `min_coverage_frac = 0.25` ([config.py:64](scan2bim/config.py#L64))
  vs SAM `sam_auto_min_coverage_frac = 0.5` ([config.py:159](scan2bim/config.py#L159)).
- different `-1` wall scaffolds per method.
No shared `harmonize_room_labels` exists yet. Until then, comparisons should be read with this
asymmetry in mind. (research-fixes [Task 05](Claude%20Prompts%20Markdowns/research-fixes/05_fair_comparison_harmonize.md).)

## 7. `rasterize_wallness` over-flags (broken wall source)
[`rasterize_wallness`](scan2bim/raster.py#L48) flags a column as wall if its point **span**
(`zmax − zmin`) ≥ 50% of room height — but any column with both a floor and a ceiling return spans
the full height, so **nearly every scanned cell reads as wall**. ARCHITECTURE.md records the same.
Consequence: every 3-D **wall export** (`boundary-ring ∩ wallness`) and the Geometry+SAM **SAM
"structure" channel** are corrupted. Room *segmentation* is protected (watershed uses the binary
slab mask; pure-SAM overrides to occupancy), so room IoU survives — but walls/IFC and the hybrid's
SAM input do not. **Hard blocker for any wall-accuracy claim.** (research-fixes
[Task 03](Claude%20Prompts%20Markdowns/research-fixes/03_fix_wallness_raster.md).)

## 8. Geometry+SAM is structurally capped and unevaluated
[`relabel_by_sam`](scan2bim/sam_refine.py) only **merges/splits existing** watershed basins — it
cannot add a room the watershed never seeded — and its gate is conservative (`sam_conf_thresh =
0.88`, [config.py:122](scan2bim/config.py#L122)), so it "rarely overrides the watershed". Net:
Geometry+SAM ≈ Geometry, collapsing the three-way story to two. It was also **skipped in the last
eval** (stale grid), so we have **no real number** for it. Don't present it as a distinct result
until it can recover residual rooms and is actually scored. (research-fixes
[Task 06](Claude%20Prompts%20Markdowns/research-fixes/06_strengthen_combined_method.md).)

## 9. n = 1 area, single run, no variance
All reported numbers are **one S3DIS area, one run**. Both SAM arms are **nondeterministic** (GPU,
run-to-run mask variation), so a single SAM number is one draw from a distribution with no error
bar. A defensible finding needs ≥3 areas with **mean ± std** and repeated SAM runs; the watershed
is deterministic (std = 0). (research-fixes [Task 07](Claude%20Prompts%20Markdowns/research-fixes/07_scale_multiarea_evaluation.md).)

## 10. End-to-end point-cloud → IFC4 is not demonstrable in *this* pipeline
Door/window detection and IFC4 export are owned by a **separate engineer** on their own pipeline
(their code lives in `src/scan2bim/`: `wall_image_processing`, `ifc_export`). They are **not wired
into this notebook pipeline**, so the full "point cloud → IFC4" claim is **not currently
demonstrable here**, and there is **no IFC/BIM ground truth** to score against. Our lane stops at
room + wall primitives, which their pipeline can consume. (Not a task we own — former research-fixes
Tasks 08/09 were removed.)

## 11. The stated headline metric (wall accuracy) does not exist yet
The research framing names **per-room wall accuracy** as the primary contribution (P/R/F1 +
position/orientation/length/thickness/height vs S3DIS `wall_N.txt`). That metric is **not built**
(research-fixes [Task 04](Claude%20Prompts%20Markdowns/research-fixes/04_wall_accuracy_evaluation.md)), and it is blocked by the
wallness bug (#7). Today we only have **room IoU + PQ**, which the framing calls *secondary*. The
current deliverable is therefore a room-segmentation comparison, not the wall/BIM result.

## 12. Reproducibility & packaging
- **Mixed-scene working tree (now):** `params.yaml` points at `data/TUB1_Cropped_June16.ply`, and
  `stage1`/`Geometry` were regenerated on it `(651×522)`, while `gt_room_labels`/`SAM` are leftover
  S3DIS Area_1 `(1606×1618)`. Nothing shares a grid → `pq_eval` can't run end-to-end until one
  scene is run cleanly through `NB1 → method(s) → gt_raster → pq_eval`.
- **Dual package:** the `main` merge left **both** root `scan2bim/` (active, imported everywhere)
  and `src/scan2bim/` (a parallel refactor). Unreconciled; risk of import ambiguity.
- **`pyproject.toml`** was merged into invalid TOML (duplicate keys) — **patched** to the root
  layout this session, but the `src`-vs-root decision is still owed.
- **Doc drift:** several docs say "Area 3" while the current GT/runs are **Area_1** (44 rooms;
  Area_3 has ~23). Pin one scene as canonical.

---

## What is already addressed
- **Frame integrity** — the GT↔scan 98%-in-grid hard-fail gate exists (research-fixes Task 01).
- **Clean room GT + paper IoU** — interior-only GT, no method aliasing, point-based paper-protocol
  scorer matched at 75% over matched rooms (Tasks 02 + 02b), unit-tested.

## Suggested order to retire these
Wallness (#7) → wall metric (#11) → fairness filter (#6) → strengthen hybrid (#8) → multi-area +
variance (#9). Metric-fidelity items (#2–#4) are mostly disclosure + an optional 0.01 m run.
Items #1 and #5 shape *how we phrase* the comparison and *what method we run*, respectively.
