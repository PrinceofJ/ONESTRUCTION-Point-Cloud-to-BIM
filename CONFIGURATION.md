# Configuration — how it works

*One file to edit, one loader to read it, one snapshot to prove what ran.*
(Introduced by research-fixes Task 12; audit trail in
[`Claude Prompts Markdowns/research-fixes/12_config_audit.md`](Claude%20Prompts%20Markdowns/research-fixes/12_config_audit.md).)

## The one-paragraph version

Every parameter in the pipeline is a field of the `Config` dataclass
([`scan2bim/config.py`](scan2bim/config.py)). You change values in **[`params.yaml`](params.yaml)
only** — it lists every field, grouped and commented. Per-method differences (e.g. pure-SAM's
`sam_image_mode='occupancy'`) are declared in the `methods:` block at the bottom of params.yaml;
notebooks select them with `scan2bim.load_config(method='sam_auto')`. Notebooks never assign
science parameters onto `CFG` — tests fail if one does. Each stage saves the exact config it ran
with into its output ZIP, and downstream stages hard-fail if the grid doesn't match.

## The four layers (precedence: later wins)

```
Config defaults        params.yaml            methods.<name>          runtime kwargs
(config.py)      <     global sections   <    block             <     load_config(..., k=v)
schema + docs          THE editable surface   per-method deltas       env paths ONLY
```

1. **`Config` dataclass** — the schema: every field, its type, default, and the docstring
   explaining it. You *read* this file; you don't edit it to change a run.
2. **`params.yaml` global sections** — the values. Section names (`raster:`, `watershed:`, …)
   are cosmetic; keys are flat `Config` field names. Every field is listed (except the legacy
   unused `out_dir`).
3. **`params.yaml → methods:`** — declared per-method overrides. Currently:
   - `geometric: {}` — pure watershed, no deltas
   - `sam_auto:` → `sam_image_mode: occupancy` (paper-faithful Ms-only SAM input; the global
     `stack` mode would make SAM grab the whole building)
   - `sam_refine:` → `sam_image_mode: stack`, `use_sam_recall: true` (explicit ==-default
     declarations, for clarity)
4. **Runtime kwargs** — `load_config(..., out_root=OUT_ROOT, sam_ckpt=SAM_CKPT)`. Reserved for
   environment facts (where Colab put the checkpoint, where outputs land), never science values.

## How a notebook gets its config

```python
CFG = scan2bim.load_config(start=PROJECT_DIR, method='sam_auto',
                           out_root=OUT_ROOT, sam_ckpt=SAM_CKPT, sam_model_cfg=SAM_CFG)
print(scan2bim.config_snapshot(CFG))    # logs every field that deviates from the defaults
```

- `method=` picks the `methods:` block; omit it for global-only stages (preprocessing, eval).
- `config_snapshot(CFG)` prints the **resolved snapshot** — the selected method plus every
  field whose value differs from the dataclass default. Every driver notebook prints it, so
  each saved run records exactly which knobs were in effect.
- The **only** sanctioned post-load mutation is `CFG.file_path = <runtime copy>` (Colab copies
  the cloud off Drive to fast local disk; the `CLOUD_OVERRIDE` switch in SAM NB1). Enforced by
  `tests/test_config.py::test_notebooks_never_mutate_science_config` (allowlist: `file_path`).

## Guarantees the loader gives you

| Guarantee | Mechanism |
|---|---|
| A typo'd key can never silently do nothing | unknown keys (global **or** inside a method block) raise `KeyError` at load |
| A method name typo can never fall back to global | unknown `method=` raises `KeyError` listing the declared methods |
| A Windows-authored path can never break Colab | the loader normalises `\` → `/` before resolving (`data\Area_1` == `data/Area_1`) |
| Method overrides can never leak into other stages | the `methods:` block is popped before the flat key collect; pinned by a regression test |
| A stage can never silently run on the wrong grid | every stage saves `config.json` into its ZIP; `assert_upstream_config` hard-fails downstream on any geometry-field mismatch (`file_path`, `pixel_m`, slab, voxel, `up_axis`, units) |
| Colab and local can never disagree invisibly | Colab clones the **pushed** params.yaml — commit + push before a Colab run, or use the notebook's explicit `CLOUD_OVERRIDE` runtime switch |

## Things that look like duplication but aren't

These stay separate **on purpose** (paper-faithfulness; see Tasks 05/12 — a test guards them):

| concept | watershed | SAM-auto | harmonized eval |
|---|---|---|---|
| min room area (m²) | `min_room_area_m2` = 1.0 | `sam_auto_min_room_area_m2` = 1.5 (paper's *A*) | `eval_min_room_area_m2` = 1.0 |
| void/coverage frac | `min_coverage_frac` = 0.25 | `sam_auto_min_coverage_frac` = 0.5 | `eval_min_coverage_frac` = 0.25 |

Each method keeps its own faithful standalone value; the `eval_*` value is the ONE shared
filter `pq_eval` applies to all methods in the `comparison` profile.

**Two different SAM models** also coexist: `sam_ckpt` (SAM 2.1 hiera-large — room
segmentation + prompted refinement) and `wproc_sam_checkpoint` (SAM 1 vit_b — door/window
detection on flattened wall images in postprocessing). Don't conflate them.

## Common tasks

- **Change a parameter** → edit it in params.yaml. If a stage upstream of your target consumed
  it (e.g. anything in `raster:`/`slab:`), re-run that stage first — `assert_upstream_config`
  will remind you.
- **Change a value for ONE method only** → add the field under that method's `methods:` block.
- **Add a brand-new parameter** → add the field to the `Config` dataclass (with default + doc
  comment), then list it in params.yaml in the matching section.
- **Add a new method/profile** → add a `methods.<name>:` block; have its notebook call
  `load_config(method='<name>')`.
- **Point at a different scan/GT** → set `input.file_path` **and** `groundtruth.gt_dir`
  together (same scene, same frame), then re-run preprocessing NB1; the frame gate in
  `gt_raster.ipynb` and the grid checks will catch mismatches.
- **See what a past run used** → open `config.json` inside the stage ZIP, or read the
  `config_snapshot` printout in the notebook.
- **Gotcha (structural-only clouds)** → no interior points ⇒ every room fails void rejection;
  set `min_coverage_frac: 0.0`.

## What's enforced by tests (`tests/test_config.py`)

Loader semantics (unknown keys/methods raise, precedence order, no method-block leak, path
portability, snapshot provenance) · the real params.yaml declares the three methods with the
right `sam_image_mode` · the paper-faithful per-method thresholds stay distinct · **no notebook
in the tree assigns a science field on CFG** · the three per-method notebooks pass their
`method=` name to `load_config`.
