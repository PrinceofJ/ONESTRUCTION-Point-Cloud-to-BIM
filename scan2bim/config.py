"""Single source of truth for all pipeline configuration.

This is a *superset* of the ``CFG`` dataclass in the original monolithic notebook.
Every field that existed before keeps its original name **and original default**, so
the rasterisation, slab extraction and watershed segmentation behave identically to
the original notebook. New fields are grouped and clearly marked ``# NEW``.

All distance thresholds are in **metres** (the cloud is converted to metres once on
load). Pixel-space equivalents are exposed as ``*_px`` properties so nothing downstream
hard-codes a pixel count.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- input ----
    # The pipeline is dataset-agnostic: point `file_path` at the real-life scan to segment and
    # `gt_dir` at the matching ground-truth model. BOTH are config inputs (no hardcoded paths),
    # both relative to the project root and resolved to absolute by load_config(). They must
    # share the same world XY frame + units; gt_raster.ipynb asserts this before scoring (a
    # mismatched pair — e.g. the area1.xyz scan with the wrong area's GT — back-projects to a
    # low in-grid fraction and hard-fails). area1.xyz pairs with data/Area_1 (verified ~100%).
    file_path: str = 'data/area1.xyz'      # the real-life scan to segment (must be a FULL cloud — interior points present)
    gt_dir: str = 'data/Area_1'            # ground-truth model: a dir of per-room point files <room>/<room>.txt
    units_per_meter: float = 1.0
    up_axis: int = 2                       # 0=X 1=Y 2=Z

    # ---- cleaning / sampling ----
    voxel_m: float = 0.02

    # ---- slab (horizontal crop) ----
    slab_relative_to: str = 'ceiling'      # 'ceiling' | 'floor' | 'absolute'
    slab_lo_m: float = 0.125                 # shallower band edge (closer to the reference)
    slab_hi_m: float = 0.5                   # deeper band edge
    ceiling_mode: str = 'local_perpoint'   # 'global' | 'local_smoothed' | 'local_perpoint'
    ceiling_cell_size_m: float = 0.1       # local-ceiling grid cell (smaller = finer lips)
    ceiling_min_pts_per_cell: int = 5      # sparse cells fall back to the global ceiling
    ceiling_smooth_cells: int = 3          # median window, 'local_smoothed' only

    # ---- rasterisation ----
    pixel_m: float = 0.03                  # the memory dial (m / px)
    min_points_per_cell: int = 3
    thicken_px: int = 1

    # ---- room seg: PASS 1 (deterministic watershed) ----
    min_wall_area_px: int = 60
    marker_h_m: float = 0.30               # h-maxima seed depth (replaces max_seal_m)
    footprint_close_m: float = 1.00        # explicit-exterior closing distance
    merge_ridge_m: float = 0.70            # merge basins joined wider than this
    min_room_area_m2: float = 1.0          # the paper's A threshold

    # ---- void rejection (a room must sit over scanned data) ----
    drop_empty_rooms: bool = True
    coverage_ceiling_margin_m: float = 0.30
    coverage_min_pts: int = 1
    coverage_close_px: int = 2
    min_coverage_frac: float = 0.25

    # ---- room seg: PASS 2 (prompted SAM refinement; Notebook 4) ----
    # SAM does NOT segment the image blindly. It is PROMPTED per watershed room (points
    # from the room's eroded interior + the room box), so every returned mask is labelled
    # by construction. A single-pass region-adjacency-graph relabel then lets a confident
    # SAM mask merge/split rooms — but only where the geometry is weak (an open
    # distance-transform ridge, never on wall pixels). The watershed is always the prior.
    use_sam_recall: bool = True            # master switch for the SAM refinement stage
    sam_arch: str = 'vit_h'                # SAM1 architecture key (sam_model_registry)
    sam_ckpt: str = '/content/sam2.1_hiera_large.pt'           # SAM checkpoint path
    sam_model_cfg: str = 'configs/sam2.1/sam2.1_hiera_l.yaml'  # SAM2/SAM3 Hydra config name

    # ---- 3-D wall export ----
    do_buffer_m: float = 0.1              # boundary buffer (~half wall thickness)
    wall_floor_margin_m: float = 0.10      # drop points within this of the floor
    wall_ceiling_margin_m: float = 0.10    # drop points within this of the ceiling

    # ---- per-room wall fit (RANSAC) ----
    ransac_dist_thresh_m: float = 0.02
    ransac_normal_tol_deg: float = 10.0
    ransac_min_inliers: int = 400
    ransac_max_planes: int = 20

    # =====================================================================
    # NEW — boundary-ring room-to-wall assignment (Notebook 2)
    # =====================================================================
    # The new wall-assignment derives each room's walls from a *boundary ring*
    # (room mask minus its eroded interior), dilated outward to reach the wall.
    #   room_erode_m  -> erosion radius used to obtain the reliable interior I_i.
    #                    Also equals the thickness of the boundary ring B_i = M_i \ I_i.
    #   wall_dilate_m -> r_w, how far the ring is dilated outward to capture wall
    #                    pixels from the slab wall mask.
    # Set either to ``None`` to auto-derive it from the estimated wall thickness
    # (the same distance-transform median heuristic the original used in
    # ``room_footprints``); see walls.estimate_wall_thickness_px / resolve_ring_radii_px.
    room_erode_m: float | None = 0.125      # NEW: erosion radius -> reliable interior (ring thickness)
    wall_dilate_m: float | None = 0.15     # NEW: r_w -> outward reach to grab wall pixels

    # =====================================================================
    # NEW — SAM backend abstraction (Notebook 4)
    # =====================================================================
    # The refinement runs on SAM2. Inference is wrapped behind a MaskGenerator adapter so
    # a different model is reached by changing this one string (and supplying the matching
    # checkpoint + config) with no change to the refinement code — e.g. 'sam3' once that
    # build is available, or 'sam1' for the original Segment-Anything predictor.
    sam_backend: str = 'sam2'              # 'sam2' (default) | 'sam3' | 'sam1'

    # ---- prompting (per watershed room) ----
    sam_image_mode: str = 'stack'          # SAM input image: 'stack' = occupancy + slab wall
                                           #   mask + coverage as 3 channels; 'occupancy' =
                                           #   replicate the binary occupancy raster into 3 channels.
    sam_pos_points: int = 8                # positive points sampled from the room's eroded interior
    sam_use_neg_points: bool = True        # also feed negative points from neighbouring rooms
    sam_neg_points: int = 4                # how many negative points (per room) when enabled

    # ---- confidence x geometry gating ----
    sam_conf_thresh: float = 0.88          # SAM is "confident" if predicted-IoU >= this
    sam_wall_frac_max: float = 0.20        # an edge is geometry-WEAK only if < this fraction of
                                           #   its interface is wall pixels (else it is wall-backed)
    sam_open_ridge_m: float = 0.40         # ...and the free interface has an open ridge >= this
                                           #   wide (a real opening, not a hairline gap)
    sam_merge_cover_frac: float = 0.60     # a SAM mask "spans" an edge if it covers >= this
                                           #   fraction of BOTH rooms' interiors -> merge vote
    sam_split_min_frac: float = 0.25       # a SAM split is considered only if each resulting
                                           #   piece is >= this fraction of the room
    sam_min_sanity_margin: float = -0.02   # accept a merge/split only if the cheap sanity score
                                           #   does not drop by more than this (safety rail)

    sam_refine_qa_cloud: bool = False      # load the cloud in N4 only for optional QA

    # =====================================================================
    # NEW — pure-SAM automatic room segmentation (Method 2; paper Table 1 / §3.1)
    # =====================================================================
    # The pure-SAM method runs SAM in AUTOMATIC "segment everything" mode (NO watershed
    # prior, NO prompts) on the SAME Stage-1 rasters the other two methods consume, so
    # every method emits room_labels.npy on one shared grid -> a clean three-way
    # comparison with zero resampling. This is distinct from the PROMPTED refinement above
    # (use_sam_recall / sam_conf_thresh / ...). The SAM backend + checkpoint fields
    # (sam_backend, sam_arch, sam_ckpt, sam_model_cfg) are SHARED with the refinement.
    #
    # SAM automatic-mask-generator parameters (paper Table 1). points_per_side is the one
    # the paper tuned per case study (11/15/30); the rest were held fixed across CS1-3.
    sam_points_per_side: int = 15               # paper 'points_': 11 (CS1) / 15 (CS2) / 30 (CS3)
    sam_pred_iou_thresh: float = 0.85           # paper 'iou_'
    sam_stability_score_thresh: float = 0.95    # paper 'stability_'
    sam_crop_n_layers: int = 1                  # paper 'n_layers'
    sam_crop_n_points_downscale_factor: int = 2  # paper 'down_factor'
    sam_min_mask_region_area: int = 100         # paper 'min_mask' (px) — raw-mask noise floor

    # room / not-room classification (paper §3.1; the 'A' threshold). Kept SEPARATE from the
    # watershed's min_room_area_m2 (1.0) so each method uses its own faithful value.
    sam_auto_min_room_area_m2: float = 1.5      # paper 'A' = 1.5 m^2
    # drop a mask whose pixels sit mostly OFF scanned coverage (exterior / unscanned void):
    sam_auto_min_coverage_frac: float = 0.5     # keep a mask only if >= this frac is on coverage

    # raster boundary buffer (paper 'do' = half wall thickness). OFF by default: our
    # downstream boundary-ring wall-assignment already recovers wall points in 3-D, and
    # keeping walls as hard -1 barriers matches the other two methods' label convention
    # exactly (apples-to-apples pq_eval). Reuses do_buffer_m / do_buffer_px when enabled.
    sam_auto_buffer_rooms: bool = False

    # corridor reprocessing (paper §4.2): re-run SAM on the residual free space with a
    # sparser point grid to catch corridors the first pass missed. OFF by default so the
    # single-pass baseline is clean; enable for the paper-faithful two-pass result.
    sam_reprocess_residual: bool = False
    sam_residual_points_per_side: int = 5       # paper points_=5 on the residual

    # =====================================================================
    # NEW — harmonized evaluation (research-fixes Task 05)
    # =====================================================================
    # The three methods drop rooms with different area/void rules and (historically) used
    # different wall masks, so a raw head-to-head partly measures the post-filter, not the
    # segmenter. ``harmonize_room_labels`` applies ONE area threshold + ONE void rule + the ONE
    # shared wall scaffold to every method's labels before any metric.
    #   eval_profile == 'comparison' : apply the harmonized filter (eval_* values below) to all.
    #   eval_profile == 'paper'      : skip harmonization; each method keeps its own standalone
    #                                  values (min_room_area_m2 / sam_auto_min_room_area_m2, …).
    # The canonical wall scaffold is the cleaned slab-occupancy wall mask (deterministic,
    # method-agnostic) — the same scaffold the room metric (Task 02) and wall metric (Task 04)
    # reference; see eval.eval_wall_scaffold.
    eval_profile: str = 'comparison'       # 'comparison' (harmonized) | 'paper' (faithful)
    eval_min_room_area_m2: float = 1.0     # one area threshold for all methods (m^2)
    eval_min_coverage_frac: float = 0.25   # one void rule for all methods (interior coverage frac)

    # =====================================================================
    # NEW — structured outputs / staging
    # =====================================================================
    out_root: str = 'scan2bim_out'  # NEW: base output dir; local relative path (set via os.path.join(PROJECT_ROOT, ...) in notebooks)
    out_dir: str = 'scan2bim_out'  # legacy field (kept for compatibility; unused)

    # ---------- pixel-space conversions ----------
    @property
    def seal_gap_px(self) -> int:
        return 0  # blanket sealing is OFF by design; use seal_at_doors() instead

    @property
    def min_room_area_px(self) -> int:
        return int(round(self.min_room_area_m2 / (self.pixel_m ** 2)))

    @property
    def do_buffer_px(self) -> int:
        return max(1, int(round(self.do_buffer_m / self.pixel_m)))

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot for reproducibility (saved into every stage ZIP)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Rebuild a Config from a saved snapshot, ignoring unknown keys."""
        fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in fields})
