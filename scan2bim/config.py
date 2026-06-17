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
    file_path: str = 'data/area1.xyz'      # canonical sample cloud; relative to project root, resolved by load_config()
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
    use_wallness: bool = False             # vertical-extent raster vs binary presence (segmentation input)
    wallness_min_span_frac: float = 0.5    # column counts as wall if it spans >= this
                                           #   fraction of the floor->ceiling height

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
    #                    pixels from the wallness raster.
    # Set either to ``None`` to auto-derive it from the estimated wall thickness
    # (the same distance-transform median heuristic the original used in
    # ``room_footprints``); see walls.estimate_wall_thickness_px / resolve_ring_radii_px.
    room_erode_m: float | None = 0.125      # NEW: erosion radius -> reliable interior (ring thickness)
    wall_dilate_m: float | None = 0.15     # NEW: r_w -> outward reach to grab wall pixels
    wall_source: str = 'wallness'          # NEW: 'wallness' (spec) | 'occupancy' (legacy compatibility)

    # =====================================================================
    # NEW — SAM backend abstraction (Notebook 4)
    # =====================================================================
    # The refinement runs on SAM2. Inference is wrapped behind a MaskGenerator adapter so
    # a different model is reached by changing this one string (and supplying the matching
    # checkpoint + config) with no change to the refinement code — e.g. 'sam3' once that
    # build is available, or 'sam1' for the original Segment-Anything predictor.
    sam_backend: str = 'sam2'              # 'sam2' (default) | 'sam3' | 'sam1'

    # ---- prompting (per watershed room) ----
    sam_image_mode: str = 'stack'          # SAM input image: 'stack' = occupancy+wallness+
                                           #   coverage as 3 channels; 'occupancy' = replicate
                                           #   the binary occupancy raster into 3 channels.
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
