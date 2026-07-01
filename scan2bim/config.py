"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- input ----
    file_path: str = 'data/penthouse_merged.ply'
    gt_dir: str = 'data/Area_1'
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

    # ---- room seg: PASS 2 (prompted SAM refinement) ----
    use_sam_recall: bool = True
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

    # ---- boundary-ring wall assignment ----
    room_erode_m: float | None = 0.125
    wall_dilate_m: float | None = 0.15
    wall_source: str = 'wallness'          # 'wallness' | 'occupancy'

    # ---- SAM backend ----
    sam_backend: str = 'sam2'              # 'sam2' | 'sam3' | 'sam1'

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

    # ---- pure-SAM automatic segmentation (Method 2) ----
    sam_points_per_side: int = 15
    sam_pred_iou_thresh: float = 0.85           # paper 'iou_'
    sam_stability_score_thresh: float = 0.95    # paper 'stability_'
    sam_crop_n_layers: int = 1                  # paper 'n_layers'
    sam_crop_n_points_downscale_factor: int = 2  # paper 'down_factor'
    sam_min_mask_region_area: int = 100

    sam_auto_min_room_area_m2: float = 1.5      # paper 'A' = 1.5 m^2
    sam_auto_min_coverage_frac: float = 0.5     # keep a mask only if >= this frac is on coverage

    sam_auto_buffer_rooms: bool = False

    sam_reprocess_residual: bool = False
    sam_residual_points_per_side: int = 5       # paper points_=5 on the residual

    # ---- wall segmentation ----
    normal_radius_m: float = 0.10
    normal_max_nn: int = 30
    normal_tol_deg: float = 15.0
    wseg_angle_tol_deg: float = 10.0
    offset_tol_m: float = 0.20
    min_wall_points: int = 200
    flat_pixel_m: float = 0.04
    flat_min_pts_per_cell: int = 1
    morph_close_px: int = 5
    morph_open_px: int = 3
    density_filter_radius: int = 0
    density_filter_threshold: int = 2
    sor_neighbours: int = 20
    sor_std_ratio: float = 2.0

    # ---- wall image processing (door/window detection) ----
    wproc_sam_checkpoint: str = 'sam_vit_b_01ec64.pth'
    wproc_sam_model_type: str = 'vit_b'
    wproc_sam_upscale: int = 4
    wproc_sam_points_per_void: int = 5
    min_void_px: int = 40
    min_rectangularity: float = 0.55
    door_min_width_m: float = 0.50
    door_max_width_m: float = 1.80
    door_min_height_m: float = 1.80
    door_max_height_m: float = 2.80
    door_floor_margin_px: int = 3
    door_max_wall_width_frac: float = 0.85
    window_min_width_m: float = 0.30
    window_max_width_m: float = 2.50
    window_min_height_m: float = 0.30
    window_max_height_m: float = 2.00
    window_min_sill_m: float = 0.30

    # ---- IFC export ----
    ifc_min_directions: int = 2
    ifc_min_wall_length_m: float = 0.3
    ifc_max_wall_length_m: float = 15.0
    ifc_min_wall_aspect_ratio: float = 0.15
    ifc_max_wall_thickness_m: float = 0.40
    ifc_min_wall_fill_ratio: float = 0.15
    ifc_default_thickness: float = 0.15
    ifc_exterior_thickness: float = 0.30
    ifc_max_merge_thickness: float = 0.45
    ifc_dedup_offset_tol: float = 0.45
    ifc_snap_tolerance_m: float = 0.15
    ifc_project_name: str = 'Scanned Building'
    ifc_floor_elevation: float = 0.0
    ifc_add_floor_slabs: bool = True
    ifc_slab_thickness: float = 0.2

    # ---- bSDD enrichment ----
    bsdd_enrich: bool = False
    bsdd_server_path: str = ''
    bsdd_add_psets: bool = True
    bsdd_add_qtos: bool = True
    bsdd_add_classifications: bool = True
    bsdd_validate_geometry: bool = True

    # ---- output ----
    out_root: str = 'scan2bim_out'
    out_dir: str = 'scan2bim_out'

    @property
    def seal_gap_px(self) -> int:
        return 0

    @property
    def min_room_area_px(self) -> int:
        return int(round(self.min_room_area_m2 / (self.pixel_m ** 2)))

    @property
    def do_buffer_px(self) -> int:
        return max(1, int(round(self.do_buffer_m / self.pixel_m)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in fields})
