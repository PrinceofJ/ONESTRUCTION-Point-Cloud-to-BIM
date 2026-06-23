"""Unified configuration for the scan2bim pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RoomSegConfig:
    """Room segmentation parameters."""

    units_per_meter: float = 1.0
    up_axis: int = 2
    voxel_m: float = 0.02

    # slab (horizontal crop)
    slab_relative_to: str = "ceiling"
    slab_lo_m: float = 0.1
    slab_hi_m: float = 1.5
    ceiling_mode: str = "local_perpoint"
    ceiling_cell_size_m: float = 0.1
    ceiling_min_pts_per_cell: int = 5
    ceiling_smooth_cells: int = 3

    # rasterisation
    pixel_m: float = 0.05
    min_points_per_cell: int = 3
    thicken_px: int = 1
    use_wallness: bool = False
    wallness_min_span_frac: float = 0.5

    # pass 1: watershed
    min_wall_area_px: int = 60
    marker_h_m: float = 0.10
    footprint_close_m: float = 1.00
    merge_ridge_m: float = 0.70
    min_room_area_m2: float = 1.0

    # void rejection
    drop_empty_rooms: bool = True
    coverage_ceiling_margin_m: float = 0.30
    coverage_min_pts: int = 1
    coverage_close_px: int = 2
    min_coverage_frac: float = 0.25

    # pass 2: SAM recall
    use_sam_recall: bool = True
    sam_points: int = 11
    sam_iou: float = 0.85
    sam_stability: float = 0.95
    sam_n_layers: int = 1
    sam_down_factor: int = 2
    sam_min_mask: int = 100
    sam_min_overlap: float = 0.5
    sam_arch: str = "vit_h"
    sam_ckpt: str = "sam_vit_h.pth"
    sam_model_cfg: str = ""

    # 3D wall export
    do_buffer_m: float = 0.05
    wall_floor_margin_m: float = 0.10
    wall_ceiling_margin_m: float = 0.10

    # per-room RANSAC wall fit
    ransac_dist_thresh_m: float = 0.02
    ransac_normal_tol_deg: float = 10.0
    ransac_min_inliers: int = 400
    ransac_max_planes: int = 20

    @property
    def seal_gap_px(self) -> int:
        return 0

    @property
    def min_room_area_px(self) -> int:
        return int(round(self.min_room_area_m2 / (self.pixel_m**2)))

    @property
    def do_buffer_px(self) -> int:
        return max(1, int(round(self.do_buffer_m / self.pixel_m)))


@dataclass
class WallSegConfig:
    """Wall segmentation parameters."""

    up_axis: int = 2
    voxel_m: float = 0.02
    normal_radius_m: float = 0.10
    normal_max_nn: int = 30
    normal_tol_deg: float = 15.0
    angle_tol_deg: float = 10.0
    offset_tol_m: float = 0.20
    min_wall_points: int = 200

    # wall image rasterisation
    flat_pixel_m: float = 0.04
    min_pts_per_cell: int = 1
    morph_close_px: int = 5
    morph_open_px: int = 3
    density_filter_radius: int = 0
    density_filter_threshold: int = 2

    # statistical outlier removal
    sor_neighbours: int = 20
    sor_std_ratio: float = 2.0


@dataclass
class WallProcConfig:
    """Wall image processing (door/window detection) parameters."""

    pixel_m: float = 0.04

    # SAM
    sam_checkpoint: str = "sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    sam_upscale: int = 4
    sam_points_per_void: int = 5

    # connected-component pre-filter
    min_void_px: int = 40

    # shape filter
    min_rectangularity: float = 0.55

    # door heuristics (metres)
    door_min_width_m: float = 0.50
    door_max_width_m: float = 1.60
    door_min_height_m: float = 1.50
    door_max_height_m: float = 2.80
    door_floor_margin_px: int = 3

    # window heuristics (metres)
    window_min_width_m: float = 0.30
    window_max_width_m: float = 2.50
    window_min_height_m: float = 0.30
    window_max_height_m: float = 2.00
    window_min_sill_m: float = 0.30


@dataclass
class IfcExportConfig:
    """IFC JSON generation and export parameters."""

    # quality filters
    min_directions: int = 2
    min_wall_length_m: float = 0.3
    max_wall_length_m: float = 15.0
    min_wall_aspect_ratio: float = 0.15
    max_wall_thickness_m: float = 0.40
    min_wall_fill_ratio: float = 0.15

    # merge / dedup
    default_thickness: float = 0.15
    exterior_thickness: float = 0.30
    max_merge_thickness: float = 0.45
    dedup_offset_tol: float = 0.45
    dedup_overlap_frac: float = 0.10

    # endpoint snapping
    snap_tolerance_m: float = 0.15

    # IFC output
    project_name: str = "Scanned Building"
    floor_elevation: float = 0.0
    add_floor_slabs: bool = True
    slab_thickness: float = 0.2


@dataclass
class PipelineConfig:
    """Top-level configuration for the full scan2bim pipeline."""

    # I/O paths
    input_path: str = "cloud.ply"
    out_dir: str = "scan2bim_out"
    output_ifc: str = "model.ifc"

    # sub-configs
    room_seg: RoomSegConfig = field(default_factory=RoomSegConfig)
    wall_seg: WallSegConfig = field(default_factory=WallSegConfig)
    wall_proc: WallProcConfig = field(default_factory=WallProcConfig)
    ifc_export: IfcExportConfig = field(default_factory=IfcExportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> PipelineConfig:
        sub_map = {
            "room_seg": RoomSegConfig,
            "wall_seg": WallSegConfig,
            "wall_proc": WallProcConfig,
            "ifc_export": IfcExportConfig,
        }
        kwargs = {}
        for key, val in d.items():
            if key in sub_map and isinstance(val, dict):
                kwargs[key] = sub_map[key](**val)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
