"""scan2bim — shared library for the Scan-to-BIM room-segmentation pipeline.

The four notebooks are thin drivers; all reusable logic lives here so nothing is
duplicated across stages. Heavy / optional dependencies (open3d, supervision, torch +
the SAM backends) are imported lazily inside the functions that need them, so importing
``scan2bim`` and running the raster / watershed / wall-assignment stages does not require
them to be installed.

Stage modules
-------------
config       : Config dataclass (single source of truth)
io_utils     : load_point_cloud
slab         : ceiling estimation + vertical slab crop
raster       : occupancy / wallness / coverage rasters + pixel<->world transforms
watershed    : Pass-1 deterministic watershed segmentation
walls        : wall-mask cleanup + NEW boundary-ring wall assignment + back-projection
sam_refine   : model-agnostic PROMPTED SAM refinement (graph relabel: merge / split)
viz          : optional debug/QA plots
artifacts    : structured save/load + ZIP packaging + shared filename constants
"""

from . import config, io_utils, slab, raster, watershed, walls, sam_refine, viz, artifacts
from . import runconfig
from .config import Config

# convenience re-exports (most-used names)
from .runconfig import (project_root, load_config,
                        assert_upstream_config, assert_points_in_grid)
from .io_utils import load_point_cloud
from .slab import estimate_ceiling, estimate_local_ceilings, crop_vertical
from .raster import (rasterize_topdown, rasterize_wallness, rasterize_coverage,
                     point_cells, label_points)
from .watershed import segment_rooms_watershed, labels_to_detections
from .walls import (clean_wall_mask, seal_at_doors, room_wall_masks_boundary_ring,
                    backproject_room_masks, height_band_mask, estimate_wall_thickness_px,
                    resolve_ring_radii_px, fit_walls_in_room,
                    room_footprints, split_rooms_to_clouds)
from .sam_refine import (MaskGenerator, build_mask_generator, refine_with_sam,
                         build_sam_image, relabel_by_sam)

__all__ = [
    'config', 'io_utils', 'slab', 'raster', 'watershed', 'walls', 'sam_refine',
    'viz', 'artifacts', 'runconfig', 'Config',
    'project_root', 'load_config', 'assert_upstream_config', 'assert_points_in_grid',
    'load_point_cloud', 'estimate_ceiling', 'estimate_local_ceilings', 'crop_vertical',
    'rasterize_topdown', 'rasterize_wallness', 'rasterize_coverage', 'point_cells',
    'label_points', 'segment_rooms_watershed', 'labels_to_detections',
    'clean_wall_mask', 'seal_at_doors', 'room_wall_masks_boundary_ring',
    'backproject_room_masks', 'height_band_mask', 'estimate_wall_thickness_px',
    'resolve_ring_radii_px', 'fit_walls_in_room', 'room_footprints',
    'split_rooms_to_clouds', 'MaskGenerator', 'build_mask_generator', 'refine_with_sam',
    'build_sam_image', 'relabel_by_sam',
]

__version__ = '1.0.0'
