"""scan2bim - shared library for the Scan-to-BIM pipeline."""

from . import config, io_utils, slab, raster, watershed, walls, sam_refine, sam_auto, viz, artifacts
from . import wall_seg, wall_proc, ifc_export
from . import runconfig
from . import eval
from .config import Config

from .runconfig import (project_root, load_config,
                        assert_upstream_config, assert_points_in_grid)
from .io_utils import load_point_cloud
from .slab import estimate_ceiling, estimate_local_ceilings, crop_vertical
from .raster import (rasterize_topdown, rasterize_wallness, rasterize_coverage,
                     point_cells, label_points, grid_world_bbox,
                     interior_coverage_fraction)
from .watershed import segment_rooms_watershed, labels_to_detections
from .walls import (clean_wall_mask, seal_at_doors, room_wall_masks_boundary_ring,
                    backproject_room_masks, height_band_mask, estimate_wall_thickness_px,
                    resolve_ring_radii_px, fit_walls_in_room,
                    room_footprints, split_rooms_to_clouds)
from .sam_refine import (MaskGenerator, build_mask_generator, refine_with_sam,
                         build_sam_image, relabel_by_sam)
from .sam_auto import (AutoMaskGenerator, build_auto_mask_generator,
                       segment_rooms_sam_auto, masks_to_room_labels,
                       classify_rooms_by_area, buffer_room_labels, reprocess_residual)
from .eval import (STRUCTURAL_CLUTTER_CLASSES, annotation_class,
                   load_room_interior_points, build_gt_room_labels,
                   overlap_stats, score_rooms, load_method_labels)
from .wall_seg import (segment_walls, flatten_wall, save_wall_images,
                       run_wall_segmentation)
from .wall_proc import (find_void_components, merge_fragments, classify_openings,
                        process_wall_array, process_wall_image,
                        run_wall_image_processing)
from .ifc_export import (build_building_json, build_ifc, compute_wall_geometry,
                         compute_room_boundaries)

__all__ = [
    'config', 'io_utils', 'slab', 'raster', 'watershed', 'walls', 'sam_refine', 'sam_auto',
    'viz', 'artifacts', 'runconfig', 'eval', 'wall_seg', 'wall_proc', 'ifc_export', 'Config',
    'project_root', 'load_config', 'assert_upstream_config', 'assert_points_in_grid',
    'load_point_cloud', 'estimate_ceiling', 'estimate_local_ceilings', 'crop_vertical',
    'rasterize_topdown', 'rasterize_wallness', 'rasterize_coverage', 'point_cells',
    'label_points', 'grid_world_bbox', 'interior_coverage_fraction',
    'segment_rooms_watershed', 'labels_to_detections',
    'clean_wall_mask', 'seal_at_doors', 'room_wall_masks_boundary_ring',
    'backproject_room_masks', 'height_band_mask', 'estimate_wall_thickness_px',
    'resolve_ring_radii_px', 'fit_walls_in_room', 'room_footprints',
    'split_rooms_to_clouds', 'MaskGenerator', 'build_mask_generator', 'refine_with_sam',
    'build_sam_image', 'relabel_by_sam',
    'AutoMaskGenerator', 'build_auto_mask_generator', 'segment_rooms_sam_auto',
    'masks_to_room_labels', 'classify_rooms_by_area', 'buffer_room_labels',
    'reprocess_residual',
    'STRUCTURAL_CLUTTER_CLASSES', 'annotation_class', 'load_room_interior_points',
    'build_gt_room_labels', 'overlap_stats', 'score_rooms', 'load_method_labels',
    'segment_walls', 'flatten_wall', 'save_wall_images', 'run_wall_segmentation',
    'find_void_components', 'merge_fragments', 'classify_openings',
    'process_wall_array', 'process_wall_image', 'run_wall_image_processing',
    'build_building_json', 'build_ifc', 'compute_wall_geometry', 'compute_room_boundaries',
]

__version__ = '1.0.0'
