#!/usr/bin/env python3
"""Run the full scan2bim pipeline from the command line.

Usage:
    python run_pipeline.py                     # all stages, geometric_SAM method
    python run_pipeline.py --skip-sam          # skip SAM refinement (no GPU needed)
    python run_pipeline.py --from-stage 5      # resume from stage 5 onward
    python run_pipeline.py --params my.yaml    # custom params file
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def stage_1_occupancy(CFG, A):
    """Rasterize point cloud to 2D occupancy, wall mask, coverage."""
    import scan2bim

    logger.info("=== Stage 1: Occupancy rasterization ===")
    pcd, pts = scan2bim.load_point_cloud(CFG)
    floor_z, ceil_z = scan2bim.estimate_ceiling(pts[:, CFG.up_axis], return_floor=True)
    logger.info("Floor=%.2f  Ceiling=%.2f", floor_z, ceil_z)

    slab_pts, slab_mask, slab_info = scan2bim.crop_vertical(pts, CFG, debug=True, return_info=True)
    occ, tf = scan2bim.rasterize_topdown(
        slab_pts, CFG.pixel_m, up_axis=CFG.up_axis,
        min_points_per_cell=CFG.min_points_per_cell, thicken=CFG.thicken_px)
    wall_mask = (occ == 0)
    coverage = scan2bim.rasterize_coverage(pts, CFG, tf)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE1))
    A.save_png(os.path.join(out_dir, A.OCC_PNG), occ)
    A.save_npy(os.path.join(out_dir, A.WALLMASK_NPY), wall_mask)
    A.save_npy(os.path.join(out_dir, A.COVERAGE_NPY), coverage)
    A.save_transform(os.path.join(out_dir, A.TRANSFORM_JSON), tf,
                     extra=dict(floor_z=floor_z, ceil_z=ceil_z,
                                units_per_meter=CFG.units_per_meter,
                                up_axis=CFG.up_axis, file_path=CFG.file_path,
                                voxel_m=CFG.voxel_m))
    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE1)
    logger.info("Stage 1 done → %s", out_dir)


def stage_2_watershed(CFG, A):
    """Watershed room segmentation."""
    import scan2bim

    logger.info("=== Stage 2: Watershed room segmentation ===")
    s1 = A.load_stage_dir(CFG.out_root, A.STAGE1)
    scan2bim.assert_upstream_config(CFG, A.load_stage_config(s1))

    wall_mask = A.load_npy(os.path.join(s1, A.WALLMASK_NPY)).astype(bool)
    coverage = A.load_npy(os.path.join(s1, A.COVERAGE_NPY)).astype(bool)
    tf = A.load_transform(os.path.join(s1, A.TRANSFORM_JSON))

    labels, aux = scan2bim.segment_rooms_watershed(
        wall_mask, CFG.pixel_m,
        marker_h_m=CFG.marker_h_m,
        footprint_close_m=CFG.footprint_close_m,
        merge_ridge_m=CFG.merge_ridge_m,
        min_room_area_m2=CFG.min_room_area_m2,
        min_wall_area_px=CFG.min_wall_area_px,
        door_seal_px=CFG.seal_gap_px,
        coverage=coverage,
        min_coverage_frac=CFG.min_coverage_frac,
        return_aux=True)

    n_rooms = len([x for x in np.unique(labels) if x >= 1])
    logger.info("Detected %d rooms", n_rooms)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE2))
    A.save_npy(os.path.join(out_dir, A.ROOM_LABELS_NPY), labels.astype("int32"))
    A.save_label_png(os.path.join(out_dir, A.ROOM_LABELS_PNG), labels)
    A.save_npy(os.path.join(out_dir, A.WATERSHED_WALLS_NPY), aux["walls"].astype(bool))
    A.save_npy(os.path.join(out_dir, A.FOOTPRINT_NPY), aux["footprint"].astype(bool))
    A.save_transform(os.path.join(out_dir, A.TRANSFORM_JSON), tf)
    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE2)
    logger.info("Stage 2 done → %s", out_dir)


def stage_3_sam_refine(CFG, A):
    """SAM 2.1 refinement of watershed labels (requires GPU)."""
    import scan2bim
    from PIL import Image

    logger.info("=== Stage 3: SAM refinement ===")
    s1 = A.load_stage_dir(CFG.out_root, A.STAGE1)
    s2 = A.load_stage_dir(CFG.out_root, A.STAGE2)
    scan2bim.assert_upstream_config(CFG, A.load_stage_config(s2))

    occ = np.array(Image.open(os.path.join(s1, A.OCC_PNG)))
    wall_mask = A.load_npy(os.path.join(s1, A.WALLMASK_NPY)).astype(bool)
    coverage = A.load_npy(os.path.join(s1, A.COVERAGE_NPY)).astype(bool)
    geom_labels = A.load_npy(os.path.join(s2, A.ROOM_LABELS_NPY)).astype("int32")
    walls = A.load_npy(os.path.join(s2, A.WATERSHED_WALLS_NPY)).astype(bool)
    footprint = A.load_npy(os.path.join(s2, A.FOOTPRINT_NPY)).astype(bool)
    tf = A.load_transform(os.path.join(s2, A.TRANSFORM_JSON))

    refined, dbg = scan2bim.refine_with_sam(
        geom_labels, occ, walls, footprint, CFG,
        wall_mask=wall_mask, coverage=coverage)
    logger.info("SAM refinement: %s", dbg)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE4))
    A.save_npy(os.path.join(out_dir, A.REFINED_LABELS_NPY), refined.astype("int32"))
    A.save_label_png(os.path.join(out_dir, A.REFINED_LABELS_PNG), refined)
    A.save_transform(os.path.join(out_dir, A.TRANSFORM_JSON), tf)
    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE4)
    logger.info("Stage 3 done → %s", out_dir)


def stage_4_wall_assignment(CFG, A):
    """Assign wall point clouds to rooms from SAM-refined masks."""
    import scan2bim
    import open3d as o3d

    logger.info("=== Stage 4: Wall assignment ===")
    s1 = A.load_stage_dir(CFG.out_root, A.STAGE1)
    s4 = A.load_stage_dir(CFG.out_root, A.STAGE4)
    scan2bim.assert_upstream_config(CFG, A.load_stage_config(s1))
    scan2bim.assert_upstream_config(CFG, A.load_stage_config(s4))

    wall_mask = A.load_npy(os.path.join(s1, A.WALLMASK_NPY)).astype(bool)
    tf = A.load_transform(os.path.join(s1, A.TRANSFORM_JSON))
    labels = A.load_npy(os.path.join(s4, A.REFINED_LABELS_NPY)).astype("int32")

    pcd, pts = scan2bim.load_point_cloud(CFG)
    scan2bim.assert_points_in_grid(pts, tf)

    wall_masks, dbg = scan2bim.room_wall_masks_boundary_ring(
        labels, wall_mask, CFG, return_debug=True)
    band, floor_z, ceil_z = scan2bim.height_band_mask(pts, CFG, tf)
    rooms3d = scan2bim.backproject_room_masks(pts, wall_masks, tf, keep_mask=band)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE5))
    A.save_room_wall_masks(os.path.join(out_dir, A.ROOM_WALL_MASKS_NPZ), wall_masks)
    for e in rooms3d:
        if len(e["points"]) > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(e["points"])
            o3d.io.write_point_cloud(
                os.path.join(out_dir, "room_%02d_walls.ply" % e["room_id"]), pc)

    A.save_npy(os.path.join(out_dir, A.ROOM_LABELS_NPY), labels)
    A.save_transform(os.path.join(out_dir, A.TRANSFORM_JSON), tf,
                     extra=dict(floor_z=float(floor_z), ceil_z=float(ceil_z)))
    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE5)
    logger.info("Stage 4 done → %s  (%d rooms)", out_dir, len(rooms3d))


def stage_4_wall_assignment_no_sam(CFG, A):
    """Wall assignment using watershed labels directly (skip-sam path)."""
    import scan2bim
    import open3d as o3d

    logger.info("=== Stage 4 (no SAM): Wall assignment from watershed ===")
    s1 = A.load_stage_dir(CFG.out_root, A.STAGE1)
    s2 = A.load_stage_dir(CFG.out_root, A.STAGE2)
    scan2bim.assert_upstream_config(CFG, A.load_stage_config(s1))

    wall_mask = A.load_npy(os.path.join(s1, A.WALLMASK_NPY)).astype(bool)
    tf = A.load_transform(os.path.join(s1, A.TRANSFORM_JSON))
    labels = A.load_npy(os.path.join(s2, A.ROOM_LABELS_NPY)).astype("int32")

    pcd, pts = scan2bim.load_point_cloud(CFG)
    scan2bim.assert_points_in_grid(pts, tf)

    wall_masks, dbg = scan2bim.room_wall_masks_boundary_ring(
        labels, wall_mask, CFG, return_debug=True)
    band, floor_z, ceil_z = scan2bim.height_band_mask(pts, CFG, tf)
    rooms3d = scan2bim.backproject_room_masks(pts, wall_masks, tf, keep_mask=band)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE5))
    A.save_room_wall_masks(os.path.join(out_dir, A.ROOM_WALL_MASKS_NPZ), wall_masks)
    for e in rooms3d:
        if len(e["points"]) > 0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(e["points"])
            o3d.io.write_point_cloud(
                os.path.join(out_dir, "room_%02d_walls.ply" % e["room_id"]), pc)

    A.save_npy(os.path.join(out_dir, A.ROOM_LABELS_NPY), labels)
    A.save_transform(os.path.join(out_dir, A.TRANSFORM_JSON), tf,
                     extra=dict(floor_z=float(floor_z), ceil_z=float(ceil_z)))
    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE5)
    logger.info("Stage 4 done → %s  (%d rooms)", out_dir, len(rooms3d))


def stage_5_wall_segmentation(CFG, A):
    """Segment individual wall planes from room wall point clouds."""
    from scan2bim.wall_seg import run_wall_segmentation

    logger.info("=== Stage 5: Wall segmentation ===")
    wall_dir = A.load_stage_dir(CFG.out_root, A.STAGE5)
    room_clouds = sorted(glob.glob(os.path.join(wall_dir, "room_*_walls.ply")))
    logger.info("Found %d room wall clouds", len(room_clouds))

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE_WALL_SEG))
    results = run_wall_segmentation(room_clouds, CFG, out_dir)

    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE_WALL_SEG)
    n_walls = sum(len(v) for v in results.values())
    logger.info("Stage 5 done → %s  (%d walls across %d rooms)", out_dir, n_walls, len(results))


def stage_6_wall_processing(CFG, A):
    """Detect doors and windows in wall images."""
    from scan2bim.wall_proc import run_wall_image_processing

    logger.info("=== Stage 6: Wall image processing (door/window detection) ===")
    wall_seg_dir = A.load_stage_dir(CFG.out_root, A.STAGE_WALL_SEG)

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE_WALL_PROC))
    summaries = run_wall_image_processing(wall_seg_dir, CFG, out_dir, use_sam=False)

    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE_WALL_PROC)
    n_openings = sum(
        sum(len(w.get("openings", [])) for w in room_walls)
        for room_walls in summaries.values()
    )
    logger.info("Stage 6 done → %s  (%d openings detected)", out_dir, n_openings)


def stage_7_ifc_export(CFG, A):
    """Build JSON representation and export IFC4 model."""
    from scan2bim.ifc_export import build_building_json, build_ifc

    logger.info("=== Stage 7: IFC export ===")
    wall_dir = A.load_stage_dir(CFG.out_root, A.STAGE5)
    room_clouds = sorted(glob.glob(os.path.join(wall_dir, "room_*_walls.ply")))

    out_dir = A.ensure_dir(A.stage_dir(CFG.out_root, A.STAGE_IFC))
    building = build_building_json(room_clouds, CFG, out_dir=out_dir)

    json_path = os.path.join(out_dir, A.BUILDING_JSON)
    save_data = {k: v for k, v in building.items() if k != "_debug"}
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Building JSON → %s", json_path)

    ifc_path = os.path.join(out_dir, A.IFC_MODEL)
    try:
        build_ifc(building, out_path=ifc_path, cfg=CFG)
        logger.info("IFC model → %s", ifc_path)
    except ImportError:
        logger.warning("ifcopenshell not installed — skipping IFC export. "
                       "Building JSON at %s is the canonical output.", json_path)

    A.save_config(os.path.join(out_dir, A.CONFIG_JSON), CFG)
    A.package_stage(CFG.out_root, A.STAGE_IFC)
    logger.info("Stage 7 done → %s", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run the scan2bim pipeline end-to-end.")
    parser.add_argument("--params", default="params.yaml",
                        help="Path to params.yaml (default: params.yaml)")
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip SAM refinement (stages 3); use watershed labels directly")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 8),
                        help="Resume from this stage (1-7, default: 1)")
    parser.add_argument("--to-stage", type=int, default=7, choices=range(1, 8),
                        help="Stop after this stage (1-7, default: 7)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scan2bim
    from scan2bim import artifacts as A

    CFG = scan2bim.load_config()
    logger.info("Config loaded: input=%s  output=%s", CFG.file_path, CFG.out_root)

    stages = []
    if args.skip_sam:
        stages = [
            (1, stage_1_occupancy),
            (2, stage_2_watershed),
            (4, stage_4_wall_assignment_no_sam),
            (5, stage_5_wall_segmentation),
            (6, stage_6_wall_processing),
            (7, stage_7_ifc_export),
        ]
    else:
        stages = [
            (1, stage_1_occupancy),
            (2, stage_2_watershed),
            (3, stage_3_sam_refine),
            (4, stage_4_wall_assignment),
            (5, stage_5_wall_segmentation),
            (6, stage_6_wall_processing),
            (7, stage_7_ifc_export),
        ]

    for num, fn in stages:
        if num < args.from_stage:
            continue
        if num > args.to_stage:
            break
        fn(CFG, A)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
