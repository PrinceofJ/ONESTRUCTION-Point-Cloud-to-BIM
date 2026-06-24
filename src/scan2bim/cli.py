"""CLI entry point for the scan2bim pipeline."""

from __future__ import annotations

import glob
import json
import logging
import os
import sys

import click

from .config import PipelineConfig

logger = logging.getLogger("scan2bim")


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str | None) -> PipelineConfig:
    if config_path and os.path.exists(config_path):
        click.echo(f"Loading config from {config_path}")
        return PipelineConfig.from_yaml(config_path)
    return PipelineConfig()


@click.group()
@click.version_option(package_name="scan2bim")
def main():
    """scan2bim — Point Cloud to BIM pipeline."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out", help="Output directory.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--no-sam", is_flag=True, help="Disable SAM (room seg + wall proc).")
@click.option("--no-ifc", is_flag=True, help="Skip IFC export (JSON only).")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def run(input_path, out_dir, config_path, no_sam, no_ifc, verbose):
    """Run the full pipeline: point cloud → IFC model."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)
    cfg.input_path = input_path
    cfg.out_dir = out_dir
    if no_sam:
        cfg.room_seg.use_sam_recall = False

    from .io import load_point_cloud
    from .room_segmentation import run_room_segmentation
    from .ifc_export import build_building_json, build_ifc

    click.echo(f"[1/2] Room segmentation: {input_path}")
    room_cloud_dir = os.path.join(out_dir, "room_clouds")
    pcd, points = load_point_cloud(
        input_path,
        units_per_meter=cfg.room_seg.units_per_meter,
        voxel_m=cfg.room_seg.voxel_m,
    )
    labels, rooms, tf = run_room_segmentation(pcd, points, cfg.room_seg, room_cloud_dir)
    n_rooms = len([r for r in rooms if len(r["points"]) > 0])
    click.echo(f"  → {n_rooms} rooms")

    click.echo("[2/2] Wall segmentation + door detection + IFC export")
    room_cloud_paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    building_json = build_building_json(
        room_cloud_paths, cfg.wall_seg, cfg.ifc_export,
        wall_proc_cfg=cfg.wall_proc, out_dir=out_dir, use_sam=not no_sam,
    )

    json_path = os.path.join(out_dir, "building.json")
    serializable = {k: v for k, v in building_json.items() if k != "_debug"}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    click.echo(f"  → JSON: {json_path}")
    click.echo(f"    {len(building_json['walls'])} walls, "
               f"{len(building_json['doors'])} doors")

    if not no_ifc:
        try:
            ifc_path = os.path.join(out_dir, cfg.output_ifc)
            build_ifc(
                building_json, ifc_path,
                add_floor_slabs=cfg.ifc_export.add_floor_slabs,
                slab_thickness=cfg.ifc_export.slab_thickness,
            )
            click.echo(f"  → IFC: {ifc_path}")
        except ImportError:
            click.echo("  ⚠ ifcopenshell not installed — skipping IFC export.")
            click.echo("    Install with: pip install scan2bim[ifc]")

    click.echo("Done.")


@main.command("room-seg")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out/room_clouds", help="Output directory.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--no-sam", is_flag=True, help="Disable SAM recall.")
@click.option("-v", "--verbose", is_flag=True)
def room_seg(input_path, out_dir, config_path, no_sam, verbose):
    """Run room segmentation only."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)
    if no_sam:
        cfg.room_seg.use_sam_recall = False

    from .io import load_point_cloud
    from .room_segmentation import run_room_segmentation

    pcd, points = load_point_cloud(
        input_path,
        units_per_meter=cfg.room_seg.units_per_meter,
        voxel_m=cfg.room_seg.voxel_m,
    )
    labels, rooms, tf = run_room_segmentation(pcd, points, cfg.room_seg, out_dir)
    n = len([r for r in rooms if len(r["points"]) > 0])
    click.echo(f"Segmented {n} rooms → {out_dir}")


@main.command("wall-seg")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out/wall_images", help="Output directory.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("-v", "--verbose", is_flag=True)
def wall_seg(room_cloud_dir, out_dir, config_path, verbose):
    """Run wall segmentation on room point clouds."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .wall_segmentation import run_wall_segmentation

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    results = run_wall_segmentation(paths, cfg.wall_seg, out_dir)
    total = sum(len(v) for v in results.values())
    click.echo(f"Segmented {total} walls across {len(results)} rooms → {out_dir}")


@main.command("wall-proc")
@click.argument("wall_image_dir", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out/openings", help="Output directory.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--no-sam", is_flag=True, help="Skip SAM refinement.")
@click.option("-v", "--verbose", is_flag=True)
def wall_proc(wall_image_dir, out_dir, config_path, no_sam, verbose):
    """Detect doors and windows from wall images."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .wall_image_processing import run_wall_image_processing

    run_wall_image_processing(wall_image_dir, cfg.wall_proc, out_dir, use_sam=not no_sam)
    click.echo(f"Results → {out_dir}")


@main.command("ifc-export")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-o", "--output", default="model.ifc", help="Output IFC file path.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--no-sam", is_flag=True, help="Disable SAM for door detection.")
@click.option("-v", "--verbose", is_flag=True)
def ifc_export(room_cloud_dir, output, config_path, no_sam, verbose):
    """Generate IFC model from room clouds."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .ifc_export import build_building_json, build_ifc

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output))
    building_json = build_building_json(
        paths, cfg.wall_seg, cfg.ifc_export,
        wall_proc_cfg=cfg.wall_proc, out_dir=out_dir, use_sam=not no_sam,
    )

    json_path = output.replace(".ifc", ".json")
    serializable = {k: v for k, v in building_json.items() if k != "_debug"}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    click.echo(f"JSON → {json_path}")

    try:
        build_ifc(
            building_json, output,
            add_floor_slabs=cfg.ifc_export.add_floor_slabs,
            slab_thickness=cfg.ifc_export.slab_thickness,
        )
        click.echo(f"IFC  → {output}")
    except ImportError:
        click.echo("ifcopenshell not installed — skipping IFC export.")
        click.echo("Install with: pip install scan2bim[ifc]")


@main.command("debug-walls")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--save", "save_path", default=None, help="Save debug plot to file instead of showing.")
@click.option("-v", "--verbose", is_flag=True)
def debug_walls(room_cloud_dir, config_path, save_path, verbose):
    """Show diagnostic view of wall generation at each pipeline stage."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .ifc_export import build_building_json
    from .viz import show_wall_debug, show_wall_detail_table

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    click.echo(f"Building wall data from {len(paths)} room clouds...")
    building_json = build_building_json(paths, cfg.wall_seg, cfg.ifc_export)

    show_wall_detail_table(building_json)
    show_wall_debug(building_json, save_path=save_path)


@main.command("debug-wall-seg")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out/debug_wall_seg", help="Output directory.")
@click.option("-n", "--max-rooms", default=3, type=int, help="Max rooms to process (0=all).")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("-v", "--verbose", is_flag=True)
def debug_wall_seg(room_cloud_dir, out_dir, max_rooms, config_path, verbose):
    """Debug wall segmentation: segment, filter, merge, then save
    per-wall point clouds and flattened images for the FINAL merged walls."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .io import load_room_cloud
    from .wall_segmentation import flatten_wall
    from .ifc_export import (
        segment_walls_for_ifc, compute_wall_geometry,
        merge_wall_faces, _combine_wall_clouds,
    )
    from .wall_image_processing import (
        find_void_components, merge_fragments, save_annotated_image,
    )

    import numpy as np
    import open3d as o3d
    import cv2

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    if max_rooms > 0:
        paths = paths[:max_rooms]

    os.makedirs(out_dir, exist_ok=True)
    click.echo(f"Processing {len(paths)} room(s) → {out_dir}")

    rng = np.random.default_rng(42)
    palette = rng.random((50, 3)) * 0.7 + 0.3

    for room_path in paths:
        fname = os.path.splitext(os.path.basename(room_path))[0]
        room_name = fname.replace("_walls", "")
        click.echo(f"\n{'='*60}")
        click.echo(f"Room: {room_name}")

        room_pcd, _ = load_room_cloud(room_path, voxel_m=cfg.wall_seg.voxel_m)
        click.echo(f"  Points after downsample: {len(room_pcd.points):,}")

        walls_raw, n_dirs = segment_walls_for_ifc(room_pcd, cfg.wall_seg)
        click.echo(f"  Raw segments: {len(walls_raw)}, directions: {n_dirs}")

        if not walls_raw:
            click.echo("  (no walls — skipping)")
            continue

        # Geometry + filtering (same as build_building_json)
        wall_geos_raw = [compute_wall_geometry(w, cfg.wall_seg.up_axis) for w in walls_raw]
        wall_geos = []
        for raw_idx, g in enumerate(wall_geos_raw):
            if g["length"] < cfg.ifc_export.min_wall_length_m:
                continue
            if g["length"] > cfg.ifc_export.max_wall_length_m:
                continue
            aspect = g["length"] / max(g["height"], 1e-3)
            if aspect < cfg.ifc_export.min_wall_aspect_ratio:
                continue
            if g["thickness"] > cfg.ifc_export.max_wall_thickness_m:
                g["thickness"] = cfg.ifc_export.default_thickness
            if g.get("is_exterior"):
                g["thickness"] = cfg.ifc_export.exterior_thickness
            if g["fill_ratio"] < cfg.ifc_export.min_wall_fill_ratio:
                continue
            g["_raw_idx"] = raw_idx
            wall_geos.append(g)

        click.echo(f"  After filters: {len(wall_geos)} (removed {len(wall_geos_raw) - len(wall_geos)})")

        # Merge opposite faces
        merged = merge_wall_faces(wall_geos, cfg.ifc_export, cfg.wall_seg.angle_tol_deg)
        click.echo(f"  After merge: {len(merged)} final walls")

        room_dir = os.path.join(out_dir, room_name)
        os.makedirs(room_dir, exist_ok=True)

        all_colored_pts = []
        all_colored_cols = []
        room_gaps = []

        for i, mw in enumerate(merged):
            color = palette[i % len(palette)]
            src_indices = mw.get("source_indices", [i])

            combined_cloud = _combine_wall_clouds(walls_raw, src_indices)
            if combined_cloud is None or len(combined_cloud.points) < 50:
                click.echo(f"  Merged wall {i+1:2d}: too few points — skipping")
                continue

            pts = np.asarray(combined_cloud.points)

            # Flatten to 2D wall image
            wall_dict = {
                "cloud": combined_cloud,
                "normal_2d": np.array(mw["normal_2d"]),
                "offset": mw["offset"],
            }
            flat = flatten_wall(wall_dict, cfg.wall_seg)
            wall_img = flat["image"]

            angle_deg = float(np.degrees(mw.get("angle", 0)))
            img_h, img_w = wall_img.shape
            click.echo(
                f"  Merged wall {i+1:2d}: {len(pts):>6,} pts | "
                f"angle={angle_deg:5.1f}° | offset={mw['offset']:.3f} | "
                f"len={mw['length']:.2f}m | {img_w}x{img_h}px | "
                f"from {len(src_indices)} segment(s)"
            )

            # The flattened image's pixel column 0 corresponds to flat u_min.
            # The merged wall's start corresponds to mw["u_min"].
            # offset_m for IFC = (bbox_x * pixel_m) + (flat_u_min - wall_u_min)
            pixel_m = cfg.wall_seg.flat_pixel_m
            flat_u_min = float(flat["u"].min())
            wall_u_min = mw["u_min"]
            u_shift = flat_u_min - wall_u_min

            # Find gaps (white voids in the wall image)
            components = find_void_components(wall_img, min_void_px=cfg.wall_proc.min_void_px)

            for comp in components:
                comp["sam_score"] = 1.0
            gaps = merge_fragments(components, merge_margin_px=cfg.wall_proc.door_floor_margin_px)

            wall_gaps = []
            for gi, gap in enumerate(gaps):
                gx, gy, gw, gh = gap["bbox"]
                w_m = gw * pixel_m
                h_m = gh * pixel_m
                bbox_bottom = gy + gh - 1
                floor_row = img_h - 1
                touches_floor = (floor_row - bbox_bottom) <= cfg.wall_proc.door_floor_margin_px
                sill_m = (floor_row - bbox_bottom) * pixel_m
                bbox_area = gw * gh
                rectangularity = gap["area"] / bbox_area if bbox_area > 0 else 0.0

                wall_length_m = img_w * pixel_m
                too_wide_for_wall = (wall_length_m > 0
                    and w_m / wall_length_m > cfg.wall_proc.door_max_wall_width_frac)

                is_door = (
                    touches_floor
                    and not too_wide_for_wall
                    and rectangularity >= cfg.wall_proc.min_rectangularity
                    and cfg.wall_proc.door_min_width_m <= w_m <= cfg.wall_proc.door_max_width_m
                    and cfg.wall_proc.door_min_height_m <= h_m <= cfg.wall_proc.door_max_height_m
                )

                offset_m = gx * pixel_m + u_shift

                wall_gaps.append({
                    "gap_id": gi + 1,
                    "bbox_px": [gx, gy, gw, gh],
                    "offset_m": round(offset_m, 3),
                    "width_m": round(w_m, 3),
                    "height_m": round(h_m, 3),
                    "area_px": gap["area"],
                    "rectangularity": round(rectangularity, 3),
                    "touches_floor": touches_floor,
                    "sill_m": round(sill_m, 3),
                    "n_fragments": gap.get("n_fragments", 1),
                    "is_door": is_door,
                })

            # Filter out non-rectangular gaps
            wall_gaps = [g for g in wall_gaps if g["rectangularity"] >= cfg.wall_proc.min_rectangularity]

            # Save annotated wall image
            # Green = door, red = rejected gap
            rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
            for g in wall_gaps:
                gx, gy, gw, gh = g["bbox_px"]
                if g["is_door"]:
                    color_bgr = (0, 255, 0)
                    tag = "DOOR"
                else:
                    color_bgr = (0, 0, 255)
                    tag = "not door"
                cv2.rectangle(rgb, (gx, gy), (gx + gw - 1, gy + gh - 1), color_bgr, 1)
                dim_label = f"{tag} {g['width_m']:.2f}x{g['height_m']:.2f}m"
                cv2.putText(rgb, dim_label, (gx + 2, gy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1)
            img_path = os.path.join(room_dir, f"merged_wall_{i+1:02d}.png")
            cv2.imwrite(img_path, rgb)

            if wall_gaps:
                for g in wall_gaps:
                    door_tag = "DOOR" if g["is_door"] else "----"
                    floor_tag = "FLOOR" if g["touches_floor"] else f"sill={g['sill_m']:.2f}m"
                    click.echo(
                        f"    gap {g['gap_id']}: [{door_tag}] {g['width_m']:.2f}x{g['height_m']:.2f}m | "
                        f"rect={g['rectangularity']:.2f} | {floor_tag}"
                    )
            else:
                click.echo("    (no gaps found)")

            room_gaps.append({
                "wall": f"merged_wall_{i+1:02d}",
                "wall_length_m": round(mw["length"], 3),
                "wall_height_m": round(mw["height"], 3),
                "gaps": wall_gaps,
            })

            all_colored_pts.append(pts)
            all_colored_cols.append(np.tile(color, (len(pts), 1)))

        if all_colored_pts:
            combined = o3d.geometry.PointCloud()
            combined.points = o3d.utility.Vector3dVector(np.vstack(all_colored_pts))
            combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colored_cols))
            ply_path = os.path.join(out_dir, f"{room_name}_walls.ply")
            o3d.io.write_point_cloud(ply_path, combined)
            click.echo(f"  → {ply_path}")

        gaps_path = os.path.join(room_dir, "gaps.json")
        with open(gaps_path, "w") as f:
            json.dump(room_gaps, f, indent=2)
        total_gaps = sum(len(w["gaps"]) for w in room_gaps)
        click.echo(f"  → {gaps_path} ({total_gaps} gaps across {len(room_gaps)} walls)")

    click.echo(f"\nDone. Inspect clouds in {out_dir}")


@main.command("debug-doors")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="scan2bim_out/debug_doors", help="Output directory.")
@click.option("-n", "--max-rooms", default=0, type=int, help="Max rooms to process (0=all).")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("-v", "--verbose", is_flag=True)
def debug_doors(room_cloud_dir, out_dir, max_rooms, config_path, verbose):
    """Generate per-room overview images showing all walls and gap detection results."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .io import load_room_cloud
    from .wall_segmentation import flatten_wall
    from .ifc_export import (
        segment_walls_for_ifc, compute_wall_geometry,
        merge_wall_faces, _combine_wall_clouds,
    )
    from .wall_image_processing import find_void_components, merge_fragments

    import numpy as np
    import cv2

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    if max_rooms > 0:
        paths = paths[:max_rooms]

    os.makedirs(out_dir, exist_ok=True)
    click.echo(f"Processing {len(paths)} room(s) → {out_dir}")

    SCALE = 4
    PAD = 12
    HEADER_H = 24
    MIN_PANEL_W = 300

    for room_path in paths:
        fname = os.path.splitext(os.path.basename(room_path))[0]
        room_name = fname.replace("_walls", "")

        room_pcd, _ = load_room_cloud(room_path, voxel_m=cfg.wall_seg.voxel_m)
        walls_raw, n_dirs = segment_walls_for_ifc(room_pcd, cfg.wall_seg)
        if not walls_raw:
            continue

        wall_geos_raw = [compute_wall_geometry(w, cfg.wall_seg.up_axis) for w in walls_raw]
        wall_geos = []
        for raw_idx, g in enumerate(wall_geos_raw):
            if g["length"] < cfg.ifc_export.min_wall_length_m:
                continue
            if g["length"] > cfg.ifc_export.max_wall_length_m:
                continue
            aspect = g["length"] / max(g["height"], 1e-3)
            if aspect < cfg.ifc_export.min_wall_aspect_ratio:
                continue
            if g["thickness"] > cfg.ifc_export.max_wall_thickness_m:
                g["thickness"] = cfg.ifc_export.default_thickness
            if g.get("is_exterior"):
                g["thickness"] = cfg.ifc_export.exterior_thickness
            if g["fill_ratio"] < cfg.ifc_export.min_wall_fill_ratio:
                continue
            g["_raw_idx"] = raw_idx
            wall_geos.append(g)

        if not wall_geos:
            continue

        merged = merge_wall_faces(wall_geos, cfg.ifc_export, cfg.wall_seg.angle_tol_deg)

        wall_panels = []
        pixel_m = cfg.wall_seg.flat_pixel_m

        for i, mw in enumerate(merged):
            src_indices = mw.get("source_indices", [i])
            combined_cloud = _combine_wall_clouds(walls_raw, src_indices)
            if combined_cloud is None or len(combined_cloud.points) < 50:
                continue

            wall_dict = {
                "cloud": combined_cloud,
                "normal_2d": np.array(mw["normal_2d"]),
                "offset": mw["offset"],
            }
            flat = flatten_wall(wall_dict, cfg.wall_seg)
            wall_img = flat["image"]
            img_h, img_w = wall_img.shape

            components = find_void_components(wall_img, min_void_px=cfg.wall_proc.min_void_px)
            for comp in components:
                comp["sam_score"] = 1.0
            gaps = merge_fragments(components, merge_margin_px=cfg.wall_proc.door_floor_margin_px)

            rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
            scaled_w = max(img_w * SCALE, MIN_PANEL_W)
            scaled_h = int(img_h * (scaled_w / img_w)) if img_w > 0 else img_h * SCALE
            rgb = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            effective_scale = scaled_w / img_w if img_w > 0 else SCALE

            for gap in gaps:
                gx, gy, gw, gh = gap["bbox"]
                w_m = gw * pixel_m
                h_m = gh * pixel_m
                bbox_bottom = gy + gh - 1
                floor_row = img_h - 1
                touches_floor = (floor_row - bbox_bottom) <= cfg.wall_proc.door_floor_margin_px
                sill_m = (floor_row - bbox_bottom) * pixel_m
                bbox_area = gw * gh
                rectangularity = gap["area"] / bbox_area if bbox_area > 0 else 0.0
                wall_length_m = img_w * pixel_m
                wall_frac = w_m / wall_length_m if wall_length_m > 0 else 0

                reasons = []
                if not touches_floor:
                    reasons.append(f"no floor (sill={sill_m:.2f}m)")
                if rectangularity < cfg.wall_proc.min_rectangularity:
                    reasons.append(f"rect={rectangularity:.2f}<{cfg.wall_proc.min_rectangularity}")
                if w_m < cfg.wall_proc.door_min_width_m:
                    reasons.append(f"narrow={w_m:.2f}<{cfg.wall_proc.door_min_width_m}")
                if w_m > cfg.wall_proc.door_max_width_m:
                    reasons.append(f"wide={w_m:.2f}>{cfg.wall_proc.door_max_width_m}")
                if h_m < cfg.wall_proc.door_min_height_m:
                    reasons.append(f"short={h_m:.2f}<{cfg.wall_proc.door_min_height_m}")
                if h_m > cfg.wall_proc.door_max_height_m:
                    reasons.append(f"tall={h_m:.2f}>{cfg.wall_proc.door_max_height_m}")
                if wall_frac > cfg.wall_proc.door_max_wall_width_frac:
                    reasons.append(f"frac={wall_frac:.0%}>{cfg.wall_proc.door_max_wall_width_frac:.0%}")

                is_door = len(reasons) == 0
                sx, sy = int(gx * effective_scale), int(gy * effective_scale)
                ex, ey = int((gx + gw - 1) * effective_scale), int((gy + gh - 1) * effective_scale)

                if is_door:
                    color = (0, 255, 0)
                    label = f"DOOR {w_m:.2f}x{h_m:.2f}m"
                else:
                    color = (0, 0, 255)
                    label = f"{w_m:.2f}x{h_m:.2f}m"

                cv2.rectangle(rgb, (sx, sy), (ex, ey), color, 2)
                cv2.putText(rgb, label, (sx + 2, sy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                if not is_door:
                    reason_str = ", ".join(reasons)
                    cv2.putText(rgb, reason_str, (sx + 2, ey + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 255), 1)

            header = np.zeros((HEADER_H, rgb.shape[1], 3), dtype=np.uint8)
            angle_deg = float(np.degrees(mw.get("angle", 0)))
            title = f"Wall {i+1} | {mw['length']:.1f}m | {angle_deg:.0f}deg | {len(combined_cloud.points)} pts"
            cv2.putText(header, title, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            panel = np.vstack([header, rgb])
            wall_panels.append(panel)

        if not wall_panels:
            continue

        max_w = max(p.shape[1] for p in wall_panels)
        padded = []
        for p in wall_panels:
            if p.shape[1] < max_w:
                pad_right = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.hstack([p, pad_right])
            padded.append(p)
            padded.append(np.zeros((PAD, max_w, 3), dtype=np.uint8))

        overview = np.vstack(padded)
        out_path = os.path.join(out_dir, f"{room_name}.png")
        cv2.imwrite(out_path, overview)
        n_doors = sum(1 for p in wall_panels for _ in [0])  # just for count
        click.echo(f"  {room_name}: {len(wall_panels)} walls → {out_path}")

    click.echo(f"\nDone. Overview images in {out_dir}")


@main.command("from-rooms")
@click.argument("room_cloud_dir", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default=None, help="Output directory (default: parent of room_cloud_dir).")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--no-sam", is_flag=True, help="Disable SAM (wall proc).")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def from_rooms(room_cloud_dir, out_dir, config_path, no_sam, verbose):
    """Run wall-seg → wall-proc → IFC export starting from room clouds folder."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(room_cloud_dir))

    from .ifc_export import build_building_json, build_ifc

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    click.echo(f"Wall segmentation + door detection + IFC export ({len(paths)} rooms)")
    building_json = build_building_json(
        paths, cfg.wall_seg, cfg.ifc_export,
        wall_proc_cfg=cfg.wall_proc, out_dir=out_dir, use_sam=not no_sam,
    )

    json_path = os.path.join(out_dir, "building.json")
    serializable = {k: v for k, v in building_json.items() if k != "_debug"}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    click.echo(f"  → JSON: {json_path}")
    click.echo(f"    {len(building_json['walls'])} walls, "
               f"{len(building_json['doors'])} doors")

    try:
        ifc_path = os.path.join(out_dir, cfg.output_ifc)
        build_ifc(
            building_json, ifc_path,
            add_floor_slabs=cfg.ifc_export.add_floor_slabs,
            slab_thickness=cfg.ifc_export.slab_thickness,
        )
        click.echo(f"  → IFC: {ifc_path}")
    except ImportError:
        click.echo("  ⚠ ifcopenshell not installed — skipping IFC export.")
        click.echo("    Install with: pip install scan2bim[ifc]")

    click.echo("Done.")


@main.command("dump-config")
@click.option("-o", "--output", default="scan2bim_config.yaml", help="Output YAML path.")
def dump_config(output):
    """Write the default configuration to a YAML file."""
    cfg = PipelineConfig()
    cfg.to_yaml(output)
    click.echo(f"Default config written to {output}")
