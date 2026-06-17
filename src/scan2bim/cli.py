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
    from .wall_segmentation import run_wall_segmentation
    from .wall_image_processing import run_wall_image_processing
    from .ifc_export import build_building_json, build_ifc

    click.echo(f"[1/4] Room segmentation: {input_path}")
    room_cloud_dir = os.path.join(out_dir, "room_clouds")
    pcd, points = load_point_cloud(
        input_path,
        units_per_meter=cfg.room_seg.units_per_meter,
        voxel_m=cfg.room_seg.voxel_m,
    )
    labels, rooms, tf = run_room_segmentation(pcd, points, cfg.room_seg, room_cloud_dir)
    n_rooms = len([r for r in rooms if len(r["points"]) > 0])
    click.echo(f"  → {n_rooms} rooms")

    click.echo("[2/4] Wall segmentation")
    wall_image_dir = os.path.join(out_dir, "wall_images")
    room_cloud_paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    wall_results = run_wall_segmentation(room_cloud_paths, cfg.wall_seg, wall_image_dir)
    total_walls = sum(len(v) for v in wall_results.values())
    click.echo(f"  → {total_walls} wall images across {len(wall_results)} rooms")

    click.echo("[3/4] Wall image processing (door/window detection)")
    openings_dir = os.path.join(out_dir, "openings")
    run_wall_image_processing(wall_image_dir, cfg.wall_proc, openings_dir, use_sam=not no_sam)

    click.echo("[4/4] IFC export")
    building_json = build_building_json(room_cloud_paths, openings_dir, cfg.wall_seg, cfg.ifc_export,
                                           wall_image_dir=wall_image_dir)

    json_path = os.path.join(out_dir, "building.json")
    serializable = {k: v for k, v in building_json.items() if k != "_debug"}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    click.echo(f"  → JSON: {json_path}")
    click.echo(f"    {len(building_json['walls'])} walls, "
               f"{len(building_json['doors'])} doors, "
               f"{len(building_json['windows'])} windows")

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
@click.option("--openings-dir", default=None, help="Openings JSON directory.")
@click.option("-o", "--output", default="model.ifc", help="Output IFC file path.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("-v", "--verbose", is_flag=True)
def ifc_export(room_cloud_dir, openings_dir, output, config_path, verbose):
    """Generate IFC model from room clouds + openings."""
    _setup_logging(verbose)
    cfg = _load_config(config_path)

    from .ifc_export import build_building_json, build_ifc

    paths = sorted(glob.glob(os.path.join(room_cloud_dir, "room_*_walls.ply")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(room_cloud_dir, "*.ply")))
    if not paths:
        click.echo(f"No .ply files found in {room_cloud_dir}", err=True)
        sys.exit(1)

    building_json = build_building_json(paths, openings_dir, cfg.wall_seg, cfg.ifc_export)

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
@click.option("--openings-dir", default=None, help="Openings JSON directory.")
@click.option("-c", "--config", "config_path", default=None, help="YAML config file.")
@click.option("--save", "save_path", default=None, help="Save debug plot to file instead of showing.")
@click.option("-v", "--verbose", is_flag=True)
def debug_walls(room_cloud_dir, openings_dir, config_path, save_path, verbose):
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
    building_json = build_building_json(paths, openings_dir, cfg.wall_seg, cfg.ifc_export)

    show_wall_detail_table(building_json)
    show_wall_debug(building_json, save_path=save_path)


@main.command("dump-config")
@click.option("-o", "--output", default="scan2bim_config.yaml", help="Output YAML path.")
def dump_config(output):
    """Write the default configuration to a YAML file."""
    cfg = PipelineConfig()
    cfg.to_yaml(output)
    click.echo(f"Default config written to {output}")
