"""Phase-0 safety net: run the local pipeline (N1 -> N2 -> N3 logic) end-to-end on a fixed
synthetic cloud and assert the room count + per-room wall-point totals stay within tolerance
of a stored baseline.

This is the gate that proves the *plumbing* refactor changed **no algorithm**: the refactor
must not touch ``raster.py`` / ``watershed.py`` / ``walls.py``, so for a fixed input these
numbers must not move. The baseline was recorded from the pipeline functions directly (the
same calls the notebooks make) and the identical totals come out of the real notebook cells.
"""

import os
import tempfile

import numpy as np
import pytest

import scan2bim
from scan2bim import Config

from tests._synth import write_synthetic_xyz

# Baseline recorded on the fixed synthetic cloud (seed 0). Room count is exact; wall-point
# totals are allowed a small tolerance for cross-platform open3d/voxel rounding  - an actual
# algorithm change would move these by far more than the tolerance.
BASELINE_ROOMS = 2
BASELINE_WALL_POINTS = {1: 41508, 2: 41316}
REL_TOL = 0.05


def _run_local_pipeline(cfg):
    """N1 -> N2 -> N3 as plain function calls (no notebook, no disk staging)."""
    _, pts = scan2bim.load_point_cloud(cfg)
    slab_pts, _, _ = scan2bim.crop_vertical(pts, cfg, return_info=True)
    occ, tf = scan2bim.rasterize_topdown(
        slab_pts, cfg.pixel_m, up_axis=cfg.up_axis,
        min_points_per_cell=cfg.min_points_per_cell, thicken=cfg.thicken_px)
    wall_mask = (occ == 0)
    wallness = scan2bim.rasterize_wallness(pts, cfg, tf)
    coverage = scan2bim.rasterize_coverage(pts, cfg, tf)

    seg_input = wallness if cfg.use_wallness else wall_mask
    labels = scan2bim.segment_rooms_watershed(
        seg_input, cfg.pixel_m, marker_h_m=cfg.marker_h_m,
        footprint_close_m=cfg.footprint_close_m, merge_ridge_m=cfg.merge_ridge_m,
        min_room_area_m2=cfg.min_room_area_m2, min_wall_area_px=cfg.min_wall_area_px,
        door_seal_px=cfg.seal_gap_px, coverage=coverage,
        min_coverage_frac=cfg.min_coverage_frac)

    wall_masks = scan2bim.room_wall_masks_boundary_ring(labels, wallness, cfg)
    band, _, _ = scan2bim.height_band_mask(pts, cfg, tf)
    rooms3d = scan2bim.backproject_room_masks(pts, wall_masks, tf, keep_mask=band)
    return labels, rooms3d


@pytest.fixture(scope='module')
def pipeline_result():
    with tempfile.TemporaryDirectory() as d:
        xyz = write_synthetic_xyz(os.path.join(d, 'synth.xyz'))
        cfg = Config(file_path=xyz, units_per_meter=1.0)
        yield _run_local_pipeline(cfg)


def test_room_count_matches_baseline(pipeline_result):
    labels, _ = pipeline_result
    n_rooms = len([r for r in np.unique(labels) if r >= 1])
    assert n_rooms == BASELINE_ROOMS


def test_every_room_has_nontrivial_walls(pipeline_result):
    """The §0 failure was rooms coming back with ~0 back-projected wall points."""
    _, rooms3d = pipeline_result
    assert rooms3d, 'no rooms produced'
    for e in rooms3d:
        assert len(e['points']) > 0, f"room {e['room_id']} has 0 wall points"


def test_wall_point_totals_within_tolerance(pipeline_result):
    _, rooms3d = pipeline_result
    got = {e['room_id']: len(e['points']) for e in rooms3d}
    assert set(got) == set(BASELINE_WALL_POINTS), got
    for rid, base in BASELINE_WALL_POINTS.items():
        assert got[rid] == pytest.approx(base, rel=REL_TOL), (rid, got[rid], base)
    total, base_total = sum(got.values()), sum(BASELINE_WALL_POINTS.values())
    assert total == pytest.approx(base_total, rel=REL_TOL)
