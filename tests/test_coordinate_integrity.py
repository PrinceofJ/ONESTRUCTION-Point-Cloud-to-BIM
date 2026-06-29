"""Tests for the coordinate-integrity helpers (research-fixes Task 01):
``scan2bim.grid_world_bbox`` and ``scan2bim.interior_coverage_fraction``, plus the
``gt_dir`` config field resolution. These back the GT frame-alignment gate
(``gt_raster.ipynb``) and the structural-only-cloud warning (Notebook 1).
"""

import os

import numpy as np
import pytest

import scan2bim
from scan2bim import Config


# --------------------------------------------------------------- grid_world_bbox
def test_grid_world_bbox_basic():
    tf = dict(a_min=-1.0, b_min=2.0, pixel_size=0.1, width=10, height=20,
              ax_a=0, ax_b=1, up_axis=2)
    a0, b0, a1, b1 = scan2bim.grid_world_bbox(tf)
    assert (a0, b0) == (-1.0, 2.0)
    assert a1 == pytest.approx(-1.0 + 10 * 0.1)     # a_min + width*pixel
    assert b1 == pytest.approx(2.0 + 20 * 0.1)      # b_min + height*pixel


def test_grid_world_bbox_returns_floats():
    tf = dict(a_min=0, b_min=0, pixel_size=1, width=3, height=4, ax_a=0, ax_b=1, up_axis=2)
    bbox = scan2bim.grid_world_bbox(tf)
    assert all(isinstance(v, float) for v in bbox)


# ------------------------------------------------- interior_coverage_fraction
def _hollow_box(n=20, margin=3):
    """A wall mask that is the outline of a rectangle (hollow box) on an n x n grid."""
    wall = np.zeros((n, n), bool)
    wall[margin, margin:n - margin] = True
    wall[n - margin - 1, margin:n - margin] = True
    wall[margin:n - margin, margin] = True
    wall[margin:n - margin, n - margin - 1] = True
    return wall


def test_full_cloud_high_interior_coverage():
    wall = _hollow_box()
    coverage = np.ones_like(wall)                   # scanned everywhere (full cloud)
    frac = scan2bim.interior_coverage_fraction(coverage, wall)
    assert frac == pytest.approx(1.0)


def test_structural_only_near_zero_interior_coverage():
    wall = _hollow_box()
    coverage = wall.copy()                          # data ONLY on structure (no interior points)
    frac = scan2bim.interior_coverage_fraction(coverage, wall)
    assert frac == pytest.approx(0.0)


def test_partial_interior_coverage_is_a_fraction():
    wall = _hollow_box()
    coverage = np.zeros_like(wall)
    coverage[8:12, :] = True                         # a band of scanned interior
    frac = scan2bim.interior_coverage_fraction(coverage, wall)
    assert 0.0 < frac < 1.0


def test_no_interior_returns_zero():
    wall = np.ones((10, 10), bool)                   # solid -> fill adds nothing -> empty interior
    coverage = np.ones((10, 10), bool)
    assert scan2bim.interior_coverage_fraction(coverage, wall) == 0.0


def test_close_px_bridges_a_doorway_gap():
    """A doorway gap in the wall would let binary_fill_holes leak to the exterior; closing the
    wall first restores a sealed footprint, so interior coverage stays high."""
    wall = _hollow_box()
    wall[10, 3] = False                              # punch a 1-px doorway in the left wall
    coverage = np.ones_like(wall)
    frac_closed = scan2bim.interior_coverage_fraction(coverage, wall, close_px=2)
    assert frac_closed == pytest.approx(1.0)


# ------------------------------------------------------------------- gt_dir config
def test_gt_dir_resolved_absolute(tmp_path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "tmp"\n')
    (tmp_path / 'params.yaml').write_text(
        'input:\n  file_path: data/scan.xyz\ngroundtruth:\n  gt_dir: data/GT\n')
    cfg = scan2bim.load_config(start=str(tmp_path))
    assert os.path.isabs(cfg.gt_dir)
    assert cfg.gt_dir == os.path.normpath(os.path.join(str(tmp_path), 'data', 'GT'))


def test_gt_dir_defaults_when_omitted(tmp_path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "tmp"\n')
    (tmp_path / 'params.yaml').write_text('raster:\n  pixel_m: 0.03\n')
    cfg = scan2bim.load_config(start=str(tmp_path))
    # default 'data/Area_1' resolved under the root
    assert cfg.gt_dir == os.path.normpath(os.path.join(str(tmp_path), Config().gt_dir))
