"""Tests for the cross-stage guards (D5 / §3.4): ``assert_upstream_config`` and
``assert_points_in_grid``. These are what make the §0 bug (one cloud back-projected through
another cloud's grid) impossible to run past.
"""

import numpy as np
import pytest

import scan2bim
from scan2bim import Config


def _grid(pixel=0.1, n=10):
    """A simple square transform covering world [0, n*pixel) on both in-plane axes."""
    return dict(a_min=0.0, b_min=0.0, pixel_size=pixel, width=n, height=n,
                ax_a=0, ax_b=1, up_axis=2)


# --------------------------------------------------------------- assert_upstream_config
def test_matching_config_passes():
    cfg = Config(pixel_m=0.03, voxel_m=0.02)
    scan2bim.assert_upstream_config(cfg, cfg.to_dict())     # no raise


def test_mismatched_geometry_field_raises():
    cfg = Config(pixel_m=0.03)
    upstream = cfg.to_dict()
    upstream['pixel_m'] = 0.05                              # produced at a different resolution
    with pytest.raises(ValueError, match='pixel_m'):
        scan2bim.assert_upstream_config(cfg, upstream)


def test_different_cloud_basename_raises():
    cfg = Config(file_path='/here/data/area1.xyz')
    upstream = cfg.to_dict()
    upstream['file_path'] = '/there/data/apt_subsampled.ply'   # a *different* cloud (the §0 bug)
    with pytest.raises(ValueError, match='file_path'):
        scan2bim.assert_upstream_config(cfg, upstream)


def test_same_cloud_different_directory_passes():
    cfg = Config(file_path='/machineA/proj/data/area1.xyz')
    upstream = cfg.to_dict()
    upstream['file_path'] = '/machineB/elsewhere/data/area1.xyz'   # same cloud, moved repo
    scan2bim.assert_upstream_config(cfg, upstream)          # basename matches -> no raise


# --------------------------------------------------------------- assert_points_in_grid
def test_points_inside_grid_pass():
    tf = _grid()
    pts = np.random.default_rng(0).uniform(0.05, 0.95, size=(500, 3))   # all in-bounds
    frac = scan2bim.assert_points_in_grid(pts, tf)
    assert frac == pytest.approx(1.0)


def test_points_out_of_grid_raise():
    tf = _grid()
    pts = np.random.default_rng(0).uniform(0, 1, size=(500, 3)) + 100.0  # shifted far away
    with pytest.raises(ValueError, match='inside the upstream grid'):
        scan2bim.assert_points_in_grid(pts, tf)


def test_mostly_out_of_bounds_raises():
    """The exact §0 ratio: ~4 % in-bounds must fail the >= 50 % default."""
    tf = _grid(n=10)                                        # world [0, 1)
    rng = np.random.default_rng(1)
    inb = rng.uniform(0, 1, size=(40, 3))                  # ~4 %
    out = rng.uniform(0, 1, size=(960, 3)) + 50.0          # ~96 %
    pts = np.vstack([inb, out])
    with pytest.raises(ValueError):
        scan2bim.assert_points_in_grid(pts, tf)
