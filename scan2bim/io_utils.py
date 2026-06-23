"""Point-cloud IO. Ported verbatim from the original notebook's section 2.

``open3d`` is imported lazily so the rest of the package (rasters, watershed) can be
imported in environments without open3d installed.
"""

from __future__ import annotations

import numpy as np


def load_point_cloud(cfg):
    """Read the cloud, convert to METRES once, voxel-downsample.

    Returns (pcd, points) where ``pcd`` is the open3d point cloud and ``points`` is the
    Nx3 float array of its coordinates. Deterministic for a given file + voxel size, so
    re-loading in a downstream notebook reproduces exactly the same points (and therefore
    the same raster transform).
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(cfg.file_path)
    raw = np.asarray(pcd.points)
    print(f"units/meter = {cfg.units_per_meter}  (raw max extent {np.ptp(raw, axis=0).max():.2f})")
    pcd.scale(1.0 / cfg.units_per_meter, center=(0, 0, 0))
    print(f"points before clean: {len(pcd.points):,}")
    pcd = pcd.voxel_down_sample(cfg.voxel_m)
    print(f"points after voxel ({cfg.voxel_m} m): {len(pcd.points):,}")
    return pcd, np.asarray(pcd.points)
