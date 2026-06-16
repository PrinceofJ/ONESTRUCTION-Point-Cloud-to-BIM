"""Point cloud I/O utilities."""

from __future__ import annotations

import glob
import os
import logging

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def load_point_cloud(
    file_path: str,
    units_per_meter: float = 1.0,
    voxel_m: float = 0.02,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Read a point cloud, convert to metres, and voxel-downsample."""
    pcd = o3d.io.read_point_cloud(file_path)
    raw = np.asarray(pcd.points)
    logger.info(
        "units/meter = %s  (raw max extent %.2f)",
        units_per_meter,
        np.ptp(raw, axis=0).max(),
    )
    pcd.scale(1.0 / units_per_meter, center=(0, 0, 0))
    logger.info("points before clean: %s", f"{len(pcd.points):,}")
    if voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_m)
    logger.info("points after voxel (%s m): %s", voxel_m, f"{len(pcd.points):,}")
    return pcd, np.asarray(pcd.points)


def load_room_cloud(
    file_path: str,
    voxel_m: float = 0.02,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Read a single room point cloud and downsample."""
    pcd = o3d.io.read_point_cloud(file_path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError(f"Empty cloud: {file_path}")
    logger.info("Loaded %s points from %s", f"{len(pts):,}", os.path.basename(file_path))
    if voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_m)
        logger.info("After voxel downsample (%s m): %s", voxel_m, f"{len(pcd.points):,}")
    return pcd, np.asarray(pcd.points)


def save_cloud(pcd: o3d.geometry.PointCloud, out_path: str) -> str:
    """Write a point cloud to disk."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = o3d.io.write_point_cloud(out_path, pcd)
    if not ok:
        raise IOError(f"open3d failed to write {out_path!r}")
    logger.info("Wrote %s  (%s points)", out_path, f"{len(pcd.points):,}")
    return out_path


def _read_xyzrgb_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a space-delimited XYZRGB text file."""
    try:
        import pandas as pd

        arr = pd.read_csv(path, sep=" ", header=None, dtype=np.float32, engine="c").to_numpy()
    except Exception:
        rows = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
        arr = np.asarray(rows, dtype=np.float32)

    if arr.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    xyz = arr[:, :3].astype(np.float64)
    rgb = np.clip(arr[:, 3:6] / 255.0, 0.0, 1.0)
    return xyz, rgb


def txt_to_cloud(path: str, with_color: bool = True) -> o3d.geometry.PointCloud:
    """Convert a single room .txt file to an Open3D PointCloud."""
    xyz, rgb = _read_xyzrgb_txt(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if with_color and len(rgb):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def area_to_cloud(
    area_dir: str,
    with_color: bool = True,
    voxel_size: float | None = None,
) -> o3d.geometry.PointCloud:
    """Load all room .txt files in a Stanford S3DIS area directory."""
    files = glob.glob(os.path.join(area_dir, "*", "*.txt"))
    room_files = sorted(
        p
        for p in files
        if os.path.basename(os.path.dirname(p)) == os.path.splitext(os.path.basename(p))[0]
    )
    if not room_files:
        raise FileNotFoundError(
            f"No room .txt files found under {area_dir!r}. "
            f"Point this at an Area_N folder of the Aligned_Version."
        )

    all_xyz, all_rgb = [], []
    for p in room_files:
        xyz, rgb = _read_xyzrgb_txt(p)
        if len(xyz):
            all_xyz.append(xyz)
            all_rgb.append(rgb)
        logger.info("  %s  %s pts", os.path.basename(p), f"{len(xyz):>10,}")

    xyz = np.vstack(all_xyz)
    rgb = np.vstack(all_rgb)
    logger.info("Stacked %d rooms -> %s points total", len(room_files), f"{len(xyz):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if with_color:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
        logger.info("After voxel_down_sample(%s): %s points", voxel_size, f"{len(pcd.points):,}")
    return pcd
