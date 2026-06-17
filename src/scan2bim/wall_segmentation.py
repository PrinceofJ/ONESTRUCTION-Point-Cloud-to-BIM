"""Wall segmentation: normal-direction + offset clustering, 2D flattening."""

from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import WallSegConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster_1d_gaps(values, tol):
    if len(values) == 0:
        return np.array([], dtype=int)
    order = np.argsort(values)
    sorted_v = values[order]
    labels_sorted = np.zeros(len(sorted_v), dtype=int)
    label = 0
    for i in range(1, len(sorted_v)):
        if sorted_v[i] - sorted_v[i - 1] > tol:
            label += 1
        labels_sorted[i] = label
    result = np.empty(len(values), dtype=int)
    result[order] = labels_sorted
    return result


def _find_angle_peaks(theta, n_bins=180, smooth_width=5, min_height_frac=0.08):
    counts, edges = np.histogram(theta, bins=n_bins, range=(0, np.pi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    tiled = np.concatenate([counts, counts, counts])
    smoothed_tiled = uniform_filter1d(tiled.astype(float), size=smooth_width)

    height_thr = smoothed_tiled[n_bins : 2 * n_bins].max() * min_height_frac
    min_dist = max(1, int(15 / (180 / n_bins)))

    all_peaks, _ = find_peaks(smoothed_tiled, height=height_thr, distance=min_dist)

    peak_bins: list[int] = []
    for p in all_peaks:
        bin_idx = p % n_bins
        if bin_idx not in peak_bins:
            peak_bins.append(bin_idx)

    final_bins: list[int] = []
    for b in peak_bins:
        merged = False
        for i, fb in enumerate(final_bins):
            circ_dist = min(abs(b - fb), n_bins - abs(b - fb))
            if circ_dist < min_dist:
                if smoothed_tiled[n_bins + b] > smoothed_tiled[n_bins + fb]:
                    final_bins[i] = b
                merged = True
                break
        if not merged:
            final_bins.append(b)

    smoothed = smoothed_tiled[n_bins : 2 * n_bins]
    peak_centres = centres[np.array(final_bins, dtype=int)] if final_bins else np.array([])
    return peak_centres, smoothed, centres


def _assign_to_nearest_peak(theta, peak_centres):
    labels = np.empty(len(theta), dtype=int)
    for i, t in enumerate(theta):
        dists = np.minimum(np.abs(peak_centres - t), np.pi - np.abs(peak_centres - t))
        labels[i] = np.argmin(dists)
    return labels


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def segment_walls(pcd, cfg: WallSegConfig):
    """Segment a room point cloud into individual wall clusters.

    Returns a list of dicts with 'cloud', 'normal_2d', 'offset'.
    """
    up_axis = cfg.up_axis
    if len(pcd.points) < cfg.min_wall_points:
        logger.warning("Too few points (%d) for wall segmentation.", len(pcd.points))
        return []

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius_m, max_nn=cfg.normal_max_nn,
        )
    )

    pts = np.asarray(pcd.points)
    norms = np.asarray(pcd.normals)

    up = np.zeros(3)
    up[up_axis] = 1.0
    cos_with_up = np.abs(norms @ up)
    sin_tol = np.sin(np.deg2rad(cfg.normal_tol_deg))
    vert_mask = cos_with_up < sin_tol

    pts_v = pts[vert_mask]
    norms_v = norms[vert_mask]
    logger.info(
        "Vertical-surface points: %s / %s (%.1f%%)",
        f"{len(pts_v):,}", f"{len(pts):,}", 100 * len(pts_v) / max(1, len(pts)),
    )

    if len(pts_v) < cfg.min_wall_points:
        logger.warning("Too few vertical points (%d).", len(pts_v))
        return []

    ha, hb = [a for a in range(3) if a != up_axis]
    nh = norms_v[:, [ha, hb]].copy()
    nh /= np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9
    theta = np.arctan2(nh[:, 1], nh[:, 0]) % np.pi

    peak_centres, smoothed, bin_centres = _find_angle_peaks(theta)
    logger.info(
        "Angle peaks: %d  at %s",
        len(peak_centres),
        [f"{np.degrees(p):.1f}" for p in peak_centres],
    )

    if len(peak_centres) == 0:
        logger.warning("No angle peaks found.")
        return []

    angle_labels = _assign_to_nearest_peak(theta, peak_centres)
    n_dirs = len(peak_centres)

    walls: list[dict] = []
    for a_label in range(n_dirs):
        a_mask = angle_labels == a_label
        a_pts = pts_v[a_mask]
        a_nh = nh[a_mask]

        mean_n = a_nh.mean(axis=0)
        mean_n /= np.linalg.norm(mean_n) + 1e-9

        offsets = a_pts[:, ha] * mean_n[0] + a_pts[:, hb] * mean_n[1]
        off_labels = _cluster_1d_gaps(offsets, cfg.offset_tol_m)

        for o_label in np.unique(off_labels):
            o_mask = off_labels == o_label
            wall_pts = a_pts[o_mask]
            if len(wall_pts) < cfg.min_wall_points:
                continue

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(wall_pts)
            if pcd.has_colors():
                colors_v = np.asarray(pcd.colors)[vert_mask]
                cloud.colors = o3d.utility.Vector3dVector(colors_v[a_mask][o_mask])

            walls.append({
                "cloud": cloud,
                "normal_2d": mean_n.copy(),
                "offset": float(offsets[o_mask].mean()),
            })

    # Attach diagnostic data to first wall for optional viz
    if walls:
        walls[0]["_theta_peaks"] = peak_centres
        walls[0]["_theta_smooth"] = smoothed
        walls[0]["_theta_centres"] = bin_centres

    logger.info("Walls found: %d", len(walls))
    return walls


# ---------------------------------------------------------------------------
# Wall flattening — 3D → 2D frontal images
# ---------------------------------------------------------------------------

def _density_filter(binary_img, radius=2, threshold=3):
    wall = (binary_img == 0).astype(np.float32)
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    neighbour_count = cv2.filter2D(wall, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbour_count -= wall
    isolated = (wall == 1) & (neighbour_count < threshold)
    result = binary_img.copy()
    result[isolated] = 255
    return result


def _clean_wall_image(image, close_px=3, open_px=2,
                      density_radius=0, density_threshold=3):
    if density_radius > 0:
        image = _density_filter(image, radius=density_radius, threshold=density_threshold)
    if close_px <= 0 and open_px <= 0:
        return image
    wall = 255 - image
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, k)
    return 255 - wall


def flatten_wall(wall, cfg: WallSegConfig):
    """Project a wall's 3D points onto its plane and rasterize to a 2D image.

    Returns a dict with 'image', 'image_raw', coordinate frame info, etc.
    """
    pts = np.asarray(wall["cloud"].points)
    n2d = wall["normal_2d"]
    n_pts_raw = len(pts)
    up_axis = cfg.up_axis

    if cfg.sor_neighbours > 0 and len(pts) > cfg.sor_neighbours:
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(pts)
        _, inlier_idx = wall_pcd.remove_statistical_outlier(
            nb_neighbors=cfg.sor_neighbours, std_ratio=cfg.sor_std_ratio,
        )
        pts = pts[inlier_idx]

    n_pts_clean = len(pts)

    up = np.zeros(3)
    up[up_axis] = 1.0
    ha, hb = [a for a in range(3) if a != up_axis]
    n3d = np.zeros(3)
    n3d[ha] = n2d[0]
    n3d[hb] = n2d[1]
    n3d /= np.linalg.norm(n3d) + 1e-9

    u_axis = np.cross(up, n3d)
    u_axis /= np.linalg.norm(u_axis) + 1e-9
    v_axis = up.copy()

    u = pts @ u_axis
    v = pts @ v_axis
    u_min, v_min = u.min(), v.min()
    u_shifted = u - u_min
    v_shifted = v - v_min

    pixel_m = cfg.flat_pixel_m
    u_px = (u_shifted / pixel_m).astype(np.int64)
    v_px = (v_shifted / pixel_m).astype(np.int64)
    width = int(u_px.max()) + 1
    height = int(v_px.max()) + 1

    counts = np.zeros((height, width), dtype=np.uint16)
    np.add.at(counts, (v_px, u_px), 1)
    occupied = counts >= cfg.min_pts_per_cell
    occupied = occupied[::-1, :]
    image_raw = np.where(occupied, 0, 255).astype(np.uint8)

    image = _clean_wall_image(
        image_raw, close_px=cfg.morph_close_px, open_px=cfg.morph_open_px,
        density_radius=cfg.density_filter_radius,
        density_threshold=cfg.density_filter_threshold,
    )

    origin = np.zeros(3)
    origin += u_min * u_axis + v_min * v_axis

    return {
        "image": image,
        "image_raw": image_raw,
        "u": u, "v": v,
        "u_axis": u_axis, "v_axis": v_axis,
        "origin": origin,
        "pixel_m": pixel_m,
        "width_m": float(u_shifted.max()),
        "height_m": float(v_shifted.max()),
        "n_pts_raw": n_pts_raw,
        "n_pts_clean": n_pts_clean,
    }


def save_wall_images(walls, room_name: str, out_dir: str, cfg: WallSegConfig):
    """Flatten every wall and save as clean PNGs. Returns list of flat dicts."""
    import json

    room_dir = os.path.join(out_dir, room_name)
    os.makedirs(room_dir, exist_ok=True)

    flats = []
    wall_meta = []
    for i, w in enumerate(walls):
        flat = flatten_wall(w, cfg)
        flats.append(flat)
        img_path = os.path.join(room_dir, f"wall_{i + 1:02d}.png")
        Image.fromarray(flat["image"]).save(img_path)
        wall_meta.append({
            "name": f"wall_{i + 1:02d}",
            "normal_2d": [float(w["normal_2d"][0]), float(w["normal_2d"][1])],
            "offset": float(w["offset"]),
            "u_min": float(flat["u"].min()),
            "u_max": float(flat["u"].max()),
            "width_m": flat["width_m"],
            "height_m": flat["height_m"],
        })

    meta_path = os.path.join(room_dir, "wall_meta.json")
    with open(meta_path, "w") as f:
        json.dump(wall_meta, f, indent=2)

    logger.info("  %s/: saved %d wall images to %s", room_name, len(flats), room_dir)
    return flats


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def run_wall_segmentation(room_cloud_paths: list[str], cfg: WallSegConfig, out_dir: str):
    """Run wall segmentation on all room clouds.

    Returns dict mapping room_name -> list of flat dicts.
    """
    from .io import load_room_cloud

    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for room_path in room_cloud_paths:
        fname = os.path.splitext(os.path.basename(room_path))[0]
        room_name = fname.replace("_walls", "")

        logger.info("Processing %s", room_name)
        try:
            room_pcd, _ = load_room_cloud(room_path, voxel_m=cfg.voxel_m)
            walls = segment_walls(room_pcd, cfg)
            if walls:
                flats = save_wall_images(walls, room_name, out_dir, cfg)
                all_results[room_name] = flats
            else:
                logger.info("  %s: no walls segmented, skipping", room_name)
        except Exception as e:
            logger.error("  %s: ERROR %s", room_name, e)

    total = sum(len(v) for v in all_results.values())
    logger.info("Done — %d rooms, %d total wall images", len(all_results), total)
    return all_results
