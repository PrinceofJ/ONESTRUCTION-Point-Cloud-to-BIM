"""Top-down rasterisation (section 4) + the pixel<->world transform helpers
(``point_cells`` / ``label_points`` from section 9). Ported verbatim.

These functions are shared by several stages (the transform produced here is the single
coordinate contract for every later notebook), which is exactly why they live in the
package rather than being copied into each notebook.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

from .slab import estimate_ceiling


def rasterize_topdown(sliced_points, pixel_size, up_axis=2,
                      min_points_per_cell=1, thicken=0, max_cells=150_000_000,
                      save_path=None):
    """Binary occupancy (0=wall, 255=free) + transform dict."""
    pts = np.asarray(sliced_points, np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0
    ax_a, ax_b = [a for a in (0, 1, 2) if a != up_axis]
    a, b = pts[:, ax_a], pts[:, ax_b]
    a_px = ((a - a.min()) / pixel_size).astype(np.int64)
    b_px = ((b - b.min()) / pixel_size).astype(np.int64)
    width, height = int(a_px.max()) + 1, int(b_px.max()) + 1
    if width * height > max_cells:
        raise MemoryError(f"Grid {width*height:,} cells — raise pixel_size (unit mismatch?).")
    counts = np.zeros((height, width), np.uint16)
    np.add.at(counts, (b_px, a_px), 1)
    wall = (counts >= min_points_per_cell)[::-1, :]            # flip: world-up = image-up
    for _ in range(int(thicken)):
        w = wall.copy()
        w[1:, :] |= wall[:-1, :]; w[:-1, :] |= wall[1:, :]
        w[:, 1:] |= wall[:, :-1]; w[:, :-1] |= wall[:, 1:]
        wall = w
    occ = np.where(wall, 0, 255).astype(np.uint8)
    tf = dict(a_min=float(a.min()), b_min=float(b.min()), pixel_size=float(pixel_size),
              width=int(width), height=int(height),
              ax_a=int(ax_a), ax_b=int(ax_b), up_axis=int(up_axis))
    if save_path:
        Image.fromarray(occ).save(save_path)
    return occ, tf


def rasterize_wallness(full_points, cfg, transform):
    """Vertical-EXTENT raster from the FULL cloud, aligned to ``transform``'s grid.
    Returns a bool wall mask: column is wall if its point span covers >=
    wallness_min_span_frac of the floor->ceiling height."""
    pts = np.asarray(full_points, np.float64)
    up = cfg.up_axis
    floor_z, ceil_z = estimate_ceiling(pts[:, up], return_floor=True)
    room_h = max(1e-6, ceil_z - floor_z)
    ax_a, ax_b, ps = transform['ax_a'], transform['ax_b'], transform['pixel_size']
    H, W = transform['height'], transform['width']
    a_px = ((pts[:, ax_a] - transform['a_min']) / ps).astype(np.int64)
    b_px = ((pts[:, ax_b] - transform['b_min']) / ps).astype(np.int64)
    inb = (a_px >= 0) & (a_px < W) & (b_px >= 0) & (b_px < H)
    a_px, b_px, z = a_px[inb], b_px[inb], pts[inb, up]
    flat = b_px * W + a_px
    zmax = np.full(H * W, -np.inf); zmin = np.full(H * W, np.inf)
    np.maximum.at(zmax, flat, z); np.minimum.at(zmin, flat, z)
    span = (zmax - zmin).reshape(H, W)
    span[~np.isfinite(span)] = 0.0
    wall = span >= cfg.wallness_min_span_frac * room_h
    return wall[::-1, :]                                       # match the row flip


def rasterize_coverage(full_points, cfg, transform):
    """Bool map of where the scan actually HAS DATA below the ceiling, aligned to
    transform's grid. A cell is 'covered' if it holds >= coverage_min_pts such points.
    Used to drop 'rooms' that are really unscanned voids."""
    pts = np.asarray(full_points, np.float64); up = cfg.up_axis
    floor_z, ceil_z = estimate_ceiling(pts[:, up], return_floor=True)
    lo = floor_z - 0.2
    hi = ceil_z - cfg.coverage_ceiling_margin_m       # exclude the ceiling slab + above
    sel = (pts[:, up] >= lo) & (pts[:, up] <= hi)
    p = pts[sel]
    ax_a, ax_b, ps = transform['ax_a'], transform['ax_b'], transform['pixel_size']
    H, W = transform['height'], transform['width']
    a_px = ((p[:, ax_a] - transform['a_min']) / ps).astype(np.int64)
    b_px = ((p[:, ax_b] - transform['b_min']) / ps).astype(np.int64)
    inb = (a_px >= 0) & (a_px < W) & (b_px >= 0) & (b_px < H)
    cov = np.zeros((H, W), np.int32)
    np.add.at(cov, (b_px[inb], a_px[inb]), 1)
    covered = (cov >= int(cfg.coverage_min_pts))
    c = int(cfg.coverage_close_px)
    if c > 0:                                          # bridge furniture-shadow holes
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c + 1, 2 * c + 1))
        covered = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    return covered[::-1, :]                            # match the row flip


# ---------------------------------------------------------------------------
# pixel <-> world transform helpers (from section 9 of the original notebook)
# ---------------------------------------------------------------------------
def point_cells(points, transform):
    """Map 3-D points to (row, col) image cells under ``transform``. Returns
    (row, col, in_bounds_mask)."""
    pts = np.asarray(points, np.float64)
    ax_a, ax_b, ps = transform['ax_a'], transform['ax_b'], transform['pixel_size']
    H, W = transform['height'], transform['width']
    a_px = ((pts[:, ax_a] - transform['a_min']) / ps).astype(np.int64)
    b_px = ((pts[:, ax_b] - transform['b_min']) / ps).astype(np.int64)
    row = H - 1 - b_px; col = a_px
    inb = (col >= 0) & (col < W) & (b_px >= 0) & (b_px < H)
    return row, col, inb


def label_points(points, labels, transform):
    """Look up the label image value at each point's cell (-2 for out-of-bounds)."""
    row, col, inb = point_cells(points, transform)
    out = np.full(len(row), -2, np.int32)
    out[inb] = labels[row[inb], col[inb]]
    return out
