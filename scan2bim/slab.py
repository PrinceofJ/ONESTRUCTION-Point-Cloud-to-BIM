"""Slab extraction: horizontal crop at wall height."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def estimate_ceiling(h, bins=256, rel_thresh=0.02, return_floor=False):
    """Histogram-based floor/ceiling height estimate."""
    h = np.asarray(h, np.float64)
    counts, edges = np.histogram(h, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = np.flatnonzero(counts >= counts.max() * rel_thresh)
    floor_z, ceiling_z = centers[keep[0]], centers[keep[-1]]
    return (floor_z, ceiling_z) if return_floor else ceiling_z


def estimate_local_ceilings(points, up_axis=2, cell_size_m=1.0,
                            min_pts_per_cell=20, smooth_cells=1):
    """Per-point ceiling height from a grid of local estimates."""
    pts = np.asarray(points, np.float64); h = pts[:, up_axis]
    ax_a, ax_b = [a for a in (0, 1, 2) if a != up_axis]
    a, b = pts[:, ax_a], pts[:, ax_b]
    g = estimate_ceiling(h)
    ai = np.floor((a - a.min()) / cell_size_m).astype(int)
    bi = np.floor((b - b.min()) / cell_size_m).astype(int)
    na, nb = int(ai.max()) + 1, int(bi.max()) + 1
    grid = np.full((na, nb), g, np.float64)
    flat = ai * nb + bi
    order = np.argsort(flat, kind='stable')
    fs, hs = flat[order], h[order]
    bnd = np.flatnonzero(np.diff(fs)) + 1
    for s, e in zip(np.concatenate([[0], bnd]), np.concatenate([bnd, [len(fs)]])):
        if e - s >= min_pts_per_cell:
            x, y = divmod(int(fs[s]), nb)
            grid[x, y] = estimate_ceiling(hs[s:e])
    if smooth_cells and smooth_cells > 1:
        grid = ndimage.median_filter(grid, size=int(smooth_cells))
    return grid[ai, bi]


def crop_vertical(points, cfg, debug=False, return_info=False):
    """Keep points in a horizontal slab. Returns (slab_points, mask[, info])."""
    pts = np.asarray(points, np.float64); h = pts[:, cfg.up_axis]
    mode = cfg.slab_relative_to
    if mode == 'absolute':
        keep_lo, keep_hi, ref = cfg.slab_lo_m, cfg.slab_hi_m, None
    elif mode == 'floor':
        floor_z, _ = estimate_ceiling(h, return_floor=True)
        keep_lo, keep_hi, ref = floor_z + cfg.slab_lo_m, floor_z + cfg.slab_hi_m, floor_z
    else:  # ceiling
        cm = getattr(cfg, 'ceiling_mode', 'global')
        if cm == 'global':
            ref = estimate_ceiling(h)
        else:
            sc = 1 if cm == 'local_perpoint' else int(getattr(cfg, 'ceiling_smooth_cells', 3))
            ref = estimate_local_ceilings(pts, cfg.up_axis, cfg.ceiling_cell_size_m,
                                          cfg.ceiling_min_pts_per_cell, smooth_cells=sc)
        keep_hi = ref - cfg.slab_lo_m       # shallower cut (closer to ceiling)
        keep_lo = ref - cfg.slab_hi_m       # deeper cut
    mask = (h >= keep_lo) & (h <= keep_hi)
    if debug:
        klo = keep_lo if np.isscalar(keep_lo) else float(np.mean(keep_lo))
        khi = keep_hi if np.isscalar(keep_hi) else float(np.mean(keep_hi))
        extra = ''
        if mode == 'ceiling' and not np.isscalar(ref):
            extra = f"  local ceiling range [{float(np.min(ref)):.2f},{float(np.max(ref)):.2f}] m"
        print(f"[crop_vertical] mode={mode}  band(m,mean)=[{klo:.2f},{khi:.2f}]  "
              f"kept {int(mask.sum()):,}/{len(pts):,} ({100*mask.mean():.1f}%){extra}")
    info = dict(mode=mode, ref=ref, keep_lo=keep_lo, keep_hi=keep_hi)
    return (pts[mask], mask, info) if return_info else (pts[mask], mask)
