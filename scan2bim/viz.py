"""Debug / QA visualisations (section 10 of the original), ported verbatim, plus one
new plot (``show_wall_assignment``) for the boundary-ring wall masks.

``supervision`` is imported lazily inside the annotator so the module imports without it.
All functions are optional QA helpers; the notebooks call them but none of the pipeline
logic depends on them.
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .slab import estimate_ceiling, estimate_local_ceilings
from .walls import room_footprints


def get_color_palette():
    import supervision as sv
    return sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"])


def annotate(image, detections, label=None):
    import supervision as sv
    ma = sv.MaskAnnotator(color=get_color_palette(),
                          color_lookup=sv.ColorLookup.INDEX, opacity=0.6)
    return ma.annotate(image.copy(), detections)


def _subsample(n, k=150_000):
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(0).choice(n, k, replace=False)


def colorize_labels(labels):
    H, W = labels.shape
    rgb = np.ones((H, W, 3))
    rgb[labels == -1] = (0, 0, 0)
    rgb[labels == 0] = (0.93, 0.93, 0.93)
    cmap = plt.get_cmap('tab20')
    for k, r in enumerate([int(x) for x in np.unique(labels) if x >= 1]):
        rgb[labels == r] = cmap(k % 20)[:3]
    return rgb


def _annotate_ids(ax, labels):
    for r in [int(x) for x in np.unique(labels) if x >= 1]:
        ys, xs = np.where(labels == r)
        ax.text(xs.mean(), ys.mean(), str(r), color='k', fontsize=9, ha='center',
                va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))


def _color_basins(ws, ext_id):
    H, W = ws.shape
    rgb = np.ones((H, W, 3))
    rng = np.random.default_rng(1)
    for v in [int(x) for x in np.unique(ws) if x > 0]:
        rgb[ws == v] = (0.85, 0.85, 0.85) if v == ext_id else (rng.random(3) * 0.7 + 0.15)
    return rgb


def show_topdown(points, up_axis, title='top-down'):
    pts = np.asarray(points)
    aa, bb = [a for a in (0, 1, 2) if a != up_axis]
    s = _subsample(len(pts), 200_000)
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(pts[s, aa], pts[s, bb], s=0.5, c=pts[s, up_axis], cmap='viridis')
    plt.colorbar(sc, label='height (m)'); plt.gca().set_aspect('equal')
    plt.title(title); plt.tight_layout(); plt.show()


def show_slab_debug(points, slab_mask, up_axis, info=None):
    pts = np.asarray(points); h = pts[:, up_axis]; kept = pts[slab_mask]
    aa, bb = [a for a in (0, 1, 2) if a != up_axis]
    nm = {0: 'X', 1: 'Y', 2: 'Z'}
    lo, hi = kept[:, up_axis].min(), kept[:, up_axis].max()
    sa = _subsample(len(pts)); sk = _subsample(len(kept))

    def arr(key):
        if info is None or key not in info or info[key] is None or np.isscalar(info[key]):
            return None
        return np.asarray(info[key])
    ref, klo, khi = arr('ref'), arr('keep_lo'), arr('keep_hi')

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    ax[0, 0].hist(h, bins=200, color='steelblue')
    ax[0, 0].axvspan(lo, hi, color='orange', alpha=0.35)
    ax[0, 0].axvline(lo, color='r', lw=1); ax[0, 0].axvline(hi, color='r', lw=1)
    ax[0, 0].set_title(f'height histogram + slab band (mean [{lo:.2f}, {hi:.2f}] m)')
    ax[0, 0].set_xlabel(f'{nm[up_axis]} (m)'); ax[0, 0].set_ylabel('point count')

    ax[0, 1].scatter(pts[sa, aa], pts[sa, bb], s=0.5, c='lightgray')
    ax[0, 1].scatter(kept[sk, aa], kept[sk, bb], s=0.5, c='crimson')
    ax[0, 1].set_aspect('equal'); ax[0, 1].set_title('top-down (slab red over all grey)')
    ax[0, 1].set_xlabel(nm[aa]); ax[0, 1].set_ylabel(nm[bb])

    for col, axis in ((0, aa), (1, bb)):
        A = ax[1, col]
        A.scatter(pts[sa, axis], pts[sa, up_axis], s=0.5, c='lightgray')
        A.scatter(kept[sk, axis], kept[sk, up_axis], s=0.6, c='crimson')
        if ref is not None:
            A.scatter(pts[sa, axis], ref[sa], s=0.4, c='#1f6fb4', alpha=0.5)
        if klo is not None and khi is not None:
            A.scatter(pts[sa, axis], khi[sa], s=0.3, c='orange', alpha=0.5)
            A.scatter(pts[sa, axis], klo[sa], s=0.3, c='orange', alpha=0.5)
        else:
            A.axhspan(lo, hi, color='orange', alpha=0.12)
            A.axhline(lo, color='r', lw=0.6); A.axhline(hi, color='r', lw=0.6)
        A.set_title(f'side: {nm[axis]} vs {nm[up_axis]}  (blue = ceiling, orange = band)')
        A.set_xlabel(nm[axis]); A.set_ylabel(f'{nm[up_axis]} (m)')
    plt.tight_layout(); plt.show()


def show_ceiling_map(points, cfg):
    pts = np.asarray(points, np.float64); up = cfg.up_axis
    aa, bb = [a for a in (0, 1, 2) if a != up]; nm = {0: 'X', 1: 'Y', 2: 'Z'}
    ceil_pp = estimate_local_ceilings(pts, up, cfg.ceiling_cell_size_m,
                                      cfg.ceiling_min_pts_per_cell, smooth_cells=1)
    g = estimate_ceiling(pts[:, up])
    s = _subsample(len(pts), 200_000)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sc = ax[0].scatter(pts[s, aa], pts[s, bb], s=1, c=ceil_pp[s], cmap='viridis')
    ax[0].set_aspect('equal'); fig.colorbar(sc, ax=ax[0], label='local ceiling height (m)')
    ax[0].set_title(f'local ceiling map (global = {g:.2f} m) — lower patches = dropped lips')
    ax[0].set_xlabel(nm[aa]); ax[0].set_ylabel(nm[bb])
    ax[1].hist(ceil_pp, bins=120, color='teal')
    ax[1].axvline(g, color='r', lw=1, label=f'global {g:.2f} m')
    ax[1].set_title('ceiling-height distribution (separate peaks = separate heights)')
    ax[1].set_xlabel('ceiling height (m)'); ax[1].set_ylabel('point count'); ax[1].legend()
    plt.tight_layout(); plt.show()


def show_raster_debug(wall_mask):
    free = ~wall_mask
    dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(wall_mask, cmap='gray_r'); ax[0].set_title('wall mask (black = wall)')
    im = ax[1].imshow(dt, cmap='magma'); ax[1].set_title('distance transform (px to wall)')
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.axis('off')
    plt.tight_layout(); plt.show()


def show_watershed_internals(aux, labels, pixel_m):
    walls, dt, foot = aux['walls'], aux['dt'], aux['footprint']
    seeds, ws, ext = aux['seeds'], aux['ws'], aux['exterior']; ext_id = aux['ext_id']
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].imshow(walls, cmap='gray_r'); ax[0, 0].set_title('1 · cleaned walls')
    im = ax[0, 1].imshow(dt, cmap='magma'); ax[0, 1].set_title('2 · distance transform')
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[0, 2].imshow(dt, cmap='gray')
    sd = cv2.dilate(((seeds > 0) & (seeds != ext_id)).astype(np.uint8),
                    np.ones((5, 5), np.uint8)).astype(bool)
    ov = np.zeros((*seeds.shape, 4)); ov[sd] = (1, 0, 0, 1)
    ax[0, 2].imshow(ov); ax[0, 2].set_title('3 · room seeds (h-maxima)')
    fp = np.zeros((*foot.shape, 3)); fp[foot] = (0.2, 0.5, 1.0); fp[ext] = (1.0, 0.6, 0.1)
    ax[1, 0].imshow(fp); ax[1, 0].set_title('4 · footprint (blue indoor / orange exterior)')
    ax[1, 1].imshow(_color_basins(ws, ext_id)); ax[1, 1].set_title('5 · raw basins (pre-merge)')
    ax[1, 2].imshow(colorize_labels(labels)); _annotate_ids(ax[1, 2], labels)
    nr = len([r for r in np.unique(labels) if r >= 1])
    ax[1, 2].set_title(f'6 · final rooms: {nr}')
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout(); plt.show()


def show_sam_debug(aux, geom_labels, fused_labels):
    walls, foot = aux['walls'], aux['footprint']
    residual = (~walls) & (geom_labels == 0) & foot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(colorize_labels(geom_labels)); _annotate_ids(ax[0], geom_labels)
    ax[0].set_title('Pass 1 — geometry')
    r = np.zeros((*residual.shape, 3)); r[walls] = (0, 0, 0); r[residual] = (1, 0.45, 0)
    ax[1].imshow(r); ax[1].set_title('residual free space (SAM searches here)')
    ax[2].imshow(colorize_labels(fused_labels)); _annotate_ids(ax[2], fused_labels)
    ax[2].set_title('refined — geometry + SAM')
    for a in ax:
        a.axis('off')
    plt.tight_layout(); plt.show()


def show_room_footprints(labels, do_buffer_px):
    """LEGACY overlay: which wall columns the *old* raster-overlap method would export."""
    foot = room_footprints(labels, margin=do_buffer_px, walls_only=True)
    H, W = labels.shape; rgb = np.ones((H, W, 3)); rgb[labels == -1] = (0.85, 0.85, 0.85)
    cmap = plt.get_cmap('tab20')
    for k, (rid, m) in enumerate(sorted(foot.items())):
        rgb[m] = cmap(k % 20)[:3]
    fig, axx = plt.subplots(figsize=(9, 7)); axx.imshow(rgb)
    _annotate_ids(axx, labels)
    axx.set_title('LEGACY per-room wall columns (raster overlap)')
    axx.axis('off'); plt.tight_layout(); plt.show()


def show_wall_assignment(labels, wall_masks, debug=None, title='boundary-ring wall assignment'):
    """NEW: visualise the boundary-ring per-room wall pixels (and, if provided, the
    ring construction for one room)."""
    H, W = labels.shape
    rgb = np.ones((H, W, 3)); rgb[labels == -1] = (0.85, 0.85, 0.85)
    cmap = plt.get_cmap('tab20')
    for k, (rid, m) in enumerate(sorted(wall_masks.items())):
        rgb[np.asarray(m, bool)] = cmap(k % 20)[:3]
    if debug is None:
        fig, axx = plt.subplots(figsize=(9, 7)); axx.imshow(rgb)
        _annotate_ids(axx, labels); axx.set_title(title); axx.axis('off')
        plt.tight_layout(); plt.show()
        return
    # also show the ring construction for the largest room
    masks = debug['masks']
    rid = max(masks, key=lambda r: int(masks[r]['M'].sum()))
    d = masks[rid]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(rgb); _annotate_ids(ax[0], labels)
    ax[0].set_title(f"{title}\n(erode={debug['erode_px']}px, r_w={debug['dilate_px']}px)")
    ax[1].imshow(d['M'], cmap='gray_r'); ax[1].set_title(f'room {rid}: M_i')
    ring = np.zeros((H, W, 3)); ring[d['I']] = (0.6, 0.6, 0.6); ring[d['B']] = (1, 0.3, 0)
    ax[2].imshow(ring); ax[2].set_title('I_i (grey) + boundary B_i (orange)')
    wv = np.zeros((H, W, 3)); wv[d['Bd']] = (0.2, 0.4, 1.0); wv[d['W']] = (1, 0, 0)
    ax[3].imshow(wv); ax[3].set_title('dilated ring (blue) ∩ wall mask = walls (red)')
    for a in ax:
        a.axis('off')
    plt.tight_layout(); plt.show()


def show_coverage_debug(aux, geom_labels, cfg):
    cov = aux.get('coverage'); pre = aux.get('pre_drop', geom_labels)
    room_cov = aux.get('room_coverage', {})
    thr = getattr(cfg, 'min_coverage_frac', 0.25)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    if cov is not None:
        ax[0].imshow(cov, cmap='gray_r')
    ax[0].set_title('scan coverage (black = floor/data present)'); ax[0].axis('off')
    rgb = colorize_labels(pre); overlay = rgb.copy()
    for r, frac in room_cov.items():
        if frac < thr:
            overlay[pre == r] = (0.9, 0.15, 0.15)
    ax[1].imshow(0.45 * rgb + 0.55 * overlay)
    for r in [int(x) for x in np.unique(pre) if x >= 1]:
        ys, xs = np.where(pre == r); f = room_cov.get(r, None)
        txt = f"{int(round(100 * f))}%" if f is not None else "?"
        ax[1].text(xs.mean(), ys.mean(), txt, ha='center', va='center', fontsize=9,
                   fontweight='bold', color='k',
                   bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))
    ax[1].set_title(f'rooms + scan coverage %  (red = below {thr:.0%}, dropped)'); ax[1].axis('off')
    ax[2].imshow(colorize_labels(geom_labels)); _annotate_ids(ax[2], geom_labels)
    n = len([r for r in np.unique(geom_labels) if r >= 1])
    ax[2].set_title(f'after void removal — {n} rooms'); ax[2].axis('off')
    plt.tight_layout(); plt.show()
