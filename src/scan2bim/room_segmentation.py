"""Room segmentation: watershed + optional SAM recall.

Two-pass geometry-first / SAM-for-recall room segmenter:
1. Deterministic distance-transform watershed (high precision).
2. SAM automatic mask generation on residual unclaimed space (high recall).
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, h_maxima
from skimage.segmentation import watershed

from .config import RoomSegConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Height estimation
# ---------------------------------------------------------------------------

def estimate_ceiling(h, bins=256, rel_thresh=0.02, return_floor=False):
    h = np.asarray(h, np.float64)
    counts, edges = np.histogram(h, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = np.flatnonzero(counts >= counts.max() * rel_thresh)
    floor_z, ceiling_z = centers[keep[0]], centers[keep[-1]]
    return (floor_z, ceiling_z) if return_floor else ceiling_z


def estimate_local_ceilings(points, up_axis=2, cell_size_m=1.0,
                            min_pts_per_cell=20, smooth_cells=1):
    pts = np.asarray(points, np.float64)
    h = pts[:, up_axis]
    ax_a, ax_b = [a for a in (0, 1, 2) if a != up_axis]
    a, b = pts[:, ax_a], pts[:, ax_b]
    g = estimate_ceiling(h)
    ai = np.floor((a - a.min()) / cell_size_m).astype(int)
    bi = np.floor((b - b.min()) / cell_size_m).astype(int)
    na, nb = int(ai.max()) + 1, int(bi.max()) + 1
    grid = np.full((na, nb), g, np.float64)
    flat = ai * nb + bi
    order = np.argsort(flat, kind="stable")
    fs, hs = flat[order], h[order]
    bnd = np.flatnonzero(np.diff(fs)) + 1
    for s, e in zip(np.concatenate([[0], bnd]), np.concatenate([bnd, [len(fs)]])):
        if e - s >= min_pts_per_cell:
            x, y = divmod(int(fs[s]), nb)
            grid[x, y] = estimate_ceiling(hs[s:e])
    if smooth_cells and smooth_cells > 1:
        grid = ndimage.median_filter(grid, size=int(smooth_cells))
    return grid[ai, bi]


# ---------------------------------------------------------------------------
# Slab extraction
# ---------------------------------------------------------------------------

def crop_vertical(points, cfg: RoomSegConfig, return_info=False):
    pts = np.asarray(points, np.float64)
    h = pts[:, cfg.up_axis]
    mode = cfg.slab_relative_to
    if mode == "absolute":
        keep_lo, keep_hi, ref = cfg.slab_lo_m, cfg.slab_hi_m, None
    elif mode == "floor":
        floor_z, _ = estimate_ceiling(h, return_floor=True)
        keep_lo, keep_hi, ref = floor_z + cfg.slab_lo_m, floor_z + cfg.slab_hi_m, floor_z
    else:
        cm = cfg.ceiling_mode
        if cm == "global":
            ref = estimate_ceiling(h)
        else:
            sc = 1 if cm == "local_perpoint" else int(cfg.ceiling_smooth_cells)
            ref = estimate_local_ceilings(
                pts, cfg.up_axis, cfg.ceiling_cell_size_m,
                cfg.ceiling_min_pts_per_cell, smooth_cells=sc,
            )
        keep_hi = ref - cfg.slab_lo_m
        keep_lo = ref - cfg.slab_hi_m
    mask = (h >= keep_lo) & (h <= keep_hi)
    klo = keep_lo if np.isscalar(keep_lo) else float(np.mean(keep_lo))
    khi = keep_hi if np.isscalar(keep_hi) else float(np.mean(keep_hi))
    logger.info(
        "[crop_vertical] mode=%s  band(m,mean)=[%.2f,%.2f]  kept %s/%s (%.1f%%)",
        mode, klo, khi, f"{int(mask.sum()):,}", f"{len(pts):,}", 100 * mask.mean(),
    )
    info = dict(mode=mode, ref=ref, keep_lo=keep_lo, keep_hi=keep_hi)
    return (pts[mask], mask, info) if return_info else (pts[mask], mask)


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def rasterize_topdown(sliced_points, pixel_size, up_axis=2,
                      min_points_per_cell=1, thicken=0, max_cells=150_000_000):
    pts = np.asarray(sliced_points, np.float64)
    ax_a, ax_b = [a for a in (0, 1, 2) if a != up_axis]
    a, b = pts[:, ax_a], pts[:, ax_b]
    a_px = ((a - a.min()) / pixel_size).astype(np.int64)
    b_px = ((b - b.min()) / pixel_size).astype(np.int64)
    width, height = int(a_px.max()) + 1, int(b_px.max()) + 1
    if width * height > max_cells:
        raise MemoryError(f"Grid {width * height:,} cells — raise pixel_size.")
    counts = np.zeros((height, width), np.uint16)
    np.add.at(counts, (b_px, a_px), 1)
    wall = (counts >= min_points_per_cell)[::-1, :]
    for _ in range(int(thicken)):
        w = wall.copy()
        w[1:, :] |= wall[:-1, :]
        w[:-1, :] |= wall[1:, :]
        w[:, 1:] |= wall[:, :-1]
        w[:, :-1] |= wall[:, 1:]
        wall = w
    occ = np.where(wall, 0, 255).astype(np.uint8)
    tf = dict(
        a_min=a.min(), b_min=b.min(), pixel_size=pixel_size,
        width=width, height=height, ax_a=ax_a, ax_b=ax_b, up_axis=up_axis,
    )
    return occ, tf


def rasterize_wallness(full_points, cfg: RoomSegConfig, transform):
    pts = np.asarray(full_points, np.float64)
    up = cfg.up_axis
    floor_z, ceil_z = estimate_ceiling(pts[:, up], return_floor=True)
    room_h = max(1e-6, ceil_z - floor_z)
    ax_a, ax_b, ps = transform["ax_a"], transform["ax_b"], transform["pixel_size"]
    H, W = transform["height"], transform["width"]
    a_px = ((pts[:, ax_a] - transform["a_min"]) / ps).astype(np.int64)
    b_px = ((pts[:, ax_b] - transform["b_min"]) / ps).astype(np.int64)
    inb = (a_px >= 0) & (a_px < W) & (b_px >= 0) & (b_px < H)
    a_px, b_px, z = a_px[inb], b_px[inb], pts[inb, up]
    flat = b_px * W + a_px
    zmax = np.full(H * W, -np.inf)
    zmin = np.full(H * W, np.inf)
    np.maximum.at(zmax, flat, z)
    np.minimum.at(zmin, flat, z)
    span = (zmax - zmin).reshape(H, W)
    span[~np.isfinite(span)] = 0.0
    wall = span >= cfg.wallness_min_span_frac * room_h
    return wall[::-1, :]


def rasterize_coverage(full_points, cfg: RoomSegConfig, transform):
    pts = np.asarray(full_points, np.float64)
    up = cfg.up_axis
    floor_z, ceil_z = estimate_ceiling(pts[:, up], return_floor=True)
    lo = floor_z - 0.2
    hi = ceil_z - cfg.coverage_ceiling_margin_m
    sel = (pts[:, up] >= lo) & (pts[:, up] <= hi)
    p = pts[sel]
    ax_a, ax_b, ps = transform["ax_a"], transform["ax_b"], transform["pixel_size"]
    H, W = transform["height"], transform["width"]
    a_px = ((p[:, ax_a] - transform["a_min"]) / ps).astype(np.int64)
    b_px = ((p[:, ax_b] - transform["b_min"]) / ps).astype(np.int64)
    inb = (a_px >= 0) & (a_px < W) & (b_px >= 0) & (b_px < H)
    cov = np.zeros((H, W), np.int32)
    np.add.at(cov, (b_px[inb], a_px[inb]), 1)
    covered = cov >= int(cfg.coverage_min_pts)
    c = int(cfg.coverage_close_px)
    if c > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c + 1, 2 * c + 1))
        covered = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    return covered[::-1, :]


# ---------------------------------------------------------------------------
# Wall cleanup
# ---------------------------------------------------------------------------

def _building_footprint(walls, close_px):
    w = walls.astype(np.uint8)
    ker = None
    if close_px > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        w = cv2.dilate(w, ker)
    cnts, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(w)
    cv2.drawContours(f, cnts, -1, 1, thickness=cv2.FILLED)
    if ker is not None:
        f = cv2.erode(f, ker)
    return f.astype(bool)


def clean_wall_mask(wall_mask, min_wall_area=60, seal_gap=0):
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(wm, connectivity=8)
    keep = np.zeros_like(wm)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_wall_area:
            keep[lbl == i] = 1
    walls = keep.astype(bool)
    if seal_gap > 0:
        walls = bridge_wall_endpoints(walls, max_gap=seal_gap)
    return walls


def seal_at_doors(wall_mask, doors_px, thickness=None):
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    if thickness is None:
        dist = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
        pos = dist[dist > 0]
        thickness = max(1, int(round(2 * np.median(pos)))) if pos.size else 2
    out = wm.copy()
    for x0, y0, x1, y1 in doors_px:
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(thickness))
    return out.astype(bool)


def _endpoints_and_tangents(skel):
    sk = skel.astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neigh = ndimage.convolve(sk, k, mode="constant")
    eps = np.argwhere((sk == 1) & (neigh == 1))
    ys, xs = np.where(sk)
    pts = np.column_stack([ys, xs])
    tree = cKDTree(pts)
    tangents = []
    for y, x in eps:
        loc = pts[tree.query_ball_point([y, x], r=6)]
        v = np.array([y, x]) - loc.mean(0)
        n = np.linalg.norm(v)
        tangents.append(v / n if n else np.zeros(2))
    return eps, (np.array(tangents) if tangents else np.zeros((0, 2)))


def bridge_wall_endpoints(wall_mask, max_gap=12, max_angle_deg=45, thickness=None):
    wm = (np.asarray(wall_mask) > 0).astype(np.uint8)
    if thickness is None:
        dist = cv2.distanceTransform(wm, cv2.DIST_L2, 5)
        thickness = max(1, int(round(2 * np.median(dist[dist > 0]))))
    eps, tang = _endpoints_and_tangents(skeletonize(wm > 0))
    if len(eps) < 2:
        return wm.astype(bool)
    tree = cKDTree(eps)
    out = wm.copy()
    cos_thr = np.cos(np.deg2rad(max_angle_deg))
    used = set()
    cand = sorted(
        (np.linalg.norm(eps[i] - eps[j]), i, j) for i, j in tree.query_pairs(r=max_gap)
    )
    for d, i, j in cand:
        if i in used or j in used:
            continue
        pi, pj = eps[i].astype(float), eps[j].astype(float)
        seg = (pj - pi) / (np.linalg.norm(pj - pi) + 1e-9)
        if np.dot(tang[i], seg) < cos_thr or np.dot(tang[j], -seg) < cos_thr:
            continue
        cv2.line(out, (int(pi[1]), int(pi[0])), (int(pj[1]), int(pj[0])), 1, thickness)
        used.add(i)
        used.add(j)
    return out.astype(bool)


# ---------------------------------------------------------------------------
# Pass 1: deterministic watershed
# ---------------------------------------------------------------------------

def _relabel_rooms(labels):
    out = labels.copy()
    for new, r in enumerate([int(x) for x in np.unique(labels) if x >= 1], start=1):
        out[labels == r] = new
    return out


def merge_wide_connections(labels, dt, merge_ridge_m, pixel_m):
    ridge_px = merge_ridge_m / pixel_m
    ids = [int(r) for r in np.unique(labels) if r >= 1]
    if len(ids) < 2:
        return labels
    parent = {r: r for r in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    contact = {}

    def add_pairs(la, lb, da, db):
        m = (la >= 1) & (lb >= 1) & (la != lb)
        if not m.any():
            return
        for x, y, d in zip(la[m], lb[m], np.maximum(da[m], db[m])):
            key = (min(int(x), int(y)), max(int(x), int(y)))
            if d > contact.get(key, 0.0):
                contact[key] = float(d)

    add_pairs(labels[:, :-1], labels[:, 1:], dt[:, :-1], dt[:, 1:])
    add_pairs(labels[:-1, :], labels[1:, :], dt[:-1, :], dt[1:, :])
    for (a, b), d in contact.items():
        if d >= ridge_px:
            union(a, b)
    out = labels.copy()
    newid = {}
    nxt = 1
    for r in ids:
        root = find(r)
        if root not in newid:
            newid[root] = nxt
            nxt += 1
    for r in ids:
        out[labels == r] = newid[find(r)]
    return out


def segment_rooms_watershed(
    wall_mask, pixel_m,
    marker_h_m=0.30, footprint_close_m=1.0, merge_ridge_m=0.70,
    min_room_area_m2=2.0, min_wall_area_px=60, door_seal_px=0,
    coverage=None, min_coverage_frac=0.25, return_aux=False,
):
    walls = clean_wall_mask(wall_mask, min_wall_area=min_wall_area_px, seal_gap=door_seal_px)
    free = ~walls
    dt = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    h = max(1.0, marker_h_m / pixel_m)
    markers, n_int = ndimage.label(h_maxima(dt, h))

    k = max(1, int(round(footprint_close_m / pixel_m)))
    footprint = _building_footprint(walls, k)
    exterior_seed = free & (~footprint)
    markers = markers.copy()
    ext_id = n_int + 1
    if exterior_seed.any():
        markers[exterior_seed] = ext_id

    ws = watershed(-dt, markers, mask=free)

    labels = np.full(wall_mask.shape, -1, np.int32)
    labels[free] = 0
    min_area_px = int(round(min_room_area_m2 / (pixel_m**2)))
    rid = 1
    for b in range(1, n_int + 1):
        m = ws == b
        if m.sum() == 0:
            continue
        inside = (m & footprint).sum() / m.sum()
        if m.sum() >= min_area_px and inside >= 0.5:
            labels[m] = rid
            rid += 1
        else:
            labels[m] = 0
    labels[ws == ext_id] = 0
    labels = merge_wide_connections(labels, dt, merge_ridge_m, pixel_m)

    pre_drop = labels.copy()
    room_cov = {}
    if coverage is not None:
        for r in [int(x) for x in np.unique(labels) if x >= 1]:
            interior = labels == r
            frac = float((interior & coverage).sum() / max(1, int(interior.sum())))
            room_cov[r] = frac
            if frac < min_coverage_frac:
                labels[interior] = 0
        labels = _relabel_rooms(labels)

    if return_aux:
        return labels, dict(
            walls=walls, dt=dt, footprint=footprint,
            seeds=markers, ws=ws, exterior=exterior_seed,
            n_int=n_int, ext_id=ext_id,
            coverage=coverage, pre_drop=pre_drop, room_coverage=room_cov,
        )
    return labels


# ---------------------------------------------------------------------------
# Pass 2: SAM recall
# ---------------------------------------------------------------------------

def build_sam_mask_generator(cfg: RoomSegConfig):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as AMG

        sam = build_sam2(cfg.sam_model_cfg, cfg.sam_ckpt, device=device)
        return AMG(
            sam, points_per_side=cfg.sam_points,
            pred_iou_thresh=cfg.sam_iou, stability_score_thresh=cfg.sam_stability,
            crop_n_layers=cfg.sam_n_layers, crop_n_points_downscale_factor=cfg.sam_down_factor,
            min_mask_region_area=cfg.sam_min_mask,
        )
    except Exception:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator as AMG

        sam = sam_model_registry[cfg.sam_arch](checkpoint=cfg.sam_ckpt).to(device)
        return AMG(
            sam, points_per_side=cfg.sam_points,
            pred_iou_thresh=cfg.sam_iou, stability_score_thresh=cfg.sam_stability,
            crop_n_layers=cfg.sam_n_layers, crop_n_points_downscale_factor=cfg.sam_down_factor,
            min_mask_region_area=cfg.sam_min_mask,
        )


def sam_auto_masks(generator, occ_gray):
    rgb = np.stack([occ_gray] * 3, -1).astype(np.uint8)
    return [m["segmentation"].astype(bool) for m in generator.generate(rgb)]


def sam_rooms_on_residual(sam_masks, walls, geom_labels, footprint, pixel_m,
                          min_room_area_m2=2.0, min_overlap=0.5):
    free = ~walls
    residual = free & (geom_labels == 0) & footprint
    min_area_px = int(round(min_room_area_m2 / (pixel_m**2)))
    out = []
    for m in sam_masks:
        m = m & free
        if m.sum() < min_area_px:
            continue
        if (m & residual).sum() < min_overlap * m.sum():
            continue
        lbl, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        out.append(lbl == (int(np.argmax(sizes)) + 1))
    return out


def fuse_labels(geom_labels, sam_room_masks):
    out = geom_labels.copy()
    nxt = int(out.max()) + 1
    for m in sam_room_masks:
        m = m & (out == 0)
        if m.sum() == 0:
            continue
        out[m] = nxt
        nxt += 1
    return out


# ---------------------------------------------------------------------------
# 3D projection + per-room cloud splitting
# ---------------------------------------------------------------------------

def point_cells(points, transform):
    pts = np.asarray(points, np.float64)
    ax_a, ax_b, ps = transform["ax_a"], transform["ax_b"], transform["pixel_size"]
    H, W = transform["height"], transform["width"]
    a_px = ((pts[:, ax_a] - transform["a_min"]) / ps).astype(np.int64)
    b_px = ((pts[:, ax_b] - transform["b_min"]) / ps).astype(np.int64)
    row = H - 1 - b_px
    col = a_px
    inb = (col >= 0) & (col < W) & (b_px >= 0) & (b_px < H)
    return row, col, inb


def label_points(points, labels, transform):
    row, col, inb = point_cells(points, transform)
    out = np.full(len(row), -2, np.int32)
    out[inb] = labels[row[inb], col[inb]]
    return out


def room_footprints(labels, margin=1, thickness=None, walls_only=False):
    wall_mask = labels == -1
    if thickness is None:
        dist = cv2.distanceTransform(wall_mask.astype(np.uint8), cv2.DIST_L2, 5)
        pos = dist[dist > 0]
        thickness = int(round(2 * np.median(pos))) if pos.size else 1
    k = max(1, thickness + margin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    foot = {}
    for r in np.unique(labels):
        if r < 1:
            continue
        interior = labels == r
        grown = cv2.dilate(interior.astype(np.uint8), kernel).astype(bool)
        band = grown & wall_mask
        foot[int(r)] = band if walls_only else (interior | band)
    return foot


def split_rooms_to_clouds(points, labels, transform, colors=None,
                          margin=1, thickness=None, walls_only=False, keep_mask=None):
    row, col, inb = point_cells(points, transform)
    foot = room_footprints(labels, margin=margin, thickness=thickness, walls_only=walls_only)
    pts = np.asarray(points)
    cols = np.asarray(colors) if colors is not None else None
    keep = None if keep_mask is None else np.asarray(keep_mask, bool)
    out = []
    for rid, fmask in sorted(foot.items()):
        sel = np.zeros(len(pts), bool)
        sel[inb] = fmask[row[inb], col[inb]]
        if keep is not None:
            sel &= keep
        e = {"room_id": rid, "mask": sel, "points": pts[sel]}
        if cols is not None:
            e["colors"] = cols[sel]
        out.append(e)
    return out


def fit_walls_in_room(pc, up_axis=2, normal_tol_deg=10, dist_thresh=0.02,
                      min_inliers=400, max_planes=20, dbscan_eps_mult=8):
    if len(pc.points) < min_inliers:
        return []
    pc.estimate_normals()
    up = np.zeros(3)
    up[up_axis] = 1.0
    sin_tol = np.sin(np.deg2rad(normal_tol_deg))
    n = np.asarray(pc.normals)
    cand = pc.select_by_index(np.where(np.abs(n @ up) < sin_tol)[0])
    walls, rest = [], cand
    while len(rest.points) >= min_inliers and len(walls) < max_planes:
        model, inl = rest.segment_plane(dist_thresh, 3, 1000)
        if len(inl) < min_inliers:
            break
        nrm = np.array(model[:3])
        nrm /= np.linalg.norm(nrm) + 1e-9
        if abs(nrm @ up) < sin_tol:
            seg = rest.select_by_index(inl)
            lbl = np.array(
                seg.cluster_dbscan(
                    eps=dist_thresh * dbscan_eps_mult,
                    min_points=max(10, min_inliers // 4),
                )
            )
            for c in sorted(set(lbl) - {-1}):
                walls.append({"model": model, "cloud": seg.select_by_index(np.where(lbl == c)[0])})
        rest = rest.select_by_index(inl, invert=True)
    return walls


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def run_room_segmentation(pcd, points, cfg: RoomSegConfig, out_dir: str):
    """Run the full room segmentation pipeline and write per-room wall clouds.

    Returns (labels_2d, rooms_list, transform).
    """
    os.makedirs(out_dir, exist_ok=True)

    slab_pts, slab_mask = crop_vertical(points, cfg)[:2]

    occ, tf = rasterize_topdown(
        slab_pts, cfg.pixel_m, up_axis=cfg.up_axis,
        min_points_per_cell=cfg.min_points_per_cell, thicken=cfg.thicken_px,
    )
    wall_mask = occ == 0
    if cfg.use_wallness:
        wall_mask = rasterize_wallness(points, cfg, tf)

    coverage = rasterize_coverage(points, cfg, tf)

    geom_labels, aux = segment_rooms_watershed(
        wall_mask, cfg.pixel_m,
        marker_h_m=cfg.marker_h_m,
        footprint_close_m=cfg.footprint_close_m,
        merge_ridge_m=cfg.merge_ridge_m,
        min_room_area_m2=cfg.min_room_area_m2,
        min_wall_area_px=cfg.min_wall_area_px,
        coverage=coverage if cfg.drop_empty_rooms else None,
        min_coverage_frac=cfg.min_coverage_frac,
        return_aux=True,
    )
    labels = geom_labels
    logger.info("Pass 1 rooms: %d", int(geom_labels.max()))

    if cfg.use_sam_recall:
        try:
            gen = build_sam_mask_generator(cfg)
            sam_masks = sam_auto_masks(gen, np.where(aux["walls"], 0, 255).astype(np.uint8))
            sam_rooms = sam_rooms_on_residual(
                sam_masks, aux["walls"], geom_labels, aux["footprint"], cfg.pixel_m,
                min_room_area_m2=cfg.min_room_area_m2, min_overlap=cfg.sam_min_overlap,
            )
            labels = fuse_labels(geom_labels, sam_rooms)
            logger.info(
                "Pass 1: %d rooms -> fused: %d (+%d from SAM)",
                int(geom_labels.max()), int(labels.max()),
                int(labels.max()) - int(geom_labels.max()),
            )
        except Exception as e:
            logger.warning("SAM recall failed (%s), using Pass-1 labels only.", e)

    n_rooms = len([r for r in np.unique(labels) if r >= 1])
    logger.info("Final room count: %d", n_rooms)

    # Export per-room wall clouds
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    floor_z, ceil_z = estimate_ceiling(xyz[:, cfg.up_axis], return_floor=True)
    z = xyz[:, cfg.up_axis]
    band = (z >= floor_z + cfg.wall_floor_margin_m) & (z <= ceil_z - cfg.wall_ceiling_margin_m)

    rooms = split_rooms_to_clouds(
        xyz, labels, tf, colors=colors,
        margin=cfg.do_buffer_px, walls_only=True, keep_mask=band,
    )

    paths = []
    for r in rooms:
        if len(r["points"]) == 0:
            continue
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(r["points"])
        if "colors" in r:
            out.colors = o3d.utility.Vector3dVector(r["colors"])
        p = os.path.join(out_dir, f"room_{r['room_id']:02d}_walls.ply")
        o3d.io.write_point_cloud(p, out)
        paths.append(p)

    logger.info("Wrote %d per-room wall clouds to %s", len(paths), out_dir)
    return labels, rooms, tf
