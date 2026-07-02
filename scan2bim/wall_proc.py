"""Wall image processing: door and window detection."""

from __future__ import annotations

import glob
import json
import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def find_void_components(wall_img, min_void_px=15, void_open_px=3):
    void_mask = (wall_img == 255).astype(np.uint8)
    if void_open_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * void_open_px + 1, 2 * void_open_px + 1))
        eroded = cv2.morphologyEx(void_mask, cv2.MORPH_OPEN, k)
        n_labels, seed_labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8)
        labels = np.zeros_like(seed_labels)
        for i in range(1, n_labels):
            seed = (seed_labels == i).astype(np.uint8)
            grown = cv2.dilate(seed, k, iterations=2)
            labels[(grown > 0) & (void_mask > 0) & (labels == 0)] = i
    else:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            void_mask, connectivity=8)

    components = []
    for i in range(1, n_labels):
        mask = labels == i
        area = int(mask.sum())
        if area < min_void_px:
            continue
        ys, xs = np.where(mask)
        x, y = int(xs.min()), int(ys.min())
        w = int(xs.max() - x + 1)
        h = int(ys.max() - y + 1)
        cx, cy = float(xs.mean()), float(ys.mean())
        components.append({
            "mask": mask, "area": area,
            "bbox": (x, y, w, h), "centroid": (cx, cy),
        })
    return components


def _sample_points_in_mask(mask, n_points=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 2))
    n = min(n_points, len(ys))
    idx = rng.choice(len(ys), n, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


def prepare_sam_image(wall_img, upscale=4):
    rgb = np.stack([wall_img, wall_img, wall_img], axis=-1)
    rgb[wall_img == 0] = 40
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)


def refine_void_with_sam(predictor, component, upscale=4, n_points=5):
    mask = component["mask"]
    orig_h, orig_w = mask.shape
    pts_orig = _sample_points_in_mask(mask, n_points=n_points)
    if len(pts_orig) == 0:
        return mask, 0.0
    pts_up = pts_orig * upscale + upscale // 2
    labels = np.ones(len(pts_up), dtype=np.int32)
    cx, cy = component["centroid"]
    neg_candidates = []
    for dy, dx in [(-5, 0), (5, 0), (0, -5), (0, 5)]:
        ny, nx = int(cy + dy), int(cx + dx)
        if 0 <= ny < orig_h and 0 <= nx < orig_w and not mask[ny, nx]:
            neg_candidates.append((nx, ny))
    if neg_candidates:
        neg_pt = np.array([neg_candidates[0]]) * upscale + upscale // 2
        pts_up = np.vstack([pts_up, neg_pt])
        labels = np.append(labels, 0)
    masks, scores, _ = predictor.predict(
        point_coords=pts_up, point_labels=labels, multimask_output=True)
    best_idx = np.argmax(scores)
    sam_mask_up = masks[best_idx]
    score = float(scores[best_idx])
    sam_mask_down = cv2.resize(
        sam_mask_up.astype(np.uint8), (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST).astype(bool)
    return sam_mask_down, score


def build_sam_predictor(cfg):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("SAM device: %s", device)
    sam = sam_model_registry[cfg.wproc_sam_model_type](
        checkpoint=cfg.wproc_sam_checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def _bbox_overlap_or_close(b1, b2, margin=3):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return not (
        x1 + w1 + margin < x2 or x2 + w2 + margin < x1
        or y1 + h1 + margin < y2 or y2 + h2 + margin < y1)


def merge_fragments(refined_components, merge_margin_px=3):
    n = len(refined_components)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    bboxes = [c["bbox"] for c in refined_components]
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_overlap_or_close(bboxes[i], bboxes[j], merge_margin_px):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        combined_mask = np.zeros_like(refined_components[0]["mask"])
        for i in indices:
            combined_mask |= refined_components[i]["mask"]
        ys, xs = np.where(combined_mask)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        best_score = max(refined_components[i]["sam_score"] for i in indices)
        merged.append({
            "mask": combined_mask, "bbox": bbox,
            "area": int(combined_mask.sum()),
            "sam_score": best_score, "n_fragments": len(indices),
        })
    return merged


def classify_openings(merged_openings, img_height, cfg):
    pixel_m = cfg.flat_pixel_m
    results = []
    for opening in merged_openings:
        x, y, w_px, h_px = opening["bbox"]
        w_m = w_px * pixel_m
        h_m = h_px * pixel_m
        bbox_bottom_row = y + h_px - 1
        floor_row = img_height - 1
        touches_floor = (floor_row - bbox_bottom_row) <= cfg.door_floor_margin_px
        sill_m = (floor_row - bbox_bottom_row) * pixel_m
        bbox_area = w_px * h_px
        rectangularity = opening["area"] / bbox_area if bbox_area > 0 else 0.0

        label = "unknown"
        reason = ""
        rect_ok = rectangularity >= cfg.min_rectangularity
        door_size_ok = (cfg.door_min_width_m <= w_m <= cfg.door_max_width_m
                        and cfg.door_min_height_m <= h_m <= cfg.door_max_height_m)
        window_size_ok = (cfg.window_min_width_m <= w_m <= cfg.window_max_width_m
                          and cfg.window_min_height_m <= h_m <= cfg.window_max_height_m)

        if not rect_ok:
            reason = f"rectangularity {rectangularity:.2f} < {cfg.min_rectangularity}"
        elif touches_floor and door_size_ok:
            label = "door"
            reason = f"touches floor, size {w_m:.2f}x{h_m:.2f}m"
        elif not touches_floor and sill_m >= cfg.window_min_sill_m and window_size_ok:
            label = "window"
            reason = f"floating at sill={sill_m:.2f}m, size {w_m:.2f}x{h_m:.2f}m"
        else:
            reason = f"unclassified: floor={touches_floor}, size={w_m:.2f}x{h_m:.2f}m"

        results.append({
            **opening, "label": label, "reason": reason,
            "width_m": w_m, "height_m": h_m, "sill_m": sill_m,
            "touches_floor": touches_floor, "rectangularity": rectangularity,
        })
    return results


def process_wall_array(wall_img, predictor, cfg):
    img_h, img_w = wall_img.shape
    components = find_void_components(wall_img, min_void_px=cfg.min_void_px)
    if not components:
        return [], wall_img

    if predictor is not None:
        rgb_up = prepare_sam_image(wall_img, upscale=cfg.wproc_sam_upscale)
        predictor.set_image(rgb_up)

    refined = []
    for comp in components:
        if predictor is not None:
            sam_mask, score = refine_void_with_sam(
                predictor, comp, upscale=cfg.wproc_sam_upscale,
                n_points=cfg.wproc_sam_points_per_void)
        else:
            sam_mask, score = comp["mask"], 1.0
        ys, xs = np.where(sam_mask)
        if len(ys) == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        refined.append({
            "mask": sam_mask, "bbox": bbox,
            "area": int(sam_mask.sum()), "sam_score": score,
        })

    if not refined:
        return [], wall_img

    merged = merge_fragments(refined, merge_margin_px=cfg.door_floor_margin_px)
    openings = classify_openings(merged, img_h, cfg)
    return openings, wall_img


def process_wall_image(img_path, predictor, cfg):
    wall_img = np.array(Image.open(img_path).convert("L"))
    return process_wall_array(wall_img, predictor, cfg)


def save_annotated_image(wall_img, openings, out_path, pixel_m=0.04):
    rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    color_map = {"door": (255, 150, 50), "window": (0, 200, 255), "unknown": (180, 180, 180)}
    for op in openings:
        bx, by, bw, bh = op["bbox"]
        color = color_map.get(op["label"], color_map["unknown"])
        cv2.rectangle(rgb, (bx, by), (bx + bw - 1, by + bh - 1), color, 1)
        cv2.putText(rgb, op["label"][0].upper(), (bx + 2, by + bh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.imwrite(out_path, rgb)


def opening_to_dict(op):
    return {
        "label": op["label"],
        "bbox_px": list(op["bbox"]),
        "width_m": round(op["width_m"], 3),
        "height_m": round(op["height_m"], 3),
        "sill_m": round(op["sill_m"], 3),
        "touches_floor": op["touches_floor"],
        "rectangularity": round(op.get("rectangularity", 0.0), 3),
        "sam_score": round(op["sam_score"], 4),
        "n_fragments": op.get("n_fragments", 1),
        "reason": op["reason"],
    }


def run_wall_image_processing(wall_image_dir, cfg, out_dir, use_sam=True):
    os.makedirs(out_dir, exist_ok=True)

    predictor = None
    if use_sam:
        try:
            predictor = build_sam_predictor(cfg)
            logger.info("SAM predictor loaded.")
        except Exception as e:
            logger.warning("Could not load SAM (%s), running without refinement.", e)

    room_dirs = sorted(glob.glob(os.path.join(wall_image_dir, "room_*")))
    room_dirs = [d for d in room_dirs if os.path.isdir(d)]

    if room_dirs:
        room_wall_map = {}
        for rd in room_dirs:
            pngs = sorted(glob.glob(os.path.join(rd, "*.png")))
            room_wall_map[os.path.basename(rd)] = pngs
    else:
        all_pngs = sorted(glob.glob(os.path.join(wall_image_dir, "*.png")))
        room_wall_map = {os.path.basename(wall_image_dir): all_pngs}

    all_summaries = {}
    for room_name, wall_paths in sorted(room_wall_map.items()):
        logger.info("  %s: %d walls", room_name, len(wall_paths))
        room_out = os.path.join(out_dir, room_name)
        os.makedirs(room_out, exist_ok=True)

        room_summary = []
        for wp in wall_paths:
            wall_name = os.path.splitext(os.path.basename(wp))[0]
            try:
                openings, wall_img = process_wall_image(wp, predictor, cfg)
                n_doors = sum(1 for o in openings if o["label"] == "door")
                n_windows = sum(1 for o in openings if o["label"] == "window")
                logger.info("    %s: %d openings (%dD, %dW)",
                            wall_name, len(openings), n_doors, n_windows)
                ann_path = os.path.join(room_out, f"{wall_name}_annotated.png")
                save_annotated_image(wall_img, openings, ann_path, cfg.flat_pixel_m)
                room_summary.append({
                    "wall": wall_name,
                    "openings": [opening_to_dict(o) for o in openings],
                })
            except Exception as e:
                logger.error("    %s: ERROR %s", wall_name, e)

        all_summaries[room_name] = room_summary
        json_path = os.path.join(room_out, "openings.json")
        with open(json_path, "w") as f:
            json.dump(room_summary, f, indent=2)

    combined_path = os.path.join(out_dir, "all_openings.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    total_doors = sum(
        sum(1 for o in entry.get("openings", []) if o["label"] == "door")
        for room in all_summaries.values() for entry in room)
    total_windows = sum(
        sum(1 for o in entry.get("openings", []) if o["label"] == "window")
        for room in all_summaries.values() for entry in room)
    logger.info("Total: %d doors, %d windows across %d rooms",
                total_doors, total_windows, len(all_summaries))
    return all_summaries
