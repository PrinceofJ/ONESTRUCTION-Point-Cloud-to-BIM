"""IFC export: wall geometry, JSON builder, IFC4 writer."""

from __future__ import annotations

import glob
import json
import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)


def segment_walls_for_ifc(pcd, cfg):
    from .wall_seg import segment_walls

    walls = segment_walls(pcd, cfg)
    n_directions = 0
    if walls and "_theta_peaks" in walls[0]:
        n_directions = len(walls[0]["_theta_peaks"])
    elif walls:
        up_axis = cfg.up_axis
        ha, hb = [a for a in range(3) if a != up_axis]
        angles = set()
        for w in walls:
            a = round(float(np.arctan2(w["normal_2d"][1], w["normal_2d"][0]) % np.pi), 2)
            angles.add(a)
        n_directions = len(angles)
    return walls, n_directions


def compute_wall_geometry(wall, up_axis=2):
    pts = np.asarray(wall["cloud"].points)
    n2d = wall["normal_2d"]
    ha, hb = [a for a in range(3) if a != up_axis]
    normal_3d = np.zeros(3)
    normal_3d[ha] = n2d[0]
    normal_3d[hb] = n2d[1]
    normal_3d /= np.linalg.norm(normal_3d) + 1e-9
    up = np.zeros(3)
    up[up_axis] = 1.0
    u_axis = np.cross(up, normal_3d)
    u_axis /= np.linalg.norm(u_axis) + 1e-9
    u = pts @ u_axis
    v = pts[:, up_axis]
    n_vals = pts[:, ha] * n2d[0] + pts[:, hb] * n2d[1]
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    offset = wall["offset"]
    thickness = float(np.percentile(n_vals, 95) - np.percentile(n_vals, 5))
    n_centered = n_vals - offset
    pos_frac = float(np.mean(n_centered > 0.02))
    neg_frac = float(np.mean(n_centered < -0.02))
    minority = min(pos_frac, neg_frac)
    is_exterior = minority < 0.10
    cell_size = 0.10
    u_range = u_max - u_min
    v_range = v_max - v_min
    if u_range > cell_size and v_range > cell_size:
        n_u = max(1, int(u_range / cell_size))
        n_v = max(1, int(v_range / cell_size))
        u_idx = np.clip(((u - u_min) / u_range * n_u).astype(int), 0, n_u - 1)
        v_idx = np.clip(((v - v_min) / v_range * n_v).astype(int), 0, n_v - 1)
        grid = np.zeros((n_v, n_u), dtype=bool)
        grid[v_idx, u_idx] = True
        fill_ratio = float(grid.sum()) / float(grid.size)
    else:
        fill_ratio = 1.0
    start_2d = [
        float(u_min * u_axis[0] + offset * normal_3d[0]),
        float(u_min * u_axis[1] + offset * normal_3d[1]),
    ]
    end_2d = [
        float(u_max * u_axis[0] + offset * normal_3d[0]),
        float(u_max * u_axis[1] + offset * normal_3d[1]),
    ]
    return {
        "start": start_2d, "end": end_2d,
        "height": v_max - v_min, "thickness": thickness,
        "floor_z": v_min, "length": u_max - u_min,
        "u_min": u_min, "u_max": u_max,
        "u_axis": u_axis.tolist(), "normal_3d": normal_3d.tolist(),
        "normal_2d": n2d.tolist(), "offset": offset,
        "angle": float(np.arctan2(n2d[1], n2d[0]) % np.pi),
        "fill_ratio": fill_ratio, "is_exterior": is_exterior,
    }


def _angle_close(a1, a2, tol_rad):
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff) < tol_rad


def _offset_distance(g1, g2):
    s1 = np.array(g1["start"])
    e1 = np.array(g1["end"])
    d1 = e1 - s1
    length1 = np.linalg.norm(d1)
    if length1 < 1e-9:
        return float("inf")
    n1 = np.array([-d1[1], d1[0]]) / length1
    mid2 = (np.array(g2["start"]) + np.array(g2["end"])) / 2
    return float(abs(np.dot(mid2 - s1, n1)))


def _u_overlap(g1, g2):
    s1 = np.array(g1["start"])
    e1 = np.array(g1["end"])
    d1 = e1 - s1
    length1 = np.linalg.norm(d1)
    if length1 < 1e-9:
        return 0.0
    u_dir = d1 / length1
    u1_max = length1
    s2 = np.array(g2["start"])
    e2 = np.array(g2["end"])
    p2a = float(np.dot(s2 - s1, u_dir))
    p2b = float(np.dot(e2 - s1, u_dir))
    u2_min, u2_max = min(p2a, p2b), max(p2a, p2b)
    return min(u1_max, u2_max) - max(0.0, u2_min)


def _walls_spatially_close(g1, g2, tol=0.5):
    s1, e1 = np.array(g1["start"]), np.array(g1["end"])
    s2, e2 = np.array(g2["start"]), np.array(g2["end"])
    if _u_overlap(g1, g2) > 0:
        return True
    return (np.linalg.norm(s1 - s2) < tol or np.linalg.norm(s1 - e2) < tol
            or np.linalg.norm(e1 - s2) < tol or np.linalg.norm(e1 - e2) < tol)


def merge_wall_faces(wall_geos, cfg, angle_tol_deg=10.0):
    n = len(wall_geos)
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

    angle_tol = np.deg2rad(angle_tol_deg)
    for i in range(n):
        for j in range(i + 1, n):
            if not _angle_close(wall_geos[i]["angle"], wall_geos[j]["angle"], angle_tol):
                continue
            if _offset_distance(wall_geos[i], wall_geos[j]) > cfg.ifc_max_merge_thickness:
                continue
            if not _walls_spatially_close(wall_geos[i], wall_geos[j]):
                continue
            overlap = _u_overlap(wall_geos[i], wall_geos[j])
            min_len = min(wall_geos[i]["length"], wall_geos[j]["length"])
            if min_len > 0 and overlap / min_len < 0.3:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        raw_indices = [wall_geos[i].get("_raw_idx", i) for i in indices]
        if len(indices) == 1:
            g = wall_geos[indices[0]].copy()
            g["source_indices"] = raw_indices
            if g["thickness"] < 0.05:
                g["thickness"] = cfg.ifc_default_thickness
            merged.append(g)
        else:
            geos = [wall_geos[i] for i in indices]
            avg_offset = np.mean([g["offset"] for g in geos])
            thickness = max(abs(g["offset"] - avg_offset) * 2 for g in geos)
            if thickness < 0.05:
                thickness = cfg.ifc_default_thickness
            longest = max(geos, key=lambda g: g["length"])
            u_min = longest["u_min"]
            u_max = longest["u_max"]
            height = max(g["height"] for g in geos)
            floor_z = min(g["floor_z"] for g in geos)
            ref = geos[0]
            u_axis = np.array(ref["u_axis"])
            normal_3d = np.array(ref["normal_3d"])
            start_2d = [
                float(u_min * u_axis[0] + avg_offset * normal_3d[0]),
                float(u_min * u_axis[1] + avg_offset * normal_3d[1]),
            ]
            end_2d = [
                float(u_max * u_axis[0] + avg_offset * normal_3d[0]),
                float(u_max * u_axis[1] + avg_offset * normal_3d[1]),
            ]
            merged.append({
                "start": start_2d, "end": end_2d,
                "height": height, "thickness": thickness,
                "floor_z": floor_z, "length": u_max - u_min,
                "u_min": u_min, "u_max": u_max,
                "u_axis": ref["u_axis"], "normal_3d": ref["normal_3d"],
                "normal_2d": ref["normal_2d"], "offset": float(avg_offset),
                "angle": ref["angle"], "source_indices": raw_indices,
                "fill_ratio": max(g.get("fill_ratio", 0) for g in geos),
                "is_exterior": any(g.get("is_exterior", False) for g in geos),
            })
    return merged


def deduplicate_walls(all_room_walls, cfg, angle_tol_deg=10.0):
    flat = []
    for room_name, walls in all_room_walls.items():
        for i, w in enumerate(walls):
            flat.append({**w, "_room": room_name, "_idx": i,
                         "_openings": w.get("_openings", [])})

    n = len(flat)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if flat[a]["length"] >= flat[b]["length"]:
                parent[b] = a
            else:
                parent[a] = b

    angle_tol = np.deg2rad(angle_tol_deg)
    for i in range(n):
        for j in range(i + 1, n):
            if not _angle_close(flat[i]["angle"], flat[j]["angle"], angle_tol):
                continue
            if _offset_distance(flat[i], flat[j]) > cfg.ifc_dedup_offset_tol:
                continue
            if not _walls_spatially_close(flat[i], flat[j]):
                continue
            u_gap = -_u_overlap(flat[i], flat[j])
            if u_gap > cfg.ifc_dedup_offset_tol:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    unique_walls = []
    for root_idx, indices in groups.items():
        keeper = flat[root_idx].copy()
        if len(indices) > 1:
            s1 = np.array(keeper["start"])
            e1 = np.array(keeper["end"])
            d1 = e1 - s1
            length1 = np.linalg.norm(d1)
            if length1 > 1e-9:
                u_dir = d1 / length1
                all_u = [0.0, length1]
                for idx in indices:
                    if idx == root_idx:
                        continue
                    other = flat[idx]
                    s2 = np.array(other["start"])
                    e2 = np.array(other["end"])
                    all_u.append(float(np.dot(s2 - s1, u_dir)))
                    all_u.append(float(np.dot(e2 - s1, u_dir)))
                new_u_min = min(all_u)
                new_u_max = max(all_u)
                keeper["start"] = (s1 + u_dir * new_u_min).tolist()
                keeper["end"] = (s1 + u_dir * new_u_max).tolist()
                keeper["length"] = new_u_max - new_u_min

        all_openings = []
        for idx in indices:
            for op in flat[idx].get("_openings", []):
                all_openings.append(op)
        keeper["_openings"] = all_openings

        rooms_set = set()
        for idx in indices:
            rooms_set.add(flat[idx]["_room"])
        keeper["_rooms"] = sorted(rooms_set)
        keeper.pop("_room", None)
        keeper.pop("_idx", None)
        unique_walls.append(keeper)
    return unique_walls


def snap_wall_endpoints(walls, tol=0.15):
    for _ in range(3):
        for i in range(len(walls)):
            for ki in ("start", "end"):
                pi = np.array(walls[i][ki])
                best_dist = tol
                best_target = None
                best_type = None
                for j in range(len(walls)):
                    if i == j:
                        continue
                    for kj in ("start", "end"):
                        pj = np.array(walls[j][kj])
                        dist = np.linalg.norm(pi - pj)
                        if 0 < dist < best_dist:
                            best_dist = dist
                            best_target = ((pi + pj) / 2).tolist()
                            best_type = ("ll", j, kj)
                    sj = np.array(walls[j]["start"])
                    ej = np.array(walls[j]["end"])
                    seg = ej - sj
                    seg_len = np.linalg.norm(seg)
                    if seg_len < 0.01:
                        continue
                    t = np.dot(pi - sj, seg) / (seg_len ** 2)
                    if t < 0.01 or t > 0.99:
                        continue
                    closest = sj + t * seg
                    dist = np.linalg.norm(pi - closest)
                    if 0 < dist < best_dist:
                        best_dist = dist
                        best_target = closest.tolist()
                        best_type = ("t", j)
                if best_target is not None:
                    walls[i][ki] = best_target
                    if best_type[0] == "ll":
                        walls[best_type[1]][best_type[2]] = best_target
    return walls


def _find_manhattan_angles(all_room_walls, angle_tol_deg=10.0):
    angles, lengths = [], []
    for walls in all_room_walls.values():
        for w in walls:
            angles.append(w["angle"])
            lengths.append(w["length"])
    if not angles:
        return []
    angles = np.array(angles)
    lengths = np.array(lengths)
    tol = np.deg2rad(angle_tol_deg)
    buckets: list[tuple[float, float]] = []
    for a, l in zip(angles, lengths):
        placed = False
        for i, (ba, bl) in enumerate(buckets):
            if min(abs(a - ba), np.pi - abs(a - ba)) < tol:
                buckets[i] = (ba, bl + l)
                placed = True
                break
        if not placed:
            buckets.append((a, l))
    buckets.sort(key=lambda x: x[1], reverse=True)
    if len(buckets) < 2:
        return [b[0] for b in buckets[:1]]
    return [buckets[0][0], buckets[0][0] + np.pi / 2]


def snap_wall_angles(all_room_walls, angle_tol_deg=10.0):
    manhattan = _find_manhattan_angles(all_room_walls, angle_tol_deg)
    if len(manhattan) < 1:
        return all_room_walls
    tol = np.deg2rad(angle_tol_deg)
    for room_name, walls in all_room_walls.items():
        for w in walls:
            wall_angle = w["angle"]
            best_ma = None
            best_diff = tol
            for ma in manhattan:
                diff = min(abs(wall_angle - ma), np.pi - abs(wall_angle - ma))
                if diff < best_diff and diff > 1e-6:
                    best_diff = diff
                    best_ma = ma
            if best_ma is None:
                continue
            s = np.array(w["start"])
            e = np.array(w["end"])
            n2d_new = np.array([np.cos(best_ma), np.sin(best_ma)])
            wall_dir = np.array([-n2d_new[1], n2d_new[0]])
            if np.dot(wall_dir, e - s) < 0:
                wall_dir = -wall_dir
            mid = (s + e) / 2.0
            new_offset = float(np.dot(mid, n2d_new))
            u_s = float(np.dot(s, wall_dir))
            u_e = float(np.dot(e, wall_dir))
            u_min, u_max = min(u_s, u_e), max(u_s, u_e)
            w["start"] = (u_min * wall_dir + new_offset * n2d_new).tolist()
            w["end"] = (u_max * wall_dir + new_offset * n2d_new).tolist()
            w["angle"] = float(best_ma)
            w["offset"] = new_offset
            w["normal_2d"] = n2d_new.tolist()
            w["u_min"] = u_min
            w["u_max"] = u_max
    return all_room_walls


def compute_room_boundaries(all_room_walls):
    boundaries = {}
    for room_name, walls in all_room_walls.items():
        pts = []
        for w in walls:
            pts.append(w["start"])
            pts.append(w["end"])
        if len(pts) < 2:
            boundaries[room_name] = []
            continue
        longest = max(walls, key=lambda w: w.get("length", 0))
        sx, sy = longest["start"]
        ex, ey = longest["end"]
        angle = math.atan2(ey - sy, ex - sx)
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        arr = np.array(pts)
        rotated = np.column_stack([
            arr[:, 0] * cos_a - arr[:, 1] * sin_a,
            arr[:, 0] * sin_a + arr[:, 1] * cos_a,
        ])
        r_min = rotated.min(axis=0)
        r_max = rotated.max(axis=0)
        corners_rot = np.array([
            [r_min[0], r_min[1]], [r_max[0], r_min[1]],
            [r_max[0], r_max[1]], [r_min[0], r_max[1]],
        ])
        cos_b, sin_b = math.cos(angle), math.sin(angle)
        corners_world = [
            [float(c[0] * cos_b - c[1] * sin_b),
             float(c[0] * sin_b + c[1] * cos_b)]
            for c in corners_rot
        ]
        boundaries[room_name] = corners_world
    return boundaries


def _detect_doors_on_merged_walls(merged_walls, walls_raw, cfg,
                                  room_name, out_dir=None):
    from .wall_seg import flatten_wall
    from .wall_proc import find_void_components, merge_fragments
    import cv2

    for i, mw in enumerate(merged_walls):
        mw["_openings"] = []
        src_indices = mw.get("source_indices", [i])
        all_pts = []
        for idx in src_indices:
            if idx < len(walls_raw):
                all_pts.append(np.asarray(walls_raw[idx]["cloud"].points))
        if not all_pts:
            continue
        import open3d as o3d
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        if len(combined.points) < 50:
            continue

        wall_dict = {"cloud": combined, "normal_2d": np.array(mw["normal_2d"]),
                     "offset": mw["offset"]}
        flat = flatten_wall(wall_dict, cfg)
        wall_img = flat["image"]
        img_h, img_w = wall_img.shape
        pixel_m = cfg.flat_pixel_m
        flat_u_min = float(flat["u"].min())
        wall_u_min = mw["u_min"]

        components = find_void_components(wall_img, min_void_px=cfg.min_void_px)
        for comp in components:
            comp["sam_score"] = 1.0
        gaps = merge_fragments(components, merge_margin_px=cfg.door_floor_margin_px)

        n_doors = 0
        for gap in gaps:
            gx, gy, gw, gh = gap["bbox"]
            w_m = gw * pixel_m
            h_m = gh * pixel_m
            bbox_bottom = gy + gh - 1
            floor_row = img_h - 1
            touches_floor = (floor_row - bbox_bottom) <= cfg.door_floor_margin_px
            bbox_area = gw * gh
            rectangularity = gap["area"] / bbox_area if bbox_area > 0 else 0.0
            wall_length_m = img_w * pixel_m
            too_wide = (wall_length_m > 0
                        and w_m / wall_length_m > cfg.door_max_wall_width_frac)
            is_door = (
                touches_floor and not too_wide
                and rectangularity >= cfg.min_rectangularity
                and cfg.door_min_width_m <= w_m <= cfg.door_max_width_m
                and cfg.door_min_height_m <= h_m <= cfg.door_max_height_m)
            if not is_door:
                continue
            offset_from_image = gx * pixel_m
            abs_u = flat_u_min + offset_from_image
            u_axis_2d = np.array(flat["u_axis"][:2])
            n2d = np.array(mw["normal_2d"])
            world_xy = abs_u * u_axis_2d + mw["offset"] * n2d
            mw["_openings"].append({
                "label": "door",
                "world_xy": [round(float(world_xy[0]), 4), round(float(world_xy[1]), 4)],
                "width": round(w_m, 3), "height": round(h_m, 3),
            })
            n_doors += 1
        if n_doors:
            logger.info("  %s wall %d: %d doors detected", room_name, i + 1, n_doors)


def build_building_json(room_cloud_paths, cfg, out_dir=None):
    import open3d as o3d

    all_room_walls: dict[str, list] = {}
    skipped = []
    _debug: dict[str, dict] = {}
    angle_tol = cfg.wseg_angle_tol_deg

    for cloud_path in room_cloud_paths:
        fname = os.path.splitext(os.path.basename(cloud_path))[0]
        room_name = fname.replace("_walls", "")
        logger.info("Processing %s", room_name)

        pcd = o3d.io.read_point_cloud(cloud_path)
        if cfg.voxel_m > 0:
            pcd = pcd.voxel_down_sample(cfg.voxel_m)
        walls, n_dirs = segment_walls_for_ifc(pcd, cfg)
        logger.info("  %d wall segments, %d directions", len(walls), n_dirs)

        if n_dirs < cfg.ifc_min_directions:
            skipped.append(room_name)
            continue

        wall_geos_raw = [compute_wall_geometry(w, cfg.up_axis) for w in walls]
        wall_geos = []
        for raw_idx, g in enumerate(wall_geos_raw):
            if g["length"] < cfg.ifc_min_wall_length_m:
                continue
            if g["length"] > cfg.ifc_max_wall_length_m:
                continue
            aspect = g["length"] / max(g["height"], 1e-3)
            if aspect < cfg.ifc_min_wall_aspect_ratio:
                continue
            if g["thickness"] > cfg.ifc_max_wall_thickness_m:
                g["thickness"] = cfg.ifc_default_thickness
            if g.get("is_exterior"):
                g["thickness"] = cfg.ifc_exterior_thickness
            if g["fill_ratio"] < cfg.ifc_min_wall_fill_ratio:
                continue
            g["_raw_idx"] = raw_idx
            wall_geos.append(g)

        if not wall_geos:
            skipped.append(room_name)
            continue

        merged = merge_wall_faces(wall_geos, cfg, angle_tol)
        _detect_doors_on_merged_walls(merged, walls, cfg, room_name, out_dir)
        _debug[room_name] = {
            "n_segments": len(walls), "n_directions": n_dirs,
            "n_raw_geos": len(wall_geos_raw),
            "n_after_filter": len(wall_geos),
            "n_after_merge": len(merged),
            "raw_geos": wall_geos_raw,
            "filtered_geos": wall_geos,
            "merged_geos": merged,
        }
        all_room_walls[room_name] = merged

    logger.info("Processed %d rooms, skipped %d", len(all_room_walls), len(skipped))
    all_room_walls = snap_wall_angles(all_room_walls, angle_tol)

    pre_dedup_count = sum(len(ws) for ws in all_room_walls.values())
    unique_walls = deduplicate_walls(all_room_walls, cfg, angle_tol)
    logger.info("Deduplicated: %d -> %d unique walls", pre_dedup_count, len(unique_walls))

    unique_walls = snap_wall_endpoints(unique_walls, tol=cfg.ifc_snap_tolerance_m)

    min_wall_len = 0.30
    unique_walls = [w for w in unique_walls
                    if np.linalg.norm(np.array(w["end"]) - np.array(w["start"])) >= min_wall_len]

    all_heights = sorted([w["height"] for w in unique_walls])
    if all_heights:
        storey_height = float(np.percentile(all_heights, 90))
        for w in unique_walls:
            w["height"] = storey_height

    storey_id = "L0"
    data: dict = {
        "project": {"name": cfg.ifc_project_name},
        "storeys": [{"id": storey_id, "name": "Ground Floor",
                      "elevation": cfg.ifc_floor_elevation}],
        "walls": [], "doors": [], "windows": [], "rooms": [],
    }

    door_counter = 0
    for i, w in enumerate(unique_walls):
        wall_id = f"W{i + 1}"
        data["walls"].append({
            "id": wall_id, "storey": storey_id,
            "start": [round(w["start"][0], 4), round(w["start"][1], 4)],
            "end": [round(w["end"][0], 4), round(w["end"][1], 4)],
            "height": round(w["height"], 3),
            "thickness": round(cfg.ifc_default_thickness, 3),
        })
        wall_start = np.array(w["start"])
        wall_end = np.array(w["end"])
        wall_dir = wall_end - wall_start
        wall_len = float(np.linalg.norm(wall_dir))
        wall_u_hat = wall_dir / wall_len if wall_len > 1e-9 else np.array([1.0, 0.0])
        for op in w.get("_openings", []):
            if op["label"] == "door":
                door_w = op["width"]
                if "world_xy" in op:
                    door_pt = np.array(op["world_xy"])
                    offset = float(np.dot(door_pt - wall_start, wall_u_hat))
                else:
                    offset = op.get("offset", 0.0)
                if door_w > wall_len:
                    continue
                offset = max(0.0, min(offset, wall_len - door_w))
                door_h = min(op["height"], w["height"] - 0.01)
                door_counter += 1
                data["doors"].append({
                    "id": f"D{door_counter}", "wall": wall_id,
                    "offset": round(offset, 3),
                    "width": round(door_w, 3), "height": round(door_h, 3),
                })

    room_boundaries = compute_room_boundaries(all_room_walls)
    for room_name, boundary in room_boundaries.items():
        if not boundary:
            continue
        heights = [rw["height"] for rw in all_room_walls.get(room_name, [])]
        room_height = max(heights) if heights else 2.5
        data["rooms"].append({
            "id": room_name, "storey": storey_id,
            "name": room_name.replace("_", " ").title(),
            "boundary": [[round(x, 4), round(y, 4)] for x, y in boundary],
            "height": round(room_height, 3),
        })

    logger.info("JSON: %d walls, %d doors, %d windows, %d rooms",
                len(data["walls"]), len(data["doors"]),
                len(data["windows"]), len(data["rooms"]))
    data["_debug"] = _debug
    return data


def build_ifc(data, out_path="model.ifc", cfg=None):
    import ifcopenshell
    import ifcopenshell.guid
    import ifcopenshell.api.context
    import ifcopenshell.api.unit
    import ifcopenshell.api.root
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.spatial

    add_floor_slabs = cfg.ifc_add_floor_slabs if cfg else True
    slab_thickness = cfg.ifc_slab_thickness if cfg else 0.2

    m = ifcopenshell.file(schema="IFC4")

    def _dir(v):
        return m.create_entity("IfcDirection", DirectionRatios=[float(x) for x in v])

    def _pt(v):
        return m.create_entity("IfcCartesianPoint", Coordinates=[float(x) for x in v])

    def _ax2(loc=(0.0, 0.0)):
        return m.create_entity("IfcAxis2Placement2D", Location=_pt(loc))

    def _ax3(loc=(0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)):
        return m.create_entity("IfcAxis2Placement3D", Location=_pt(loc),
                               Axis=_dir(z), RefDirection=_dir(x))

    def _place(loc=(0, 0, 0), x=(1, 0, 0), rel_to=None):
        return m.create_entity("IfcLocalPlacement", PlacementRelTo=rel_to,
                               RelativePlacement=_ax3(loc, (0, 0, 1), x))

    def _box(body, xdim, ydim, height, center=(0.0, 0.0)):
        prof = m.create_entity("IfcRectangleProfileDef", ProfileType="AREA",
                               Position=_ax2(center), XDim=float(xdim), YDim=float(ydim))
        solid = m.create_entity("IfcExtrudedAreaSolid", SweptArea=prof, Position=_ax3(),
                                ExtrudedDirection=_dir((0, 0, 1)), Depth=float(height))
        rep = m.create_entity("IfcShapeRepresentation", ContextOfItems=body,
                              RepresentationIdentifier="Body",
                              RepresentationType="SweptSolid", Items=[solid])
        return m.create_entity("IfcProductDefinitionShape", Representations=[rep])

    def _polygon(body, pts, height):
        ifc_pts = [_pt((p[0], p[1])) for p in pts]
        ifc_pts.append(ifc_pts[0])
        curve = m.create_entity("IfcPolyline", Points=ifc_pts)
        prof = m.create_entity("IfcArbitraryClosedProfileDef",
                               ProfileType="AREA", OuterCurve=curve)
        solid = m.create_entity("IfcExtrudedAreaSolid", SweptArea=prof, Position=_ax3(),
                                ExtrudedDirection=_dir((0, 0, 1)), Depth=float(height))
        rep = m.create_entity("IfcShapeRepresentation", ContextOfItems=body,
                              RepresentationIdentifier="Body",
                              RepresentationType="SweptSolid", Items=[solid])
        return m.create_entity("IfcProductDefinitionShape", Representations=[rep])

    project = ifcopenshell.api.root.create_entity(
        m, ifc_class="IfcProject",
        name=data.get("project", {}).get("name", "Project"))
    ifcopenshell.api.unit.assign_unit(m, units=[
        ifcopenshell.api.unit.add_si_unit(m, unit_type="LENGTHUNIT"),
        ifcopenshell.api.unit.add_si_unit(m, unit_type="AREAUNIT"),
        ifcopenshell.api.unit.add_si_unit(m, unit_type="VOLUMEUNIT"),
    ])
    ctx = ifcopenshell.api.context.add_context(m, context_type="Model")
    body = ifcopenshell.api.context.add_context(
        m, context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=ctx)

    site = ifcopenshell.api.root.create_entity(m, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.root.create_entity(m, ifc_class="IfcBuilding", name="Building")
    ifcopenshell.api.aggregate.assign_object(m, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(m, products=[building], relating_object=site)

    storeys, elevations = {}, {}
    for s in data.get("storeys", []):
        st = ifcopenshell.api.root.create_entity(
            m, ifc_class="IfcBuildingStorey", name=s.get("name"))
        st.Elevation = float(s.get("elevation", 0.0))
        ifcopenshell.api.aggregate.assign_object(m, products=[st], relating_object=building)
        storeys[s["id"]] = st
        elevations[s["id"]] = float(s.get("elevation", 0.0))

    def storey_of(key):
        return key if key in storeys else next(iter(storeys))

    contained: dict[str, list] = {k: [] for k in storeys}
    walls_map = {}

    for w in data.get("walls", []):
        sk = storey_of(w.get("storey"))
        z0 = elevations[sk]
        sx, sy = w["start"]
        ex, ey = w["end"]
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        ux, uy = (dx / length, dy / length) if length else (1.0, 0.0)
        wall = ifcopenshell.api.root.create_entity(
            m, ifc_class="IfcWall", name=f"Wall-{w['id']}")
        wall.Representation = _box(body, length, w["thickness"], w["height"],
                                   center=(length / 2.0, 0.0))
        wall.ObjectPlacement = _place(loc=(sx, sy, z0), x=(ux, uy, 0.0))
        walls_map[w["id"]] = {"e": wall, "storey": sk, "thickness": w["thickness"]}
        contained[sk].append(wall)

    def add_opening(spec, kind):
        host = walls_map[spec["wall"]]
        wall = host["e"]
        sk = host["storey"]
        sill = float(spec.get("sill_height", 0.0))
        depth = host.get("thickness", 0.15)
        op = ifcopenshell.api.root.create_entity(
            m, ifc_class="IfcOpeningElement", name=f"Opening-{spec['id']}")
        op.PredefinedType = "OPENING"
        op.Representation = _box(body, spec["width"], depth, spec["height"],
                                  center=(spec["width"] / 2.0, 0.0))
        op.ObjectPlacement = _place(loc=(spec["offset"], 0.0, sill),
                                    rel_to=wall.ObjectPlacement)
        m.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(),
                        RelatingBuildingElement=wall, RelatedOpeningElement=op)
        cls = "IfcDoor" if kind == "door" else "IfcWindow"
        elem = ifcopenshell.api.root.create_entity(
            m, ifc_class=cls, name=f"{kind.title()}-{spec['id']}")
        elem.OverallHeight = float(spec["height"])
        elem.OverallWidth = float(spec["width"])
        elem.Representation = _box(body, spec["width"],
                                   0.05 if kind == "door" else 0.03,
                                   spec["height"],
                                   center=(spec["width"] / 2.0, 0.0))
        elem.ObjectPlacement = _place(loc=(0.0, 0.0, 0.0), rel_to=op.ObjectPlacement)
        m.create_entity("IfcRelFillsElement", GlobalId=ifcopenshell.guid.new(),
                        RelatingOpeningElement=op, RelatedBuildingElement=elem)
        contained[sk].append(elem)

    for d in data.get("doors", []):
        add_opening(d, "door")

    for r in data.get("rooms", []):
        sk = storey_of(r.get("storey"))
        z0 = elevations[sk]
        space = ifcopenshell.api.root.create_entity(
            m, ifc_class="IfcSpace", name=r.get("name", r["id"]))
        space.PredefinedType = "INTERNAL"
        space.Representation = _polygon(body, r["boundary"], r["height"])
        space.ObjectPlacement = _place(loc=(0.0, 0.0, z0))
        ifcopenshell.api.aggregate.assign_object(
            m, products=[space], relating_object=storeys[sk])
        if add_floor_slabs:
            slab = ifcopenshell.api.root.create_entity(
                m, ifc_class="IfcSlab", name=f"Floor-{r['id']}")
            slab.PredefinedType = "FLOOR"
            slab.Representation = _polygon(body, r["boundary"], slab_thickness)
            slab.ObjectPlacement = _place(loc=(0.0, 0.0, z0 - slab_thickness))
            contained[sk].append(slab)

    for sk, items in contained.items():
        if items:
            ifcopenshell.api.spatial.assign_container(
                m, products=items, relating_structure=storeys[sk])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.write(out_path)
    counts = {c: len(m.by_type(c)) for c in
              ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcSpace"]}
    logger.info("Wrote %s  ->  %s", out_path, counts)
    return out_path
