"""IFC export: wall geometry → JSON → IFC4 model."""

from __future__ import annotations

import glob
import json
import logging
import math
import os

import numpy as np
import open3d as o3d

from .config import WallSegConfig, IfcExportConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wall segmentation bridge
# ---------------------------------------------------------------------------

def segment_walls_for_ifc(pcd, wall_seg_cfg: WallSegConfig):
    """Segment walls from a room cloud (returns walls list + n_directions).

    Delegates to wall_segmentation.segment_walls so that wall ordering
    matches the wall images used for opening detection.
    """
    from .wall_segmentation import segment_walls

    walls = segment_walls(pcd, wall_seg_cfg)
    n_directions = 0
    if walls and "_theta_peaks" in walls[0]:
        n_directions = len(walls[0]["_theta_peaks"])
    elif walls:
        up_axis = wall_seg_cfg.up_axis
        ha, hb = [a for a in range(3) if a != up_axis]
        angles = set()
        for w in walls:
            a = round(float(np.arctan2(w["normal_2d"][1], w["normal_2d"][0]) % np.pi), 2)
            angles.add(a)
        n_directions = len(angles)
    return walls, n_directions


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

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
        "fill_ratio": fill_ratio,
        "is_exterior": is_exterior,
    }


# ---------------------------------------------------------------------------
# Merge / dedup
# ---------------------------------------------------------------------------

def _angle_close(a1, a2, tol_rad):
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff) < tol_rad


def _normals_opposed(g1, g2):
    return np.dot(g1["normal_2d"], g2["normal_2d"]) < 0


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
    u1_min, u1_max = 0.0, length1
    s2 = np.array(g2["start"])
    e2 = np.array(g2["end"])
    p2a = float(np.dot(s2 - s1, u_dir))
    p2b = float(np.dot(e2 - s1, u_dir))
    u2_min, u2_max = min(p2a, p2b), max(p2a, p2b)
    return min(u1_max, u2_max) - max(u1_min, u2_min)


def _walls_spatially_close(g1, g2, tol=0.5):
    s1, e1 = np.array(g1["start"]), np.array(g1["end"])
    s2, e2 = np.array(g2["start"]), np.array(g2["end"])
    if _u_overlap(g1, g2) > 0:
        return True
    return (np.linalg.norm(s1 - s2) < tol or np.linalg.norm(s1 - e2) < tol
            or np.linalg.norm(e1 - s2) < tol or np.linalg.norm(e1 - e2) < tol)


def merge_wall_faces(wall_geos, cfg: IfcExportConfig, angle_tol_deg: float = 10.0):
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
            if _offset_distance(wall_geos[i], wall_geos[j]) > cfg.max_merge_thickness:
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
                g["thickness"] = cfg.default_thickness
            merged.append(g)
        else:
            geos = [wall_geos[i] for i in indices]
            avg_offset = np.mean([g["offset"] for g in geos])
            thickness = max(abs(g["offset"] - avg_offset) * 2 for g in geos)
            if thickness < 0.05:
                thickness = cfg.default_thickness
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


def _opening_pos_close(a, b, tol=0.20):
    """Check if two openings are at the same position (world or offset)."""
    if "world_xy" in a and "world_xy" in b:
        return (abs(a["world_xy"][0] - b["world_xy"][0]) < tol
                and abs(a["world_xy"][1] - b["world_xy"][1]) < tol)
    return abs(a.get("offset", 0) - b.get("offset", 0)) < tol


def _opening_duplicate(op, existing, pos_tol=0.15, size_tol=0.15):
    for e in existing:
        if e["label"] != op["label"]:
            continue
        if (_opening_pos_close(e, op, pos_tol)
                and abs(e["width"] - op["width"]) < size_tol
                and abs(e["height"] - op["height"]) < size_tol):
            return True
    return False


def _openings_match(a, b, pos_tol=0.20, size_tol=0.20):
    """Check if two openings refer to the same physical door/window."""
    if a["label"] != b["label"]:
        return False
    return (_opening_pos_close(a, b, pos_tol)
            and abs(a["width"] - b["width"]) < size_tol
            and abs(a["height"] - b["height"]) < size_tol)


def deduplicate_walls(all_room_walls, cfg: IfcExportConfig, angle_tol_deg: float = 10.0):
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
            if _offset_distance(flat[i], flat[j]) > cfg.dedup_offset_tol:
                continue
            if not _walls_spatially_close(flat[i], flat[j]):
                continue
            u_gap = -_u_overlap(flat[i], flat[j])
            if u_gap > cfg.dedup_offset_tol:
                continue
            logger.debug(
                "  DEDUP MERGE: %s[%d] (len=%.2f) + %s[%d] (len=%.2f)  "
                "off_diff=%.3f  u_gap=%.2f",
                flat[i]["_room"], flat[i]["_idx"], flat[i]["length"],
                flat[j]["_room"], flat[j]["_idx"], flat[j]["length"],
                _offset_distance(flat[i], flat[j]), u_gap,
            )
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    unique_walls = []
    for root_idx, indices in groups.items():
        keeper = flat[root_idx].copy()
        keeper_room = keeper["_room"]

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
                keeper["u_min"] = keeper["u_min"] + new_u_min
                keeper["u_max"] = keeper["u_min"] + keeper["length"]

        keeper_u_min = keeper["u_min"]

        sides: dict[str, list] = {}
        for op in keeper.get("_openings", []):
            tagged = op.copy()
            tagged["_src_room"] = keeper_room
            sides.setdefault(keeper_room, []).append(tagged)

        for idx in indices:
            if idx == root_idx:
                continue
            other = flat[idx]
            other_room = other["_room"]
            for op in other.get("_openings", []):
                tagged = op.copy()
                tagged["_src_room"] = other_room
                sides.setdefault(other_room, []).append(tagged)

        all_openings_tagged = [op for ops in sides.values() for op in ops]
        n_sides = len(sides)

        if n_sides >= 2:
            confirmed = []
            used = set()
            for i, a in enumerate(all_openings_tagged):
                for j, b in enumerate(all_openings_tagged):
                    if j <= i or j in used:
                        continue
                    if a["_src_room"] == b["_src_room"]:
                        continue
                    if _openings_match(a, b):
                        merged_op = a.copy()
                        if "world_xy" in a and "world_xy" in b:
                            merged_op["world_xy"] = [
                                round((a["world_xy"][0] + b["world_xy"][0]) / 2, 4),
                                round((a["world_xy"][1] + b["world_xy"][1]) / 2, 4),
                            ]
                        elif "offset" in a and "offset" in b:
                            merged_op["offset"] = round((a["offset"] + b["offset"]) / 2, 3)
                        merged_op["width"] = round((a["width"] + b["width"]) / 2, 3)
                        merged_op["height"] = round((a["height"] + b["height"]) / 2, 3)
                        merged_op.pop("_src_room", None)
                        if not _opening_duplicate(merged_op, confirmed):
                            confirmed.append(merged_op)
                        used.add(i)
                        used.add(j)
                        break
            n_dropped = len(all_openings_tagged) - len(used)
            if n_dropped > 0:
                logger.info("  Wall consensus: kept %d confirmed openings, "
                            "dropped %d single-side detections",
                            len(confirmed), n_dropped)
            keeper["_openings"] = confirmed
        else:
            deduped = []
            for op in all_openings_tagged:
                clean = {k: v for k, v in op.items() if k != "_src_room"}
                if not _opening_duplicate(clean, deduped):
                    deduped.append(clean)
            keeper["_openings"] = deduped

        rooms_set = set()
        for idx in indices:
            rooms_set.add(flat[idx]["_room"])
        keeper["_rooms"] = sorted(rooms_set)
        keeper.pop("_room", None)
        keeper.pop("_idx", None)
        unique_walls.append(keeper)
    return unique_walls


# ---------------------------------------------------------------------------
# Openings loading
# ---------------------------------------------------------------------------

def load_openings(openings_dir: str):
    rooms: dict[str, list] = {}
    pixel_m = 0.04

    all_path = os.path.join(openings_dir, "all_openings.json")
    if os.path.exists(all_path):
        with open(all_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "rooms" in data:
            pixel_m = data.get("pixel_m", 0.04)
            raw_rooms = data["rooms"]
        elif isinstance(data, dict):
            raw_rooms = data
        else:
            raw_rooms = {}
        for room_name, walls_data in raw_rooms.items():
            rooms[room_name] = walls_data if isinstance(walls_data, list) else []
        return rooms, pixel_m

    room_dirs = sorted(glob.glob(os.path.join(openings_dir, "room_*")))
    for rd in room_dirs:
        jp = os.path.join(rd, "openings.json")
        if not os.path.exists(jp):
            continue
        room_name = os.path.basename(rd)
        with open(jp) as f:
            data = json.load(f)
        if isinstance(data, dict) and "walls" in data:
            pixel_m = data.get("pixel_m", pixel_m)
            rooms[room_name] = data["walls"]
        elif isinstance(data, list):
            rooms[room_name] = data
    return rooms, pixel_m


def _match_wall_to_merged(wall_meta_entry, merged_walls, angle_tol_deg=15.0, offset_tol=0.30):
    """Find the merged wall that best matches a source wall by angle and offset."""
    src_n = np.array(wall_meta_entry["normal_2d"])
    src_angle = float(np.arctan2(src_n[1], src_n[0]) % np.pi)
    src_offset = wall_meta_entry["offset"]
    src_u_min = wall_meta_entry["u_min"]
    src_u_max = wall_meta_entry["u_max"]
    angle_tol = np.deg2rad(angle_tol_deg)

    u_axis = np.array([-src_n[1], src_n[0]])
    src_start = (src_u_min * u_axis + src_offset * src_n).tolist()
    src_end = (src_u_max * u_axis + src_offset * src_n).tolist()
    src_geo = {"normal_2d": src_n, "offset": src_offset,
               "u_min": src_u_min, "u_max": src_u_max,
               "start": src_start, "end": src_end}

    best_idx = None
    best_score = float("inf")
    for mi, mw in enumerate(merged_walls):
        mw_angle = mw.get("angle", float(np.arctan2(mw["normal_2d"][1], mw["normal_2d"][0]) % np.pi))
        angle_diff = abs(mw_angle - src_angle)
        angle_diff = min(angle_diff, np.pi - angle_diff)
        if angle_diff > angle_tol:
            continue
        off_diff = _offset_distance(src_geo, mw)
        if off_diff > offset_tol:
            continue
        overlap = _u_overlap(src_geo, mw)
        if overlap < -offset_tol:
            continue
        score = off_diff + angle_diff
        if score < best_score:
            best_score = score
            best_idx = mi
    return best_idx


def attach_openings(merged_walls, wall_meta, room_openings, pixel_m):
    for mw in merged_walls:
        mw["_openings"] = []

    wall_meta_by_name = {}
    for wm in wall_meta:
        wall_meta_by_name[wm["name"]] = wm

    use_geo_match = len(wall_meta_by_name) > 0

    for entry in room_openings:
        wall_name = entry.get("wall", "")

        if use_geo_match:
            wm = wall_meta_by_name.get(wall_name)
            if wm is None:
                continue
            merged_idx = _match_wall_to_merged(wm, merged_walls)
            if merged_idx is None:
                logger.debug("  Opening on %s: no matching merged wall found", wall_name)
                continue
            mw = merged_walls[merged_idx]
            src_n = np.array(wm["normal_2d"])
            opposed = np.dot(src_n, mw["normal_2d"]) < 0
            if opposed:
                src_u_min = -wm["u_max"]
            else:
                src_u_min = wm["u_min"]
            u_shift = src_u_min - mw["u_min"]
        else:
            seg_idx = int(wall_name.replace("wall_", "")) - 1 if wall_name.startswith("wall_") else -1
            merged_idx = None
            opposed = False
            for mi, mw in enumerate(merged_walls):
                if seg_idx in mw.get("source_indices", []):
                    merged_idx = mi
                    break
            if merged_idx is None:
                continue
            u_shift = 0.0

        for op in entry.get("openings", []):
            if op["label"] != "door":
                continue
            offset_m = op.get("offset_m", op["bbox_px"][0] * pixel_m)
            if opposed:
                src_length = wm["u_max"] - wm["u_min"]
                offset_m = src_length - offset_m - op["width_m"]
            opening = {
                "label": op["label"],
                "offset": round(offset_m + u_shift, 3),
                "width": op["width_m"],
                "height": op["height_m"],
                "sill_height": op.get("sill_m", 0.0),
            }
            merged_walls[merged_idx]["_openings"].append(opening)


# ---------------------------------------------------------------------------
# Room boundaries
# ---------------------------------------------------------------------------

def _convex_hull_2d(points):
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts.tolist()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p.tolist())
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p.tolist())
    return lower[:-1] + upper[:-1]


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
            [r_min[0], r_min[1]],
            [r_max[0], r_min[1]],
            [r_max[0], r_max[1]],
            [r_min[0], r_max[1]],
        ])
        cos_b, sin_b = math.cos(angle), math.sin(angle)
        corners_world = [
            [float(c[0] * cos_b - c[1] * sin_b),
             float(c[0] * sin_b + c[1] * cos_b)]
            for c in corners_rot
        ]
        boundaries[room_name] = corners_world
    return boundaries


# ---------------------------------------------------------------------------
# Manhattan-direction detection
# ---------------------------------------------------------------------------

def _find_manhattan_angles(all_room_walls, angle_tol_deg=10.0):
    """Find the 2 dominant wall angles across all rooms."""
    angles = []
    lengths = []
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
        manhattan = [b[0] for b in buckets[:1]]
    else:
        weighted_angles = []
        for a, l in zip(angles, lengths):
            for i, (ba, _) in enumerate(buckets[:2]):
                if min(abs(a - ba), np.pi - abs(a - ba)) < tol:
                    weighted_angles.append((i, a, l))
                    break
        bucket_avg = [0.0, 0.0]
        bucket_wt = [0.0, 0.0]
        for bi, a, l in weighted_angles:
            bucket_avg[bi] += a * l
            bucket_wt[bi] += l
        for i in range(2):
            if bucket_wt[i] > 0:
                bucket_avg[i] /= bucket_wt[i]
        primary = bucket_avg[0]
        secondary = primary + np.pi / 2
        if secondary > np.pi:
            secondary -= np.pi
        if min(abs(bucket_avg[1] - secondary), np.pi - abs(bucket_avg[1] - secondary)) > np.pi / 4:
            secondary = primary - np.pi / 2
            if secondary < 0:
                secondary += np.pi
        manhattan = [primary, secondary]

    logger.info(
        "Manhattan angles: %s (total length: %s)",
        [f"{np.degrees(a):.1f}°" for a in manhattan],
        [f"{b[1]:.1f}m" for b in buckets[:2]],
    )
    return manhattan


def _filter_non_manhattan(all_room_walls, angle_tol_deg, ifc_cfg):
    """Apply stricter thresholds to walls not aligned with dominant directions."""
    manhattan = _find_manhattan_angles(all_room_walls, angle_tol_deg)
    if len(manhattan) < 2:
        return all_room_walls

    tol = np.deg2rad(angle_tol_deg)

    non_manhattan_angles: dict[int, list[str]] = {}
    for room_name, walls in all_room_walls.items():
        for w in walls:
            on_manhattan = any(
                min(abs(w["angle"] - ma), np.pi - abs(w["angle"] - ma)) < tol
                for ma in manhattan
            )
            if not on_manhattan:
                bucket = round(np.degrees(w["angle"]))
                non_manhattan_angles.setdefault(bucket, []).append(room_name)

    multi_room_angles = set()
    for bucket, rooms in non_manhattan_angles.items():
        if len(set(rooms)) >= 2:
            multi_room_angles.add(bucket)
            logger.info("  Non-Manhattan angle ~%d° seen in %d rooms — keeping",
                        bucket, len(set(rooms)))

    strict_fill = max(ifc_cfg.min_wall_fill_ratio * 2.5, 0.35)
    strict_min_length = max(ifc_cfg.min_wall_length_m * 2, 0.8)

    filtered = {}
    for room_name, walls in all_room_walls.items():
        kept = []
        for w in walls:
            on_manhattan = any(
                min(abs(w["angle"] - ma), np.pi - abs(w["angle"] - ma)) < tol
                for ma in manhattan
            )
            if on_manhattan:
                kept.append(w)
                continue

            bucket = round(np.degrees(w["angle"]))
            seen_multi_room = bucket in multi_room_angles
            fill = w.get("fill_ratio", 1.0)
            length = w.get("length", 0)

            if seen_multi_room and fill >= strict_fill and length >= strict_min_length:
                logger.info(
                    "  %s: keeping non-Manhattan wall (angle=%.1f°, "
                    "fill=%.2f, len=%.2f) — multi-room + strict OK",
                    room_name, np.degrees(w["angle"]), fill, length,
                )
                kept.append(w)
            else:
                logger.info(
                    "  %s: removing non-Manhattan wall (angle=%.1f°, "
                    "fill=%.2f, len=%.2f, multi_room=%s)",
                    room_name, np.degrees(w["angle"]), fill, length,
                    seen_multi_room,
                )
        filtered[room_name] = kept

    before = sum(len(ws) for ws in all_room_walls.values())
    after = sum(len(ws) for ws in filtered.values())
    if before != after:
        logger.info("Non-Manhattan filter: %d → %d walls (removed %d)",
                    before, after, before - after)
    return filtered


def snap_wall_angles(all_room_walls, angle_tol_deg=10.0):
    """Snap slightly-off-axis walls to the nearest Manhattan direction.

    Preserves the wall's lateral position (offset from origin along the normal)
    and recomputes start/end on the new axis so the wall doesn't drift inward
    or outward.
    """
    manhattan = _find_manhattan_angles(all_room_walls, angle_tol_deg)
    if len(manhattan) < 1:
        return all_room_walls

    tol = np.deg2rad(angle_tol_deg)
    snapped = 0

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

            # Preserve offset: project the wall midpoint onto the new normal
            mid = (s + e) / 2.0
            new_offset = float(np.dot(mid, n2d_new))

            # Project original endpoints onto the new wall direction
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
            if "u_axis" in w:
                w["u_axis"] = [float(wall_dir[0]), float(wall_dir[1]), 0.0]
            if "normal_3d" in w:
                n3d = np.array(w["normal_3d"])
                axes = [i for i in range(3) if abs(n3d[i]) > 1e-6 or i != 2]
                ha, hb = [a for a in range(3) if a != 2][:2]
                n3d_new = np.zeros(3)
                n3d_new[ha] = n2d_new[0]
                n3d_new[hb] = n2d_new[1]
                n3d_new /= np.linalg.norm(n3d_new) + 1e-9
                w["normal_3d"] = n3d_new.tolist()

            snapped += 1
            logger.debug(
                "  %s: snapped wall angle %.1f° → %.1f° (diff=%.2f°)",
                room_name, np.degrees(wall_angle), np.degrees(best_ma),
                np.degrees(best_diff),
            )

    if snapped:
        logger.info("Snapped %d wall angles to Manhattan directions", snapped)
    return all_room_walls


# ---------------------------------------------------------------------------
# Endpoint snapping
# ---------------------------------------------------------------------------

def snap_wall_endpoints(walls, tol=0.15):
    """Snap nearby wall endpoints together and to wall bodies (T-junctions)."""
    snapped_ll = 0
    snapped_t = 0

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
                        snapped_ll += 1
                    else:
                        snapped_t += 1

    if snapped_ll or snapped_t:
        logger.info("Snapped endpoints: %d L-joints, %d T-joints (tol=%.2fm)",
                    snapped_ll, snapped_t, tol)

    extended = 0
    for i in range(len(walls)):
        si = np.array(walls[i]["start"])
        ei = np.array(walls[i]["end"])
        wall_dir = ei - si
        wall_len = np.linalg.norm(wall_dir)
        if wall_len < 0.01:
            continue
        wall_dir /= wall_len

        for ki, pt, sign in [("start", si, -1.0), ("end", ei, 1.0)]:
            for j in range(len(walls)):
                if i == j:
                    continue
                sj = np.array(walls[j]["start"])
                ej = np.array(walls[j]["end"])
                seg = ej - sj
                seg_len = np.linalg.norm(seg)
                if seg_len < 0.01:
                    continue
                t = np.dot(pt - sj, seg) / (seg_len ** 2)
                if t < -0.05 or t > 1.05:
                    continue
                closest = sj + t * seg
                dist = np.linalg.norm(pt - closest)
                if dist < 0.01:
                    half_thick = walls[j].get("thickness", 0.15) / 2.0
                    new_pt = (pt + sign * wall_dir * half_thick).tolist()
                    walls[i][ki] = new_pt
                    extended += 1
                    break

    if extended:
        logger.info("Extended %d endpoints past junctions for thickness", extended)
    return walls


def extend_walls_to_corners(walls, max_extend_m=0.50):
    """Extend wall endpoints to meet perpendicular walls at corners.

    For each free endpoint (not already touching another wall), find the
    nearest perpendicular wall line and extend this wall to intersect it.
    """
    extended = 0
    for i in range(len(walls)):
        si = np.array(walls[i]["start"])
        ei = np.array(walls[i]["end"])
        di = ei - si
        len_i = np.linalg.norm(di)
        if len_i < 0.01:
            continue
        di_hat = di / len_i

        for ki, pt, sign in [("start", si, -1.0), ("end", ei, 1.0)]:
            touching = False
            for j in range(len(walls)):
                if i == j:
                    continue
                for kj in ("start", "end"):
                    if np.linalg.norm(np.array(walls[j][kj]) - pt) < 0.05:
                        touching = True
                        break
                if touching:
                    break
            if touching:
                continue

            best_dist = max_extend_m
            best_pt = None
            for j in range(len(walls)):
                if i == j:
                    continue
                sj = np.array(walls[j]["start"])
                ej = np.array(walls[j]["end"])
                dj = ej - sj
                len_j = np.linalg.norm(dj)
                if len_j < 0.01:
                    continue

                cross_val = di_hat[0] * dj[1] - di_hat[1] * dj[0]
                if abs(cross_val) < 0.3 * len_j:
                    continue

                diff = sj - pt
                t_i = (diff[0] * dj[1] - diff[1] * dj[0]) / cross_val
                t_j = (diff[0] * di_hat[1] - diff[1] * di_hat[0]) / cross_val

                if t_j < -0.1 or t_j > 1.1:
                    continue
                if sign * t_i < -max_extend_m:
                    continue
                dist = abs(t_i)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (pt + t_i * di_hat).tolist()

            if best_pt is not None:
                walls[i][ki] = best_pt
                extended += 1

    if extended:
        logger.info("Extended %d wall endpoints to meet corners (max %.2fm)",
                    extended, max_extend_m)
    return walls


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------

def _load_wall_meta(wall_image_dir: str | None) -> dict[str, list]:
    """Load wall_meta.json for each room from the wall images directory."""
    meta_all: dict[str, list] = {}
    if not wall_image_dir or not os.path.isdir(wall_image_dir):
        return meta_all
    for rd in sorted(glob.glob(os.path.join(wall_image_dir, "room_*"))):
        mp = os.path.join(rd, "wall_meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta_all[os.path.basename(rd)] = json.load(f)
    return meta_all


def _combine_wall_clouds(walls, source_indices):
    """Combine point clouds from source wall segments into one."""
    all_pts = []
    all_cols = []
    has_colors = False
    for idx in source_indices:
        if idx < len(walls):
            cloud = walls[idx]["cloud"]
            all_pts.append(np.asarray(cloud.points))
            if cloud.has_colors():
                has_colors = True
                all_cols.append(np.asarray(cloud.colors))
    if not all_pts:
        return None
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    if has_colors and all_cols:
        combined.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    return combined


def _detect_doors_on_merged_walls(merged_walls, walls_raw, wall_seg_cfg, wall_proc_cfg,
                                  room_name, out_dir=None, use_sam=False):
    """Flatten each merged wall, detect door-shaped gaps, attach to wall."""
    from .wall_segmentation import flatten_wall
    from .wall_image_processing import find_void_components, merge_fragments

    import cv2

    for i, mw in enumerate(merged_walls):
        mw["_openings"] = []
        src_indices = mw.get("source_indices", [i])
        combined = _combine_wall_clouds(walls_raw, src_indices)
        if combined is None or len(combined.points) < 50:
            continue

        wall_dict = {"cloud": combined, "normal_2d": np.array(mw["normal_2d"]),
                     "offset": mw["offset"]}
        flat = flatten_wall(wall_dict, wall_seg_cfg)
        wall_img = flat["image"]
        img_h, img_w = wall_img.shape
        logger.info("  %s wall %d: %d pts → %dx%d img (%.2fm x %.2fm)  src=%s",
                     room_name, i + 1, len(combined.points), img_w, img_h,
                     img_w * wall_seg_cfg.flat_pixel_m, img_h * wall_seg_cfg.flat_pixel_m,
                     src_indices)

        pixel_m = wall_seg_cfg.flat_pixel_m
        flat_u_min = float(flat["u"].min())
        wall_u_min = mw["u_min"]
        u_shift = flat_u_min - wall_u_min

        components = find_void_components(wall_img, min_void_px=wall_proc_cfg.min_void_px)
        for comp in components:
            comp["sam_score"] = 1.0
        gaps = merge_fragments(components, merge_margin_px=wall_proc_cfg.door_floor_margin_px)

        n_doors = 0
        for gap in gaps:
            gx, gy, gw, gh = gap["bbox"]
            w_m = gw * pixel_m
            h_m = gh * pixel_m
            bbox_bottom = gy + gh - 1
            floor_row = img_h - 1
            touches_floor = (floor_row - bbox_bottom) <= wall_proc_cfg.door_floor_margin_px
            bbox_area = gw * gh
            rectangularity = gap["area"] / bbox_area if bbox_area > 0 else 0.0

            wall_length_m = img_w * pixel_m
            too_wide_for_wall = (wall_length_m > 0
                and w_m / wall_length_m > wall_proc_cfg.door_max_wall_width_frac)

            is_door = (
                touches_floor
                and not too_wide_for_wall
                and rectangularity >= wall_proc_cfg.min_rectangularity
                and wall_proc_cfg.door_min_width_m <= w_m <= wall_proc_cfg.door_max_width_m
                and wall_proc_cfg.door_min_height_m <= h_m <= wall_proc_cfg.door_max_height_m
            )
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
                "width": round(w_m, 3),
                "height": round(h_m, 3),
            })
            n_doors += 1

        if out_dir:
            room_dir = os.path.join(out_dir, room_name)
            os.makedirs(room_dir, exist_ok=True)
            rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
            u_axis_2d = np.array(flat["u_axis"][:2])
            for op in mw["_openings"]:
                door_world = np.array(op["world_xy"])
                door_u = float(np.dot(door_world, u_axis_2d))
                ox_px = int(round((door_u - flat_u_min) / pixel_m))
                ow_px = int(round(op["width"] / pixel_m))
                oh_px = int(round(op["height"] / pixel_m))
                oy_px = img_h - oh_px
                cv2.rectangle(rgb, (ox_px, oy_px), (ox_px + ow_px - 1, img_h - 1), (0, 255, 0), 1)
                cv2.putText(rgb, f"DOOR {op['width']:.2f}x{op['height']:.2f}m",
                            (ox_px + 2, oy_px - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(room_dir, f"merged_wall_{i + 1:02d}.png"), rgb)

        if n_doors:
            logger.info("  %s wall %d: %d doors detected", room_name, i + 1, n_doors)


def build_building_json(
    room_cloud_paths: list[str],
    wall_seg_cfg: WallSegConfig,
    ifc_cfg: IfcExportConfig,
    wall_proc_cfg=None,
    out_dir: str | None = None,
    use_sam: bool = False,
) -> dict:
    """Build the canonical building JSON from room clouds.

    Walls are segmented, merged, then flattened for door detection — so doors
    are detected on the final merged geometry rather than pre-merge fragments.
    """
    all_room_walls: dict[str, list] = {}
    skipped = []
    _debug: dict[str, dict] = {}

    for cloud_path in room_cloud_paths:
        fname = os.path.splitext(os.path.basename(cloud_path))[0]
        room_name = fname.replace("_walls", "")
        logger.info("Processing %s", room_name)

        pcd = o3d.io.read_point_cloud(cloud_path)
        if wall_seg_cfg.voxel_m > 0:
            pcd = pcd.voxel_down_sample(wall_seg_cfg.voxel_m)

        walls, n_dirs = segment_walls_for_ifc(pcd, wall_seg_cfg)
        logger.info("  %d wall segments, %d directions", len(walls), n_dirs)

        if n_dirs < ifc_cfg.min_directions:
            skipped.append(room_name)
            continue

        wall_geos_raw = [compute_wall_geometry(w, wall_seg_cfg.up_axis) for w in walls]
        wall_geos = []
        filtered_reasons: list[str] = []
        for raw_idx, g in enumerate(wall_geos_raw):
            if g["length"] < ifc_cfg.min_wall_length_m:
                filtered_reasons.append(f"  FILTERED: length {g['length']:.2f}m < {ifc_cfg.min_wall_length_m}")
                continue
            if g["length"] > ifc_cfg.max_wall_length_m:
                filtered_reasons.append(f"  FILTERED: length {g['length']:.2f}m > {ifc_cfg.max_wall_length_m}")
                continue
            aspect = g["length"] / max(g["height"], 1e-3)
            if aspect < ifc_cfg.min_wall_aspect_ratio:
                filtered_reasons.append(f"  FILTERED: aspect {aspect:.2f} < {ifc_cfg.min_wall_aspect_ratio}")
                continue
            if g["thickness"] > ifc_cfg.max_wall_thickness_m:
                logger.debug("  CLAMPED: thickness %.2fm → %.2fm", g["thickness"], ifc_cfg.default_thickness)
                g["thickness"] = ifc_cfg.default_thickness
            if g.get("is_exterior"):
                logger.debug("  EXTERIOR wall (one-sided points): thickness → %.2fm", ifc_cfg.exterior_thickness)
                g["thickness"] = ifc_cfg.exterior_thickness
            if g["fill_ratio"] < ifc_cfg.min_wall_fill_ratio:
                filtered_reasons.append(
                    f"  FILTERED: fill_ratio {g['fill_ratio']:.2f} < {ifc_cfg.min_wall_fill_ratio} "
                    f"(len={g['length']:.2f}m, h={g['height']:.2f}m)"
                )
                continue
            g["_raw_idx"] = raw_idx
            wall_geos.append(g)

        for reason in filtered_reasons:
            logger.debug(reason)
        logger.info("  %s: %d raw → %d after filters (%d removed)",
                     room_name, len(wall_geos_raw), len(wall_geos), len(filtered_reasons))

        if not wall_geos:
            skipped.append(room_name)
            continue

        merged = merge_wall_faces(wall_geos, ifc_cfg, wall_seg_cfg.angle_tol_deg)
        logger.info("  %s: %d after merge_wall_faces", room_name, len(merged))

        if wall_proc_cfg is not None:
            wall_images_dir = os.path.join(out_dir, "wall_images") if out_dir else None
            _detect_doors_on_merged_walls(
                merged, walls, wall_seg_cfg, wall_proc_cfg,
                room_name, out_dir=wall_images_dir, use_sam=use_sam,
            )

        _debug[room_name] = {
            "n_segments": len(walls),
            "n_directions": n_dirs,
            "n_raw_geos": len(wall_geos_raw),
            "n_filtered": len(filtered_reasons),
            "n_after_filter": len(wall_geos),
            "n_after_merge": len(merged),
            "raw_geos": wall_geos_raw,
            "filtered_geos": wall_geos,
            "merged_geos": merged,
        }

        all_room_walls[room_name] = merged

    logger.info("Processed %d rooms, skipped %d", len(all_room_walls), len(skipped))

    all_room_walls = _filter_non_manhattan(all_room_walls, wall_seg_cfg.angle_tol_deg, ifc_cfg)
    all_room_walls = snap_wall_angles(all_room_walls, wall_seg_cfg.angle_tol_deg)

    pre_dedup_count = sum(len(ws) for ws in all_room_walls.values())
    unique_walls = deduplicate_walls(all_room_walls, ifc_cfg, wall_seg_cfg.angle_tol_deg)
    logger.info("Deduplicated: %d → %d unique walls (removed %d)",
                pre_dedup_count, len(unique_walls), pre_dedup_count - len(unique_walls))

    unique_walls = snap_wall_endpoints(unique_walls, tol=ifc_cfg.snap_tolerance_m)
    unique_walls = extend_walls_to_corners(unique_walls)

    manhattan = _find_manhattan_angles(all_room_walls, wall_seg_cfg.angle_tol_deg)
    if len(manhattan) >= 2:
        tol_rad = np.deg2rad(wall_seg_cfg.angle_tol_deg)
        for w in unique_walls:
            s = np.array(w["start"])
            e = np.array(w["end"])
            wall_angle = w.get("angle", 0.0)
            best_ma = None
            best_diff = tol_rad
            for ma in manhattan:
                diff = min(abs(wall_angle - ma), np.pi - abs(wall_angle - ma))
                if diff < best_diff:
                    best_diff = diff
                    best_ma = ma
            if best_ma is None:
                continue
            n2d = np.array([np.cos(best_ma), np.sin(best_ma)])
            wall_dir = np.array([-n2d[1], n2d[0]])
            if np.dot(wall_dir, e - s) < 0:
                wall_dir = -wall_dir
            mid = (s + e) / 2.0
            offset = float(np.dot(mid, n2d))
            u_s = float(np.dot(s, wall_dir))
            u_e = float(np.dot(e, wall_dir))
            w["start"] = (u_s * wall_dir + offset * n2d).tolist()
            w["end"] = (u_e * wall_dir + offset * n2d).tolist()

    min_wall_len = 0.30
    before_cull = len(unique_walls)
    unique_walls = [w for w in unique_walls
                    if np.linalg.norm(np.array(w["end"]) - np.array(w["start"])) >= min_wall_len]
    if len(unique_walls) < before_cull:
        logger.info("Removed %d degenerate walls (< %.2fm)",
                    before_cull - len(unique_walls), min_wall_len)

    all_heights = sorted([w["height"] for w in unique_walls])
    if all_heights:
        storey_height = float(np.percentile(all_heights, 90))
        for w in unique_walls:
            w["height"] = storey_height
        logger.info("Normalized wall heights to %.2fm (90th percentile)", storey_height)

    storey_id = "L0"
    data: dict = {
        "project": {"name": ifc_cfg.project_name},
        "storeys": [{"id": storey_id, "name": "Ground Floor", "elevation": ifc_cfg.floor_elevation}],
        "walls": [], "doors": [], "windows": [], "rooms": [],
    }

    door_counter = 0

    for i, w in enumerate(unique_walls):
        wall_id = f"W{i + 1}"
        thickness = ifc_cfg.default_thickness

        data["walls"].append({
            "id": wall_id, "storey": storey_id,
            "start": [round(w["start"][0], 4), round(w["start"][1], 4)],
            "end": [round(w["end"][0], 4), round(w["end"][1], 4)],
            "height": round(w["height"], 3), "thickness": round(thickness, 3),
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

    logger.info(
        "JSON: %d walls, %d doors, %d windows, %d rooms",
        len(data["walls"]), len(data["doors"]), len(data["windows"]), len(data["rooms"]),
    )
    data["_debug"] = _debug
    return data


# ---------------------------------------------------------------------------
# IFC4 writer (requires ifcopenshell)
# ---------------------------------------------------------------------------

def build_ifc(data: dict, out_path: str = "model.ifc",
              add_floor_slabs: bool = True, slab_thickness: float = 0.2) -> str:
    """Translate the canonical building JSON into an IFC4 file."""
    import ifcopenshell
    import ifcopenshell.guid
    import ifcopenshell.api.context
    import ifcopenshell.api.unit
    import ifcopenshell.api.root
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.spatial

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
                              RepresentationIdentifier="Body", RepresentationType="SweptSolid",
                              Items=[solid])
        return m.create_entity("IfcProductDefinitionShape", Representations=[rep])

    def _polygon(body, pts, height):
        ifc_pts = [_pt((p[0], p[1])) for p in pts]
        ifc_pts.append(ifc_pts[0])
        curve = m.create_entity("IfcPolyline", Points=ifc_pts)
        prof = m.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=curve)
        solid = m.create_entity("IfcExtrudedAreaSolid", SweptArea=prof, Position=_ax3(),
                                ExtrudedDirection=_dir((0, 0, 1)), Depth=float(height))
        rep = m.create_entity("IfcShapeRepresentation", ContextOfItems=body,
                              RepresentationIdentifier="Body", RepresentationType="SweptSolid",
                              Items=[solid])
        return m.create_entity("IfcProductDefinitionShape", Representations=[rep])

    project = ifcopenshell.api.root.create_entity(
        m, ifc_class="IfcProject", name=data.get("project", {}).get("name", "Project"))
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
        st = ifcopenshell.api.root.create_entity(m, ifc_class="IfcBuildingStorey", name=s.get("name"))
        st.Elevation = float(s.get("elevation", 0.0))
        ifcopenshell.api.aggregate.assign_object(m, products=[st], relating_object=building)
        storeys[s["id"]] = st
        elevations[s["id"]] = float(s.get("elevation", 0.0))
    if not storeys:
        st = ifcopenshell.api.root.create_entity(m, ifc_class="IfcBuildingStorey", name="Storey")
        st.Elevation = 0.0
        ifcopenshell.api.aggregate.assign_object(m, products=[st], relating_object=building)
        storeys["__default__"] = st
        elevations["__default__"] = 0.0

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
        wall = ifcopenshell.api.root.create_entity(m, ifc_class="IfcWall", name=f"Wall-{w['id']}")
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
        op = ifcopenshell.api.root.create_entity(m, ifc_class="IfcOpeningElement",
                                                 name=f"Opening-{spec['id']}")
        op.PredefinedType = "OPENING"
        op.Representation = _box(body, spec["width"], depth, spec["height"],
                                  center=(spec["width"] / 2.0, 0.0))
        op.ObjectPlacement = _place(loc=(spec["offset"], 0.0, sill),
                                    rel_to=wall.ObjectPlacement)
        m.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(),
                        RelatingBuildingElement=wall, RelatedOpeningElement=op)
        cls = "IfcDoor" if kind == "door" else "IfcWindow"
        elem = ifcopenshell.api.root.create_entity(m, ifc_class=cls, name=f"{kind.title()}-{spec['id']}")
        elem.OverallHeight = float(spec["height"])
        elem.OverallWidth = float(spec["width"])
        elem.Representation = _box(body, spec["width"], 0.05 if kind == "door" else 0.03,
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
        space = ifcopenshell.api.root.create_entity(m, ifc_class="IfcSpace", name=r.get("name", r["id"]))
        space.PredefinedType = "INTERNAL"
        space.Representation = _polygon(body, r["boundary"], r["height"])
        space.ObjectPlacement = _place(loc=(0.0, 0.0, z0))
        ifcopenshell.api.aggregate.assign_object(m, products=[space], relating_object=storeys[sk])
        if add_floor_slabs:
            slab = ifcopenshell.api.root.create_entity(m, ifc_class="IfcSlab", name=f"Floor-{r['id']}")
            slab.PredefinedType = "FLOOR"
            slab.Representation = _polygon(body, r["boundary"], slab_thickness)
            slab.ObjectPlacement = _place(loc=(0.0, 0.0, z0 - slab_thickness))
            contained[sk].append(slab)

    for sk, items in contained.items():
        if items:
            ifcopenshell.api.spatial.assign_container(m, products=items, relating_structure=storeys[sk])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.write(out_path)
    counts = {c: len(m.by_type(c)) for c in
              ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcSpace", "IfcBuildingStorey"]}
    logger.info("Wrote %s  ->  %s", out_path, counts)
    return out_path
