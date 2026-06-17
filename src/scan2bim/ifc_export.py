"""IFC export: wall geometry → JSON → IFC4 model."""

from __future__ import annotations

import glob
import json
import logging
import math
import os

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import WallSegConfig, IfcExportConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wall segmentation (lightweight copy for re-segmenting room clouds)
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
    if final_bins:
        return centres[np.array(final_bins, dtype=int)]
    return np.array([])


def _assign_to_nearest_peak(theta, peak_centres):
    labels = np.empty(len(theta), dtype=int)
    for i, t in enumerate(theta):
        dists = np.minimum(np.abs(peak_centres - t), np.pi - np.abs(peak_centres - t))
        labels[i] = np.argmin(dists)
    return labels


def segment_walls_for_ifc(pcd, wall_seg_cfg: WallSegConfig):
    """Segment walls from a room cloud (returns walls list + n_directions)."""
    up_axis = wall_seg_cfg.up_axis
    if len(pcd.points) < wall_seg_cfg.min_wall_points:
        return [], 0

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=wall_seg_cfg.normal_radius_m, max_nn=wall_seg_cfg.normal_max_nn,
        )
    )
    pts = np.asarray(pcd.points)
    norms = np.asarray(pcd.normals)
    up = np.zeros(3)
    up[up_axis] = 1.0
    sin_tol = np.sin(np.deg2rad(wall_seg_cfg.normal_tol_deg))
    vert_mask = np.abs(norms @ up) < sin_tol
    pts_v = pts[vert_mask]
    norms_v = norms[vert_mask]
    if len(pts_v) < wall_seg_cfg.min_wall_points:
        return [], 0

    ha, hb = [a for a in range(3) if a != up_axis]
    nh = norms_v[:, [ha, hb]].copy()
    nh /= np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9
    theta = np.arctan2(nh[:, 1], nh[:, 0]) % np.pi

    peak_centres = _find_angle_peaks(theta)
    n_directions = len(peak_centres)
    if n_directions == 0:
        return [], 0

    angle_labels = _assign_to_nearest_peak(theta, peak_centres)
    walls = []
    for a_label in range(n_directions):
        a_mask = angle_labels == a_label
        a_pts = pts_v[a_mask]
        a_nh = nh[a_mask]
        mean_n = a_nh.mean(axis=0)
        mean_n /= np.linalg.norm(mean_n) + 1e-9
        offsets = a_pts[:, ha] * mean_n[0] + a_pts[:, hb] * mean_n[1]
        off_labels = _cluster_1d_gaps(offsets, wall_seg_cfg.offset_tol_m)
        ifc_min_pts = max(wall_seg_cfg.min_wall_points, 500)
        for o_label in np.unique(off_labels):
            o_mask = off_labels == o_label
            wall_pts = a_pts[o_mask]
            if len(wall_pts) < ifc_min_pts:
                continue
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(wall_pts)
            walls.append({
                "cloud": cloud,
                "normal_2d": mean_n.copy(),
                "offset": float(offsets[o_mask].mean()),
            })
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
    }


# ---------------------------------------------------------------------------
# Merge / dedup
# ---------------------------------------------------------------------------

def _angle_close(a1, a2, tol_rad):
    diff = abs(a1 - a2)
    return min(diff, np.pi - diff) < tol_rad


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
            if abs(wall_geos[i]["offset"] - wall_geos[j]["offset"]) > cfg.max_merge_thickness:
                continue
            u_overlap = (min(wall_geos[i]["u_max"], wall_geos[j]["u_max"])
                         - max(wall_geos[i]["u_min"], wall_geos[j]["u_min"]))
            min_len = min(wall_geos[i]["length"], wall_geos[j]["length"])
            if min_len > 0 and u_overlap / min_len < 0.3:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            g = wall_geos[indices[0]].copy()
            g["source_indices"] = indices
            if g["thickness"] < 0.05:
                g["thickness"] = cfg.default_thickness
            merged.append(g)
        else:
            geos = [wall_geos[i] for i in indices]
            avg_offset = np.mean([g["offset"] for g in geos])
            thickness = max(abs(g["offset"] - avg_offset) * 2 for g in geos)
            if thickness < 0.05:
                thickness = cfg.default_thickness
            u_min = min(g["u_min"] for g in geos)
            u_max = max(g["u_max"] for g in geos)
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
                "angle": ref["angle"], "source_indices": indices,
            })
    return merged


def _opening_duplicate(op, existing, pos_tol=0.15, size_tol=0.15):
    for e in existing:
        if e["label"] != op["label"]:
            continue
        if (abs(e["offset"] - op["offset"]) < pos_tol
                and abs(e["width"] - op["width"]) < size_tol
                and abs(e["height"] - op["height"]) < size_tol):
            return True
    return False


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
            if flat[i]["_room"] == flat[j]["_room"]:
                continue
            if not _angle_close(flat[i]["angle"], flat[j]["angle"], angle_tol):
                continue
            if abs(flat[i]["offset"] - flat[j]["offset"]) > cfg.dedup_offset_tol:
                continue
            u_overlap = (min(flat[i]["u_max"], flat[j]["u_max"])
                         - max(flat[i]["u_min"], flat[j]["u_min"]))
            min_len = min(flat[i]["length"], flat[j]["length"])
            if min_len <= 0 or u_overlap / min_len < cfg.dedup_overlap_frac:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    unique_walls = []
    for root_idx, indices in groups.items():
        keeper = flat[root_idx].copy()
        all_openings = list(keeper.get("_openings", []))
        keeper_u_min = keeper["u_min"]
        for idx in indices:
            if idx == root_idx:
                continue
            other = flat[idx]
            normals_opposed = np.dot(keeper["normal_2d"], other["normal_2d"]) < 0
            for op in other.get("_openings", []):
                remapped = op.copy()
                if normals_opposed:
                    remapped["offset"] = keeper["length"] - op["offset"] - op["width"]
                else:
                    remapped["offset"] = op["offset"] + other["u_min"] - keeper_u_min
                if not _opening_duplicate(remapped, all_openings):
                    all_openings.append(remapped)
        keeper["_openings"] = all_openings
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


def attach_openings(merged_walls, source_wall_count, room_openings, pixel_m):
    for mw in merged_walls:
        mw["_openings"] = []

    wall_name_to_index = {}
    for entry in room_openings:
        name = entry.get("wall", "")
        idx = int(name.replace("wall_", "")) - 1 if name.startswith("wall_") else -1
        wall_name_to_index[name] = idx

    seg_idx_to_merged = {}
    for mi, mw in enumerate(merged_walls):
        for si in mw.get("source_indices", []):
            seg_idx_to_merged[si] = mi

    for entry in room_openings:
        wall_name = entry.get("wall", "")
        seg_idx = wall_name_to_index.get(wall_name, -1)
        merged_idx = seg_idx_to_merged.get(seg_idx)
        if merged_idx is None:
            continue
        for op in entry.get("openings", []):
            if op["label"] not in ("door", "window"):
                continue
            offset_m = op.get("offset_m", op["bbox_px"][0] * pixel_m)
            opening = {
                "label": op["label"],
                "offset": round(offset_m, 3),
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
        if len(pts) < 3:
            boundaries[room_name] = []
            continue
        boundaries[room_name] = _convex_hull_2d(np.array(pts))
    return boundaries


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------

def build_building_json(
    room_cloud_paths: list[str],
    openings_dir: str | None,
    wall_seg_cfg: WallSegConfig,
    ifc_cfg: IfcExportConfig,
) -> dict:
    """Build the canonical building JSON from room clouds + openings."""
    room_openings_all: dict[str, list] = {}
    pixel_m = wall_seg_cfg.flat_pixel_m
    if openings_dir and os.path.isdir(openings_dir):
        room_openings_all, pixel_m = load_openings(openings_dir)
        logger.info("Loaded openings for %d rooms", len(room_openings_all))

    all_room_walls: dict[str, list] = {}
    skipped = []

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

        wall_geos = [compute_wall_geometry(w, wall_seg_cfg.up_axis) for w in walls]
        wall_geos = [
            g for g in wall_geos
            if ifc_cfg.min_wall_length_m <= g["length"] <= ifc_cfg.max_wall_length_m
            and g["length"] / max(g["height"], 1e-3) >= ifc_cfg.min_wall_aspect_ratio
        ]
        if not wall_geos:
            skipped.append(room_name)
            continue

        merged = merge_wall_faces(wall_geos, ifc_cfg, wall_seg_cfg.angle_tol_deg)

        room_openings = room_openings_all.get(room_name, [])
        if room_openings:
            attach_openings(merged, len(walls), room_openings, pixel_m)

        all_room_walls[room_name] = merged

    logger.info("Processed %d rooms, skipped %d", len(all_room_walls), len(skipped))

    unique_walls = deduplicate_walls(all_room_walls, ifc_cfg, wall_seg_cfg.angle_tol_deg)
    logger.info("Deduplicated: %d unique walls", len(unique_walls))

    storey_id = "L0"
    data: dict = {
        "project": {"name": ifc_cfg.project_name},
        "storeys": [{"id": storey_id, "name": "Ground Floor", "elevation": ifc_cfg.floor_elevation}],
        "walls": [], "doors": [], "windows": [], "rooms": [],
    }

    door_counter = 0
    window_counter = 0

    for i, w in enumerate(unique_walls):
        wall_id = f"W{i + 1}"
        thickness = w["thickness"]
        if thickness < 0.05:
            thickness = ifc_cfg.default_thickness

        data["walls"].append({
            "id": wall_id, "storey": storey_id,
            "start": [round(w["start"][0], 4), round(w["start"][1], 4)],
            "end": [round(w["end"][0], 4), round(w["end"][1], 4)],
            "height": round(w["height"], 3), "thickness": round(thickness, 3),
        })

        for op in w.get("_openings", []):
            if op["label"] == "door":
                door_counter += 1
                data["doors"].append({
                    "id": f"D{door_counter}", "wall": wall_id,
                    "offset": round(op["offset"], 3),
                    "width": round(op["width"], 3), "height": round(op["height"], 3),
                })
            elif op["label"] == "window":
                window_counter += 1
                data["windows"].append({
                    "id": f"Win{window_counter}", "wall": wall_id,
                    "offset": round(op["offset"], 3),
                    "width": round(op["width"], 3), "height": round(op["height"], 3),
                    "sill_height": round(op.get("sill_height", 0.9), 3),
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
        walls_map[w["id"]] = {"e": wall, "storey": sk}
        contained[sk].append(wall)

    def add_opening(spec, kind):
        host = walls_map[spec["wall"]]
        wall = host["e"]
        sk = host["storey"]
        sill = float(spec.get("sill_height", 0.0))
        depth = 1.0
        op = ifcopenshell.api.root.create_entity(m, ifc_class="IfcOpeningElement",
                                                 name=f"Opening-{spec['id']}")
        op.PredefinedType = "OPENING"
        op.Representation = _box(body, spec["width"], depth, spec["height"])
        op.ObjectPlacement = _place(loc=(spec["offset"], 0.0, sill),
                                    rel_to=wall.ObjectPlacement)
        m.create_entity("IfcRelVoidsElement", GlobalId=ifcopenshell.guid.new(),
                        RelatingBuildingElement=wall, RelatedOpeningElement=op)
        cls = "IfcDoor" if kind == "door" else "IfcWindow"
        elem = ifcopenshell.api.root.create_entity(m, ifc_class=cls, name=f"{kind.title()}-{spec['id']}")
        elem.OverallHeight = float(spec["height"])
        elem.OverallWidth = float(spec["width"])
        elem.Representation = _box(body, spec["width"], 0.05 if kind == "door" else 0.03,
                                   spec["height"])
        elem.ObjectPlacement = _place(loc=(0.0, 0.0, 0.0), rel_to=op.ObjectPlacement)
        m.create_entity("IfcRelFillsElement", GlobalId=ifcopenshell.guid.new(),
                        RelatingOpeningElement=op, RelatedBuildingElement=elem)
        contained[sk].append(elem)

    for d in data.get("doors", []):
        add_opening(d, "door")
    for w in data.get("windows", []):
        add_opening(w, "window")

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
