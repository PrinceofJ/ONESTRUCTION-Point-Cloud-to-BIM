import os
import glob
import numpy as np
import open3d as o3d


def _read_xyzrgb_txt(path):
    try:
        import pandas as pd
        arr = pd.read_csv(
            path, sep=" ", header=None, dtype=np.float32, engine="c",
        ).to_numpy()
    except Exception:
        rows = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:          # corrupted token -> skip the row
                    continue
        arr = np.asarray(rows, dtype=np.float32)

    if arr.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    xyz = arr[:, :3].astype(np.float64)
    rgb = np.clip(arr[:, 3:6] / 255.0, 0.0, 1.0)   # open3d wants colors in [0,1]
    return xyz, rgb


def _make_pcd(xyz, rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None and len(rgb):
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def txt_to_cloud(path, with_color=True):
    """One room .txt -> open3d PointCloud."""
    xyz, rgb = _read_xyzrgb_txt(path)
    return _make_pcd(xyz, rgb if with_color else None)


def _room_txt_files(area_dir):
    files = glob.glob(os.path.join(area_dir, "*", "*.txt"))
    keep = []
    for p in files:
        folder = os.path.basename(os.path.dirname(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        if folder == stem:                  # skips Annotations/<object>.txt
            keep.append(p)
    return sorted(keep)


def area_to_cloud(area_dir, with_color=True, voxel_size=None, verbose=True):
    room_files = _room_txt_files(area_dir)
    if not room_files:
        raise FileNotFoundError(
            f"No room .txt files found under {area_dir!r}. "
            f"Point this at an Area_N folder of the Aligned_Version.")

    all_xyz, all_rgb = [], []
    for p in room_files:
        xyz, rgb = _read_xyzrgb_txt(p)
        if len(xyz):
            all_xyz.append(xyz)
            all_rgb.append(rgb)
        if verbose:
            print(f"  {os.path.basename(p):35s} {len(xyz):>10,} pts")

    xyz = np.vstack(all_xyz)
    rgb = np.vstack(all_rgb)
    if verbose:
        print(f"Stacked {len(room_files)} rooms -> {len(xyz):,} points total")

    pcd = _make_pcd(xyz, rgb if with_color else None)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
        if verbose:
            print(f"After voxel_down_sample({voxel_size}): "
                  f"{len(pcd.points):,} points")
    return pcd


def save_cloud(pcd, out_path):
    ok = o3d.io.write_point_cloud(out_path, pcd)
    if not ok:
        raise IOError(f"open3d failed to write {out_path!r}")
    print(f"Wrote {out_path}  ({len(pcd.points):,} points)")
    return out_path


if __name__ == "__main__":
    AREA = "Area_6"   # <- edit path

    floor = area_to_cloud(AREA, with_color=True, voxel_size=0.03)
    #save_cloud(floor, "area3_floor.ply")          # color preserved
    save_cloud(floor, "area6.xyz")          # coords only (smaller)