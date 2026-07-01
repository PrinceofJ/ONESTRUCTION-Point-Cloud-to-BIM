"""Artifact IO and ZIP packaging."""

from __future__ import annotations

import os
import json
import zipfile

import numpy as np
from PIL import Image

STAGE1 = 'stage1_occupancy'
STAGE2 = 'stage2_watershed'
STAGE3 = 'stage3_walls'
STAGE4 = 'stage4_sam_refined'
STAGE5 = 'stage5_walls_sam_refined'

STAGE_SAM_AUTO = 'stage_sam_auto'
STAGE_SAM_WALLS = 'stage_sam_walls'

STAGE_WALL_SEG = 'stage_wall_seg'
STAGE_WALL_PROC = 'stage_wall_proc'
STAGE_IFC = 'stage_ifc'

TRANSFORM_JSON = 'transform.json'
CONFIG_JSON = 'config.json'

OCC_PNG = 'occupancy.png'
WALLMASK_NPY = 'wall_mask.npy'
WALLNESS_NPY = 'wallness.npy'
COVERAGE_NPY = 'coverage.npy'

ROOM_LABELS_NPY = 'room_labels.npy'
ROOM_LABELS_PNG = 'room_labels_color.png'
WATERSHED_WALLS_NPY = 'walls.npy'
FOOTPRINT_NPY = 'footprint.npy'

ROOM_WALL_MASKS_NPZ = 'room_wall_masks.npz'
WALL_ASSIGN_PNG = 'wall_assignment.png'

REFINED_LABELS_NPY = 'room_labels_refined.npy'
REFINED_LABELS_PNG = 'room_labels_refined_color.png'

BUILDING_JSON = 'building.json'
IFC_MODEL = 'model.ifc'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def stage_dir(out_root, stage):
    return os.path.join(out_root, stage)


def stage_zip(out_root, stage):
    return os.path.join(out_root, f"{stage}.zip")


def save_transform(path, transform, extra=None):
    d = dict(transform)
    if extra:
        d.update(extra)
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    return path


def load_transform(path):
    with open(path) as f:
        return json.load(f)


def save_config(path, cfg):
    with open(path, 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return path


def load_stage_config(stage_dir):
    with open(os.path.join(stage_dir, CONFIG_JSON)) as f:
        return json.load(f)


def save_npy(path, arr):
    np.save(path, np.asarray(arr))
    return path


def load_npy(path):
    return np.load(path)


def save_png(path, image_uint8):
    Image.fromarray(np.asarray(image_uint8).astype(np.uint8)).save(path)
    return path


def save_label_png(path, labels):
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    H, W = labels.shape
    rgb = np.ones((H, W, 3))
    rgb[labels == -1] = (0, 0, 0)
    rgb[labels == 0] = (0.93, 0.93, 0.93)
    cmap = plt.get_cmap('tab20')
    for k, r in enumerate([int(x) for x in np.unique(labels) if x >= 1]):
        rgb[labels == r] = cmap(k % 20)[:3]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


def zip_dir(src_dir, zip_path):
    src_dir = os.path.abspath(src_dir)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                fp = os.path.join(root, name)
                if os.path.abspath(fp) == os.path.abspath(zip_path):
                    continue
                zf.write(fp, os.path.relpath(fp, src_dir))
    return zip_path


def package_stage(out_root, stage):
    d = stage_dir(out_root, stage)
    z = stage_zip(out_root, stage)
    zip_dir(d, z)
    return z


def load_stage_dir(out_root, stage):
    d = stage_dir(out_root, stage)
    if os.path.isdir(d) and os.listdir(d):
        return d
    z = stage_zip(out_root, stage)
    if os.path.isfile(z):
        ensure_dir(d)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        return d
    raise FileNotFoundError(
        f"Stage '{stage}' not found: neither {d} nor {z} exists. "
        f"Run the producing notebook first (check CFG.out_root).")


def save_room_wall_masks(path, masks_dict):
    np.savez_compressed(path, **{f"room_{int(k)}": np.asarray(v, bool)
                                 for k, v in masks_dict.items()})
    return path


def load_room_wall_masks(path):
    z = np.load(path)
    return {int(k.split('_')[1]): z[k] for k in z.files}
