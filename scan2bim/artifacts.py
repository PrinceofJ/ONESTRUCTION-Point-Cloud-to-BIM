"""Structured artifact IO and ZIP packaging shared by all four notebooks.

A single place defines the stage directory names and the canonical artifact filenames,
so a later notebook consumes an earlier stage's outputs *by name* with no manual wiring.
Each stage writes into ``{out_root}/{stage}/`` and is packaged into ``{out_root}/{stage}.zip``.
``load_stage_dir`` resolves a stage's directory, transparently extracting the ZIP if only
the archive is present (so any notebook can run from just the previous stage's ZIP).
"""

from __future__ import annotations

import os
import json
import zipfile

import numpy as np
from PIL import Image

# ---- stage directory names (consistent across notebooks) ----
# Numbered to match the run order 1 -> 2 -> 3 -> 4:
#   1 occupancy · 2 watershed · 3 wall-assignment · 4 SAM refinement.
# (Wall assignment needs the watershed's room masks, so the watershed is stage 2.)
STAGE1 = 'stage1_occupancy'
STAGE2 = 'stage2_watershed'
STAGE3 = 'stage3_walls'
STAGE4 = 'stage4_sam_refined'
STAGE5 = 'stage5_walls_sam_refined'   # N3 re-run on SAM-refined masks (Option B)

# ---- pure-SAM method (Method 2) stage names ----
# Runs OFF the shared Stage-1 rasters (no watershed), so it has its own two-stage chain:
#   stage_sam_auto  : automatic-mask room labels (room_labels.npy on the Stage-1 grid)
#   stage_sam_walls : boundary-ring wall assignment on those masks (same logic as STAGE3)
STAGE_SAM_AUTO = 'stage_sam_auto'
STAGE_SAM_WALLS = 'stage_sam_walls'

# ---- postprocessing stage names (wall seg -> door/window -> IFC export) ----
STAGE_WALL_SEG = 'stage_wall_seg'      # per-room wall-plane segmentation + flattened images
STAGE_WALL_PROC = 'stage_wall_proc'    # door/window detection on the wall images
STAGE_IFC = 'stage_ifc'                # building.json + IFC4 model export

# ---- canonical artifact filenames ----
TRANSFORM_JSON = 'transform.json'      # grid transform + floor/ceil + provenance
CONFIG_JSON = 'config.json'            # full Config snapshot

OCC_PNG = 'occupancy.png'              # binary occupancy image (0=wall, 255=free)
WALLMASK_NPY = 'wall_mask.npy'         # bool: slab-occupancy wall pixels (the wall source +
                                       #   segmentation input + canonical eval scaffold)
COVERAGE_NPY = 'coverage.npy'          # bool: scan-coverage raster (void rejection)

ROOM_LABELS_NPY = 'room_labels.npy'    # int32: -1 wall, 0 exterior, >=1 rooms (watershed)
ROOM_LABELS_PNG = 'room_labels_color.png'
WATERSHED_WALLS_NPY = 'walls.npy'      # bool: cleaned walls used by the watershed
FOOTPRINT_NPY = 'footprint.npy'        # bool: building footprint (needed by SAM residual)

ROOM_WALL_MASKS_NPZ = 'room_wall_masks.npz'   # stage2: per-room wall-pixel masks
WALL_ASSIGN_PNG = 'wall_assignment.png'

REFINED_LABELS_NPY = 'room_labels_refined.npy'
REFINED_LABELS_PNG = 'room_labels_refined_color.png'

BUILDING_JSON = 'building.json'        # postprocessing: structured walls/openings/rooms
IFC_MODEL = 'model.ifc'                # postprocessing: exported IFC4 model


# ---------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def stage_dir(out_root, stage):
    return os.path.join(out_root, stage)


def stage_zip(out_root, stage):
    return os.path.join(out_root, f"{stage}.zip")


# ---- transform / config ----
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
    """Read a stage directory's saved ``config.json`` as a plain dict.

    Used by downstream stages to validate (via ``scan2bim.assert_upstream_config``) that
    they share the same cloud + grid as the stage they are consuming.
    """
    with open(os.path.join(stage_dir, CONFIG_JSON)) as f:
        return json.load(f)


# ---- arrays / images ----
def save_npy(path, arr):
    np.save(path, np.asarray(arr))
    return path


def load_npy(path):
    return np.load(path)


def save_png(path, image_uint8):
    Image.fromarray(np.asarray(image_uint8).astype(np.uint8)).save(path)
    return path


def save_label_png(path, labels):
    """Colourised label image for QA: -1 black, 0 light-grey, rooms via tab20."""
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


# ---- ZIP packaging ----
def zip_dir(src_dir, zip_path):
    """Zip every file under ``src_dir`` (flat-ish, preserving relative paths)."""
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
    """Zip ``{out_root}/{stage}`` -> ``{out_root}/{stage}.zip`` and return the zip path."""
    d = stage_dir(out_root, stage)
    z = stage_zip(out_root, stage)
    zip_dir(d, z)
    return z


def load_stage_dir(out_root, stage):
    """Return the directory for ``stage``. If the directory is missing but the ZIP
    exists, extract the ZIP into the directory first. Raises if neither exists."""
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
    """Save {room_id: bool HxW} as an .npz keyed 'room_<id>'."""
    np.savez_compressed(path, **{f"room_{int(k)}": np.asarray(v, bool)
                                 for k, v in masks_dict.items()})
    return path


def load_room_wall_masks(path):
    z = np.load(path)
    return {int(k.split('_')[1]): z[k] for k in z.files}
