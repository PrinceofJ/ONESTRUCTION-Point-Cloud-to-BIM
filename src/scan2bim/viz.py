"""Optional debug visualizations (requires matplotlib)."""

from __future__ import annotations

import numpy as np


def _subsample(n, k=150_000):
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(0).choice(n, k, replace=False)


def colorize_labels(labels):
    import matplotlib.pyplot as plt

    H, W = labels.shape
    rgb = np.ones((H, W, 3))
    rgb[labels == -1] = (0, 0, 0)
    rgb[labels == 0] = (0.93, 0.93, 0.93)
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate([int(x) for x in np.unique(labels) if x >= 1]):
        rgb[labels == r] = cmap(k % 20)[:3]
    return rgb


def show_topdown(points, up_axis, title="top-down"):
    import matplotlib.pyplot as plt

    pts = np.asarray(points)
    aa, bb = [a for a in (0, 1, 2) if a != up_axis]
    s = _subsample(len(pts), 200_000)
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(pts[s, aa], pts[s, bb], s=0.5, c=pts[s, up_axis], cmap="viridis")
    plt.colorbar(sc, label="height (m)")
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_room_labels(labels, title="Room labels"):
    import matplotlib.pyplot as plt

    rgb = colorize_labels(labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    for r in [int(x) for x in np.unique(labels) if x >= 1]:
        ys, xs = np.where(labels == r)
        ax.text(
            xs.mean(), ys.mean(), str(r),
            color="k", fontsize=9, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )
    n = len([r for r in np.unique(labels) if r >= 1])
    ax.set_title(f"{title} ({n} rooms)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_wall_images(flats, room_name="", cols=3):
    """Show a grid of flattened wall images."""
    import math
    import matplotlib.pyplot as plt

    n = len(flats)
    if n == 0:
        print("No walls to display.")
        return
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, flat in enumerate(flats):
        ax = axes[i // cols][i % cols]
        ax.imshow(
            flat["image"], cmap="gray", aspect="equal",
            extent=[0, flat["width_m"], 0, flat["height_m"]],
        )
        ax.set_xlabel("along wall (m)")
        ax.set_ylabel("height (m)")
        ax.set_title(f"wall {i + 1} — {flat['width_m']:.1f} x {flat['height_m']:.1f} m", fontsize=10)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"{room_name}: {n} wall images", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def show_floor_plan(building_json):
    """Quick top-down floor-plan sketch from the building JSON."""
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    data = building_json
    fig, ax = plt.subplots(figsize=(12, 10))

    for wall in data["walls"]:
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        ax.plot([sx, ex], [sy, ey], "k-", linewidth=2)

    for door in data.get("doors", []):
        wall = next((w for w in data["walls"] if w["id"] == door["wall"]), None)
        if not wall:
            continue
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        length = math.hypot(ex - sx, ey - sy)
        if length == 0:
            continue
        ux, uy = (ex - sx) / length, (ey - sy) / length
        dx = sx + ux * door["offset"]
        dy = sy + uy * door["offset"]
        ax.plot(dx, dy, "s", color="dodgerblue", markersize=8)

    for win in data.get("windows", []):
        wall = next((w for w in data["walls"] if w["id"] == win["wall"]), None)
        if not wall:
            continue
        sx, sy = wall["start"]
        ex, ey = wall["end"]
        length = math.hypot(ex - sx, ey - sy)
        if length == 0:
            continue
        ux, uy = (ex - sx) / length, (ey - sy) / length
        wx = sx + ux * win["offset"]
        wy = sy + uy * win["offset"]
        ax.plot(wx, wy, "D", color="orange", markersize=8)

    for room in data.get("rooms", []):
        bnd = room.get("boundary", [])
        if len(bnd) >= 3:
            poly = plt.Polygon(
                bnd, fill=True, facecolor=(0.9, 0.95, 1.0, 0.3),
                edgecolor="steelblue", linewidth=0.8, linestyle="--",
            )
            ax.add_patch(poly)
            cx = np.mean([p[0] for p in bnd])
            cy = np.mean([p[1] for p in bnd])
            ax.text(cx, cy, room.get("name", ""), fontsize=8, ha="center", va="center",
                    color="steelblue", fontstyle="italic")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{data.get('project', {}).get('name', 'Building')} — "
        f"{len(data['walls'])} walls, {len(data.get('doors', []))} doors, "
        f"{len(data.get('windows', []))} windows"
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="black", label="Wall"),
            mpatches.Patch(color="dodgerblue", label="Door"),
            mpatches.Patch(color="orange", label="Window"),
        ],
        fontsize=8, loc="upper right",
    )
    plt.tight_layout()
    plt.show()
