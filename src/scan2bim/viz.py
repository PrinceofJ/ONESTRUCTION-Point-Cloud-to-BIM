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


def show_wall_debug(building_json, save_path=None):
    """Show a multi-stage diagnostic view of wall generation.

    Requires building_json to contain '_debug' (set by build_building_json).
    Shows: raw segments, after filtering, after merge, and final deduped.
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    debug = building_json.get("_debug", {})
    if not debug:
        print("No debug data — re-run build_building_json to populate _debug.")
        return

    room_colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    titles = ["1) Raw segments (per room)", "2) After length/aspect filter",
              "3) After merge_wall_faces", "4) Final (after dedup)"]

    for ax, title in zip(axes.flat, titles):
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    def _draw_walls(ax, geos, color, alpha=0.6, lw=1.5, label=None):
        for i, g in enumerate(geos):
            sx, sy = g["start"]
            ex, ey = g["end"]
            ax.plot([sx, ex], [sy, ey], "-", color=color, alpha=alpha, linewidth=lw,
                    label=label if i == 0 else None)
            ax.plot(sx, sy, "o", color=color, alpha=alpha, markersize=3)
            ax.plot(ex, ey, "o", color=color, alpha=alpha, markersize=3)

    total_counts = [0, 0, 0, 0]
    for ri, (room_name, rd) in enumerate(sorted(debug.items())):
        c = room_colors(ri % 10)

        _draw_walls(axes[0, 0], rd["raw_geos"], c, label=room_name)
        total_counts[0] += len(rd["raw_geos"])

        _draw_walls(axes[0, 1], rd["filtered_geos"], c, label=room_name)
        total_counts[1] += len(rd["filtered_geos"])

        _draw_walls(axes[1, 0], rd["merged_geos"], c, label=room_name)
        total_counts[2] += len(rd["merged_geos"])

    final_walls = building_json.get("walls", [])
    for w in final_walls:
        sx, sy = w["start"]
        ex, ey = w["end"]
        axes[1, 1].plot([sx, ex], [sy, ey], "k-", linewidth=2)
        axes[1, 1].plot(sx, sy, "ko", markersize=3)
        axes[1, 1].plot(ex, ey, "ko", markersize=3)
    total_counts[3] = len(final_walls)

    for ax, title, count in zip(axes.flat, titles, total_counts):
        ax.set_title(f"{title} — {count} walls", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(
        f"Wall Pipeline Debug — {len(debug)} rooms",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Debug plot saved to {save_path}")
    plt.show()


def show_wall_detail_table(building_json):
    """Print a per-room summary table of wall counts at each stage."""
    debug = building_json.get("_debug", {})
    if not debug:
        print("No debug data available.")
        return

    print(f"\n{'Room':<20} {'Seg':>4} {'Dirs':>5} {'Raw':>4} {'Filt':>5} "
          f"{'Kept':>5} {'Merged':>7}")
    print("-" * 65)
    totals = [0, 0, 0, 0, 0, 0]
    for room_name in sorted(debug):
        rd = debug[room_name]
        vals = [rd["n_segments"], rd["n_directions"], rd["n_raw_geos"],
                rd["n_filtered"], rd["n_after_filter"], rd["n_after_merge"]]
        print(f"{room_name:<20} {vals[0]:>4} {vals[1]:>5} {vals[2]:>4} "
              f"{vals[3]:>5} {vals[4]:>5} {vals[5]:>7}")
        for i in range(6):
            totals[i] += vals[i]

    print("-" * 65)
    print(f"{'TOTAL':<20} {totals[0]:>4} {'':>5} {totals[2]:>4} "
          f"{totals[3]:>5} {totals[4]:>5} {totals[5]:>7}")

    final_count = len(building_json.get("walls", []))
    print(f"\nAfter dedup: {totals[5]} → {final_count} walls "
          f"(removed {totals[5] - final_count})\n")


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
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.annotate(wall["id"], (mx, my), fontsize=6, color="red",
                    ha="center", va="bottom", fontweight="bold")

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
        ax.annotate(f'{door["id"]}({door["wall"]})', (dx, dy), fontsize=5,
                    color="dodgerblue", ha="center", va="top")

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
        ax.annotate(f'{win["id"]}({win["wall"]})', (wx, wy), fontsize=5,
                    color="orange", ha="center", va="top")

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
