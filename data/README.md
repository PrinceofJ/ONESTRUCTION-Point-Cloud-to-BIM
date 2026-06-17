# data

Drop your **segmented point cloud** here (walls / windows / doors), named `area1.xyz`.

- Default expected path: `data/area1.xyz` (set in `params.yaml` as `input.file_path`). To use
  a different name or location, edit that one line in `params.yaml` — never a notebook cell.
- Same format and units as the original pipeline (`.xyz`, etc.). `input.units_per_meter` in
  `params.yaml` controls the scale conversion on load.

This folder is git-ignored (except this file) so large clouds aren't committed.
