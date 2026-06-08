# Building JSON Schema — Authoring Guide

This is the input format for the JSON → IFC4 translator. If a file follows the rules below, the translator turns it into a valid BIM model. Read the **Mental model** and **Global conventions** sections first — most mistakes come from those, not from the field details.

---

## 1. Mental model

The JSON describes a building the way IFC thinks about one, not the way a floor plan looks. Four ideas drive everything:

1. **A wall is a centerline, not a box.** You give its two endpoints, plus a height and a thickness. The translator builds the solid; the thickness is spread evenly on both sides of the line you draw.
2. **Doors and windows belong to a wall.** You don't give them absolute coordinates. You name the wall they sit in and how far along that wall they are. The translator cuts the hole and places the panel.
3. **A room is a footprint.** You give an ordered list of corner points; the translator extrudes it upward into a space (and an optional floor slab).
4. **Everything is linked by `id`.** Walls and rooms name their storey by `id`; doors and windows name their wall by `id`. These names are how the model is stitched together, so they must match exactly.

---

## 2. Global conventions (read these)

- **Units are metres.** Every coordinate, length, width, height, thickness, and elevation is in metres. A 3-metre wall is `3.0`, not `3000`. If your source data is in millimetres, divide by 1000 before writing the JSON.
- **Coordinates are 2D plan points** written `[x, y]`. There is **no z in a point** — vertical position comes from the storey `elevation` and from `height`/`sill_height`. Z is "up".
- **One global coordinate system.** Every wall endpoint and room corner is measured in the same flat XY frame. Keep the numbers small (near the origin); if your scan is in large real-world coordinates, subtract a base point first.
- **`id`s are case-sensitive strings** and must be unique within their list. `"W1"` and `"w1"` are different. A reference (`door.wall`, `wall.storey`) must match an existing `id` character-for-character.
- **Numbers must be numbers**, not strings. Write `0.2`, not `"0.2"`.

---

## 3. Top-level structure

```json
{
  "project":  { ... },        // optional
  "storeys":  [ ... ],        // one entry per level
  "walls":    [ ... ],        // the walls
  "doors":    [ ... ],        // doors (fill walls)
  "windows":  [ ... ],        // windows (fill walls)
  "rooms":    [ ... ]         // spaces / floor footprints
}
```

Any of `doors`, `windows`, or `rooms` may be omitted if there are none. You should always include `storeys` and `walls`.

---

## 4. Field reference

### `project` (optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | no | Project name shown in the model. Defaults to `"Project"`. |

### `storeys` — one object per level

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `id` | string | **yes** | — | Unique handle. Walls and rooms reference this. |
| `name` | string | no | — | Display name, e.g. `"Ground Floor"`. |
| `elevation` | number | no | m | Floor level height. Defaults to `0.0`. Walls and rooms on this storey sit at this Z. |

### `walls` — one object per wall

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `id` | string | **yes** | — | Unique handle. Doors/windows reference this. |
| `storey` | string | **yes** | — | The `id` of the storey this wall is on. |
| `start` | `[x, y]` | **yes** | m | One end of the centerline. |
| `end` | `[x, y]` | **yes** | m | The other end. Must differ from `start`. |
| `height` | number | **yes** | m | Wall height, extruded up from the storey elevation. |
| `thickness` | number | **yes** | m | Total thickness, centered on the centerline (half on each side). |

### `doors` — one object per door

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `id` | string | **yes** | — | Unique handle. |
| `wall` | string | **yes** | — | The `id` of the wall this door sits in. |
| `offset` | number | **yes** | m | Distance **along the wall from its `start`** to the **center** of the door. |
| `width` | number | **yes** | m | Door width. |
| `height` | number | **yes** | m | Door height (measured from the floor). |

### `windows` — one object per window

Same as doors, plus a sill:

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `id` | string | **yes** | — | Unique handle. |
| `wall` | string | **yes** | — | The `id` of the wall this window sits in. |
| `offset` | number | **yes** | m | Distance along the wall from `start` to the window center. |
| `width` | number | **yes** | m | Window width. |
| `height` | number | **yes** | m | Window height (the glazed opening). |
| `sill_height` | number | recommended | m | Height of the window's bottom above the floor. Defaults to `0.0` (which would sit it on the floor), so set it. |

### `rooms` — one object per space

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `id` | string | **yes** | — | Unique handle. |
| `storey` | string | **yes** | — | The `id` of the storey this room is on. |
| `name` | string | no | — | Room name, e.g. `"Room 101"`. |
| `boundary` | `[[x,y], ...]` | **yes** | m | Ordered corner points of the footprint. **Do not repeat the first point** — the loop is closed automatically. Any simple (non-self-intersecting) polygon works, including L-shapes. List points in counter-clockwise order. |
| `height` | number | **yes** | m | Ceiling height of the space. |

---

## 5. How `offset` and `thickness` work

The single most error-prone part is positioning a door or window. `offset` is measured **along the wall, starting at the wall's `start` point, to the center of the opening** — not from a building corner, and not as a fraction.

```
        offset
   |<----------->|
   |             |   width
   |          |<------->|
 start o=======[ opening ]=========o end
   (wall.start)                  (wall.end)

   The wall's centerline runs start -> end.
   thickness is spread evenly on both sides of this line.
```

For the opening to fit inside the wall, keep:

```
   width / 2  <=  offset  <=  wall_length - width / 2
```

where `wall_length` is the distance between `start` and `end`.

`sill_height` is how high the **bottom** of the opening sits above the floor. Doors leave it at `0` (they reach the floor); windows raise it (e.g. `0.9`):

```
        ___              ___  <- opening top   (sill_height + height)
       |   |            |   |
       |   |            |   |
       |   |       sill |___| <- opening bottom (sill_height)
   ____|___|____    ________
       floor            floor
        door            window
```

---

## 6. Complete example

A single 5 × 4 m room, 3 m high, with one door and two windows:

```json
{
  "project": { "name": "Simple Room" },
  "storeys": [
    { "id": "L0", "name": "Ground Floor", "elevation": 0.0 }
  ],
  "walls": [
    { "id": "S", "storey": "L0", "start": [0.0, 0.0], "end": [5.0, 0.0], "height": 3.0, "thickness": 0.2 },
    { "id": "E", "storey": "L0", "start": [5.0, 0.0], "end": [5.0, 4.0], "height": 3.0, "thickness": 0.2 },
    { "id": "N", "storey": "L0", "start": [5.0, 4.0], "end": [0.0, 4.0], "height": 3.0, "thickness": 0.2 },
    { "id": "W", "storey": "L0", "start": [0.0, 4.0], "end": [0.0, 0.0], "height": 3.0, "thickness": 0.2 }
  ],
  "doors": [
    { "id": "D1", "wall": "S", "offset": 2.5, "width": 0.9, "height": 2.1 }
  ],
  "windows": [
    { "id": "Win1", "wall": "S", "offset": 4.0, "width": 1.0, "height": 1.2, "sill_height": 0.9 },
    { "id": "Win2", "wall": "E", "offset": 2.0, "width": 1.2, "height": 1.2, "sill_height": 0.9 }
  ],
  "rooms": [
    { "id": "R1", "storey": "L0", "name": "Room 101",
      "boundary": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]], "height": 3.0 }
  ]
}
```

Note how `D1` names wall `"S"` at `offset` 2.5 (the middle of a 5 m wall), and `Win2` names wall `"E"` at `offset` 2.0 (the middle of that 4 m wall).

---

## 7. Minimal template to copy

```json
{
  "project": { "name": "" },
  "storeys": [
    { "id": "L0", "name": "", "elevation": 0.0 }
  ],
  "walls": [
    { "id": "", "storey": "L0", "start": [0.0, 0.0], "end": [0.0, 0.0], "height": 3.0, "thickness": 0.2 }
  ],
  "doors": [
    { "id": "", "wall": "", "offset": 0.0, "width": 0.9, "height": 2.1 }
  ],
  "windows": [
    { "id": "", "wall": "", "offset": 0.0, "width": 1.2, "height": 1.2, "sill_height": 0.9 }
  ],
  "rooms": [
    { "id": "", "storey": "L0", "name": "", "boundary": [[0.0, 0.0]], "height": 3.0 }
  ]
}
```

---

## 8. Validation checklist

Before handing a file to the translator, confirm:

- [ ] Every number is in **metres** (no millimetre values like `3000`).
- [ ] Points are `[x, y]` with **two** numbers — no z.
- [ ] Every `door.wall` / `window.wall` matches an existing `wall.id` exactly.
- [ ] Every `wall.storey` / `room.storey` matches an existing `storey.id` exactly.
- [ ] All `id`s are unique within their list.
- [ ] No wall has `start` equal to `end`.
- [ ] For each opening, `width/2 <= offset <= wall_length - width/2` (it fits inside the wall).
- [ ] Windows have a `sill_height` set (otherwise they sit on the floor).
- [ ] Room `boundary` has at least 3 points, in order, and the first point is **not** repeated at the end.
- [ ] Numbers are written as numbers, not quoted strings.
- [ ] The file is valid JSON (no trailing commas, all keys quoted).

---

## 9. What this schema does not cover (yet)

The translator currently produces geometry and the spatial hierarchy (project → site → building → storey → walls/openings/spaces/slabs). It does **not** yet read materials, wall/door/window **types**, or property sets. If your source data carries those (e.g. material per wall, a classification code), flag it — those are the natural next fields to add, and the schema can be extended without breaking existing files.
