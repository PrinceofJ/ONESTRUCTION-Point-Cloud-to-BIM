"""Enrich IFC models with bSDD data via the bSDD MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

IFC_DICT_URI = "https://identifier.buildingsmart.org/uri/buildingsmart/ifc/4.3"

ELEMENT_PSET_MAP: dict[str, dict] = {
    "IfcWall": {
        "pset": "Pset_WallCommon",
        "qto": "Qto_WallBaseQuantities",
        "defaults": {
            "Reference": "",
            "IsExternal": False,
            "FireRating": "",
            "ThermalTransmittance": 0.0,
            "LoadBearing": False,
            "AcousticRating": "",
            "Combustible": False,
            "SurfaceSpreadOfFlame": "",
            "ExtendToStructure": False,
        },
    },
    "IfcDoor": {
        "pset": "Pset_DoorCommon",
        "qto": "Qto_DoorBaseQuantities",
        "defaults": {
            "Reference": "",
            "IsExternal": False,
            "FireRating": "",
            "AcousticRating": "",
            "SecurityRating": "",
            "HandicapAccessible": False,
            "FireExit": False,
            "SelfClosing": False,
            "SmokeStop": False,
            "HasDrive": False,
        },
    },
    "IfcWindow": {
        "pset": "Pset_WindowCommon",
        "qto": "Qto_WindowBaseQuantities",
        "defaults": {
            "Reference": "",
            "IsExternal": True,
            "FireRating": "",
            "AcousticRating": "",
            "ThermalTransmittance": 0.0,
            "GlazingAreaFraction": 0.7,
            "SmokeStop": False,
        },
    },
    "IfcSlab": {
        "pset": "Pset_SlabCommon",
        "qto": "Qto_SlabBaseQuantities",
        "defaults": {
            "Reference": "",
            "IsExternal": False,
            "FireRating": "",
            "LoadBearing": True,
            "AcousticRating": "",
            "Combustible": False,
        },
    },
    "IfcSpace": {
        "pset": "Pset_SpaceCommon",
        "qto": "Qto_SpaceBaseQuantities",
        "defaults": {
            "Reference": "",
            "IsExternal": False,
            "PubliclyAccessible": False,
            "HandicapAccessible": False,
            "GrossPlannedArea": 0.0,
            "NetPlannedArea": 0.0,
        },
    },
}

GEOMETRY_LIMITS = {
    "IfcDoor": {"width": (0.5, 2.5), "height": (1.5, 3.5)},
    "IfcWindow": {"width": (0.3, 3.0), "height": (0.3, 2.5)},
    "IfcWall": {"thickness": (0.05, 1.0), "length": (0.2, 30.0)},
}


@dataclass
class ValidationWarning:
    element_name: str
    ifc_class: str
    property_name: str
    actual_value: float
    expected_min: float | None
    expected_max: float | None
    message: str


@dataclass
class EnrichmentReport:
    elements_enriched: int = 0
    property_sets_added: int = 0
    quantity_sets_added: int = 0
    classification_refs_added: int = 0
    validation_warnings: list[ValidationWarning] = field(default_factory=list)


class BsddMcpClient:
    """Connects to the bSDD MCP server via stdio and calls its tools."""

    def __init__(self, server_path: str):
        self._server_path = server_path
        self._session = None
        self._cache: dict[str, object] = {}

    async def connect(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command="node",
            args=[self._server_path],
        )
        self._transport_ctx = stdio_client(params, errlog=open(os.devnull, "w"))
        streams = await self._transport_ctx.__aenter__()
        read_stream, write_stream = streams
        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        logger.info("Connected to bSDD MCP server")

    async def disconnect(self):
        if self._session:
            await self._session_ctx.__aexit__(None, None, None)
        if hasattr(self, "_transport_ctx"):
            await self._transport_ctx.__aexit__(None, None, None)
        logger.info("Disconnected from bSDD MCP server")

    async def _call(self, tool_name: str, arguments: dict) -> object:
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        if key in self._cache:
            return self._cache[key]
        result = await self._session.call_tool(tool_name, arguments)
        parsed = _parse_tool_result(result)
        self._cache[key] = parsed
        return parsed

    async def search_class(self, label: str) -> list[dict]:
        result = await self._call("bsdd_search_classes", {
            "searchText": label,
            "dictionaryUri": IFC_DICT_URI,
            "limit": 5,
        })
        if isinstance(result, dict):
            return result.get("classes", result.get("results", []))
        if isinstance(result, list):
            return result
        return []

    async def get_class(self, uri: str) -> dict:
        result = await self._call("bsdd_get_class", {
            "uri": uri,
            "includeProperties": True,
        })
        return result if isinstance(result, dict) else {}

    async def get_class_properties(self, class_uri: str) -> list[dict]:
        result = await self._call("bsdd_get_class_properties", {
            "classUri": class_uri,
            "limit": 200,
        })
        if isinstance(result, dict):
            return result.get("properties", result.get("classProperties", []))
        if isinstance(result, list):
            return result
        return []


def _parse_tool_result(result) -> object:
    """Extract JSON data from an MCP CallToolResult."""
    if not result or not result.content:
        return {}
    for block in result.content:
        if hasattr(block, "text"):
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, TypeError):
                return block.text
    return {}


async def _fetch_bsdd_class_data(
    client: BsddMcpClient,
    ifc_class: str,
) -> tuple[str | None, list[dict]]:
    """Search for an IFC class in bSDD and return (uri, properties)."""
    results = await client.search_class(ifc_class)
    if not results:
        logger.warning("bSDD: no results for %s", ifc_class)
        return None, []

    class_uri = None
    for r in results:
        name = r.get("name", "")
        uri = r.get("uri", "")
        if name == ifc_class or ifc_class.lower() in uri.lower():
            class_uri = uri
            break
    if not class_uri and results:
        class_uri = results[0].get("uri")

    if not class_uri:
        return None, []

    props = await client.get_class_properties(class_uri)
    return class_uri, props


def _build_property_values(
    bsdd_properties: list[dict],
    defaults: dict,
) -> dict:
    """Merge bSDD property definitions with static defaults."""
    props = dict(defaults)
    for bp in bsdd_properties:
        name = bp.get("name", bp.get("propertyName", ""))
        if not name:
            continue
        if name in props:
            continue
        data_type = bp.get("dataType", "").lower()
        if "bool" in data_type:
            props[name] = False
        elif "real" in data_type or "number" in data_type or "integer" in data_type:
            props[name] = 0.0
        else:
            props[name] = ""
    return props


def _compute_wall_quantities(wall_data: dict) -> dict:
    sx, sy = wall_data["start"]
    ex, ey = wall_data["end"]
    length = math.hypot(ex - sx, ey - sy)
    height = wall_data["height"]
    thickness = wall_data["thickness"]
    return {
        "Length": round(length, 4),
        "Height": round(height, 4),
        "Width": round(thickness, 4),
        "GrossFootprintArea": round(length * thickness, 4),
        "GrossSideArea": round(length * height, 4),
        "GrossVolume": round(length * height * thickness, 6),
        "NetVolume": round(length * height * thickness, 6),
    }


def _compute_door_quantities(door_data: dict) -> dict:
    w = door_data["width"]
    h = door_data["height"]
    return {
        "Width": round(w, 4),
        "Height": round(h, 4),
        "Area": round(w * h, 4),
    }


def _compute_window_quantities(window_data: dict) -> dict:
    w = window_data["width"]
    h = window_data["height"]
    return {
        "Width": round(w, 4),
        "Height": round(h, 4),
        "Area": round(w * h, 4),
    }


def _compute_slab_quantities(room_data: dict, slab_thickness: float) -> dict:
    pts = room_data.get("boundary", [])
    area = _polygon_area(pts) if pts else 0.0
    return {
        "Width": round(slab_thickness, 4),
        "GrossArea": round(area, 4),
        "GrossVolume": round(area * slab_thickness, 6),
    }


def _compute_space_quantities(room_data: dict) -> dict:
    pts = room_data.get("boundary", [])
    area = _polygon_area(pts) if pts else 0.0
    height = room_data.get("height", 0.0)
    perimeter = _polygon_perimeter(pts) if pts else 0.0
    return {
        "GrossFloorArea": round(area, 4),
        "NetFloorArea": round(area, 4),
        "GrossVolume": round(area * height, 6),
        "GrossPerimeter": round(perimeter, 4),
        "Height": round(height, 4),
    }


def _polygon_area(pts: list) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _polygon_perimeter(pts: list) -> float:
    n = len(pts)
    if n < 2:
        return 0.0
    perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        perim += math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
    return perim


def _validate_geometry(
    model,
    data: dict,
) -> list[ValidationWarning]:
    """Check detected dimensions against typical ranges."""
    warnings = []

    for w in data.get("walls", []):
        sx, sy = w["start"]
        ex, ey = w["end"]
        length = math.hypot(ex - sx, ey - sy)
        thickness = w["thickness"]
        name = f"Wall-{w['id']}"
        limits = GEOMETRY_LIMITS["IfcWall"]

        lo, hi = limits["thickness"]
        if thickness < lo or thickness > hi:
            warnings.append(ValidationWarning(
                element_name=name, ifc_class="IfcWall",
                property_name="thickness", actual_value=thickness,
                expected_min=lo, expected_max=hi,
                message=f"{name} thickness {thickness:.2f}m outside typical range [{lo}, {hi}]m",
            ))
        lo, hi = limits["length"]
        if length < lo or length > hi:
            warnings.append(ValidationWarning(
                element_name=name, ifc_class="IfcWall",
                property_name="length", actual_value=length,
                expected_min=lo, expected_max=hi,
                message=f"{name} length {length:.2f}m outside typical range [{lo}, {hi}]m",
            ))

    for d in data.get("doors", []):
        name = f"Door-{d['id']}"
        limits = GEOMETRY_LIMITS["IfcDoor"]
        for dim in ("width", "height"):
            val = d[dim]
            lo, hi = limits[dim]
            if val < lo or val > hi:
                warnings.append(ValidationWarning(
                    element_name=name, ifc_class="IfcDoor",
                    property_name=dim, actual_value=val,
                    expected_min=lo, expected_max=hi,
                    message=f"{name} {dim} {val:.2f}m outside typical range [{lo}, {hi}]m",
                ))

    return warnings


def _match_elements_to_data(model, data: dict, ifc_class: str, data_key: str) -> list[tuple]:
    """Match IFC elements to building JSON entries by name."""
    elements = model.by_type(ifc_class)
    data_items = data.get(data_key, [])
    id_to_data = {str(item["id"]): item for item in data_items}

    pairs = []
    for elem in elements:
        name = elem.Name or ""
        suffix = name.split("-", 1)[-1] if "-" in name else ""
        matched = id_to_data.get(suffix)
        pairs.append((elem, matched))
    return pairs


async def _enrich_async(model, cfg, data: dict) -> EnrichmentReport:
    """Core async enrichment logic."""
    import ifcopenshell.api.classification
    import ifcopenshell.api.pset

    report = EnrichmentReport()

    server_path = cfg.bsdd_server_path
    if not os.path.isabs(server_path):
        candidates = [
            os.path.join(os.getcwd(), server_path),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), server_path),
        ]
        for c in candidates:
            if os.path.isfile(c):
                server_path = c
                break

    if not os.path.isfile(server_path):
        raise FileNotFoundError(
            f"bSDD MCP server not found at '{server_path}'. "
            f"Set bsdd_server_path in params.yaml to the absolute path of bSDD-mcp/build/index.js")

    logger.info("bSDD: starting MCP server at %s", server_path)
    client = BsddMcpClient(server_path)
    try:
        await client.connect()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to bSDD MCP server at '{server_path}'. "
            f"Ensure the server is built and Node.js 18+ is available. Error: {exc}"
        ) from exc

    try:
        class_uris: dict[str, str | None] = {}
        class_properties: dict[str, list[dict]] = {}

        for ifc_class in ELEMENT_PSET_MAP:
            uri, props = await _fetch_bsdd_class_data(client, ifc_class)
            class_uris[ifc_class] = uri
            class_properties[ifc_class] = props
            if uri:
                logger.info("bSDD: %s → %s (%d properties)", ifc_class, uri, len(props))
            else:
                logger.warning("bSDD: %s → no URI found, using static defaults", ifc_class)
    finally:
        await client.disconnect()

    if cfg.bsdd_add_classifications:
        classification = ifcopenshell.api.classification.add_classification(
            model, classification="buildingSMART Data Dictionary (bSDD)")

    slab_thickness = getattr(cfg, "ifc_slab_thickness", 0.2)

    element_configs = [
        ("IfcWall", "walls"),
        ("IfcDoor", "doors"),
        ("IfcWindow", "windows"),
        ("IfcSlab", "rooms"),
        ("IfcSpace", "rooms"),
    ]

    for ifc_class, data_key in element_configs:
        pset_info = ELEMENT_PSET_MAP[ifc_class]
        uri = class_uris.get(ifc_class)
        bsdd_props = class_properties.get(ifc_class, [])
        prop_values = _build_property_values(bsdd_props, pset_info["defaults"])

        pairs = _match_elements_to_data(model, data, ifc_class, data_key)

        logger.info("bSDD: enriching %d %s elements", len(pairs), ifc_class)

        for elem, elem_data in pairs:
            report.elements_enriched += 1
            elem_name = elem.Name or f"<unnamed {ifc_class}>"

            if cfg.bsdd_add_classifications and uri:
                ifcopenshell.api.classification.add_reference(
                    model,
                    products=[elem],
                    classification=classification,
                    identification=uri,
                    name=ifc_class,
                )
                report.classification_refs_added += 1
                logger.debug("  %s → classification ref added", elem_name)

            if cfg.bsdd_add_psets:
                pset = ifcopenshell.api.pset.add_pset(
                    model, product=elem, name=pset_info["pset"])
                ifcopenshell.api.pset.edit_pset(
                    model, pset=pset, properties=prop_values)
                report.property_sets_added += 1
                logger.debug("  %s → %s (%d properties)", elem_name, pset_info["pset"], len(prop_values))

            if cfg.bsdd_add_qtos and elem_data:
                qto_props = {}
                if ifc_class == "IfcWall":
                    qto_props = _compute_wall_quantities(elem_data)
                elif ifc_class == "IfcDoor":
                    qto_props = _compute_door_quantities(elem_data)
                elif ifc_class == "IfcWindow":
                    qto_props = _compute_window_quantities(elem_data)
                elif ifc_class == "IfcSlab":
                    qto_props = _compute_slab_quantities(elem_data, slab_thickness)
                elif ifc_class == "IfcSpace":
                    qto_props = _compute_space_quantities(elem_data)

                if qto_props:
                    qto = ifcopenshell.api.pset.add_qto(
                        model, product=elem, name=pset_info["qto"])
                    ifcopenshell.api.pset.edit_qto(
                        model, qto=qto, properties=qto_props)
                    report.quantity_sets_added += 1
                    logger.debug("  %s → %s %s", elem_name, pset_info["qto"], qto_props)

    if cfg.bsdd_validate_geometry:
        report.validation_warnings = _validate_geometry(model, data)

    return report


def enrich_ifc(model, cfg, data: dict) -> EnrichmentReport:
    """Enrich an IFC model with bSDD classification, property sets, and quantities.

    Requires the bSDD MCP server to be available at cfg.bsdd_server_path.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()

    return asyncio.run(_enrich_async(model, cfg, data))
