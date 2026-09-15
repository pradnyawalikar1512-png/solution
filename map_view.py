"""
Interactive Folium map for IRANav.

Uses Esri imagery (no personal API key). All hazard layers are simulated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
from folium.plugins import HeatMap

from simulation import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, cell_to_latlon


def _div_icon(emoji: str, size: int = 22) -> folium.DivIcon:
    return folium.DivIcon(
        html=(
            f'<div style="font-size:{size}px;line-height:1;text-shadow:0 1px 4px #000;">'
            f"{emoji}</div>"
        )
    )


def build_map(
    ship: Dict,
    destination: Dict,
    icebergs: List[Dict],
    zones: List[Dict],
    risk_grid: np.ndarray,
    route_latlon: List[Tuple[float, float]],
    previous_latlon: Optional[List[Tuple[float, float]]] = None,
    storm: Optional[Dict] = None,
    predictions: Optional[List[Dict]] = None,
) -> folium.Map:
    """Build the Antarctic Peninsula navigation map."""
    center_lat = (ship["lat"] + destination["lat"]) / 2
    center_lon = (ship["lon"] + destination["lon"]) / 2

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=(
            "Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
            "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        ),
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(fmap)
    fmap.fit_bounds([[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]])

    heat = []
    rows, cols = risk_grid.shape
    for row in range(rows):
        for col in range(cols):
            score = float(risk_grid[row, col])
            if score >= 28:
                lat, lon = cell_to_latlon(row, col)
                heat.append([lat, lon, score / 100.0])
    if heat:
        HeatMap(
            heat,
            min_opacity=0.22,
            radius=20,
            blur=24,
            max_zoom=6,
            name="Simulated risk heatmap",
        ).add_to(fmap)

    for zone in zones:
        folium.Circle(
            location=[zone["lat"], zone["lon"]],
            radius=zone["radius_km"] * 1000,
            color="#38bdf8",
            weight=1,
            fill=True,
            fill_color="#0ea5e9",
            fill_opacity=0.14,
            popup=folium.Popup(
                f"<b>{zone['name']}</b><br>SIMULATED sea-ice field<br>"
                f"Concentration: {zone['concentration']:.0f}%",
                max_width=260,
            ),
        ).add_to(fmap)

    if storm is not None:
        folium.Circle(
            location=[storm["lat"], storm["lon"]],
            radius=storm["radius_km"] * 1000,
            color="#a855f7",
            weight=2,
            fill=True,
            fill_color="#7e22ce",
            fill_opacity=0.18,
            popup=folium.Popup(
                f"<b>{storm['name']}</b><br>SIMULATED storm<br>"
                f"Wind {storm['wind_kt']:.0f} kt · vis {storm['visibility_km']:.1f} km",
                max_width=260,
            ),
        ).add_to(fmap)

    if previous_latlon and len(previous_latlon) >= 2:
        folium.PolyLine(
            previous_latlon,
            color="#94a3b8",
            weight=4,
            opacity=0.7,
            dash_array="8, 10",
            tooltip="Previous route (before reroute)",
        ).add_to(fmap)

    if route_latlon and len(route_latlon) >= 2:
        folium.PolyLine(
            route_latlon,
            color="#22d3ee",
            weight=6,
            opacity=0.95,
            tooltip="Recommended A* route",
        ).add_to(fmap)

    if predictions:
        for item in predictions:
            folium.PolyLine(
                [item["from"], item["to"]],
                color="#fb923c",
                weight=2,
                opacity=0.85,
                dash_array="4, 7",
                tooltip=f"Prototype prediction {item['hours']}h · {item['name']}",
            ).add_to(fmap)
            folium.CircleMarker(
                location=item["to"],
                radius=4,
                color="#fb923c",
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Predicted {item['hours']}h: {item['name']}",
            ).add_to(fmap)

    for berg in icebergs:
        is_new = bool(berg.get("is_new"))
        color = "#facc15" if is_new else "#ef4444"
        folium.Circle(
            location=[berg["lat"], berg["lon"]],
            radius=berg["radius_km"] * 1000,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.28,
            popup=folium.Popup(
                f"<b>{berg['name']}</b><br>SIMULATED iceberg"
                f"{' (NEW)' if is_new else ''}<br>"
                f"Radius {berg['radius_km']:.0f} km",
                max_width=260,
            ),
        ).add_to(fmap)
        folium.Marker(
            location=[berg["lat"], berg["lon"]],
            icon=_div_icon("🔴", 16),
            tooltip=f"{berg['name']} (simulated iceberg)",
        ).add_to(fmap)

    folium.Marker(
        location=[destination["lat"], destination["lon"]],
        icon=_div_icon("🏁", 24),
        tooltip=f"Destination: {destination['name']}",
        popup=folium.Popup(
            f"<b>{destination['name']}</b><br>SIMULATED destination",
            max_width=240,
        ),
    ).add_to(fmap)

    folium.Marker(
        location=[ship["lat"], ship["lon"]],
        icon=_div_icon("🚢", 26),
        tooltip=f"Ship: {ship['name']}",
        popup=folium.Popup(
            f"<b>{ship['name']}</b><br>SIMULATED position",
            max_width=240,
        ),
    ).add_to(fmap)

    legend = """
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;
                background:rgba(8,18,36,0.92);color:#e2e8f0;padding:10px 12px;
                border-radius:8px;border:1px solid #334155;font-size:12px;
                line-height:1.55;box-shadow:0 2px 10px rgba(0,0,0,0.35);">
      <b>Map legend (simulated)</b><br>
      🚢 Ship &nbsp; 🏁 Destination<br>
      🔴 Icebergs &nbsp; 🔵 Sea-ice fields<br>
      <span style="color:#22d3ee;">━</span> Recommended route<br>
      <span style="color:#94a3b8;">┄</span> Previous route<br>
      <span style="color:#fb923c;">┄</span> Prototype iceberg prediction
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap
