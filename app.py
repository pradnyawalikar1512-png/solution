"""
IRANav — Icebound-Resilient Autonomous Navigator

Streamlit dashboard for a simulated Antarctic navigation decision-support
prototype. Entry point:

    streamlit run app.py
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from map_view import build_map
from pathfinding import find_route
from risk_engine import build_risk_grid, risk_band, route_is_high_risk
from simulation import (
    create_base_sim,
    iceberg_along_route,
    make_storm,
    predict_iceberg_track,
)

st.set_page_config(
    page_title="IRANav | Icebound-Resilient Autonomous Navigator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {background-color: #07111d;}
      .block-container {padding-top: 1.05rem; padding-bottom: 2rem;}
      h1, h2, h3 {color: #f1f5f9;}
      .sim-banner {
        background: #3b1d08;
        border: 1px solid #f59e0b;
        color: #fde68a;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.35rem 0 0.9rem 0;
      }
      .alert-banner {
        background: #3f1d24;
        border: 1px solid #fb7185;
        color: #fecdd3;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0 0 0.8rem 0;
      }
      .explain-box {
        background: #0f1c2e;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        color: #cbd5e1;
      }
      div[data-testid="stMetricValue"] {color: #e2e8f0;}
      div[data-testid="stMetricLabel"] {color: #94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "sim" not in st.session_state:
        st.session_state.sim = create_base_sim()
        st.session_state.previous_route = None
        st.session_state.alert = None


def route_explanation(route: dict, polar_class: str, mode: str, rerouted: bool) -> list[str]:
    bullets = [
        f"Selected mode is **{mode}**, so A* weighted that objective more heavily.",
        "High-risk iceberg cores are treated as blocked where a bypass exists.",
        f"Prototype Polar Class **{polar_class}** changes ice-related cost (demo parameter only).",
        "Travel time and a relative fuel index are estimated from distance and average risk.",
    ]
    if rerouted:
        bullets.insert(0, "A new simulated hazard raised risk on the previous path, so A* was run again.")
    if route["found"] and route["avg_risk"] <= 30:
        bullets.append("Average path risk stayed in the Low prototype band.")
    elif route["found"] and route["avg_risk"] <= 60:
        bullets.append("Average path risk is Moderate; the search still avoided the highest iceberg cells.")
    return bullets


def prediction_rows(icebergs, weather, storm, hours: int) -> list[dict]:
    rows = []
    lines = []
    for berg in icebergs:
        pred = predict_iceberg_track(berg, weather, hours, storm)
        rows.append(
            {
                "Iceberg": berg["name"],
                "Now lat": round(berg["lat"], 3),
                "Now lon": round(berg["lon"], 3),
                f"+{hours}h lat": round(pred["lat"], 3),
                f"+{hours}h lon": round(pred["lon"], 3),
                "Bearing °": round(pred["bearing_deg"], 0),
                "Drift km/h": round(pred["speed_kmh"], 2),
            }
        )
        lines.append(
            {
                "name": berg["name"],
                "hours": hours,
                "from": (berg["lat"], berg["lon"]),
                "to": (pred["lat"], pred["lon"]),
            }
        )
    return rows, lines


init_state()
sim = st.session_state.sim

st.title("IRANav")
st.subheader("Icebound-Resilient Autonomous Navigator")
st.caption("Antarctic sea-ice, iceberg, and navigation decision-support prototype.")

st.markdown(
    '<div class="sim-banner">'
    "⚠️ Prototype uses simulated environmental data. Not live ice-chart data. "
    "Icebergs, sea ice, wind, current, temperature, visibility, and risk scores "
    "are generated for this demo. They are not live satellite or real-time Antarctic data."
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🚢 Ship Profile")
    st.caption("Prototype parameters — not official Polar Class performance.")
    ship_name = st.text_input("Ship name", value=sim["ship"]["name"])
    polar_class = st.selectbox(
        "Polar Class",
        options=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"],
        index=4,
        help="Used only as a demo cost multiplier in A*. Not a certified ice-class model.",
    )
    max_speed = st.slider("Maximum speed (knots)", min_value=6, max_value=16, value=12)
    fuel = st.slider("Fuel available (%)", min_value=20, max_value=100, value=78)
    risk_pref = st.selectbox("Risk preference", options=["Cautious", "Balanced", "Direct"], index=1)

    st.markdown("### Route mode")
    mode = st.radio(
        "Route options",
        options=["Safest", "Fastest", "Greenest"],
        index=0,
        captions=["🛡️ Lowest risk", "⚡ Shorter time", "🌱 Lower fuel index"],
    )

    st.markdown("### Prediction horizon")
    horizon = st.selectbox("Prototype iceberg trajectory", options=[6, 12, 24], index=1, format_func=lambda h: f"{h} hours")
    st.caption("Simple drift from simulated wind/current. Not a validated prediction model.")

    st.divider()
    st.markdown("### Simulation controls")
    col_a, col_b = st.columns(2)
    add_berg = col_a.button("🧊 Simulate New Iceberg", use_container_width=True)
    add_storm = col_b.button("🌪️ Simulate Storm", use_container_width=True)
    reset_sim = st.button("Reset Simulation", use_container_width=True)

if reset_sim:
    st.session_state.sim = create_base_sim()
    st.session_state.previous_route = None
    st.session_state.alert = None
    st.rerun()

risk_grid = build_risk_grid(
    sim["icebergs"],
    sim["zones"],
    sim["weather"],
    sim["pressure"],
    sim["storm"],
)

route = find_route(
    risk_grid,
    sim["ship"]["lat"],
    sim["ship"]["lon"],
    sim["destination"]["lat"],
    sim["destination"]["lon"],
    mode=mode,
    polar_class=polar_class,
    risk_pref=risk_pref,
    max_speed_kt=float(max_speed),
    fuel_percent=float(fuel),
)

if add_berg:
    previous = deepcopy(route)
    berg = iceberg_along_route(route["path_latlon"], sim["next_iceberg_id"])
    sim["icebergs"].append(berg)
    sim["next_iceberg_id"] += 1
    risk_grid = build_risk_grid(
        sim["icebergs"], sim["zones"], sim["weather"], sim["pressure"], sim["storm"]
    )
    route_hit = route_is_high_risk(previous["path_cells"], risk_grid)
    new_route = find_route(
        risk_grid,
        sim["ship"]["lat"],
        sim["ship"]["lon"],
        sim["destination"]["lat"],
        sim["destination"]["lon"],
        mode=mode,
        polar_class=polar_class,
        risk_pref=risk_pref,
        max_speed_kt=float(max_speed),
        fuel_percent=float(fuel),
    )
    st.session_state.previous_route = previous
    route = new_route
    st.session_state.alert = "⚠️ New iceberg detected. Route recalculated for safety."
    if route_hit:
        st.session_state.alert += " Previous path entered the High risk band."
    sim["event_log"].append(st.session_state.alert)
    st.rerun()

if add_storm:
    previous = deepcopy(route)
    sim["storm"] = make_storm(route["path_latlon"])
    risk_grid = build_risk_grid(
        sim["icebergs"], sim["zones"], sim["weather"], sim["pressure"], sim["storm"]
    )
    new_route = find_route(
        risk_grid,
        sim["ship"]["lat"],
        sim["ship"]["lon"],
        sim["destination"]["lat"],
        sim["destination"]["lon"],
        mode=mode,
        polar_class=polar_class,
        risk_pref=risk_pref,
        max_speed_kt=float(max_speed),
        fuel_percent=float(fuel),
    )
    st.session_state.previous_route = previous
    route = new_route
    st.session_state.alert = "⚠️ Simulated storm added. Risk map updated and route recalculated."
    sim["event_log"].append(st.session_state.alert)
    st.rerun()

if st.session_state.alert:
    st.markdown(f'<div class="alert-banner">{st.session_state.alert}</div>', unsafe_allow_html=True)

ship_for_map = {"name": ship_name, "lat": sim["ship"]["lat"], "lon": sim["ship"]["lon"]}
n_hazards = len(sim["icebergs"]) + len(sim["zones"]) + (1 if sim["storm"] else 0)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Risk score", f"{route['avg_risk']:.0f} / 100" if route["found"] else "—", route["risk_band"] if route["found"] else None)
m2.metric("Distance", f"{route['distance_km']:.0f} km" if route["found"] else "No route")
m3.metric("ETA", f"{route['eta_hours']:.1f} h" if route["found"] else "—")
m4.metric("Hazards", f"{n_hazards}")
m5.metric("Peak route risk", f"{route['max_risk']:.0f} ({risk_band(route['max_risk'])})" if route["found"] else "—")

w1, w2, w3, w4 = st.columns(4)
w1.metric("Wind", f"{sim['weather']['wind_kt']:.0f} kt")
w2.metric("Current", f"{sim['weather']['current_kt']:.2f} kt")
w3.metric("Temperature", f"{sim['weather']['temperature_c']:.1f} °C")
w4.metric("Visibility", f"{sim['weather']['visibility_km']:.1f} km")
st.caption("Environmental values above are SIMULATED and held constant unless you add a storm.")

pred_table, pred_lines = prediction_rows(sim["icebergs"], sim["weather"], sim["storm"], int(horizon))
previous_latlon = None
if st.session_state.previous_route and st.session_state.previous_route.get("found"):
    previous_latlon = st.session_state.previous_route["path_latlon"]

st.markdown("### Antarctic navigation map")
st.caption("Interactive map of a simulated Antarctic Peninsula / Southern Ocean operating area. Zoom and pan freely.")

fmap = build_map(
    ship=ship_for_map,
    destination=sim["destination"],
    icebergs=sim["icebergs"],
    zones=sim["zones"],
    risk_grid=risk_grid,
    route_latlon=route["path_latlon"] if route["found"] else [],
    previous_latlon=previous_latlon,
    storm=sim["storm"],
    predictions=pred_lines,
)
st_folium(
    fmap,
    width=None,
    height=560,
    returned_objects=[],
    key=f"map-{len(sim['icebergs'])}-{sim['storm'] is not None}-{mode}-{horizon}",
)

left, right = st.columns((1.15, 1))

with left:
    st.markdown("### Recommended route")
    if route["found"]:
        st.success(
            f"A* ({mode}) · {route['steps']} grid steps · "
            f"{route['distance_km']:.0f} km · ETA {route['eta_hours']:.1f} h · "
            f"fuel index {route['fuel_index']:.0f}"
        )
        rerouted = st.session_state.previous_route is not None
        bullets = route_explanation(route, polar_class, mode, rerouted)
        st.markdown(
            "<div class='explain-box'><b>Recommended because:</b><ul>"
            + "".join(f"<li>{b}</li>" for b in bullets)
            + "</ul>"
            "<i>This explanation describes the prototype cost function. "
            "It is not an AI model prediction.</i></div>",
            unsafe_allow_html=True,
        )
        if route["found"]:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=[float(risk_grid[c]) for c in route["path_cells"]],
                    mode="lines",
                    line=dict(color="#22d3ee", width=3),
                    name="Risk along route",
                )
            )
            fig.update_layout(
                title="Simulated risk along recommended route",
                paper_bgcolor="#07111d",
                plot_bgcolor="#0f1c2e",
                font=dict(color="#cbd5e1"),
                height=240,
                margin=dict(l=40, r=20, t=40, b=30),
                yaxis_title="Risk (0–100)",
                xaxis_title="Grid step",
            )
            fig.update_xaxes(gridcolor="#1e293b")
            fig.update_yaxes(gridcolor="#1e293b", range=[0, 100])
            st.plotly_chart(fig, width="stretch")
    else:
        st.error("A* could not find a valid route for this simulated layout. Try Reset Simulation or another route mode.")

with right:
    st.markdown("### Prototype iceberg trajectory prediction")
    st.caption("Current vs predicted position from simulated wind and current. Not a scientifically validated model.")
    st.dataframe(pd.DataFrame(pred_table), hide_index=True, width="stretch")
    st.markdown("### Alerts")
    if sim["event_log"]:
        for item in sim["event_log"][-5:]:
            st.warning(item)
    else:
        st.info("No dynamic hazards yet. Use Simulate New Iceberg or Simulate Storm.")
    st.markdown("### Destination")
    st.write(sim["destination"]["name"])
    st.caption(f"{sim['destination']['lat']:.2f}°, {sim['destination']['lon']:.2f}°")

st.markdown("---")
st.caption(
    "IRANav prototype  ·  Python + Streamlit + Folium + A*  ·  "
    "0–30 Low, 31–60 Moderate, 61–100 High are demo thresholds, not official maritime standards.  ·  "
    "SIMULATED DATA"
)
