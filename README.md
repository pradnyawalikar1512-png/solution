# IRANav — Icebound-Resilient Autonomous Navigator

Prototype dashboard for an **AI-enabled Antarctic sea-ice, iceberg trajectory, and navigation decision-support** demo.

IRANav shows simulated hazards on an interactive map, scores grid risk, and uses **A\*** to recommend a safer route. You can inject a new iceberg or storm and watch the route recalculate.

**This is a prototype, not a real navigation system.**  
All environmental layers are **simulated**. The app does not use live satellite, ice-chart, or real-time Antarctic data.

## Problem

Ships operating in polar waters need a clear picture of changing sea ice and icebergs, plus a way to replan when a new hazard appears. Live data and certified models belong in later versions. This first version demonstrates the decision-support flow with sample data.

## Solution

A lightweight Streamlit web app that:

1. Displays a ship, destination, icebergs, and sea-ice fields on a real interactive map
2. Builds a 0–100 risk grid from simulated ice, weather, visibility, and ice pressure
3. Runs A* with **Safest / Fastest / Greenest** cost weights
4. Recalculates when you simulate a new iceberg or storm
5. Shows a simple **prototype iceberg trajectory** from simulated wind and current

## Main features

- Interactive Folium map (zoom and pan) around a simulated Antarctic Peninsula / Southern Ocean box
- Risk heatmap and 0–30 Low / 31–60 Moderate / 61–100 High bands (demo thresholds, not official standards)
- Ship profile: name, Polar Class PC1–PC7 (prototype parameters only), speed, fuel, risk preference
- Three route modes with adjustable weights inside A*
- Simulate New Iceberg, Simulate Storm, Reset Simulation
- Previous vs new route after a reroute
- Route explanation (cost-function reasons, not a trained AI claim)
- Prototype 6h / 12h / 24h iceberg drift overlay

## Technology stack

- Python
- Streamlit
- Folium + streamlit-folium
- NumPy, Pandas, Plotly
- A* pathfinding (no GPU, no paid APIs, no database)

## Project files

| File | Role |
| --- | --- |
| `app.py` | Streamlit dashboard (main entry point) |
| `simulation.py` | Simulated icebergs, sea ice, weather, drift |
| `risk_engine.py` | Grid risk scores 0–100 |
| `pathfinding.py` | A* and route statistics |
| `map_view.py` | Interactive Folium map |
| `requirements.txt` | Python packages |
| `README.md` | This file |

## How to run locally

From the project folder:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to deploy on Streamlit Community Cloud

1. Put this project in a public GitHub repository (include `app.py` and `requirements.txt` at the repo root, or set the app path to `app.py`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, choose the repo, branch, and main file `app.py`.
4. Deploy. Streamlit installs packages from `requirements.txt`.
5. Share the public `*.streamlit.app` URL. It can be opened from another computer, phone, Wi-Fi, or mobile data.

Do not add API keys, passwords, localhost URLs, or personal file paths. None are required.

## Simulated data statement

Iceberg positions, radii, sea-ice concentration, wind, current, temperature, visibility, ice pressure, risk scores, and iceberg “predictions” are **generated in code** for the demo. They are **not** live ice charts and **not** a scientifically validated drift model.

Polar Class values only change prototype routing costs. They are not claims about real ship capability.

## Future scope

- Connect public ice-chart / satellite products
- Replace the simple drift formula with a researched trajectory model
- Add machine-learning sea-ice nowcasting (clearly labeled)
- Offline shipboard packaging
