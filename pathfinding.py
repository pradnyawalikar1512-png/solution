"""
A* pathfinding with prototype multi-objective costs.

Safest / Fastest / Greenest change the weights. Polar Class values are
demo parameters only — not official ice-class performance.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from risk_engine import UNSAFE_RISK, risk_band
from simulation import cell_to_latlon, haversine_km, latlon_to_cell

GridPos = Tuple[int, int]

NEIGHBOURS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

ROUTE_MODES = {
    "Safest": {"distance": 0.45, "risk": 5.4, "fuel": 0.9},
    "Fastest": {"distance": 2.4, "risk": 1.15, "fuel": 0.35},
    "Greenest": {"distance": 0.85, "risk": 2.2, "fuel": 3.1},
}

# Prototype Polar Class multipliers for ice-related cost. Not capability claims.
POLAR_CLASS_ICE_FACTOR = {
    "PC1": 0.78,
    "PC2": 0.84,
    "PC3": 0.90,
    "PC4": 0.96,
    "PC5": 1.02,
    "PC6": 1.08,
    "PC7": 1.14,
}

RISK_PREF_FACTOR = {
    "Cautious": 1.35,
    "Balanced": 1.00,
    "Direct": 0.72,
}


def _in_bounds(pos: GridPos, grid: np.ndarray) -> bool:
    rows, cols = grid.shape
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def _step_distance(a: GridPos, b: GridPos) -> float:
    if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1:
        return math.sqrt(2.0)
    return 1.0


def astar(
    risk_grid: np.ndarray,
    start: GridPos,
    goal: GridPos,
    mode: str = "Safest",
    polar_class: str = "PC5",
    risk_pref: str = "Balanced",
    blocked_threshold: float = UNSAFE_RISK,
) -> Optional[List[GridPos]]:
    """Return a list of grid cells from start to goal, or None."""
    if start == goal:
        return [start]
    if not _in_bounds(start, risk_grid) or not _in_bounds(goal, risk_grid):
        return None

    weights = ROUTE_MODES.get(mode, ROUTE_MODES["Safest"])
    ice_f = POLAR_CLASS_ICE_FACTOR.get(polar_class, 1.0)
    pref_f = RISK_PREF_FACTOR.get(risk_pref, 1.0)

    def is_blocked(cell: GridPos) -> bool:
        if cell == start or cell == goal:
            return False
        return float(risk_grid[cell]) >= blocked_threshold

    open_heap: List[Tuple[float, int, GridPos]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start))
    came_from: Dict[GridPos, GridPos] = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _f, _i, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _rebuild(came_from, current)
        closed.add(current)

        for dr, dc in NEIGHBOURS:
            neighbour = (current[0] + dr, current[1] + dc)
            if not _in_bounds(neighbour, risk_grid) or is_blocked(neighbour):
                continue
            if neighbour in closed:
                continue

            risk = float(risk_grid[neighbour]) / 100.0
            step = _step_distance(current, neighbour)
            dist_cost = weights["distance"] * step
            risk_cost = weights["risk"] * pref_f * ice_f * (risk ** 2) * 8.0
            fuel_cost = weights["fuel"] * step * (1.0 + 1.4 * risk)
            tentative = g_score[current] + dist_cost + risk_cost + fuel_cost

            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heuristic = math.hypot(neighbour[0] - goal[0], neighbour[1] - goal[1])
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic * weights["distance"], counter, neighbour))

    return None


def _rebuild(came_from: Dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def estimate_eta_hours(distance_km: float, max_speed_kt: float, avg_risk: float, fuel_percent: float) -> float:
    """Simple prototype ETA. Fuel and ice risk slightly reduce assumed speed."""
    speed = max(4.0, float(max_speed_kt))
    fuel = min(max(float(fuel_percent), 15.0), 100.0)
    speed *= 0.72 + 0.28 * (fuel / 100.0)
    speed *= 1.0 - 0.22 * (avg_risk / 100.0)
    kmh = max(speed * 1.852, 3.0)
    return distance_km / kmh


def estimate_fuel_index(distance_km: float, avg_risk: float) -> float:
    """Relative fuel index for the dashboard (not real consumption)."""
    return distance_km * (1.0 + avg_risk / 140.0)


def find_route(
    risk_grid: np.ndarray,
    ship_lat: float,
    ship_lon: float,
    dest_lat: float,
    dest_lon: float,
    mode: str = "Safest",
    polar_class: str = "PC5",
    risk_pref: str = "Balanced",
    max_speed_kt: float = 12.0,
    fuel_percent: float = 78.0,
) -> Dict:
    """Run A* and collect dashboard statistics. Never raises on a missing path."""
    start = latlon_to_cell(ship_lat, ship_lon)
    goal = latlon_to_cell(dest_lat, dest_lon)
    path = astar(risk_grid, start, goal, mode=mode, polar_class=polar_class, risk_pref=risk_pref)
    if path is None:
        path = astar(
            risk_grid,
            start,
            goal,
            mode=mode,
            polar_class=polar_class,
            risk_pref=risk_pref,
            blocked_threshold=96.0,
        )

    empty = {
        "found": False,
        "path_cells": [],
        "path_latlon": [],
        "distance_km": 0.0,
        "steps": 0,
        "max_risk": 0.0,
        "avg_risk": 0.0,
        "eta_hours": 0.0,
        "fuel_index": 0.0,
        "risk_band": "—",
        "mode": mode,
    }
    if not path:
        return empty

    points = [cell_to_latlon(r, c) for r, c in path]
    points[0] = (ship_lat, ship_lon)
    points[-1] = (dest_lat, dest_lon)

    distance = 0.0
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        distance += haversine_km(a[0], a[1], b[0], b[1])

    risks = [float(risk_grid[cell]) for cell in path]
    max_risk = max(risks)
    avg_risk = sum(risks) / len(risks)
    eta = estimate_eta_hours(distance, max_speed_kt, avg_risk, fuel_percent)

    return {
        "found": True,
        "path_cells": path,
        "path_latlon": points,
        "distance_km": distance,
        "steps": len(path),
        "max_risk": max_risk,
        "avg_risk": avg_risk,
        "eta_hours": eta,
        "fuel_index": estimate_fuel_index(distance, avg_risk),
        "risk_band": risk_band(avg_risk),
        "mode": mode,
    }
