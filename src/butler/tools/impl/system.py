from __future__ import annotations

import platform
import time
import math
import requests
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext


class NowArgs(BaseModel):
    pass

class DistanceArgs(BaseModel):
    location1: str = Field(description="First location (e.g. 'New York', 'home')")
    location2: str = Field(description="Second location")


def _now(ctx: ToolContext, args: NowArgs) -> dict[str, Any]:
    now = datetime.now()
    return {
        "datetime_iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "epoch_ms": int(time.time() * 1000),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _geocode(loc: str, home: str) -> tuple[float, float, str]:
    if loc.lower() == "home":
        loc = home
    if not loc:
        raise Exception("Location not provided (or home location not set)")
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": loc, "count": 1, "format": "json"}
    )
    resp.raise_for_status()
    data = resp.json()
    if "results" not in data or not data["results"]:
        raise Exception(f"Location not found: {loc}")
    res = data["results"][0]
    return res["latitude"], res["longitude"], res.get("name", loc)

def _distance(ctx: ToolContext, args: DistanceArgs) -> dict[str, Any]:
    home = ctx.config.home_location
    try:
        lat1, lon1, name1 = _geocode(args.location1, home)
        lat2, lon2, name2 = _geocode(args.location2, home)
    except Exception as e:
        return {"error": str(e)}
        
    km = _haversine(lat1, lon1, lat2, lon2)
    mi = km * 0.621371
    return {
        "location1": name1,
        "location2": name2,
        "distance_km": round(km, 2),
        "distance_miles": round(mi, 2)
    }

def build() -> list[Tool]:
    return [
        Tool(
            name="system.now",
            description="Get current time and basic runtime info.",
            input_model=NowArgs,
            handler=_now,
            side_effect=False,
        ),
        Tool(
            name="system.distance",
            description="Calculate distance between two locations (can use 'home').",
            input_model=DistanceArgs,
            handler=_distance,
            side_effect=False,
        )
    ]
