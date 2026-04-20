from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext, ToolError


class WeatherCurrentArgs(BaseModel):
    location: str | None = Field(default=None, max_length=200)


_WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_GEOCODE_CACHE: dict[str, dict[str, Any] | None] = {}


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        res = requests.get(url, params=params, timeout=(3, 10), headers={"User-Agent": "BUTLER/1.0"})
        if res.status_code != 200:
            raise ToolError(f"HTTP {res.status_code} from {url}")
        data = res.json()
        if not isinstance(data, dict):
            raise ToolError(f"Unexpected response from {url}")
        return data
    except requests.RequestException as e:
        raise ToolError(f"Request failed: {e}") from e
    except ValueError as e:
        raise ToolError(f"Invalid JSON from {url}") from e


def _geocode_location(query: str) -> dict[str, Any] | None:
    key = query.lower().strip()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    data = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query, "count": 1, "language": "en", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        _GEOCODE_CACHE[key] = None
        return None
    first = results[0]
    label_parts = [
        first.get("name"),
        first.get("admin1"),
        first.get("country"),
    ]
    label = ", ".join(part for part in label_parts if isinstance(part, str) and part.strip())
    result = {
        "latitude": first.get("latitude"),
        "longitude": first.get("longitude"),
        "label": label or query,
    }
    _GEOCODE_CACHE[key] = result
    return result


def _ip_geolocation() -> dict[str, Any]:
    data = _get_json(
        "http://ip-api.com/json/",
        params={"fields": "status,message,city,regionName,country,lat,lon,timezone,query"},
    )
    if data.get("status") != "success":
        message = data.get("message") or "unknown error"
        raise ToolError(f"IP geolocation failed: {message}")

    label_parts = [
        data.get("city"),
        data.get("regionName"),
        data.get("country"),
    ]
    label = ", ".join(part for part in label_parts if isinstance(part, str) and part.strip())
    return {
        "latitude": data.get("lat"),
        "longitude": data.get("lon"),
        "label": label or "your area",
    }


def _resolve_location(ctx: ToolContext, location_hint: str | None) -> dict[str, Any]:
    candidates: list[tuple[str, str]] = []
    if isinstance(location_hint, str) and location_hint.strip():
        candidates.append(("query", location_hint.strip()))
    if ctx.config.home_location.strip():
        candidates.append(("home_location", ctx.config.home_location.strip()))

    seen: set[str] = set()
    for source, candidate in candidates:
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        geocoded = _geocode_location(candidate)
        if geocoded and geocoded.get("latitude") is not None and geocoded.get("longitude") is not None:
            geocoded["resolved_from"] = source
            return geocoded

    ip_location = _ip_geolocation()
    ip_location["resolved_from"] = "ip"
    return ip_location


def _weather_current(ctx: ToolContext, args: WeatherCurrentArgs) -> dict[str, Any]:
    resolved = _resolve_location(ctx, args.location)
    latitude = resolved["latitude"]
    longitude = resolved["longitude"]

    forecast = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,is_day",
            "timezone": "auto",
        },
    )
    current = forecast.get("current") or {}
    weather_code = current.get("weather_code")

    return {
        "location": resolved["label"],
        "resolved_from": resolved["resolved_from"],
        "latitude": latitude,
        "longitude": longitude,
        "timezone": forecast.get("timezone"),
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kph": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
            "is_day": current.get("is_day"),
            "weather_code": weather_code,
            "condition": _WEATHER_CODE_LABELS.get(weather_code, "Unknown"),
        },
    }


def build() -> list[Tool]:
    return [
        Tool(
            name="weather.current",
            description="Get the current weather for a location, home location, or your IP-based area.",
            input_model=WeatherCurrentArgs,
            handler=_weather_current,
            side_effect=False,
        ),
    ]
