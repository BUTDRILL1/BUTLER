from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext
from butler.tools.impl import weather


@dataclass
class _Resp:
    status_code: int
    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload


def _ctx(tmp_path, monkeypatch, home_location: str = "") -> ToolContext:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig(home_location=home_location)
    db = ButlerDB.open(cfg)
    weather._GEOCODE_CACHE.clear()
    return ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))


def test_weather_current_prefers_query_location_then_forecast(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, monkeypatch, home_location="Somewhere Else")
    seen_urls: list[str] = []

    def fake_get(url, params=None, timeout=None, headers=None):  # noqa: A002
        seen_urls.append(url)
        if "geocoding-api.open-meteo.com" in url:
            assert params["name"] == "Noida Sector 74, Uttar Pradesh, India"
            return _Resp(
                200,
                {
                    "results": [
                        {
                            "name": "Noida",
                            "admin1": "Uttar Pradesh",
                            "country": "India",
                            "latitude": 28.5355,
                            "longitude": 77.3910,
                        }
                    ]
                },
            )
        if "api.open-meteo.com" in url:
            return _Resp(
                200,
                {
                    "timezone": "Asia/Kolkata",
                    "current": {
                        "temperature_2m": 29.2,
                        "apparent_temperature": 31.0,
                        "relative_humidity_2m": 68,
                        "wind_speed_10m": 11.4,
                        "precipitation": 0.0,
                        "weather_code": 2,
                        "is_day": 1,
                    },
                },
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(weather.requests, "get", fake_get)

    result = weather._weather_current(ctx, weather.WeatherCurrentArgs(location="Noida Sector 74, Uttar Pradesh, India"))

    assert result["location"] == "Noida, Uttar Pradesh, India"
    assert result["resolved_from"] == "query"
    assert result["current"]["condition"] == "Partly cloudy"
    assert "geocoding-api.open-meteo.com" in seen_urls[0]
    assert "api.open-meteo.com" in seen_urls[1]


def test_weather_current_falls_back_to_home_then_ip(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, monkeypatch, home_location="Noida Sector 74, Uttar Pradesh, India")
    seen_urls: list[str] = []

    def fake_get(url, params=None, timeout=None, headers=None):  # noqa: A002
        seen_urls.append(url)
        if "geocoding-api.open-meteo.com" in url:
            assert params["name"] == "Noida Sector 74, Uttar Pradesh, India"
            return _Resp(200, {"results": []})
        if "ip-api.com" in url:
            return _Resp(
                200,
                {
                    "status": "success",
                    "city": "Noida",
                    "regionName": "Uttar Pradesh",
                    "country": "India",
                    "lat": 28.5355,
                    "lon": 77.3910,
                },
            )
        if "api.open-meteo.com" in url:
            return _Resp(
                200,
                {
                    "timezone": "Asia/Kolkata",
                    "current": {
                        "temperature_2m": 30.0,
                        "apparent_temperature": 33.0,
                        "relative_humidity_2m": 60,
                        "wind_speed_10m": 10.0,
                        "precipitation": 0.0,
                        "weather_code": 1,
                        "is_day": 1,
                    },
                },
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(weather.requests, "get", fake_get)

    result = weather._weather_current(ctx, weather.WeatherCurrentArgs(location=None))

    assert result["resolved_from"] == "ip"
    assert result["location"] == "Noida, Uttar Pradesh, India"
    assert seen_urls[0].startswith("https://geocoding-api.open-meteo.com")
    assert seen_urls[1].startswith("http://ip-api.com/json/")
    assert seen_urls[2].startswith("https://api.open-meteo.com")
