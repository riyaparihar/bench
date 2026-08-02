"""Shared NSRDB + pvlib computation — the single pvlib pipeline.

Used by both the in-layout energy provider (solar/panel_energy_nsrdb.py) and the
standalone provider (energy/nsrdb.py) so the pipeline lives in exactly one place.
Weather is supplied by the caller (prefetched in TS via fetchNsrdbWeather.py / the
hsds module) — this module never touches HSDS.
"""
import numpy as np
import pandas as pd
import pvlib
import pvlib.irradiance


# Clearness (epsilon) bin edges and the zenith weight kappa, exactly as defined
# by Perez et al. 1990 and used inside pvlib.irradiance.perez.
_PEREZ_EPS_BINS = (0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)
_PEREZ_KAPPA = 1.041  # zenith expressed in radians


def perez_f1f2(
    dhi,
    dni,
    zenith_rad,
    airmass,
    dni_extra,
    model: str = "allsitescomposite1990",
):
    """Vectorized Perez F1 (circumsolar) and F2 (horizon) brightness coefficients.

    Same math and coefficients as :func:`pvlib.irradiance.perez`. pvlib computes
    F1/F2 internally but never exposes them, so we source its authoritative
    coefficient table (``_get_perez_coefficients``) and apply the standard
    formula ourselves. This lets a caller reuse F1/F2 in a custom transposition
    (e.g. a per-pixel sky-view-factor / shadow kernel) without re-deriving — or
    mis-copying — the coefficient table.

    Parameters
    ----------
    dhi, dni : array-like
        Diffuse-horizontal and direct-normal irradiance (W/m²).
    zenith_rad : array-like
        Apparent solar zenith angle in radians.
    airmass : array-like
        Relative (not pressure-corrected) airmass.
    dni_extra : array-like
        Extraterrestrial DNI (W/m²) for the day of year.
    model : str
        Any coefficient set accepted by pvlib (default ``allsitescomposite1990``).

    Returns
    -------
    (F1, F2) : tuple of np.ndarray
        F1 clamped to ≥ 0; F2 may be negative. Nighttime / invalid samples
        (dhi ≤ 0 or sun at/below the horizon) are returned as 0 so they
        contribute no diffuse.
    """
    dhi = np.asarray(dhi, dtype=float)
    dni = np.asarray(dni, dtype=float)
    z = np.asarray(zenith_rad, dtype=float)
    am = np.maximum(np.asarray(airmass, dtype=float), 0.0)
    dx = np.maximum(np.asarray(dni_extra, dtype=float), 1.0)

    valid = (dhi > 0) & (z < np.pi / 2)
    dhi_safe = np.where(valid, dhi, 1.0)  # avoid div-by-zero in eps

    delta = dhi * am / dx
    with np.errstate(invalid="ignore", divide="ignore"):
        eps = (
            (dhi_safe + np.maximum(dni, 0.0)) / dhi_safe + _PEREZ_KAPPA * z**3
        ) / (1 + _PEREZ_KAPPA * z**3)

    # digitize then shift to 0-based — identical to pvlib's ebin construction.
    ebin = np.clip(np.digitize(eps, _PEREZ_EPS_BINS) - 1, 0, 7)

    f1c, f2c = pvlib.irradiance._get_perez_coefficients(model)
    F1 = np.maximum(f1c[ebin, 0] + f1c[ebin, 1] * delta + f1c[ebin, 2] * z, 0.0)
    F2 = f2c[ebin, 0] + f2c[ebin, 1] * delta + f2c[ebin, 2] * z

    return np.where(valid, F1, 0.0), np.where(valid, F2, 0.0)


# NSRDB attribute (camelCase, as emitted by fetchNsrdbWeather.py) -> pvlib column.
_WEATHER_RENAME = {
    "ghi": "ghi",
    "dni": "dni",
    "dhi": "dhi",
    "airTemperature": "temp_air",
    "windSpeed": "wind_speed",
    "relativeHumidity": "relative_humidity",
}


def weather_dict_to_dataframe(weather: dict) -> pd.DataFrame:
    """Rebuild the tz-aware hourly weather DataFrame from the JSON payload."""
    index = pd.to_datetime(weather["timeIndex"])
    cols = {
        _WEATHER_RENAME[k]: weather[k]
        for k in _WEATHER_RENAME
        if k in weather and weather[k] is not None
    }
    df = pd.DataFrame(cols, index=index)
    df.index.name = "Timestamp"
    return df


def hourly_dc_kwh_per_kw(
    weather: pd.DataFrame,
    lat: float,
    lon: float,
    tilt: float,
    azimuth: float,
) -> pd.Series:
    """Hourly (8760) DC kWh produced per 1 kW DC installed at this orientation.

    Keeps the weather's tz-aware index, so the caller can sum it to an annual
    figure or roll it up to local-calendar months (monthly_from_hourly).
    """
    # Don't pass weather.index.tz to Location: after the JSON round-trip the index
    # carries a FIXED offset (e.g. 'UTC-07:00'), which pvlib feeds to
    # zoneinfo.ZoneInfo() and fails (not an IANA key). The solar-position calc
    # uses the tz-aware `weather.index` directly, so the result is unchanged;
    # Location.tz is only metadata. Default ('UTC') is a valid key.
    location = pvlib.location.Location(lat, lon)
    solpos = location.get_solarposition(weather.index)
    airmass = location.get_airmass(weather.index, solar_position=solpos)
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
        albedo=0.2,
        model="perez",
    )
    # shaded_poa_per_px = apply_shading(dhi,dni,ghi,dsm)

    # []
    temp_air = weather.get("temp_air", pd.Series(25.0, index=weather.index))
    wind = weather.get("wind_speed", pd.Series(1.0, index=weather.index))
    cell_temp = pvlib.temperature.faiman(poa["poa_global"], temp_air, wind)

    [...]

# houlry_irradiance_per_px
#     hourly_dc_px 


# monthly_irr/dc, annual/irr/dc

# annual_irr -> annaul_dc  conversion: per [1000x1000] []
# w/m2


    # pdc0=1000 W -> dc series is "W per 1 kW DC installed"; each hourly sample is
    # ~Wh over that hour, so /1000 gives kWh/kW for that hour.
    dc_watts = pvlib.pvsystem.pvwatts_dc(
        poa["poa_global"], cell_temp, pdc0=1000.0, gamma_pdc=-0.004
    )
    # The Perez transposition yields NaN at twilight hours (sun at/below the
    # horizon). fillna(0) makes those hours explicit zero production: keeps the
    # 8760 hourly array valid JSON (json.dumps would otherwise emit the bare `NaN`
    # token, which strict JSON.parse on the Node bridge rejects) and is numerically
    # a no-op for the annual/monthly sums, which already skipped NaN.
    return (dc_watts.clip(lower=0) / 1000.0).fillna(0.0)


def annual_dc_kwh_per_kw(
    weather: pd.DataFrame,
    lat: float,
    lon: float,
    tilt: float,
    azimuth: float,
) -> float:
    """Annual DC kWh produced per 1 kW DC installed at this orientation."""
    return float(hourly_dc_kwh_per_kw(weather, lat, lon, tilt, azimuth).sum())


def monthly_from_hourly(hourly: pd.Series) -> list:
    """Roll an 8760 hourly Series up to 12 local-calendar-month sums (Jan..Dec).

    Buckets by the tz-aware index's local month, consistent with the local
    conversion applied when the weather was fetched.
    """
    by_month = hourly.groupby(hourly.index.month).sum()
    return [float(by_month.get(m, 0.0)) for m in range(1, 13)]


def compute_optimal_poa_annual(
    latitude: float,
    longitude: float,
    tmy: pd.DataFrame,
) -> float:
    """Annual POA sum on optimal tilt + south-facing surface. No shading."""
    location = pvlib.location.Location(latitude, longitude)
    solpos = pvlib.solarposition.get_solarposition(tmy.index, latitude, longitude)
    airmass = location.get_airmass(tmy.index, solar_position=solpos)
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=abs(latitude),
        surface_azimuth=180.0 if latitude >= 0 else 0.0,  # south in NH, north in SH
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=tmy["dni"],
        ghi=tmy["ghi"],
        dhi=tmy["dhi"],
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
        albedo=0.2,
        model="perez",
    )
    return float(poa["poa_global"].fillna(0.0).sum())

def compute_facet_poa_annual(
    latitude: float,
    longitude: float,
    tmy: pd.DataFrame,
    tilt: float,      # facet tilt in degrees (from roof segment)
    azimuth: float,   # facet azimuth in degrees (from roof segment)
) -> float:
    location = pvlib.location.Location(latitude, longitude)
    solpos = pvlib.solarposition.get_solarposition(tmy.index, latitude, longitude)
    airmass = location.get_airmass(tmy.index, solar_position=solpos)
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=tmy["dni"],
        ghi=tmy["ghi"],
        dhi=tmy["dhi"],
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
        albedo=0.2,
        model="perez",
    )
    return float(poa["poa_global"].fillna(0.0).sum())

