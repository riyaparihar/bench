"""
solar_positions.py — Solar geometry and Perez F1/F2 coefficients.

Public API
----------
compute_solar_positions_pre()  — everything that does NOT need the horizon
                                  (solpos, airmass, F1/F2, met arrays)
                                  safe to run in a background thread concurrently
                                  with run_terrain_analysis().

assemble_solar_result()        — adds beam_blk using the now-available horizon
                                  and returns the final SolarResult.
                                  Called on the main thread after the barrier.

compute_solar_positions()      — convenience wrapper: pre + assemble in one call.
                                  Used by panel / masked pipelines that already
                                  have terrain before they start solar work.

SolarPreResult                 — intermediate dataclass from the pre step.
SolarResult                    — final dataclass consumed by all kernels.

Parallelism design
------------------
Terrain (SVF ray march + PVGIS horizon) and solar positions (pvlib NREL +
Perez F1/F2) are completely independent until beam_blk needs the horizon.
Splitting into pre / assemble lets run_full_dsm run both in parallel:

    Thread A: run_terrain_analysis()         → TerrainResult
    Thread B: compute_solar_positions_pre()  → SolarPreResult
    Barrier:  wait for both
    Main:     assemble_solar_result(pre, terrain.horizon)  → SolarResult  (fast)

beam_blk is a single vectorised comparison — negligible cost after the barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import pvlib

# ── constants used by accumulation kernels ────────────────────────────────
_FAIMAN_U0: float = 25.0
_FAIMAN_U1: float = 6.84

# 0.18 = industry default for standard mono-Si (most common panel type)
# 0.20 = premium mono-Si
# 0.21 = HJT / heterojunction
# 0.23 = top-tier IBC (SunPower etc.)
_ETA_REF: float = 0.18  # standard mono-Si; use 0.21 for HJT

# How much panel efficiency drops per degree Celsius above reference temperature.
# -0.004 means efficiency drops 0.4% for every 1°C the panel heats up above its rated temperature.
# Negative because hotter = less efficient.
_GAMMA_PDC_DEFAULT: float = -0.004  # /°C; use -0.0026 for HJT / bifacial

# How much light the ground reflects.
# 0.2 means 20% of sunlight hitting the ground bounces back up toward the panel.
# Grass and typical urban surfaces are around 0.2.
_DEFAULT_ALBEDO: float = 0.2

# ============================================================
# RESULT TYPES
# ============================================================


@dataclass
class SolarPreResult:
    """
    Intermediate result from compute_solar_positions_pre().

    Contains everything computed without the horizon — safe to produce
    in a background thread while terrain analysis runs concurrently.
    beam_blk is absent; it is added by assemble_solar_result().
    """

    # ------------ geometry -------------------

    # Solar zenith angle for each hour
    # ------------------------------------
    # — how far the sun is from straight above.
    # 0 = directly overhead, π/2 = on the horizon.
    # One value per hour of the year.
    zen_rad: np.ndarray  # (n_hours,) float64

    # Solar azimuth for each hour
    # ------------------------------
    #  — which compass direction the sun is in.
    # 0 = North, π/2 = East, π = South, 3π/2 = West.
    az_rad: np.ndarray  # (n_hours,) float64

    # Solar elevation for each hour
    # ------------------------------------
    # — how high above the horizon the sun is.
    # 0 = on the horizon, π/2 = directly overhead.
    # This is simply 90° minus the zenith angle.
    el_rad: np.ndarray  # (n_hours,) float64

    # ------ Perez ---------------------------------
    # Perez sky brightness coefficients for each hour.
    # F1 = how much brighter the area around the sun is compared to the rest of the sky.
    # F2 = how bright the horizon band is.
    F1: np.ndarray  # (n_hours,) float32
    F2: np.ndarray  # (n_hours,) float32
    # irradiance
    ghi: np.ndarray  # (n_hours,) float32
    dni: np.ndarray  # (n_hours,) float32
    dhi: np.ndarray  # (n_hours,) float32
    # met
    temp_air: np.ndarray  # (n_hours,) float32
    wind_speed: np.ndarray  # (n_hours,) float32
    # time
    months: np.ndarray  # (n_hours,) int32  0-based
    daylight_hours: np.ndarray  # indices where ghi > 0


@dataclass
class SolarResult:
    """
    Final per-hour solar arrays consumed by all accumulation kernels.

    Produced by assemble_solar_result() after the terrain barrier.
    """

    # geometry
    zen_rad: np.ndarray  # (n_hours,) float64
    az_rad: np.ndarray  # (n_hours,) float64
    el_rad: np.ndarray  # (n_hours,) float64
    # Perez
    F1: np.ndarray  # (n_hours,) float32
    F2: np.ndarray  # (n_hours,) float32
    # horizon beam blocking  ← the only field that needs terrain
    beam_blk: np.ndarray  # (n_hours,) float32
    # irradiance
    ghi: np.ndarray  # (n_hours,) float32
    dni: np.ndarray  # (n_hours,) float32
    dhi: np.ndarray  # (n_hours,) float32
    # met
    temp_air: np.ndarray  # (n_hours,) float32
    wind_speed: np.ndarray  # (n_hours,) float32
    # time
    months: np.ndarray  # (n_hours,) int32  0-based
    daylight_hours: np.ndarray  # indices where ghi > 0


# ============================================================
# PRE STEP  (horizon-independent — runs in background thread)
# ============================================================


def compute_solar_positions_pre(
    tmy: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> SolarPreResult:
    """
    where is the sun, hour by hour?
    -------------------------------------
    Compute everything that does NOT require the horizon.

    Safe to run concurrently with run_terrain_analysis() in a separate
    thread.

    Parameters
    ----------
    tmy       : tz-aware hourly DataFrame — must contain ghi, dni, dhi
    latitude  : decimal degrees
    longitude : decimal degrees

    Returns
    -------
    SolarPreResult — passed to assemble_solar_result() after the barrier
    """
    for col in ('ghi', 'dni', 'dhi'):
        if col not in tmy.columns:
            raise ValueError(f"TMY DataFrame missing '{col}' column")
    if tmy.index.tz is None:
        raise ValueError('TMY DataFrame index must be tz-aware')

    n_hours = len(tmy)

    # ── Solar positions (most expensive step, releases GIL) ───────────
    solpos = pvlib.solarposition.get_solarposition(
        tmy.index, latitude, longitude, method='nrel_numpy'
    )
    zen_rad = np.radians(solpos['apparent_zenith'].values)  # how far from straight up
    az_rad = np.radians(solpos['azimuth'].values)  # compass direction
    el_rad = np.radians(solpos['apparent_elevation'].values)  # height above horizon

    # ── Airmass + DNI extra ───────────────────────────────────────────
    # Airmass = how much atmosphere the sunlight passes through.
    # When sun is directly overhead, airmass = 1.0.
    # When sun is near the horizon it passes through much more atmosphere so airmass is much higher.
    # NaN values (nighttime, sun below horizon) become 0, extreme values (sun right at horizon) are capped at 40.
    am = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'].values)
    am = np.nan_to_num(am, nan=0.0, posinf=40.0)

    # how strong the sun would be with no atmosphere.
    # The base value 1367 W/m² is the solar constant.
    # The cosine term adjusts for Earth's elliptical orbit
    # — Earth is slightly closer to the sun
    # in January so dni_extra is slightly higher then.
    doy = tmy.index.dayofyear.values
    dni_extra = 1367.0 * (1 + 0.033 * np.cos(np.radians(360 * doy / 365)))

    # ── Perez F1/F2 ───────────────────────────────────────────────────
    F1, F2 = _perez_f1f2(tmy, solpos, am, dni_extra)

    # ── Met columns ───────────────────────────────────────────────────
    _tc = 'temp_air' if 'temp_air' in tmy.columns else 'Temperature'
    _wc = 'wind_speed' if 'wind_speed' in tmy.columns else 'Wind Speed'
    temp_air = (
        tmy[_tc].values.astype(np.float32)
        if _tc in tmy.columns
        else np.full(n_hours, 25.0, np.float32)
    )
    wind_speed = (
        tmy[_wc].values.astype(np.float32)
        if _wc in tmy.columns
        else np.full(n_hours, 1.0, np.float32)
    )

    ghi_v = tmy['ghi'].values.astype(np.float32)

    return SolarPreResult(
        zen_rad=zen_rad,
        az_rad=az_rad,
        el_rad=el_rad,
        F1=F1,
        F2=F2,
        ghi=ghi_v,
        dni=tmy['dni'].values.astype(np.float32),
        dhi=tmy['dhi'].values.astype(np.float32),
        temp_air=temp_air,
        wind_speed=wind_speed,
        months=(tmy.index.month.values - 1).astype(np.int32),
        daylight_hours=np.where(ghi_v > 0)[0].astype(np.int32),
    )


# ============================================================
# ASSEMBLE STEP  (trivial — runs on main thread after barrier)
# ============================================================


def assemble_solar_result(
    pre: SolarPreResult,
    horizon: np.ndarray,
) -> SolarResult:
    """
    is the sun physically hidden behind a mountain this hour?
    -----------------------------------------------------------
    Add beam_blk to a SolarPreResult and return the final SolarResult.
    Parameters
    ----------
    pre     : SolarPreResult from compute_solar_positions_pre()
    horizon : (360,) float32 from TerrainResult.horizon
    """
    from .terrain import horizon_beam_blocking

    # For each hour check if the sun's elevation is below
    # the terrain horizon in that direction.
    # Returns 1.0 (blocked) or 0.0 (clear) for each hour.
    # This is the only step that needs terrain data.
    beam_blk = horizon_beam_blocking(pre.el_rad, pre.az_rad, horizon)

    return SolarResult(
        zen_rad=pre.zen_rad,  # how far sun is from straight up (0=overhead)
        az_rad=pre.az_rad,  # compass direction of sun (N/E/S/W)
        el_rad=pre.el_rad,  # sun height above horizon in radians
        F1=pre.F1,  # how bright the sky is around the sun
        F2=pre.F2,  # how bright the horizon band is
        beam_blk=beam_blk,  # 1 = sun behind mountain, 0 = sun visible, (multiplies DNI to zero it out)
        ghi=pre.ghi,
        dni=pre.dni,
        dhi=pre.dhi,
        temp_air=pre.temp_air,
        wind_speed=pre.wind_speed,
        months=pre.months,
        daylight_hours=pre.daylight_hours,  # only the hours where GHI > 0 (sun is up)
    )


# ============================================================
# CONVENIENCE WRAPPER  (pre + assemble in one call)
# ============================================================


def compute_solar_positions(
    tmy: pd.DataFrame,
    latitude: float,
    longitude: float,
    horizon: np.ndarray,
) -> SolarResult:
    """
    Compute solar geometry, Perez F1/F2, and horizon beam blocking.

    Convenience wrapper for pipelines that already have terrain before
    starting solar work (panel shading, masked accumulation).  For the
    full-DSM pipeline use compute_solar_positions_pre() + assemble_solar_result()
    to run concurrently with run_terrain_analysis().

    Parameters
    ----------
    tmy       : tz-aware hourly DataFrame (ghi, dni, dhi, temp_air, wind_speed)
    latitude  : decimal degrees
    longitude : decimal degrees
    horizon   : (360,) float32 from TerrainResult.horizon
    """
    pre = compute_solar_positions_pre(tmy, latitude, longitude)
    return assemble_solar_result(pre, horizon)


# ============================================================
# PEREZ F1/F2
# ============================================================


def _perez_f1f2(
    tmy: pd.DataFrame,
    solpos: pd.DataFrame,
    am: np.ndarray,
    dni_extra: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    how much of the diffuse light is coming from near the sun vs the whole sky?
    --------------------------------------------------------------------------
    Perez (1990) circumsolar (F1) and horizon (F2) brightness coefficients.

    Perez F1 and F2 are two numbers per hour that capture this shape.
    F1 = how much brighter the area around the sun is compared to average sky.
    F2 = how much brighter the horizon strip is.
    These matter because a tilted solar panel "sees" different parts of the sky
    dome differently —
    so you need to know where the light is coming from, not just how much.

    The algorithm buckets each hour into one of 8 sky clarity categories (called epsilon bins)
    and looks up the F1/F2 coefficients from a published table

    """
    # A fixed constant from the Perez (1990) paper.
    # Part of the sky clearness formula.
    kappa = 1.041
    z = np.radians(solpos['apparent_zenith'].values)
    dhi_v = tmy['dhi'].values.astype(np.float64)
    dni_v = tmy['dni'].values.astype(np.float64)

    # --- delta: sky brightness parameter-----
    # How bright the diffuse sky is relative to the extraterrestrial sun.
    # High delta means a very bright cloudy sky. maximum(..., 1e-6) prevents division by zero at night.
    delta = dhi_v * am / np.maximum(dni_extra, 1e-6)

    # --- eps: sky clearness parameter---------
    # Ranges from 1 (completely overcast) to very large numbers (perfectly clear).
    # The kappa * z**3 term adjusts for sun angle —
    # the formula behaves differently near sunset.
    # maximum(dhi_v, 1e-6) prevents division by zero when DHI is zero.
    with np.errstate(invalid='ignore'):
        eps = ((dhi_v + dni_v) / np.maximum(dhi_v, 1e-6) + kappa * z**3) / (
            1.0 + kappa * z**3
        )

    # Sort each hour into one of 8 sky categories (bins) based on clearness.
    # Bin 0 = very overcast, bin 7 = perfectly clear blue sky.
    # np.digitize does the sorting automatically given the boundary values from the Perez paper.
    # NaN hours (nighttime) get bin -1.
    ebin = np.digitize(eps, (0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)) - 1
    ebin[np.isnan(eps)] = -1

    # Load the published lookup table of Perez coefficients.
    # This table has 8 rows (one per sky bin) and 3 columns each.
    # Values come from measurements at many sites worldwide combined.

    F1c, F2c = pvlib.irradiance._get_perez_coefficients('allsitescomposite1990')
    nans = np.array([np.nan, np.nan, np.nan])
    F1c = np.vstack((F1c, nans))
    F2c = np.vstack((F2c, nans))

    # Apply the Perez formula using the looked-up coefficients.
    # Each of F1 and F2 is a linear combination of three terms —
    # a base value, a delta-dependent term, and a zenith-dependent term.
    # F1 is clipped to minimum 0 because negative circumsolar brightness is physically meaningless.
    # F2 can be negative (horizon can be darker than average sky).

    F1 = np.maximum(F1c[ebin, 0] + F1c[ebin, 1] * delta + F1c[ebin, 2] * z, 0.0)
    F2 = F2c[ebin, 0] + F2c[ebin, 1] * delta + F2c[ebin, 2] * z

    return F1.astype(np.float32), F2.astype(np.float32)
