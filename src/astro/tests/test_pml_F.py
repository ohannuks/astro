"""Accuracy tests for point-mass amplification F(ω, y) across regimes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from astro.wo import df_lut, pml
from astro.wo.df_lut import _F_geo_np, _F_wave_np, _delta_T_np


@pytest.fixture(scope="module", autouse=True)
def _ensure_lut():
    """Rebuild LUT if missing or on an old smaller domain."""
    path = df_lut.DATA_PATH
    need = True
    if path.is_file():
        data = np.load(path)
        need = float(data["y_min"]) > df_lut.Y_MIN * 1.01 or float(data["omega_min"]) > df_lut.OMEGA_MIN * 1.01
    if need:
        df_lut.build_df_lut(path)
    df_lut._default_lut = None
    df_lut.get_df_lut()


def _ref(om: float, y: float) -> complex:
    """Reference F: series when safe, else geometric optics."""
    z_abs = 0.5 * om * y * y
    if om <= df_lut.WAVE_OMEGA_MAX:
        return 1.0 + 0.0j
    if om * _delta_T_np(y) >= df_lut.GEO_PHASE_THRESH:
        return _F_geo_np(om, y)
    if z_abs >= df_lut.SERIES_Z_MAX:
        return _F_geo_np(om, y)
    return _F_wave_np(om, y)


def _err(om: float, y: float) -> float:
    got = complex(df_lut.F(jnp.asarray(om), jnp.asarray(y)))
    return abs(got - _ref(om, y))


# ---------------------------------------------------------------------------
# Analytic pml.F (no LUT / regime switching)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("y", [1e-6, 1e-4, 0.1, 1.0])
@pytest.mark.parametrize("om", [1e-3, 0.1, 1.0, 3.0])
def test_pml_F_matches_numpy_series(om, y):
    if 0.5 * om * y * y >= df_lut.SERIES_Z_MAX:
        pytest.skip("|z| too large for series")
    got = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    assert abs(got - _F_wave_np(om, y)) < 1e-8


def test_pml_Fgeo_matches_numpy():
    om, y = 5.0, 0.7
    got = complex(pml.Fgeo(jnp.asarray(om), jnp.asarray(y)))
    assert abs(got - _F_geo_np(om, y)) < 1e-12


# ---------------------------------------------------------------------------
# Regime selection (df_lut.F)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("y", [1e-6, 1e-4, 0.1, 1.0, 3.0])
def test_wave_limit_returns_one(y):
    om = jnp.asarray(1e-5)
    assert bool(df_lut.in_wave_limit(om))
    F = df_lut.F(om, jnp.asarray(y))
    assert abs(complex(F) - 1.0) < 1e-15


@pytest.mark.parametrize(
    "om,y",
    [
        (20.0, 0.5),
        (10.0, 1.0),
        (5.0, 2.0),
        (100.0, 0.1),
        (200.0, 0.05),
    ],
)
def test_geo_limit_matches_Fgeo(om, y):
    assert om * _delta_T_np(y) >= df_lut.GEO_PHASE_THRESH
    om_j, y_j = jnp.asarray(om), jnp.asarray(y)
    assert bool(df_lut.in_geo_limit(om_j, y_j))
    assert abs(complex(df_lut.F(om_j, y_j) - pml.Fgeo(om_j, y_j))) < 1e-14


def test_geo_not_used_for_tiny_y_moderate_omega():
    """Caustic: Fgeo ≫ 1 while wave F is O(1); must not return Fgeo."""
    om, y = 1.0, 1e-6
    assert om * _delta_T_np(y) < df_lut.GEO_PHASE_THRESH
    F = complex(df_lut.F(jnp.asarray(om), jnp.asarray(y)))
    Fg = complex(pml.Fgeo(jnp.asarray(om), jnp.asarray(y)))
    assert abs(F) < 10.0
    assert abs(Fg) > 100.0
    assert abs(F - Fg) > 10.0


# ---------------------------------------------------------------------------
# Accuracy vs reference across parameter space
# ---------------------------------------------------------------------------

YS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.0]
OMEGAS = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("om", OMEGAS)
def test_F_accuracy_grid(om, y):
    if om <= df_lut.WAVE_OMEGA_MAX or om * _delta_T_np(y) >= df_lut.GEO_PHASE_THRESH:
        tol = 1e-12
    elif 0.5 * om * y * y >= df_lut.SERIES_Z_MAX:
        tol = 1e-12
    else:
        tol = 5e-3
    assert _err(om, y) <= tol


@pytest.mark.parametrize("y", [1e-6, 1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("om", [1e-3, 0.01, 0.1, 1.0, 5.0, 10.0])
def test_small_y_matches_series(om, y):
    """Very small y: series is trustworthy (|z|=ω y²/2 ≪ 1)."""
    assert 0.5 * om * y * y < 1.0
    got = complex(df_lut.F(jnp.asarray(om), jnp.asarray(y)))
    exact = _F_wave_np(om, y)
    assert abs(got - exact) < 1e-6


def test_lut_interior_dense_sample():
    rng = np.random.default_rng(0)
    oms = rng.uniform(df_lut.OMEGA_MIN, df_lut.OMEGA_MAX, size=40)
    ys = 10 ** rng.uniform(np.log10(df_lut.Y_MIN), np.log10(df_lut.Y_MAX), size=40)
    max_err = 0.0
    for om, y in zip(oms, ys, strict=True):
        if om * _delta_T_np(float(y)) >= df_lut.GEO_PHASE_THRESH:
            continue
        max_err = max(
            max_err,
            abs(complex(df_lut.F(jnp.asarray(om), jnp.asarray(y))) - _F_wave_np(float(om), float(y))),
        )
    assert max_err < 5e-3


def test_geo_limit_close_to_series_when_series_safe():
    om, y = 15.0, 1.0
    assert om * _delta_T_np(y) >= df_lut.GEO_PHASE_THRESH
    assert 0.5 * om * y * y < df_lut.SERIES_Z_MAX
    assert abs(_F_geo_np(om, y) - _F_wave_np(om, y)) / abs(_F_wave_np(om, y)) < 0.02


def test_wave_limit_close_to_exact():
    om = df_lut.WAVE_OMEGA_MAX
    for y in [1e-6, 0.1, 1.0, 3.0]:
        assert abs(1.0 - _F_wave_np(om, y)) < 5e-3


def test_vector_omega_matches_scalar():
    y = jnp.asarray(0.3)
    oms = jnp.array([1e-5, 0.01, 1.0, 4.0, 50.0])
    F_vec = df_lut.F(oms, y)
    for i, om in enumerate(oms):
        F_s = df_lut.F(om, y)
        assert abs(complex(F_vec[i] - F_s)) < 1e-12


def test_jit_F():
    Fj = jax.jit(df_lut.F)
    oms = jnp.array([1e-5, 0.5, 2.0, 30.0])
    y = jnp.asarray(0.4)
    out = Fj(oms, y)
    assert out.shape == oms.shape
    for i, om in enumerate(oms):
        assert abs(complex(out[i]) - complex(df_lut.F(om, y))) < 1e-10


def test_F_approaches_one_as_omega_to_zero():
    y = jnp.asarray(0.7)
    for om in [1e-6, 1e-5, 1e-4]:
        assert abs(complex(df_lut.F(jnp.asarray(om), y)) - 1.0) < 1e-12


def test_F_approaches_Fgeo_at_high_omega():
    y = jnp.asarray(1.2)
    om = jnp.asarray(30.0)
    assert abs(complex(df_lut.F(om, y) - pml.Fgeo(om, y))) < 1e-14
