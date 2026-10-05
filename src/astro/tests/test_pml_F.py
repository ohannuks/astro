"""Accuracy tests for point-mass amplification F(ω, y)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma as sp_gamma
from specialfunctions import hyp1f1

jax.config.update("jax_enable_x64", True)

from astro.wo import pml


def _delta_T_np(y: float) -> float:
    s = np.sqrt(y * y + 4.0)
    return 0.5 * y * s + np.log((s + y) / (s - y))


def _phi_m_np(y: float) -> float:
    x_m = 0.5 * (y + np.sqrt(y * y + 4.0))
    return 0.5 * (x_m - y) ** 2 - np.log(x_m)


def _F_geo_np(omega: float, y: float) -> complex:
    s = np.sqrt(y * y + 4.0)
    mu_p = 0.5 + (y * y + 2.0) / (2.0 * y * s)
    mu_m = 0.5 - (y * y + 2.0) / (2.0 * y * s)
    dT = _delta_T_np(y)
    return np.sqrt(abs(mu_p)) - 1j * np.sqrt(abs(mu_m)) * np.exp(1j * omega * dT)


def _F_wave_ref(omega: float, y: float) -> complex:
    """Independent wave-optics F via specialfunctions.hyp1f1 + SciPy Γ."""
    a = 1j * omega / 2.0
    z = 1j * omega * y * y / 2.0
    pref = np.exp(np.pi * omega / 4.0 + 1j * omega / 2.0 * (np.log(omega / 2.0) - 2.0 * _phi_m_np(y)))
    h = complex(
        hyp1f1(
            jnp.asarray(a),
            jnp.asarray(1.0 + 0.0j),
            jnp.asarray(z),
        )
    )
    return complex(pref * sp_gamma(1.0 - 1j * omega / 2.0) * h)


# ---------------------------------------------------------------------------
# Analytic pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [1e-6, 1e-4, 0.1, 1.0])
@pytest.mark.parametrize("om", [1e-3, 0.1, 1.0, 3.0])
def test_pml_F_matches_ref(om, y):
    got = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    assert abs(got - _F_wave_ref(om, y)) < 1e-10


def test_pml_Fgeo_matches_numpy():
    om, y = 5.0, 0.7
    got = complex(pml.Fgeo(jnp.asarray(om), jnp.asarray(y)))
    assert abs(got - _F_geo_np(om, y)) < 1e-12


def test_pml_Deltat_matches_numpy():
    for y in [0.1, 0.5, 1.0, 2.0]:
        assert abs(float(pml.Deltat(jnp.asarray(y))) - _delta_T_np(y)) < 1e-14


# ---------------------------------------------------------------------------
# Limits / behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [1e-6, 1e-4, 0.1, 1.0, 3.0])
def test_F_approaches_one_as_omega_to_zero(y):
    for om in [1e-6, 1e-5, 1e-4]:
        # leading correction is O(ω)
        assert abs(complex(pml.F(jnp.asarray(om), jnp.asarray(y))) - 1.0) < 10.0 * om


def test_geo_not_used_for_tiny_y_moderate_omega():
    """Caustic: Fgeo ≫ 1 while wave F is O(1)."""
    om, y = 1.0, 1e-6
    F = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    Fg = complex(pml.Fgeo(jnp.asarray(om), jnp.asarray(y)))
    assert abs(F) < 10.0
    assert abs(Fg) > 100.0
    assert abs(F - Fg) > 10.0


def test_F_approaches_Fgeo_at_high_omega():
    """Large ω ΔT: wave F tracks geometric optics (few-percent)."""
    y = 0.5
    om = 100.0
    got = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    ref = _F_geo_np(om, y)
    assert abs(got - ref) / max(1.0, abs(ref)) < 0.05


# ---------------------------------------------------------------------------
# Grid + vector / JIT
# ---------------------------------------------------------------------------

YS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.0]
OMEGAS = [1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]


@pytest.mark.parametrize("y", YS)
@pytest.mark.parametrize("om", OMEGAS)
def test_F_matches_ref_grid(om, y):
    got = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    ref = _F_wave_ref(om, y)
    assert abs(got - ref) <= 1e-10 * max(1.0, abs(ref))


@pytest.mark.parametrize("y", [1e-6, 1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("om", [1e-3, 0.01, 0.1, 1.0, 5.0, 10.0])
def test_small_y_finite(om, y):
    got = complex(pml.F(jnp.asarray(om), jnp.asarray(y)))
    assert np.isfinite(got.real) and np.isfinite(got.imag)
    assert abs(got) < 20.0


def test_vector_omega_matches_scalar():
    y = jnp.asarray(0.3)
    oms = jnp.array([1e-5, 0.01, 1.0, 4.0, 50.0])
    F_vec = pml.F(oms, y)
    for i, om in enumerate(oms):
        F_s = pml.F(om, y)
        assert abs(complex(F_vec[i] - F_s)) < 1e-12


def test_jit_F():
    Fj = jax.jit(pml.F)
    oms = jnp.array([1e-5, 0.5, 2.0, 30.0])
    y = jnp.asarray(0.4)
    out = Fj(oms, y)
    assert out.shape == oms.shape
    for i, om in enumerate(oms):
        assert abs(complex(out[i]) - complex(pml.F(om, y))) < 1e-10
