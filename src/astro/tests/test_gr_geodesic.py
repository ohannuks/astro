"""Consistency tests: analytical Fujita–Hikida geodesics vs KerrGeoPy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

kerrgeopy = pytest.importorskip("kerrgeopy")
from kerrgeopy import StableOrbit

from astro.gr import BoyerLindquistGeodesic


ORBITS = [
    # a, p, e, theta_inc_deg
    (0.9, 6.0, 0.7, 20.0),
    (0.5, 10.0, 0.2, 30.0),
    (0.9, 100.0, 0.05, 45.0),
]


def _x(theta_inc_deg: float) -> float:
    return float(np.cos(np.deg2rad(theta_inc_deg)))


def _as(p, e, x, M, a):
    return map(jnp.asarray, [p, e, x, M, a])


@pytest.fixture(scope="module")
def geo() -> BoyerLindquistGeodesic:
    return BoyerLindquistGeodesic()


@pytest.mark.parametrize("a,p,e,inc", ORBITS)
def test_constants_match_kerrgeopy(geo, a, p, e, inc):
    x = _x(inc)
    orb_kg = StableOrbit(a, p, e, x)
    E, L, Q = geo.constants_of_motion(*_as(p, e, x, 1.0, a))
    assert float(E) == pytest.approx(orb_kg.E, rel=1e-12, abs=1e-12)
    assert float(L) == pytest.approx(orb_kg.L, rel=1e-12, abs=1e-12)
    assert float(Q) == pytest.approx(orb_kg.Q, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("a,p,e,inc", ORBITS)
def test_mino_frequencies_match_kerrgeopy(geo, a, p, e, inc):
    x = _x(inc)
    orb_kg = StableOrbit(a, p, e, x)
    ups_r, ups_th, ups_ph, Gamma = geo.mino_frequencies(*_as(p, e, x, 1.0, a))
    assert float(ups_r) == pytest.approx(orb_kg.upsilon_r, rel=1e-11, abs=1e-14)
    assert float(ups_th) == pytest.approx(orb_kg.upsilon_theta, rel=1e-11, abs=1e-14)
    assert float(ups_ph) == pytest.approx(orb_kg.upsilon_phi, rel=1e-11, abs=1e-14)
    assert float(Gamma) == pytest.approx(orb_kg.gamma, rel=1e-11, abs=1e-12)


@pytest.mark.parametrize("a,p,e,inc", ORBITS)
def test_fundamental_frequencies_match_kerrgeopy(geo, a, p, e, inc):
    x = _x(inc)
    orb_kg = StableOrbit(a, p, e, x)
    Om_r, Om_th, Om_ph = geo.fundamental_frequencies(*_as(p, e, x, 1.0, a))
    assert float(Om_r) == pytest.approx(orb_kg.omega_r, rel=1e-11, abs=1e-14)
    assert float(Om_th) == pytest.approx(orb_kg.omega_theta, rel=1e-11, abs=1e-14)
    assert float(Om_ph) == pytest.approx(orb_kg.omega_phi, rel=1e-11, abs=1e-14)


@pytest.mark.parametrize("a,p,e,inc", ORBITS)
def test_trajectory_matches_kerrgeopy(geo, a, p, e, inc):
    x = _x(inc)
    orb_kg = StableOrbit(a, p, e, x)
    t_fn, r_fn, th_fn, ph_fn = orb_kg.trajectory()
    p_, e_, x_, M_, a_ = tuple(_as(p, e, x, 1.0, a))
    lams = jnp.linspace(0.0, 20.0, 16)
    t = jax.vmap(lambda lam: geo.t(lam, p_, e_, x_, M_, a_))(lams)
    r = jax.vmap(lambda lam: geo.r(lam, p_, e_, x_, M_, a_))(lams)
    th = jax.vmap(lambda lam: geo.theta(lam, p_, e_, x_, M_, a_))(lams)
    ph = jax.vmap(lambda lam: geo.phi(lam, p_, e_, x_, M_, a_))(lams)
    lams_np = np.asarray(lams)
    kg = np.column_stack([t_fn(lams_np), r_fn(lams_np), th_fn(lams_np), ph_fn(lams_np)])
    ours = np.column_stack([np.asarray(t), np.asarray(r), np.asarray(th), np.asarray(ph)])
    np.testing.assert_allclose(ours, kg, rtol=1e-10, atol=1e-9)


def test_paper_table3_frequencies(geo):
    """Fujita & Hikida Table 3: a=0.9M, p=6M, e=0.7, theta_inc=20 deg."""
    Om_r, Om_th, Om_ph = geo.fundamental_frequencies(
        *_as(6.0, 0.7, _x(20.0), 1.0, 0.9)
    )
    assert float(Om_r) == pytest.approx(1.8928532285101992e-2, rel=1e-12)
    assert float(Om_th) == pytest.approx(2.7299110395017517e-2, rel=1e-12)
    assert float(Om_ph) == pytest.approx(3.0550463796964692e-2, rel=1e-12)


def test_trajectory_initial_conditions(geo):
    p, e, a, M = 6.0, 0.7, 0.9, 1.0
    x = _x(20.0)
    lam0 = jnp.asarray(0.0)
    p_, e_, x_, M_, a_ = _as(p, e, x, M, a)
    assert float(geo.t(lam0, p_, e_, x_, M_, a_)) == pytest.approx(0.0, abs=1e-12)
    assert float(geo.phi(lam0, p_, e_, x_, M_, a_)) == pytest.approx(0.0, abs=1e-12)
    assert float(geo.r(lam0, p_, e_, x_, M_, a_)) == pytest.approx(p * M / (1 + e), rel=1e-12)
    assert float(geo.theta(lam0, p_, e_, x_, M_, a_)) == pytest.approx(float(np.arccos(np.sqrt(1 - x**2))), rel=1e-12)


@pytest.mark.parametrize("a,p,e,inc", ORBITS)
def test_vmapped_trajectory_matches_kerrgeopy(geo, a, p, e, inc):
    x = _x(inc)
    orb_kg = StableOrbit(a, p, e, x)
    t_fn, r_fn, th_fn, ph_fn = orb_kg.trajectory()
    p_, e_, x_, M_, a_ = tuple(_as(p, e, x, 1.0, a))
    lams = jnp.linspace(0.0, 20.0, 16)
    t = jax.vmap(lambda lam: geo.t(lam, p_, e_, x_, M_, a_))(lams)
    r = jax.vmap(lambda lam: geo.r(lam, p_, e_, x_, M_, a_))(lams)
    th = jax.vmap(lambda lam: geo.theta(lam, p_, e_, x_, M_, a_))(lams)
    ph = jax.vmap(lambda lam: geo.phi(lam, p_, e_, x_, M_, a_))(lams)
    lams_np = np.asarray(lams)
    kg = np.column_stack([t_fn(lams_np), r_fn(lams_np), th_fn(lams_np), ph_fn(lams_np)])
    ours = np.column_stack([np.asarray(t), np.asarray(r), np.asarray(th), np.asarray(ph)])
    np.testing.assert_allclose(ours, kg, rtol=1e-10, atol=1e-9)
