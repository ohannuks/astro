"""Point-mass lens amplification factor with asymptotic limits + dF LUT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from astro.common import Scalar, Vec, typed
from astro.wo import df_lut
from astro.wo.df_lut import F_lut

# ω → 0: F → 1 (wavelength ≫ Einstein scale).
WAVE_OMEGA_MAX = 1e-4
# Geometric optics when phase between images is large: ω ΔT ≳ this.
GEO_PHASE_THRESH = 8.0
# Series ₁F₁ reliable only for moderate |z| = ω y² / 2; beyond → Fgeo.
SERIES_Z_MAX = 25.0
# Prefer series over LUT when |z| small (few terms, no interp error).
SERIES_PREF_Z = 2.0


@typed
def t_L(Mlz: Scalar) -> Scalar:
    """ Time scale of the lens.

    Args:
        Mlz: lens mass enclosed inside of einstein radius, redshifted, in units of seconds

    Returns:
        Time scale of the lens t_L = 4 * Mlz
    """
    return 4 * Mlz


@typed
def theta_L(Mlz: Scalar, Dl: Scalar, Ds: Scalar, Dls: Scalar) -> Scalar:
    """ Angular scale of the lens.

    Args:
        Mlz: lens mass enclosed inside of einstein radius, redshifted, in units of seconds
        Dl: angular diameter distance to the lens in seconds
        Ds: angular diameter distance to the source in seconds
        Dls: angular diameter distance from the lens to the source in seconds

    Returns:
        Angular scale of the lens theta_L = sqrt(4 * Mlz * Dls / (Dl * Ds))
    """
    return jnp.sqrt(4 * Mlz * Dls / (Dl * Ds))


@typed
def omega(Mlz: Scalar, f: Vec) -> Vec:
    """ Dimensionless frequency.

    Args:
        Mlz: lens mass in units of solar mass
        f: frequency in Hz

    Returns:
        Dimensionless frequency omega = 8 * pi * Mlz * f
    """
    return 8 * jnp.pi * Mlz * f


@typed
def phi_m(y: Scalar) -> Scalar:
    """ Phase at the minimum of the Fermat potential.

    Args:
        y: dimensionless source position

    Returns:
        Phase at the minimum of the Fermat potential
    """
    x_m = (y + jnp.sqrt(y**2 + 4)) / 2
    return (x_m - y) ** 2 / 2 - jnp.log(x_m)


def delta_T(y):
    """Time-delay between + and − images (point mass)."""
    s = jnp.sqrt(y**2 + 4)
    return 0.5 * y * s + jnp.log((s + y) / (s - y))


def in_wave_limit(omega):
    """Deep wave-optics: ω → 0 ⇒ F → 1."""
    return omega <= WAVE_OMEGA_MAX


def in_geo_limit(omega, y):
    """Geometric optics: large image phase ω ΔT."""
    return omega * delta_T(y) >= GEO_PHASE_THRESH


def F_wave_limit(omega, y):
    """Leading low-frequency amplification: F → 1 (independent of y)."""
    return jnp.ones_like(jnp.asarray(omega), dtype=jnp.float64) + 0.0j


def Fgeo(omega, y):
    """ Geometric-optics amplification factor for a point mass lens.

    Args:
        omega: dimensionless frequency
        y: dimensionless source position

    Returns:
        F_geo(omega, y) = sqrt|μ₊| - i sqrt|μ₋| exp(i ω ΔT)
    """
    s = jnp.sqrt(y**2 + 4)
    mu_p = 0.5 + (y**2 + 2) / (2 * y * s)
    mu_m = 0.5 - (y**2 + 2) / (2 * y * s)
    dT = delta_T(y)
    return jnp.sqrt(jnp.abs(mu_p)) - 1j * jnp.sqrt(jnp.abs(mu_m)) * jnp.exp(1j * omega * dT)


def _hyp1f1_series_scalar(a, b, z, tol=1e-13, maxn=5000):
    """Scalar ₁F₁(a; b; z) power series."""

    def body(carry):
        k, term, s = carry
        term = term * (a + k - 1) / (b + k - 1) * z / k
        return k + 1, term, s + term

    def cond(carry):
        k, term, s = carry
        return (k <= maxn) & (jnp.abs(term) >= tol * jnp.maximum(1.0, jnp.abs(s)))

    init_term = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
    _, _, s = jax.lax.while_loop(cond, body, (1, init_term, init_term))
    return s


def _F_series_scalar(om, y):
    z_abs = 0.5 * om * y * y
    om_s = jnp.where(z_abs < SERIES_Z_MAX, jnp.maximum(om, 1e-300), 1.0)
    a = 1j * om_s / 2
    z = 1j * om_s * y * y / 2
    log_pre = jnp.pi * om_s / 4 + 1j * om_s / 2 * (jnp.log(om_s / 2) - 2 * phi_m(y))
    F_s = jnp.exp(log_pre) * gamma(1.0 - 1j * om_s / 2.0) * _hyp1f1_series_scalar(a, 1.0 + 0.0j, z)
    return jnp.where(z_abs < SERIES_Z_MAX, F_s, Fgeo(om, y))


def F_series(omega, y):
    """Exact wave-optics F via Γ(1 − iω/2) ₁F₁ (JAX series).

    For |z| = ω y²/2 too large for the series, falls back to Fgeo.
    """
    omega = jnp.asarray(omega, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if omega.ndim == 0:
        return _F_series_scalar(omega, y)
    return jax.vmap(lambda om: _F_series_scalar(om, y))(omega)


def _in_lut(omega, y):
    lut = df_lut.get_df_lut()
    return (
        (omega >= lut.omega_min)
        & (omega <= lut.omega_max)
        & (y >= lut.y_min)
        & (y <= lut.y_max)
    )


def F(omega, y):
    """Amplification factor for a point mass lens.

    Regime selection (jit-friendly ``where``):
      * deep wave (ω ≤ WAVE_OMEGA_MAX): F → 1
      * geometric (ω ΔT ≥ GEO_PHASE_THRESH): F → Fgeo
      * else if inside F LUT box: F = F_lut
      * else: JAX series (or Fgeo if |z| too large for series)

    Note: LUT stores F (not dF). Use ``df_lut.dF`` for F−Fgeo.
    """
    omega = jnp.asarray(omega, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    wave = in_wave_limit(omega)
    geo = in_geo_limit(omega, y)
    z_abs = 0.5 * omega * y * y
    prefer_series = z_abs < SERIES_PREF_Z
    lut = _in_lut(omega, y) & (~wave) & (~geo) & (~prefer_series)

    F_w = F_wave_limit(omega, y)
    F_g = Fgeo(omega, y)
    F_l = F_lut(omega, y)
    F_s = F_series(omega, y)

    mid = jnp.where(lut, F_l, F_s)
    return jnp.where(wave, F_w, jnp.where(geo, F_g, mid))


if __name__ == "__main__":
    Msun = 4.925490947e-6  # solar mass in seconds
    Ml = 30 * Msun  # lens mass in solar masses
    zl = 0.5  # lens redshift
    Mlz = jnp.array(Ml * (1 + zl))  # lens mass in solar masses times (1+zl)
    f = jnp.geomspace(1, 1000, 100)  # frequency in Hz

    omega_val = omega(Mlz, f)
    y = jnp.array(0.1)  # dimensionless source position
    F_val = F(omega_val, y)

    print(f"Dimensionless frequency (omega): {omega_val}")
    print(f"Amplification factor (F): {F_val}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(f, jnp.abs(F_val))
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplification Factor |F|")
    plt.title("Amplification Factor vs Frequency")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(f, jnp.angle(F_val))
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase of F (radians)")
    plt.title("Phase of Amplification Factor vs Frequency")
    plt.grid()
    plt.tight_layout()
    plt.show()
