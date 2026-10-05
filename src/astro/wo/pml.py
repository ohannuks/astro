"""Point-mass lens (PML): scales, Fermat phase, and amplification factors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from astro.common import CScalar, CScalarOrVec, Scalar, ScalarOrVec, Vec, typed
from astro.wo.hyp1f1 import hyp1f1


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
@typed
def Deltat(y: Scalar) -> Scalar:
    """Time delay between the + and − images in geometric optics limit."""
    s = jnp.sqrt(y**2 + 4)
    return 0.5 * y * s + jnp.log((s + y) / (s - y))
@typed
def Fgeo(omega: ScalarOrVec, y: Scalar) -> CScalarOrVec:
    """Geometric-optics amplification factor.

    F_geo = sqrt|μ₊| - i sqrt|μ₋| exp(i ω ΔT)
    """
    s = jnp.sqrt(y**2 + 4)
    mu_p = 0.5 + (y**2 + 2) / (2 * y * s)
    mu_m = 0.5 - (y**2 + 2) / (2 * y * s)
    return jnp.sqrt(jnp.abs(mu_p)) - 1j * jnp.sqrt(jnp.abs(mu_m)) * jnp.exp(1j * omega * Deltat(y))
@typed
def _F_scalar(omega: Scalar, y: Scalar) -> CScalar:
    """Wave-optics F at one frequency (see F)."""
    a = 1j * omega / 2
    z = 1j * omega * y * y / 2
    pref = jnp.exp(jnp.pi * omega / 4 + 1j * omega / 2 * (jnp.log(omega / 2) - 2 * phi_m(y)))
    return pref * gamma(1.0 - 1j * omega / 2.0) * hyp1f1(a, 1.0 + 0.0j, z)
@typed
def F(omega: ScalarOrVec, y: Scalar) -> CScalarOrVec:
    """Wave-optics amplification factor for a point mass lens.

    F(ω, y) = exp(π ω/4 + i ω/2 [log(ω/2) − 2 φ_m(y)])
              × Γ(1 − i ω/2) × ₁F₁(i ω/2; 1; i ω y²/2)

    For a JIT-friendly path with low/high-frequency limits and a lookup
    table, use ``astro.wo.df_lut.F``.
    """
    omega = jnp.asarray(omega, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if omega.ndim == 0:
        return _F_scalar(omega, y)
    return jax.vmap(lambda om: _F_scalar(om, y))(omega)

if __name__ == "__main__":
    # Demo over a wide band: use the LUT/limit wrapper (keeps this file clean).
    from astro.wo.df_lut import F as F_eval

    Msun = 4.925490947e-6  # solar mass in seconds
    Ml = 30 * Msun
    zl = 0.5
    Mlz = jnp.array(Ml * (1 + zl))
    f = jnp.geomspace(1, 1000, 100)

    omega_val = omega(Mlz, f)
    y = jnp.array(0.1)
    F_val = F_eval(omega_val, y)

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
