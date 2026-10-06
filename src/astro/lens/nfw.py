"""Point-mass lens (PML): scales, Fermat phase, and amplification factors."""
from astro.common import CScalar, CVec, Scalar, Vec, typed
import cosmology
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from astro.common import CScalar, Scalar, Vec, typed
from astro.lens import general

jax.config.update("jax_enable_x64", True)  # Use double precision for better accuracy

def rho_crit(z: Scalar) -> Scalar:
    """Critical density of the universe at redshift z in kg/m^3.

    Args:
        z: Redshift.

    Returns:
        Critical density rho_crit(z).
    """
    return cosmology.rho_crit(z)

def rs(M200: Scalar, c: Scalar, z: Scalar) -> Scalar:
    """Scale radius of NFW profile.
    
    Args:
        M200: Mass within radius where density is 200 times the critical density.
        c: Concentration parameter.
        z: Redshift.

    Returns:
        Scale radius rs.
    """
    return (3 * M200 / (4 * jnp.pi * 200 * rho_crit(z))) ** (1 / 3) / c

def density(r: Scalar, rs: Scalar, rho0: Scalar) -> Scalar:
    """NFW density profile.

    Args:
        r: Radial distance from the center.
        rs: Scale radius.
        rho0: Characteristic density.

    Returns:
        Density at radius r.
    """
    return rho0 / ((r / rs) * (1 + r / rs) ** 2)

def Sigma(R: Scalar, rs: Scalar, rho0: Scalar) -> Scalar:
    """Projected surface density of NFW profile.

    Args:
        R: Projected radius.
        rs: Scale radius.
        rho0: Characteristic density.

    Returns:
        Projected surface density Sigma(R).
    """
    x = R / rs
    return 2 * rho0 * rs * jnp.where(
        x < 1,
        jnp.arccosh(1 / x) / jnp.sqrt(1 - x ** 2), # x < 1
        jnp.arccos(1 / x) / jnp.sqrt(x ** 2 - 1) # x > 1
    )

def Menc(R: Scalar, rs: Scalar, rho0: Scalar) -> Scalar:
    """Enclosed mass within projected radius R for NFW profile.

    Args:
        R: Projected radius.
        rs: Scale radius.
        rho0: Characteristic density.

    Returns:
        Enclosed mass Menc(<R).
    """
    x = R / rs
    return 4 * jnp.pi * rho0 * rs**3 * jnp.where(
        x < 1,
        jnp.log(x / 2) + jnp.arccosh(1 / x) / jnp.sqrt(1 - x**2),
        jnp.log(x / 2) + jnp.arccos(1 / x) / jnp.sqrt(x**2 - 1),
    )

if __name__ == "__main__":
    Msun = 4.925490947e-6  # solar mass in seconds
    Ml = 30 * Msun
    zl = 0.5
    Mlz = jnp.array(Ml * (1 + zl))
    f = jnp.geomspace(1, 1000000, 1000)

    omega_val = general.omega(Mlz, f)
    y = jnp.array(0.1)
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
