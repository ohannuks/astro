"""Point-mass lens (PML): scales, Fermat phase, and amplification factors."""
from astro.common import CScalar, CVec, Scalar, Vec, typed
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from astro.common import CScalar, Scalar, Vec, typed
from astro.lens import general

jax.config.update("jax_enable_x64", True)  # Use double precision for better accuracy

def rho_crit(z: Scalar) -> Scalar:
    """Critical density of the universe at redshift z."""
    H0 = 70.0  # Hubble constant in km/s/Mpc
    H0 = H0 * 1000 / (3.085677581e22)  # Convert to s^-1
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    rho_crit_0 = 3 * H0**2 / (8 * jnp.pi * G)  # Critical density at z=0 in kg/m^3
    return rho_crit_0 * (1 + z) ** 3  # Scale with redshift

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
    """NFW density profile."""
    return rho0 / ((r / rs) * (1 + r / rs) ** 2)



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
