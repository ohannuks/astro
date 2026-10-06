"""Point-mass lens (PML): scales, Fermat phase, and amplification factors."""
from astro.common import CScalar, CVec, Scalar, Vec, typed
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from astro.common import CScalar, Scalar, Vec, typed
from astro.lens import general

jax.config.update("jax_enable_x64", True)  # Use double precision for better accuracy


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
