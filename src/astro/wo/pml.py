from astro.common import Scalar, Vec, Mat, typed
from jax.scipy.special import hyp1f1, gamma

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
    x_m = (y+jnp.sqrt(y**2+4))/2
    return (x_m-y)**2/2-jnp.log(x_m)
@typed
def F(omega: Vec, y: Scalar) -> Vec:
    """ Amplification factor for a point mass lens.

    Args:
        omega: dimensionless frequency
        y: dimensionless source position

    Returns:
        Amplification factor F(omega, y)
    """
    exp = jnp.exp(jnp.pi*omega/4 + 1j*omega/2*(jnp.log(omega/2)-2*phi_m(y)))
    Gamma = gamma(1-1j*omega)
    hyp = hyp1f1(1j*omega/2, 1, 1j*omega*y**2/2)
    return exp * Gamma * hyp


if __name__ == "__main__":
    import jax.numpy as jnp
    Msun = 4.925490947e-6  # solar mass in seconds
    Ml= 30*Msun  # lens mass in solar masses
    zl= 0.5  # lens redshift
    Mlz = jnp.array(Ml*(1+zl))  # lens mass in solar masses times (1+zl)
    f = jnp.geomspace(1, 1000, 100)  # frequency in Hz

    omega_val = omega(Mlz, f)
    y     = jnp.array(0.1)  # dimensionless source position
    F_val = F(omega_val, y)

    print(f"Dimensionless frequency (omega): {omega_val}")
    print(f"Amplification factor (F): {F_val}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(f, jnp.abs(F_val))
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplification Factor |F|')
    plt.title('Amplification Factor vs Frequency')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(f, jnp.angle(F_val))
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase of F (radians)')
    plt.title('Phase of Amplification Factor vs Frequency')
    plt.grid()
    plt.tight_layout()
    plt.show()
