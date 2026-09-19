from jax import jit
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
import jax.numpy as jnp
from constants import h, c, k_B

@jaxtyped(typechecker=beartype)
def B_nu(
        nu: Float[Array, "N"], 
        T: Float[Array, "N"]
        ) -> Float[Array, "N"]:
    """
    Compute the Planck function B_nu(nu, T) for a given frequency nu and temperature T.

    Parameters
    nu: Frequency array in Hz.
    T: Temperature array in Kelvin.

    Returns: 
        Planck function values for the given frequencies and temperatures. Units: [W/m^2/Hz]
    """
    # Prevent numerical overflow
    log_numerator = jnp.log(2) + jnp.log(h) - 2 * jnp.log(c) + 3 * jnp.log(nu) 
    log_denominator = jnp.log(jnp.exp((h / k_B) * (nu / T)) - 1.)
    return jnp.exp(log_numerator - log_denominator)

@jaxtyped(typechecker=beartype)
def B_lambda(
        lam: Float[Array, "N"],
        T: Float[Array, "N"]
        ) -> Float[Array, "N"]:
    """
    Compute the Planck function B_lambda(lambda, T) for a given wavelength lambda and temperature T.

    Args:
    lam: Wavelength array in meters.
    T : Temperature array in Kelvin.

    Returns:
        Planck function values for the given wavelengths and temperatures. Units: [W/m^2/m]
    """
    # Prevent numerical overflow
    log_numerator = jnp.log(2) + jnp.log(h) + 2 * jnp.log(c) - 5 * jnp.log(lam)
    log_denominator = jnp.log(jnp.exp((h * c) / (lam * k_B * T)) - 1.)
    return jnp.exp(log_numerator - log_denominator)

# Convert to spectral luminosity:
@jaxtyped(typechecker=beartype)
def L_nu(
        nu: Float[Array, "N"],
        T: Float[Array, "N"],
        R: Float[Array, "N"]
        ) -> Float[Array, "N"]:
    """
    Compute the spectral luminosity L_nu(nu, T, R) for a given frequency nu, temperature T, and radius R.

    Args:
    nu: Frequency array in Hz.
    T : Temperature array in Kelvin.
    R : Radius array in meters.

    Returns:
        Spectral luminosity values for the given frequencies, temperatures, and radii. Units: [W/m^2/Hz]
    """
    return 4 * jnp.pi * R**2 * B_nu(nu, T)

def L_lambda(
        lam: Float[Array, "N"],
        T: Float[Array, "N"],
        R: Float[Array, "N"]
        ) -> Float[Array, "N"]:
    """
    Compute the spectral luminosity L_lambda(lambda, T, R) for a given wavelength lambda, temperature T, and radius R.

    Args:
    lam: Wavelength array in meters.
    T : Temperature array in Kelvin.
    R : Radius array in meters.

    Returns:
        Spectral luminosity values for the given wavelengths, temperatures, and radii. Units: [W/m^2/m]
    """
    return 4 * jnp.pi * R**2 * B_lambda(lam, T)

if __name__ == "__main__":
    # Example usage
    nu = jnp.array([1e14, 2e14, 3e14])  # Frequencies in Hz
    T = jnp.array([3000., 4000., 5000.])   # Temperatures in Kelvin

    B_nu_jit = jit(B_nu)
    B_nu_values = B_nu_jit(nu, T)
    print("B_nu values:", B_nu_values)

    lam = jnp.array([1e-6, 2e-6, 3e-6])  # Wavelengths in meters
    B_lambda_jit = jit(B_lambda)
    B_lambda_values = B_lambda_jit(lam, T)
    print("B_lambda values:", B_lambda_values)

