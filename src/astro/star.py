from jax import jit, vmap
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
from typing import TypeVar
import jax.numpy as jnp
from constants import h, c, k_B, sigma

@jaxtyped(typechecker=beartype)
def _B_nu( T: Float[Array, ""], nu: Float[Array, "N"] ) -> Float[Array, "N"]:
    log_numerator = jnp.log(2) + jnp.log(h) - 2 * jnp.log(c) + 3 * jnp.log(nu) # Prevent numerical overflow
    log_denominator = jnp.log(jnp.exp((h / k_B) * (nu / T)) - 1.)
    return jnp.exp(log_numerator - log_denominator)

@jaxtyped(typechecker=beartype)
def _B_lambda(
        T: Float[Array, ""],
        lam: Float[Array, "N"]
        ) -> Float[Array, "N"]:
    log_numerator = jnp.log(2) + jnp.log(h) + 2 * jnp.log(c) - 5 * jnp.log(lam) # Prevent numerical overflow
    log_denominator = jnp.log(jnp.exp((h * c) / (lam * k_B * T)) - 1.)
    return jnp.exp(log_numerator - log_denominator)

# Convert to spectral luminosity:
@jaxtyped(typechecker=beartype)
def _L_nu( T: Float[Array, ""], R: Float[Array, ""], nu: Float[Array, "N"]) -> Float[Array, "N"]:
    return 4 * jnp.pi * R**2 * B_nu(nu, T)

def _L_lambda( T: Float[Array, ""], R: Float[Array, ""], lam: Float[Array, "N"],) -> Float[Array, "N"]:
    return 4 * jnp.pi * R**2 * B_lambda(lam, T)
# Stefann Boltzmann law: L = 4 * pi * R^2 * sigma * T^4
def _L_bolometric(T: Float[Array, ""], R: Float[Array, ""]) -> Float[Array, ""]:
    return 4 * jnp.pi * R**2 * sigma * T**4

# Vectorize all of the above functions using vmap
def B_nu(
        T: Float[Array, "M"],
        nu: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Planck function B_nu(nu, T) for a given frequency nu and temperature T.

    Args:
    nu: Frequency in Hz.
    T : Temperature in Kelvin.

    Returns:
        Planck function values for the given frequencies and temperatures. Units: [W/m^2/Hz]
    """
    vmap_B_nu = vmap(_B_nu, in_axes=(0, None), out_axes=0)
    return vmap_B_nu(T, nu)

def B_lambda(
        T: Float[Array, "M"],
        lam: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Planck function B_lambda(lam, T) for a given wavelength lam and temperature T.

    Args:
    lam: Wavelength in meters.
    T : Temperature in Kelvin.

    Returns:
        Planck function values for the given wavelengths and temperatures. Units: [W/m^2/m]
    """
    vmap_B_lambda = vmap(_B_lambda, in_axes=(0, None), out_axes=0)
    return vmap_B_lambda(T, lam)

def L_nu(
        T: Float[Array, "M"],
        R: Float[Array, "M"],
        nu: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Spectral luminosity L_nu(nu, T) for a given frequency nu, temperature T, and radius R.

    Args:
    nu: Frequency in Hz.
    T : Temperature in Kelvin.
    R : Radius in meters.

    Returns:
        Spectral luminosity values for the given frequencies, temperatures, and radii. Units: [W/Hz]
    """
    vmap_L_nu = vmap(_L_nu, in_axes=(0, 0, None), out_axes=0)
    return vmap_L_nu(T, R, nu)

def L_lambda(
        T: Float[Array, "M"],
        R: Float[Array, "M"],
        lam: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Spectral luminosity L_lambda(lam, T) for a given wavelength lam, temperature T, and radius R.

    Args:
    lam: Wavelength in meters.
    T : Temperature in Kelvin.
    R : Radius in meters.

    Returns:
        Spectral luminosity values for the given wavelengths, temperatures, and radii. Units: [W/m]
    """
    vmap_L_lambda = vmap(_L_lambda, in_axes=(0, 0, None), out_axes=0)
    return vmap_L_lambda(T, R, lam)
def L_bolometric(
        T: Float[Array, "M"],
        R: Float[Array, "M"]
        ) -> Float[Array, "M"]:
    """
    Bolometric luminosity L_bolometric(T, R) for a given temperature T and radius R.

    Args:
    T : Temperature in Kelvin.
    R : Radius in meters.

    Returns:
        Bolometric luminosity values for the given temperatures and radii. Units: [W]
    """
    vmap_L_bolometric = vmap(_L_bolometric, in_axes=(0, 0), out_axes=0)
    return vmap_L_bolometric(T, R)


if __name__ == "__main__":
    # Example usage
    T = jnp.array([3e3, 4e3, 5e3])
    R = jnp.array([1e9, 2e9, 3e9])  # Radii in meters
    nu = jnp.array([1e14, 2e14, 3e14])  # Frequencies in Hz
    lam = jnp.array([1e-6, 2e-6, 3e-6])  # Wavelengths in meters
    print("B_nu:", B_nu(T, nu))
    print("B_lambda:", B_lambda(T, lam)) 
    print("L_nu:", L_nu(T, R, nu))
    print("L_lambda:", L_lambda(T, R, lam)) 


