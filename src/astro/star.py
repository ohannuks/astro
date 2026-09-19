from jax import jit, vmap
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
from typing import TypeVar
import jax.numpy as jnp
from constants import h, c, k_B

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

if __name__ == "__main__":
    # Example usage
    T = jnp.array([3e3, 4e3, 5e3])
    R = jnp.array([1e9, 2e9, 3e9])  # Radii in meters
    nu = jnp.array([1e14, 2e14, 3e14])  # Frequencies in Hz
    print("B_nu:", B_nu(T, nu))

