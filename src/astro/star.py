from jaxtyping import Array, Float64, Int32, jaxtyped
from beartype import beartype
import jax.numpy as jnp
from constants import h, c, k_B

@jaxtyped(typechecker=beartype)
def B_nu(
        nu: Float64[Array, "N"], 
        T: Float64[Array, "N"]
        ) -> Float64[Array, "N"]:
    """
    Compute the Planck function B_nu(nu, T) for a given frequency nu and temperature T.

    Parameters
    nu: Frequency array in Hz.
    T: Temperature array in Kelvin.

    Returns: 
        Planck function values for the given frequencies and temperatures.
    """
    return (2 * h * nu**3 / c**2) / (jnp.exp(h * nu / (k_B * T)) - 1)

@jaxtyped(typechecker=beartype)
def B_lambda(
        lam: Float64[Array, "N"],
        T: Float64[Array, "N"]
        ) -> Float64[Array, "N"]:
    """
    Compute the Planck function B_lambda(lambda, T) for a given wavelength lambda and temperature T.

    Args:
    lam: Wavelength array in meters.
    T : Temperature array in Kelvin.

    Returns:
        Planck function values for the given wavelengths and temperatures.
    """
    return (2 * h * c**2 / lam**5) / (jnp.exp(h * c / (lam * k_B * T)) - 1)



