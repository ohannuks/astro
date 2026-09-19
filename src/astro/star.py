from jaxtyping import Array, Float64, Int32, jaxtyped
from beartype import beartype
import jax.numpy as jnp

def B_nu(
        nu: Float64[Array, "N"], 
        T: Float64[Array, "N"]
        ) -> Float64[Array, "N"]:
    """
    Compute the Planck function B_nu(nu, T) for a given frequency nu and temperature T.

    Parameters
    nu : Float64[Array, "N"]
        Frequency array in Hz.
    T : Float64[Array, "N"]
        Temperature array in Kelvin.

    Returns
    Float64[Array, "N"]:
        Planck function values for the given frequencies and temperatures.
    """

