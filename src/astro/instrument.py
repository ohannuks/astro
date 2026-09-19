from jaxtyping import Float64, Int32, jaxtyped
from beartype import beartype
import jax.numpy as jnp

@jaxtyped(checker=beartype)
def spectral_flux(

