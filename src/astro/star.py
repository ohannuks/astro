import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
from typing import TypeVar
import jax.numpy as jnp
from astro.constants import *

@jaxtyped(typechecker=beartype)
def _B_nu( T: Float[Array, ""], nu: Float[Array, "N"] ) -> Float[Array, "N"]:
    ''' Planck function B_nu(T, nu) for a given frequency nu and temperature T. Units: [W/m^2/Hz/sr] -> [L_sun / R_sun^2 / Hz / sr] '''
    log_numerator = jnp.log(2) + jnp.log(h) - 2 * jnp.log(c) + 3 * jnp.log(nu) # Prevent numerical overflow
    log_denominator = jnp.log(jnp.exp((h / k_B) * (nu / T)) - 1.)
    return jnp.exp(log_numerator - log_denominator - log_solar_luminosity + 2 * log_solar_radius)
@jaxtyped(typechecker=beartype)
def _B_lambda( T: Float[Array, ""], lam: Float[Array, "N"]) -> Float[Array, "N"]:
    ''' Planck function B_lambda(T, lam) for a given wavelength lam and temperature T. Units: [W/m^2/m/sr] -> [L_sun / R_sun^2 / nm / sr] '''
    lam_meters = lam * 1e-9  # Convert wavelength from nm to meters
    log_numerator = jnp.log(2) + jnp.log(h) + 2 * jnp.log(c) - 5 * jnp.log(lam_meters) # Prevent numerical overflow
    log_denominator = jnp.log(jnp.exp((h * c) / (lam_meters * k_B * T)) - 1.)
    return jnp.exp(log_numerator - log_denominator - log_solar_luminosity + 2 * log_solar_radius + log_nm)

# Vectorize all of the above functions using vmap
def B_nu(
        T: Float[Array, "M"],
        nu: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Planck function B_nu(T, nu) for a given frequency nu and temperature T.

    Args:
    nu: Frequency in Hz.
    T : Temperature in Kelvin.

    Returns:
        Planck function values for the given frequencies and temperatures. Units: [L_sun / R_sun^2 / Hz / sr]
    """
    vmap_B_nu = vmap(_B_nu, in_axes=(0, None), out_axes=0)
    return vmap_B_nu(T, nu)
def B_lambda(
        T: Float[Array, "M"],
        lam: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Planck function B_lambda(T, lam) for a given wavelength lam and temperature T.

    Args:
    lam: Wavelength in nanometers
    T : Temperature in Kelvin.

    Returns:
        Planck function values for the given wavelengths and temperatures. Units: [L_sun / R_sun^2 / nm / sr]
    """
    vmap_B_lambda = vmap(_B_lambda, in_axes=(0, None), out_axes=0)
    return vmap_B_lambda(T, lam)

def L_nu(
        T: Float[Array, "M"],
        R: Float[Array, "M"],
        nu: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Spectral luminosity L_nu(T, nu) for a given frequency nu, temperature T, and radius R.

    Args:
    nu: Frequency in Hz.
    T : Temperature in Kelvin.
    R : Radius in solar radii

    Returns:
        Spectral luminosity values for the given frequencies, temperatures, and radii. Units: [Lsun / Hz]
    """
    return 4 * jnp.pi * R**2 * B_nu(T, nu)

def L_lambda(
        T: Float[Array, "M"],
        R: Float[Array, "M"],
        lam: Float[Array, "N"]
        ) -> Float[Array, "M N"]:
    """
    Spectral luminosity L_lambda(T, lam) for a given wavelength lam, temperature T, and radius R.

    Args:
    lam: Wavelength in nm.
    T : Temperature in Kelvin.
    R : Radius in solar radii

    Returns:
        Spectral luminosity values for the given wavelengths, temperatures, and radii. Units: [Lsun / m]
    """
    return 4 * jnp.pi * R**2 * B_lambda(T, lam)

def L_bolometric(
        T: Float[Array, "M"],
        R: Float[Array, "M"]
        ) -> Float[Array, "M"]:
    """
    Bolometric luminosity L_bolometric(T, R) for a given temperature T and radius R.

    Args:
    T : Temperature in Kelvin.
    R : Radius in solar radii

    Returns:
        Bolometric luminosity values for the given temperatures and radii. Units: [W] -> [Lsun]
    """
    R_meters = R * solar_radius  # Convert radius from solar radii to meters
    log_L_bolometric = jnp.log(4 * jnp.pi) + 2 * jnp.log(R) + jnp.log(sigma) + 4 * jnp.log(T) # Units: W
    return jnp.exp(log_L_bolometric - log_solar_luminosity)  # Convert to solar luminosities
def F_bolometric(
        T: Float[Array, "M"],
        R: Float[Array, "M"],
        d: Float[Array, "M"]
        ) -> Float[Array, "M"]:
    """
    Bolometric flux F_bolometric(T, R, d) for a given temperature T, radius R, and distance d.

    Args:
    T : Temperature in Kelvin.
    R : Radius in solar radii
    d : Distance in *kpc*

    Returns:
        Bolometric flux values for the given temperatures, radii, and distances. Units: [Lsun/kpc] -> [Jy*Hz]
    """
    L_bol = L_bolometric(T, R)
    log_F_bolometric = jnp.log(L_bol) - jnp.log(4 * jnp.pi) - 2 * jnp.log(d) # Units: [Lsun/kpc]
    # Convert Lsun/kpc^2 to W/m^2 and then to Jy*Hz
    log_F_bolometric = log_F_bolometric + log_solar_luminosity - 2 * log_kpc - log_jansky
    return jnp.exp(log_F_bolometric)

if __name__ == "__main__":
    # Example usage
    T = jnp.array([3e3, 4e3, 5e3]) # Temperatures in Kelvin
    R = jnp.array([1.0, 2.0, 3.0]) # Radii in solar radii
    nu = jnp.array([1e14, 2e14, 3e14])  # Frequencies in Hz
    lam = jnp.array([400., 500., 600.])  # Wavelengths in nm
    print("B_nu:", B_nu(T, nu))
    print("B_lambda:", B_lambda(T, lam)) 
    print("L_nu:", L_nu(T, R, nu))
    print("L_lambda:", L_lambda(T, R, lam)) 


