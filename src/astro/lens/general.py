from astro.common import Scalar, typed, Vec
import jax.numpy as jnp


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
def theta_L(Mlz: Scalar, Dl: Scalar, Ds: Scalar, Dls: Scalar) -> Scalar:
    """ Angular scale of the lens.

    Args:
        Mlz: lens mass enclosed inside of einstein radius, redshifted, in units of seconds
        Dl: angular diameter distance to the lens in seconds
        Ds: angular diameter distance to the source in seconds
        Dls: angular diameter distance from the lens to the source in seconds

    Returns:
        Angular scale of the lens theta_L = sqrt(4 * Mlz * Dls / (Dl * Ds))
    """
    return jnp.sqrt(4 * Mlz * Dls / (Dl * Ds))


@typed
def t_L(Mlz: Scalar) -> Scalar:
    """ Time scale of the lens.

    Args:
        Mlz: lens mass enclosed inside of einstein radius, redshifted, in units of seconds

    Returns:
        Time scale of the lens t_L = 4 * Mlz
    """
    return 4 * Mlz
