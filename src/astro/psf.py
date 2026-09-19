from collections.abc import Callable
from jaxtyping import Float, Int, jaxtyped, Array
from beartype import beartype
import jax.numpy as jnp
from astro.instrument import create_grid
from tfp.substrates.jax.math import bessel_kve

def jax_bessel_k(v, z):
    # K_v(z) = bessel_kve(v, z) / exp(z)
    return bessel_kve(v, z) * jnp.exp(-jnp.abs(z))

# Implement the Mossat PSF:
def mossat_psf(
        alpha: Float[Array, ""],
        beta: Float[Array, ""],
        x: Float[Array, "ny nx"],
        y: Float[Array, "ny nx"]
        ) -> Float[Array, "ny nx"]:
    """
    Compute the Mossat PSF for given parameters and coordinates.

    Args:
        alpha: The alpha parameter of the Mossat PSF.
        beta: The beta parameter of the Mossat PSF.
        x: The x-coordinates of the grid.
        y: The y-coordinates of the grid.

    Returns:
        A 2D array representing the Mossat PSF.
    """
    r2 = x**2 + y**2
    psf = (1 + r2 * (1 / alpha)**2)**(-beta)
    I0 = (beta - 1) / (jnp.pi * alpha**2)  # Normalization constant
    psf *= I0
    return psf

def mossat_psf_fourier(
        alpha: Float[Array, ""],
        beta: Float[Array, ""],
        kx: Float[Array, "ny nx"],
        ky: Float[Array, "ny nx"]
        ) -> Float[Array, "ny nx"]:
    """
    Compute the Fourier transform of the Mossat PSF for given parameters and frequency coordinates.

    Args:
        alpha: The alpha parameter of the Mossat PSF.
        beta: The beta parameter of the Mossat PSF.
        kx: The x-coordinates in frequency space.
        ky: The y-coordinates in frequency space.

    Returns:
        A 2D array representing the Fourier transform of the Mossat PSF.
    """
    k2 = kx**2 + ky**2
    psf_ft = (1 + k2 * (alpha**2))**(-beta)
    return psf_ft

if __name__ == "__main__":
    # Example usage
    alpha = 1.0  # Example alpha parameter
    beta = 2.0   # Example beta parameter
    x0, y0 = 0.0, 0.0  # Center of the grid
    pix_scale = 0.05   # Pixel scale in arcseconds/pixel
    nx, ny = 100, 100  # Number of pixels in x and y directions

    (X, Y), (x,y) = create_grid(x0, y0, pix_scale, nx, ny)
    psf = mossat_psf(alpha, beta, X, Y)
    # Plot the PSF using matplotlib:
    import matplotlib.pyplot as plt
    plt.imshow(psf, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar(label='PSF Intensity')
    plt.title('Mossat PSF')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    psf_fft_ready = jnp.fft.ifftshift(psf)
    psf_fft = jnp.fft.fft2(psf_fft_ready)
    fft_magnitude_centered = jnp.fft.fftshift(jnp.abs(psf_fft))
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=pix_scale))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=pix_scale))
    plt.imshow(
        fft_magnitude_centered, 
        extent=[kx.min(), kx.max(), ky.min(), ky.max()], 
        origin='lower'
    )
    plt.colorbar(label='FFT Magnitude')
    plt.title('FFT of Mossat PSF (Centered Spectrum)')
    plt.xlabel(r"$k_x$ [cycles / arcsec]")
    plt.ylabel(r"$k_y$ [cycles / arcsec]")
    plt.show()

