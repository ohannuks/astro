import jax
from collections.abc import Callable
from jaxtyping import Float, Int, jaxtyped, Array
from beartype import beartype
import jax.numpy as jnp
from astro.instrument import create_grid
from tensorflow_probability.substrates.jax.math import bessel_kve

@jaxtyped(typechecker=beartype)
def bessel_second(v: float, z: Float[Array, "nx ny"]) -> Float[Array, "nx ny"]:
    # K_v(z) = bessel_kve(v, z) / exp(z)
    z_safe = jnp.where(z == 0, 1e-7, z)
    return bessel_kve(v, z_safe) * jnp.exp(-jnp.abs(z_safe))

# Implement the Mossat PSF:
@jaxtyped(typechecker=beartype)
def mossat_psf(
        alpha: float|Float[Array, ""],
        beta: float|Float[Array, ""],
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

@jaxtyped(typechecker=beartype)
def mossat_psf_fourier(
        alpha: float|Float[Array, ""],
        beta: float|Float[Array, ""],
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
    k = jnp.sqrt(k2)
    psf_ft = 2./jax.scipy.special.gamma(beta-1) * (jnp.pi * alpha * k)**(beta-1) * bessel_second(beta-1, 2 * jnp.pi * alpha * k)
    return psf_ft

def run_verification_test():
    """Verify analytical Moffat FT against the scaled numerical FFT."""
    print("--- Running Moffat PSF FT Verification Test ---")
    
    # 1. Setup grid and parameters
    alpha = 1.2
    beta = 2.5
    pix_scale = 0.03
    nx, ny = 256, 256  # Larger, even power-of-2 grid minimizes aliasing edge-effects
    x0, y0 = 0.0, 0.0

    # Generate spatial coordinates
    (X, Y), _ = create_grid(x0, y0, pix_scale, nx, ny)
    
    # 2. Compute spatial PSF and its numerical FFT
    psf = mossat_psf(alpha, beta, X, Y)
    
    # Shift center to (0,0) index before FFT
    psf_fft_ready = jnp.fft.ifftshift(psf)
    
    # Continuous FT approximation requires scaling by pixel area element (dx * dy)
    numerical_ft = jnp.fft.fft2(psf_fft_ready) * (pix_scale ** 2)
    numerical_ft_centered = jnp.fft.fftshift(numerical_ft)
    
    # 3. Generate frequency coordinates and analytical FT
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=pix_scale))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=pix_scale))
    KX, KY = jnp.meshgrid(kx, ky)
    
    analytical_ft = mossat_psf_fourier(alpha, beta, KX, KY)
    
    # 4. Compute differences (excluding edges prone to discrete aliasing)
    # We slice out the central 50% of the frequency domain where energy is concentrated
    trim_y, trim_x = ny // 4, nx // 4
    num_core = numerical_ft_centered[trim_y:-trim_y, trim_x:-trim_x]
    ana_core = analytical_ft[trim_y:-trim_y, trim_x:-trim_x]
    
    max_absolute_diff = jnp.max(jnp.abs(jnp.abs(num_core) - jnp.abs(ana_core)))
    mean_absolute_error = jnp.mean(jnp.abs(jnp.abs(num_core) - jnp.abs(ana_core)))
    
    print(f"DC Component (k=0) -> Analytical: {analytical_ft[ny//2, nx//2]:.4f} | Numerical: {jnp.abs(numerical_ft_centered[ny//2, nx//2]):.4f}")
    print(f"Core Domain Max Abs Difference: {max_absolute_diff:.6f}")
    print(f"Core Domain Mean Abs Error:    {mean_absolute_error:.6f}")
    
    # Tolerances are subject to grid size and pixel scale limits due to discrete sampling
    assert mean_absolute_error < 1e-2, f"Verification failed! Mean error {mean_absolute_error} is too high."
    print("Verification Passed successfully!")


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
    #import matplotlib.pyplot as plt
    #plt.imshow(psf, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    #plt.colorbar(label='PSF Intensity')
    #plt.title('Mossat PSF')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.show()

    psf_fft_ready = jnp.fft.ifftshift(psf)
    psf_fft = jnp.fft.fft2(psf_fft_ready)
    fft_magnitude_centered = jnp.fft.fftshift(jnp.abs(psf_fft))
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=pix_scale))
    ky = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=pix_scale))
    #plt.imshow(
    #    fft_magnitude_centered, 
    #    extent=[kx.min(), kx.max(), ky.min(), ky.max()], 
    #    origin='lower'
    #)
    #plt.colorbar(label='FFT Magnitude')
    #plt.title('FFT of Mossat PSF (Centered Spectrum)')
    #plt.xlabel(r"$k_x$ [cycles / arcsec]")
    #plt.ylabel(r"$k_y$ [cycles / arcsec]")
    #plt.show()

    ## Run verification test
    run_verification_test()
