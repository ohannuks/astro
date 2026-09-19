from jaxtyping import Float, Int, jaxtyped, Array
from beartype import beartype
import jax.numpy as jnp

# HST: To perform deep follow-up imaging of galaxy-galaxy gravitational lenses, the ideal choice is almost universally F160W (H-band), often paired with F105W (Y-band) or F125W (J-band) for color profiling.

@jaxtyped(typechecker=beartype)
def create_grid(
        x0: float, 
        y0: float,
        pix_scale: float,
        nx: int,
        ny: int
        ) -> tuple[tuple[Float[Array, "ny nx"], Float[Array, "ny nx"]], tuple[Float[Array, "flat"], Float[Array, "flat"]]]:
    """
    Create a grid of pixel coordinates centered at (x0, y0) with given pixel scale and dimensions.

    Args:
        x0: The x-coordinate of the center of the grid.
        y0: The y-coordinate of the center of the grid.
        pix_scale: The pixel scale in arcseconds per pixel.
        nx: The number of pixels in the x-direction.
        ny: The number of pixels in the y-direction.
    
    Returns:
        A tuple containing two 2D arrays representing the x and y coordinates of the grid.
    """
    x = jnp.linspace(x0 - (nx // 2) * pix_scale, x0 + (nx // 2) * pix_scale, nx)
    y = jnp.linspace(y0 - (ny // 2) * pix_scale, y0 + (ny // 2) * pix_scale, ny)
    return jnp.meshgrid(x, y), (jnp.ravel(x), jnp.ravel(y))

if __name__ == "__main__":
    # Example usage
    x0, y0 = 0.0, 0.0  # Center of the grid
    pix_scale = 0.05   # Pixel scale in arcseconds/pixel
    nx, ny = 100, 100  # Number of pixels in x and y directions

    (X, Y), (x,y) = create_grid(x0, y0, pix_scale, nx, ny)
    print("2D Grid Coordinates (X):", X)
    print("2D Grid Coordinates (Y):", Y)
    print("1D Grid Coordinates (x):", x)
    print("1D Grid Coordinates (y):", y)
