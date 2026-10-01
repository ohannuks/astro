import jax.numpy as jnp
from astro.common import Scalar, Vec, Mat, typed

class BoyerLindquist:
    @typed
    def g_mu_nu(self, r:Scalar, theta:Scalar, M:Scalar, a:Scalar) -> Mat:
        """ Returns the metric tensor g_{mu nu} in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            theta (Scalar): Polar angle coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Mat: Metric tensor g_{mu nu} in Boyer-Lindquist coordinates.
        """
        Sigma = r**2 + (a * jnp.cos(theta))**2
        Delta = r**2 - 2 * M * r + a**2
        g_t_t = -1 + 2 * M * r / Sigma
        g_r_r = Sigma / Delta
        g_theta_theta = Sigma
        g_phi_phi = (r**2 + a**2 + 2 * M * r * a**2 * jnp.sin(theta)**2 / Sigma) * jnp.sin(theta)**2
        g_t_phi = -2 * M * r * a * jnp.sin(theta)**2 / Sigma
        gmunu = jnp.array([
            [g_t_t, 0, 0, g_t_phi],
            [0, g_r_r, 0, 0],
            [0, 0, g_theta_theta, 0],
            [g_t_phi, 0, 0, g_phi_phi]
        ])
        return gmunu
    @typed
    def g_inv_mu_nu(self, r:Scalar, theta:Scalar, M:Scalar, a:Scalar) -> Mat:
        """ Returns the inverse metric tensor g^{mu nu} in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            theta (Scalar): Polar angle coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Mat: Inverse metric tensor g^{mu nu} in Boyer-Lindquist coordinates
        """
        Sigma = r**2 + (a * jnp.cos(theta))**2
        Delta = r**2 - 2 * M * r + a**2
        ginv_t_t = -((r**2 + a**2)**2 - Delta * a**2 * jnp.sin(theta)**2) / (Sigma * Delta)
        ginv_r_r = Delta / Sigma
        ginv_theta_theta = 1 / Sigma
        ginv_phi_phi = (Delta - a**2 * jnp.sin(theta)**2) / (Sigma * Delta * jnp.sin(theta)**2)
        ginv_t_phi = -2 * M * r * a / (Sigma * Delta)
        ginvmunu = jnp.array([
            [ginv_t_t, 0, 0, ginv_t_phi],
            [0, ginv_r_r, 0, 0],
            [0, 0, ginv_theta_theta, 0],
            [ginv_t_phi, 0, 0, ginv_phi_phi]
        ])
        return ginvmunu


if __name__ == "__main__":
    bl = BoyerLindquist()
    r = jnp.array(2.0)
    theta = jnp.array(jnp.pi / 4)
    M = jnp.array(1.0)
    a = jnp.array(0.5)
    gmunu = bl.g_mu_nu(r, theta, M, a)
    ginvmunu = bl.g_inv_mu_nu(r, theta, M, a)
    print("Metric tensor g_{mu nu}:\n", gmunu)
    print("Inverse metric tensor g^{mu nu}:\n", ginvmunu)
