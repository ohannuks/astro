import jax.numpy as jnp
from astro.common import Scalar, Vec, Mat, typed
import elliptic  # includes Jacobi sine function sn, Legendre incomplete elliptic integral of the first kind
# sn, cn, dn, am = elliptic.ellipj(u, m)
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
        Sigma = self.Sigma(r, theta, a)
        Delta = self.Delta(r, M, a)
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
        Sigma = self.Sigma(r, theta, a)
        Delta = self.Delta(r, M, a)
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
    # Equations on motion using Mino time (for massive particles mu > 0)
    @typed
    def rdot2(self, r:Scalar, M:Scalar, a:Scalar, E:Scalar, Lz:Scalar, Q:Scalar) -> Scalar:
        """ Returns the radial component of the equations of motion in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific Energy of the particle (E = E_standard / mu)
            Lz (Scalar): Specific Angular momentum of the particle (Lz = Lz_standard / mu)
            Q (Scalar): Specific Carter constant of the particle (Q = Q_standard / mu^2)

        Returns:
            Scalar: Radial component of the equations of motion in Boyer-Lindquist coordinates.
        """
        Delta = self.Delta(r, M, a)
        R = ((r**2 + a**2) * E - a * Lz)**2 - Delta * (r**2 + Q + (Lz - a * E)**2)
        return R
    @typed
    def thetadot2(self, theta:Scalar, M:Scalar, a:Scalar, E:Scalar, Lz:Scalar, Q:Scalar) -> Scalar:
        """ Returns the polar angle component of the equations of motion in Boyer-Lindquist coordinates.

        Args:
            theta (Scalar): Polar angle coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific Energy of the particle (E = E_standard / mu)
            Lz (Scalar): Specific Angular momentum of the particle (Lz = Lz_standard / mu)
            Q (Scalar): Specific Carter constant of the particle (Q = Q_standard / mu^2)

        Returns:
            Scalar: Polar angle component of the equations of motion in Boyer-Lindquist coordinates.
        """
        Theta = Q - (a**2 * (1 - E**2) + Lz**2 / jnp.sin(theta)**2) * jnp.cos(theta)**2
        return Theta
    @typed
    def phidot(self, r:Scalar, theta:Scalar, M:Scalar, a:Scalar, E:Scalar, Lz:Scalar) -> Scalar:
        """ Returns the azimuthal component of the equations of motion in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            theta (Scalar): Polar angle coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific Energy of the particle (E = E_standard / mu)
            Lz (Scalar): Specific Angular momentum of the particle (Lz = Lz_standard / mu)

        Returns:
            Scalar: Azimuthal component of the equations of motion in Boyer-Lindquist coordinates.
        """
        Sigma = self.Sigma(r, theta, a)
        Delta = self.Delta(r, M, a)
        Phi_r = a * ((r**2 + a**2) * E - a * Lz) / Delta
        Phi_theta = Lz / jnp.sin(theta)**2 - a * E
        phidot = (Phi_r + Phi_theta) / Sigma
        return phidot
    @typed
    def tdot(self, r:Scalar, theta:Scalar, M:Scalar, a:Scalar, E:Scalar, Lz:Scalar) -> Scalar:
        """ Returns the time component of the equations of motion in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            theta (Scalar): Polar angle coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific Energy of the particle (E = E_standard / mu)
            Lz (Scalar): Specific Angular momentum of the particle (Lz = Lz_standard / mu)

        Returns:
            Scalar: Time component of the equations of motion in Boyer-Lindquist coordinates.
        """
        Sigma = self.Sigma(r, theta, a)
        Delta = self.Delta(r, M, a)
        T_r = ((r**2 + a**2)**2 * E - 2 * M * r * a * Lz) / Delta
        T_theta = a * (Lz - a * E * jnp.sin(theta)**2)
        tdot = (T_r + T_theta) / Sigma
        return tdot
    @typed
    def Sigma(self, r:Scalar, theta:Scalar, a:Scalar) -> Scalar:
        """ Returns the Sigma function in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            theta (Scalar): Polar angle coordinate.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: Sigma function in Boyer-Lindquist coordinates.
        """
        return r**2 + (a * jnp.cos(theta))**2
    @typed
    def Delta(self, r:Scalar, M:Scalar, a:Scalar) -> Scalar:
        """ Returns the Delta function in Boyer-Lindquist coordinates.

        Args:
            r (Scalar): Radial coordinate.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: Delta function in Boyer-Lindquist coordinates.
        """
        return r**2 - 2 * M * r + a**2

class BoyerLindquistGeodesic(BoyerLindquist):
    """ Struct for geodesics in Boyer-Lindquist coordinates using semi-analytical methods following https://arxiv.org/abs/0906.1420 . """
    @typed
    def r_psi(self, psi:Scalar, r1:Scalar, r2:Scalar, r3:Scalar, r4:Scalar) -> Scalar:
        """ Returns the radial coordinate r as a function of the Jacobi amplitude psi (eq. 62).

        Args:
            psi (Scalar): Jacobi amplitude (am), with sn(u|m) = sin(psi).
            r1 (Scalar): First root of the radial potential (apoapsis).
            r2 (Scalar): Second root of the radial potential (periapsis).
            r3 (Scalar): Third root of the radial potential.
            r4 (Scalar): Fourth root of the radial potential.

        Returns:
            Scalar: Radial coordinate r(psi).
        """
        sin_psi = jnp.sin(psi)
        r = (r3 * (r1 - r2) * sin_psi**2 - r2 * (r1 - r3)) / ((r1 - r2) * sin_psi**2 - (r1 - r3))
        return r
    @typed
    def find_radial_roots(self, p:Scalar, e:Scalar, M:Scalar, a:Scalar, E:Scalar, Q:Scalar) -> tuple[Scalar, Scalar, Scalar, Scalar]:
        """ Returns the four roots (r1, r2, r3, r4) of the radial potential R(r) (eqs. 21-22).

        Args:
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific energy.
            Q (Scalar): Specific Carter constant.

        Returns:
            tuple: (r1, r2, r3, r4) with r1 = r_max, r2 = r_min.
        """
        r1 = p * M / (1 - e)
        r2 = p * M / (1 + e)
        A_plus_B = 2 * M / (1 - E**2) - (r1 + r2)
        AB = a**2 * Q / ((1 - E**2) * r1 * r2)
        r3 = 0.5 * (A_plus_B + jnp.sqrt(A_plus_B**2 - 4 * AB))
        r4 = AB / r3
        return r1, r2, r3, r4
    @typed
    def find_polar_roots(self, a:Scalar, E:Scalar, Lz:Scalar, Q:Scalar, x:Scalar) -> tuple[Scalar, Scalar]:
        """ Returns the polar roots (z_minus, z_plus) (eq. 20).

        Args:
            a (Scalar): Spin parameter of the black hole.
            E (Scalar): Specific energy.
            Lz (Scalar): Specific angular momentum.
            Q (Scalar): Specific Carter constant.
            x (Scalar): Cosine of the inclination angle.

        Returns:
            tuple: (z_minus, z_plus) with z_minus = cos^2(theta_min).
        """
        z_minus = 1 - x**2
        eps0 = a**2 * (1 - E**2) / Lz**2
        z_plus = Q / (Lz**2 * eps0 * z_minus)
        return z_minus, z_plus

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
    E, Lz, Q = jnp.array(1.0), jnp.array(0.5), jnp.array(0.1)
    print("rdot2:", bl.rdot2(r, M, a, E, Lz, Q))
    print("thetadot2:", bl.thetadot2(theta, M, a, E, Lz, Q))
    print("phidot:", bl.phidot(r, theta, M, a, E, Lz))
    print("tdot:", bl.tdot(r, theta, M, a, E, Lz))
