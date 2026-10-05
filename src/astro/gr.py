import jax
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
    def ellipk(self, m:Scalar) -> Scalar:
        """ Complete elliptic integral of the first kind K(m). """
        F, _, _ = elliptic.elliptic12(jnp.pi / 2, m)
        return F
    @typed
    def ellipe(self, m:Scalar) -> Scalar:
        """ Complete elliptic integral of the second kind E(m). """
        _, Einc, _ = elliptic.elliptic12(jnp.pi / 2, m)
        return Einc
    @typed
    def ellippi(self, n:Scalar, m:Scalar) -> Scalar:
        """ Complete elliptic integral of the third kind Pi(n, m). """
        return elliptic.elliptic3(jnp.pi / 2, m, n)
    @typed
    def ellipeinc(self, phi:Scalar, m:Scalar) -> Scalar:
        """ Incomplete elliptic integral of the second kind E(phi, m). """
        _, Einc, _ = elliptic.elliptic12(phi, m)
        return Einc
    @typed
    def ellippiinc(self, phi:Scalar, n:Scalar, m:Scalar) -> Scalar:
        """ Incomplete elliptic integral of the third kind Pi(phi, n, m). """
        return elliptic.elliptic3(phi, m, n)
    @typed
    def ellipj(self, u:Scalar, m:Scalar) -> tuple[Scalar, Scalar, Scalar, Scalar]:
        """ Jacobi elliptic functions (sn, cn, dn, am). """
        return elliptic.ellipj(u, m)
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
    @typed
    def constants_of_motion(self, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> tuple[Scalar, Scalar, Scalar]:
        """ Specific (E, Lz, Q) from R(r1)=R(r2)=0 (Schmidt App. B).

        Args:
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            tuple: (E, Lz, Q).
        """
        r1 = p * M / (1 - e)
        r2 = p * M / (1 + e)
        z2 = 1 - x**2
        Delta1 = self.Delta(r1, M, a)
        Delta2 = self.Delta(r2, M, a)
        f1 = r1**4 + a**2 * (r1 * (r1 + 2 * M) + z2 * Delta1)
        g1 = 2 * a * M * r1
        h1 = r1 * (r1 - 2 * M) + z2 / x**2 * Delta1
        d1 = (r1**2 + a**2 * z2) * Delta1
        f2 = r2**4 + a**2 * (r2 * (r2 + 2 * M) + z2 * Delta2)
        g2 = 2 * a * M * r2
        h2 = r2 * (r2 - 2 * M) + z2 / x**2 * Delta2
        d2 = (r2**2 + a**2 * z2) * Delta2
        kappa = d1 * h2 - h1 * d2
        rho = f1 * h2 - h1 * f2
        sigma = g1 * h2 - h1 * g2
        epsilon = d1 * g2 - g1 * d2
        eta = f1 * g2 - g1 * f2
        disc = sigma * (sigma * epsilon**2 + rho * epsilon * kappa - eta * kappa**2)
        E = jnp.sqrt((kappa * rho + 2 * epsilon * sigma - jnp.sign(x) * 2 * jnp.sqrt(disc)) / (rho**2 + 4 * eta * sigma))
        Lz = (-E * g1 + jnp.sign(x) * jnp.sqrt(-d1 * h1 + E**2 * (g1**2 + f1 * h1))) / h1
        Q = z2 * (a**2 * (1 - E**2) + Lz**2 / x**2)
        return E, Lz, Q
    @typed
    def mino_frequencies(self, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> tuple[Scalar, Scalar, Scalar, Scalar]:
        """ Mino-time frequencies (Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma).

        Args:
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            tuple: (Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma).
        """
        E, Lz, Q = self.constants_of_motion(p, e, x, M, a)
        r1, r2, r3, r4 = self.find_radial_roots(p, e, M, a, E, Q)
        z_minus, z_plus = self.find_polar_roots(a, E, Lz, Q, x)
        r_plus = M + jnp.sqrt(M**2 - a**2)
        r_minus = M - jnp.sqrt(M**2 - a**2)
        k_r2 = (r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4))
        k_th2 = z_minus / z_plus
        h_r = (r1 - r2) / (r1 - r3)
        h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
        h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))
        K_r = self.ellipk(k_r2)
        K_th = self.ellipk(k_th2)
        E_r = self.ellipe(k_r2)
        E_th = self.ellipe(k_th2)
        Pi_r = self.ellippi(h_r, k_r2)
        Pi_plus = self.ellippi(h_plus, k_r2)
        Pi_minus = self.ellippi(h_minus, k_r2)
        Pi_z = self.ellippi(z_minus, k_th2)
        e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + Lz**2) / (Lz**2 * (1 - z_minus))
        eps0 = a**2 * (1 - E**2) / Lz**2
        radial_pref = jnp.sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
        Upsilon_r = jnp.pi * radial_pref / (2 * K_r)
        Upsilon_theta = jnp.pi * Lz * jnp.sqrt(e0zp) / (2 * K_th)
        Upsilon_phi = 2 * Upsilon_theta / (jnp.pi * jnp.sqrt(e0zp)) * Pi_z + 2 * a * Upsilon_r / (jnp.pi * (r_plus - r_minus) * radial_pref) * (
            (2 * E * r_plus - a * Lz) / (r3 - r_plus) * (K_r - (r2 - r3) / (r2 - r_plus) * Pi_plus)
            - (2 * E * r_minus - a * Lz) / (r3 - r_minus) * (K_r - (r2 - r3) / (r2 - r_minus) * Pi_minus)
        )
        Gamma = (
            4 * E
            + 2 * a**2 * z_plus / jnp.sqrt(eps0 * z_plus) * E * Upsilon_theta * (K_th - E_th) / (jnp.pi * Lz)
            + 2 * Upsilon_r / (jnp.pi * radial_pref) * (
                E / 2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * K_r + (r2 - r3) * (r1 + r2 + r3 + r4) * Pi_r + (r1 - r3) * (r2 - r4) * E_r)
                + 2 * E * (r3 * K_r + (r2 - r3) * Pi_r)
                + 2 / (r_plus - r_minus) * (
                    ((4 * E - a * Lz) * r_plus - 2 * a**2 * E) / (r3 - r_plus) * (K_r - (r2 - r3) / (r2 - r_plus) * Pi_plus)
                    - ((4 * E - a * Lz) * r_minus - 2 * a**2 * E) / (r3 - r_minus) * (K_r - (r2 - r3) / (r2 - r_minus) * Pi_minus)
                )
            )
        )
        return Upsilon_r, jnp.abs(Upsilon_theta), Upsilon_phi, Gamma
    @typed
    def fundamental_frequencies(self, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> tuple[Scalar, Scalar, Scalar]:
        """ Boyer-Lindquist frequencies Omega_i = Upsilon_i / Gamma.

        Args:
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            tuple: (Omega_r, Omega_theta, Omega_phi).
        """
        Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma = self.mino_frequencies(p, e, x, M, a)
        return Upsilon_r / Gamma, Upsilon_theta / Gamma, Upsilon_phi / Gamma
    @typed
    def r(self, lam:Scalar, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> Scalar:
        """ Radial coordinate r(lambda) (Fujita & Hikida eq. 27).

        Args:
            lam (Scalar): Mino time.
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: r(lambda), with r(0) = periapsis.
        """
        E, Lz, Q = self.constants_of_motion(p, e, x, M, a)
        r1, r2, r3, r4 = self.find_radial_roots(p, e, M, a, E, Q)
        Upsilon_r, _, _, _ = self.mino_frequencies(p, e, x, M, a)
        k_r2 = (r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4))
        q_r = Upsilon_r * lam
        _, _, _, psi = self.ellipj(self.ellipk(k_r2) * q_r / jnp.pi, k_r2)
        return self.r_psi(psi, r1, r2, r3, r4)
    @typed
    def theta(self, lam:Scalar, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> Scalar:
        """ Polar angle theta(lambda) (Fujita & Hikida eq. 38).

        Args:
            lam (Scalar): Mino time.
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: theta(lambda), with theta(0) = theta_min.
        """
        E, Lz, Q = self.constants_of_motion(p, e, x, M, a)
        z_minus, z_plus = self.find_polar_roots(a, E, Lz, Q, x)
        _, Upsilon_theta, _, _ = self.mino_frequencies(p, e, x, M, a)
        k_th2 = z_minus / z_plus
        q_theta = Upsilon_theta * lam
        sn, _, _, _ = self.ellipj(2 / jnp.pi * self.ellipk(k_th2) * (q_theta + jnp.pi / 2), k_th2)
        return jnp.arccos(jnp.sqrt(z_minus) * sn)
    @typed
    def t(self, lam:Scalar, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> Scalar:
        """ Coordinate time t(lambda) (Fujita & Hikida eqs. 6, 28, 39).

        Args:
            lam (Scalar): Mino time.
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: t(lambda), with t(0) = 0.
        """
        E, Lz, Q = self.constants_of_motion(p, e, x, M, a)
        r1, r2, r3, r4 = self.find_radial_roots(p, e, M, a, E, Q)
        z_minus, z_plus = self.find_polar_roots(a, E, Lz, Q, x)
        Upsilon_r, Upsilon_theta, _, Gamma = self.mino_frequencies(p, e, x, M, a)
        r_plus = M + jnp.sqrt(M**2 - a**2)
        r_minus = M - jnp.sqrt(M**2 - a**2)
        k_r2 = (r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4))
        k_th2 = z_minus / z_plus
        h_r = (r1 - r2) / (r1 - r3)
        h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
        h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))
        q_r = Upsilon_r * lam
        q_theta = Upsilon_theta * lam
        sn, cn, dn, psi_r = self.ellipj(self.ellipk(k_r2) * q_r / jnp.pi, k_r2)
        _, _, _, psi_th = self.ellipj(2 / jnp.pi * self.ellipk(k_th2) * (q_theta + jnp.pi / 2), k_th2)
        dPi_r = self.ellippiinc(psi_r, h_r, k_r2) - q_r / jnp.pi * self.ellippi(h_r, k_r2)
        dPi_plus = self.ellippiinc(psi_r, h_plus, k_r2) - q_r / jnp.pi * self.ellippi(h_plus, k_r2)
        dPi_minus = self.ellippiinc(psi_r, h_minus, k_r2) - q_r / jnp.pi * self.ellippi(h_minus, k_r2)
        dE_r = self.ellipeinc(psi_r, k_r2) + h_r * sn * cn * dn / (h_r * sn**2 - 1) - q_r / jnp.pi * self.ellipe(k_r2)
        radial_pref = 2 / jnp.sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
        t_r = radial_pref * (
            E / 2 * ((r2 - r3) * (r1 + r2 + r3 + r4) * dPi_r + (r1 - r3) * (r2 - r4) * dE_r)
            + 2 * E * (r2 - r3) * dPi_r
            - 2 / (r_plus - r_minus) * (
                ((4 * E - a * Lz) * r_plus - 2 * a**2 * E) * (r2 - r3) / ((r3 - r_plus) * (r2 - r_plus)) * dPi_plus
                - ((4 * E - a * Lz) * r_minus - 2 * a**2 * E) * (r2 - r3) / ((r3 - r_minus) * (r2 - r_minus)) * dPi_minus
            )
        )
        eps0 = a**2 * (1 - E**2) / Lz**2
        t_theta = jnp.sign(Lz) * a**2 * z_plus / jnp.sqrt(eps0 * z_plus) * E / Lz * (
            2 / jnp.pi * self.ellipe(k_th2) * (q_theta + jnp.pi / 2) - self.ellipeinc(psi_th, k_th2)
        )
        return Gamma * lam + t_r + t_theta
    @typed
    def phi(self, lam:Scalar, p:Scalar, e:Scalar, x:Scalar, M:Scalar, a:Scalar) -> Scalar:
        """ Azimuthal angle phi(lambda) (Fujita & Hikida eqs. 6, 28, 39).

        Args:
            lam (Scalar): Mino time.
            p (Scalar): Dimensionless semi-latus rectum (semi-latus rectum / M).
            e (Scalar): Eccentricity.
            x (Scalar): Cosine of the inclination.
            M (Scalar): Mass of the black hole.
            a (Scalar): Spin parameter of the black hole.

        Returns:
            Scalar: phi(lambda), with phi(0) = 0.
        """
        E, Lz, Q = self.constants_of_motion(p, e, x, M, a)
        r1, r2, r3, r4 = self.find_radial_roots(p, e, M, a, E, Q)
        z_minus, z_plus = self.find_polar_roots(a, E, Lz, Q, x)
        Upsilon_r, Upsilon_theta, Upsilon_phi, _ = self.mino_frequencies(p, e, x, M, a)
        r_plus = M + jnp.sqrt(M**2 - a**2)
        r_minus = M - jnp.sqrt(M**2 - a**2)
        k_r2 = (r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4))
        k_th2 = z_minus / z_plus
        h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
        h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))
        q_r = Upsilon_r * lam
        q_theta = Upsilon_theta * lam
        _, _, _, psi_r = self.ellipj(self.ellipk(k_r2) * q_r / jnp.pi, k_r2)
        _, _, _, psi_th = self.ellipj(2 / jnp.pi * self.ellipk(k_th2) * (q_theta + jnp.pi / 2), k_th2)
        dPi_plus = self.ellippiinc(psi_r, h_plus, k_r2) - q_r / jnp.pi * self.ellippi(h_plus, k_r2)
        dPi_minus = self.ellippiinc(psi_r, h_minus, k_r2) - q_r / jnp.pi * self.ellippi(h_minus, k_r2)
        phi_r = -2 * a / ((r_plus - r_minus) * jnp.sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))) * (
            (2 * E * r_plus - a * Lz) * (r2 - r3) / ((r3 - r_plus) * (r2 - r_plus)) * dPi_plus
            - (2 * E * r_minus - a * Lz) * (r2 - r3) / ((r3 - r_minus) * (r2 - r_minus)) * dPi_minus
        )
        e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + Lz**2) / (Lz**2 * (1 - z_minus))
        phi_theta = jnp.sign(Lz) / jnp.sqrt(e0zp) * (
            self.ellippiinc(psi_th, z_minus, k_th2) - 2 / jnp.pi * self.ellippi(z_minus, k_th2) * (q_theta + jnp.pi / 2)
        )
        return Upsilon_phi * lam + phi_r + phi_theta

if __name__ == "__main__":
    import jax
    import matplotlib.pyplot as plt
    jax.config.update("jax_enable_x64", True)
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
    geo = BoyerLindquistGeodesic()
    p = jnp.array(6.0)
    e = jnp.array(0.7)
    x = jnp.array(jnp.cos(jnp.deg2rad(20.0)))
    M = jnp.array(1.0)
    a = jnp.array(0.9)
    lams = jnp.linspace(0.0, 100.0, 10000)
    r = jax.vmap(lambda lam: geo.r(lam, p, e, x, M, a))(lams)
    theta = jax.vmap(lambda lam: geo.theta(lam, p, e, x, M, a)) (lams)
    phi = jax.vmap(lambda lam: geo.phi(lam, p, e, x, M, a)) (lams)
    t = jax.vmap(lambda lam: geo.t(lam, p, e, x, M, a)) (lams)
    lams = jnp.asarray(lams)
    r = jnp.asarray(r)
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    t = jnp.asarray(t)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(lams, r)
    axes[0, 0].set_ylabel(r"$r(\lambda)$")
    axes[0, 1].plot(lams, theta)
    axes[0, 1].set_ylabel(r"$\theta(\lambda)$")
    axes[1, 0].plot(lams, phi)
    axes[1, 0].set_ylabel(r"$\phi(\lambda)$")
    axes[1, 0].set_xlabel(r"Mino time $\lambda$")
    axes[1, 1].plot(lams, t)
    axes[1, 1].set_ylabel(r"$t(\lambda)$")
    axes[1, 1].set_xlabel(r"Mino time $\lambda$")
    fig.suptitle(r"Coordinates vs Mino time ($a=0.9M$, $p=6M$, $e=0.7$, $i=20^\circ$)")
    fig.tight_layout()
    X = r * jnp.sin(theta) * jnp.cos(phi)
    Y = r * jnp.sin(theta) * jnp.sin(phi)
    Z = r * jnp.cos(theta)
    r_plus = float(M + jnp.sqrt(M**2 - a**2))
    u = jnp.linspace(0.0, jnp.pi, 40)
    v = jnp.linspace(0.0, 2 * jnp.pi, 80)
    U, V = jnp.meshgrid(u, v)
    Xh = r_plus * jnp.sin(U) * jnp.cos(V)
    Yh = r_plus * jnp.sin(U) * jnp.sin(V)
    Zh = r_plus * jnp.cos(U)
    fig3 = plt.figure(figsize=(8, 7))
    ax = fig3.add_subplot(111, projection="3d")
    ax.plot_surface(Xh, Yh, Zh, color="k", alpha=0.35, linewidth=0)
    ax.plot(X, Y, Z, color="C0", lw=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title(r"3D trajectory ($x=r\sin\theta\cos\phi$, $y=r\sin\theta\sin\phi$, $z=r\cos\theta$)")
    lim = jnp.max(jnp.abs(jnp.concatenate([X, Y, Z])))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    fig3.tight_layout()
    plt.show()
