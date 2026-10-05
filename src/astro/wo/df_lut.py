"""Fast PML amplification: lookup table + asymptotic regime switching.

Keeps the mess out of ``pml.py``. Table is built on first use (SciPy/NumPy
series; JAX has no complex ₁F₁). Stores F directly (not dF): at small y,
Fgeo∼y^{-1/2} so interpolating dF cancels badly.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from astro.wo import pml

DATA_PATH = Path(__file__).resolve().parent / "data" / "df_lut.npz"

# LUT box (intermediate wave-optics strip).
OMEGA_MAX = 5.0
Y_MAX = 3.0
OMEGA_MIN = 1e-4
Y_MIN = 1e-6

# Regime thresholds for F() below.
WAVE_OMEGA_MAX = 1e-4  # ω → 0 ⇒ F → 1
GEO_PHASE_THRESH = 8.0  # ω ΔT ≳ this ⇒ Fgeo
SERIES_Z_MAX = 25.0  # |z|=ω y²/2; beyond, series overflows → Fgeo
SERIES_PREF_Z = 2.0  # prefer pml.F over LUT when |z| small


def _hyp1f1_series(a: complex, b: complex, z: complex, tol: float = 1e-13, maxn: int = 5000) -> complex:
    term = 1.0 + 0.0j
    s = term
    for k in range(1, maxn + 1):
        term *= (a + k - 1) / (b + k - 1) * z / k
        s += term
        if abs(term) < tol * max(1.0, abs(s)):
            return s
    return s


def _phi_m_np(y: float) -> float:
    x_m = 0.5 * (y + np.sqrt(y * y + 4.0))
    return 0.5 * (x_m - y) ** 2 - np.log(x_m)


def _delta_T_np(y: float) -> float:
    s = np.sqrt(y * y + 4.0)
    return 0.5 * y * s + np.log((s + y) / (s - y))


def _F_wave_np(omega: float, y: float) -> complex:
    from scipy.special import gamma

    exp = np.exp(np.pi * omega / 4.0 + 1j * omega / 2.0 * (np.log(omega / 2.0) - 2.0 * _phi_m_np(y)))
    return complex(
        exp * gamma(1.0 - 1j * omega / 2.0) * _hyp1f1_series(1j * omega / 2.0, 1.0, 1j * omega * y * y / 2.0)
    )


def _F_geo_np(omega: float, y: float) -> complex:
    s = np.sqrt(y * y + 4.0)
    mu_p = 0.5 + (y * y + 2.0) / (2.0 * y * s)
    mu_m = 0.5 - (y * y + 2.0) / (2.0 * y * s)
    dT = _delta_T_np(y)
    return np.sqrt(abs(mu_p)) - 1j * np.sqrt(abs(mu_m)) * np.exp(1j * omega * dT)


def build_df_lut(
    path: Path | None = None,
    n_omega: int = 512,
    n_y: int = 320,
    omega_min: float = OMEGA_MIN,
    omega_max: float = OMEGA_MAX,
    y_min: float = Y_MIN,
    y_max: float = Y_MAX,
) -> Path:
    """Build F(ω, y) table with NumPy/SciPy and save .npz."""
    path = Path(path) if path is not None else DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    omega = np.geomspace(omega_min, omega_max, n_omega)
    y = np.geomspace(y_min, y_max, n_y)
    F = np.empty((n_omega, n_y), dtype=np.complex128)
    for i, om in enumerate(omega):
        for j, yj in enumerate(y):
            F[i, j] = _F_wave_np(float(om), float(yj))

    np.savez(
        path,
        omega=omega.astype(np.float64),
        y=y.astype(np.float64),
        F_re=F.real,
        F_im=F.imag,
        omega_min=np.array(omega_min, dtype=np.float64),
        omega_max=np.array(omega_max, dtype=np.float64),
        y_min=np.array(y_min, dtype=np.float64),
        y_max=np.array(y_max, dtype=np.float64),
    )
    return path


def _F_bilinear(
    omega,
    y,
    omega_grid,
    y_grid,
    F_re,
    F_im,
    omega_min: float,
    omega_max: float,
    y_min: float,
    y_max: float,
):
    """Bilinear interp of F; 0 outside the LUT rectangle (caller must mask)."""
    inside = (omega >= omega_min) & (omega <= omega_max) & (y >= y_min) & (y <= y_max)
    om = jnp.clip(omega, omega_min, omega_max)
    yy = jnp.clip(y, y_min, y_max)

    i = jnp.searchsorted(omega_grid, om) - 1
    j = jnp.searchsorted(y_grid, yy) - 1
    i = jnp.clip(i, 0, omega_grid.shape[0] - 2)
    j = jnp.clip(j, 0, y_grid.shape[0] - 2)

    om0 = omega_grid[i]
    om1 = omega_grid[i + 1]
    y0 = y_grid[j]
    y1 = y_grid[j + 1]
    tx = (om - om0) / (om1 - om0)
    ty = (yy - y0) / (y1 - y0)

    def cell(re_or_im):
        f00 = re_or_im[i, j]
        f10 = re_or_im[i + 1, j]
        f01 = re_or_im[i, j + 1]
        f11 = re_or_im[i + 1, j + 1]
        return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11

    F = cell(F_re) + 1j * cell(F_im)
    return jnp.where(inside, F, jnp.zeros_like(F))


class DFLookup:
    """JAX bilinear interpolator for complex F on a rectilinear (ω, y) grid."""

    def __init__(self, path: Path | None = None):
        path = Path(path) if path is not None else DATA_PATH
        if not path.is_file():
            build_df_lut(path)
        data = np.load(path)
        self.omega = jnp.asarray(data["omega"])
        self.y = jnp.asarray(data["y"])
        # Backward compatible with older dF_re/dF_im tables.
        self._legacy_dF = "F_re" not in data.files
        if self._legacy_dF:
            self.F_re = jnp.asarray(data["dF_re"])
            self.F_im = jnp.asarray(data["dF_im"])
        else:
            self.F_re = jnp.asarray(data["F_re"])
            self.F_im = jnp.asarray(data["F_im"])
        self.omega_min = float(np.asarray(data["omega_min"]))
        self.omega_max = float(np.asarray(data["omega_max"]))
        self.y_min = float(np.asarray(data["y_min"]))
        self.y_max = float(np.asarray(data["y_max"]))

    def F(self, omega, y):
        return _F_bilinear(
            omega,
            y,
            self.omega,
            self.y,
            self.F_re,
            self.F_im,
            self.omega_min,
            self.omega_max,
            self.y_min,
            self.y_max,
        )

    def __call__(self, omega, y):
        """Return dF = F_lut − Fgeo (or legacy interpolated dF)."""
        if self._legacy_dF:
            return _F_bilinear(
                omega,
                y,
                self.omega,
                self.y,
                self.F_re,
                self.F_im,
                self.omega_min,
                self.omega_max,
                self.y_min,
                self.y_max,
            )
        return self.F(omega, y) - pml.Fgeo(omega, y)


_default_lut: DFLookup | None = None


def get_df_lut(path: Path | None = None) -> DFLookup:
    global _default_lut
    if path is not None:
        return DFLookup(path)
    if _default_lut is None:
        _default_lut = DFLookup()
    return _default_lut


def F_lut(omega, y):
    """Interpolated wave-optics F(ω, y) from the LUT (0 outside box)."""
    return get_df_lut().F(omega, y)


def dF(omega, y):
    """Interpolated residual dF(ω, y) = F_lut − Fgeo."""
    return get_df_lut()(omega, y)


def in_wave_limit(omega):
    """Deep wave-optics: ω → 0 ⇒ F → 1."""
    return omega <= WAVE_OMEGA_MAX


def in_geo_limit(omega, y):
    """Geometric optics: large image phase ω ΔT."""
    return omega * pml.Deltat(y) >= GEO_PHASE_THRESH


def _in_lut(omega, y):
    lut = get_df_lut()
    return (
        (omega >= lut.omega_min)
        & (omega <= lut.omega_max)
        & (y >= lut.y_min)
        & (y <= lut.y_max)
    )


def _F_series_safe(omega, y):
    """``pml.F`` with a guard when |z| is too large for the ₁F₁ series."""
    z_abs = 0.5 * omega * y * y
    om_s = jnp.where(z_abs < SERIES_Z_MAX, jnp.maximum(omega, 1e-300), 1.0)
    return jnp.where(z_abs < SERIES_Z_MAX, pml.F(om_s, y), pml.Fgeo(omega, y))


def F(omega, y):
    """Fast amplification factor: wave/geo limits + LUT + ``pml.F`` fallback.

    * ω ≤ WAVE_OMEGA_MAX → 1
    * ω ΔT ≥ GEO_PHASE_THRESH → ``pml.Fgeo``
    * else inside LUT and |z| ≥ SERIES_PREF_Z → table
    * else → ``pml.F`` (series), or ``pml.Fgeo`` if |z| too large
    """
    omega = jnp.asarray(omega, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    wave = in_wave_limit(omega)
    geo = in_geo_limit(omega, y)
    z_abs = 0.5 * omega * y * y
    lut = _in_lut(omega, y) & (~wave) & (~geo) & (z_abs >= SERIES_PREF_Z)

    F_w = jnp.ones_like(omega, dtype=jnp.float64) + 0.0j
    F_g = pml.Fgeo(omega, y)
    F_l = F_lut(omega, y)
    F_s = _F_series_safe(omega, y)

    mid = jnp.where(lut, F_l, F_s)
    return jnp.where(wave, F_w, jnp.where(geo, F_g, mid))


if __name__ == "__main__":
    out = build_df_lut()
    print(f"wrote {out}")
