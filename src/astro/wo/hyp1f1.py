"""Complex confluent hypergeometric ₁F₁ for JAX.

JAX's ``jax.scipy.special.hyp1f1`` is real-only; this is a power-series
fallback that accepts complex ``a``, ``b``, ``z``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def hyp1f1(a, b, z, tol=1e-13, maxn=5000):
    """₁F₁(a; b; z) by power series (scalar arguments)."""

    def body(carry):
        k, term, s = carry
        term = term * (a + k - 1) / (b + k - 1) * z / k
        return k + 1, term, s + term

    def cond(carry):
        k, term, s = carry
        return (k <= maxn) & (jnp.abs(term) >= tol * jnp.maximum(1.0, jnp.abs(s)))

    one = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
    _, _, s = jax.lax.while_loop(cond, body, (1, one, one))
    return s
