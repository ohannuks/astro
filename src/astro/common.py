from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype
typed = jaxtyped(typechecker=beartype)
type Scalar = Float[Array, ""]
type Vec = Float[Array, "N"]
type Mat = Float[Array, "N N"]
type BScalar = Float[Array, "B"]
type BVec = Float[Array, "B N"]
type BMat = Float[Array, "B N N"]
