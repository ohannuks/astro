from jaxtyping import Array, Complex, Float, Int, jaxtyped
from beartype import beartype
typed = jaxtyped(typechecker=beartype)
type Scalar = Float[Array, ""]
type Vec = Float[Array, "N"]
type Mat = Float[Array, "N N"]
type BScalar = Float[Array, "B"]
type BVec = Float[Array, "B N"]
type BMat = Float[Array, "B N N"]
type CScalar = Complex[Array, ""]
type CVec = Complex[Array, "N"]
# Real / complex arrays that may be scalar or length-N (freq grids).
type ScalarOrVec = Scalar | Vec
type CScalarOrVec = CScalar | CVec
