//! # Differential Operators
//!
//! Kwavers owns no first-derivative stencils. Central differences at 2nd, 4th
//! and 6th order, the Yee staggered forward/backward pair, and the
//! arbitrary-even-order staggered gradient/divergence pair are Leto's, as
//! [`leto_ops::FiniteDifference3D`] and [`leto_ops::StaggeredLeapfrog3D`]; the
//! derived tap coefficients are
//! [`leto_ops::staggered_first_derivative_coefficients`] and
//! [`leto_ops::central_first_derivative_coefficients`]. A stencil needed here
//! and missing there is added there.
//!
//! What remains in this module is the operator Leto does not own: the
//! summation-by-parts family, whose boundary blocks are derived against a
//! per-axis norm rather than a single interior stencil, and whose contract is
//! the discrete energy estimate rather than a pointwise truncation order.
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//! - Shubin, G. R., & Bell, J. B. (1987). "A modified equation approach to
//!   constructing fourth order methods for acoustic wave propagation."
//!   *SIAM Journal on Scientific and Statistical Computing*, 8(2), 135-151.

pub mod summation_by_parts;
