pub mod axisymmetric;
mod pml_bypass;
pub mod pressure;
pub mod velocity;

#[inline(always)]
pub(super) fn apply_axis_derivative_update<const FUSED: bool>(
    current: f64,
    derivative_dt: f64,
    pml: f64,
) -> f64 {
    if FUSED {
        pml * (pml * current - derivative_dt)
    } else {
        current - derivative_dt
    }
}
