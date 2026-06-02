/// Optimal central-difference step for f32 computation (dimensionless).
///
/// ## Theorem — Optimal FD Step (Gill, Murray & Wright 1981, §8.2)
///
/// For a central-difference approximation of the first derivative,
///   f'(x) ≈ [f(x+h) − f(x−h)] / (2h)
/// the error has two competing terms:
/// - **Truncation error**: O(h²) — decreases with smaller h
/// - **Cancellation error**: O(ε_mach / h) — increases with smaller h
///
/// The step that minimises total error satisfies d/dh [h² + ε/h] = 0, giving:
///   h_opt = ε^(1/3)
///
/// For f32 (24-bit mantissa, ε ≈ 1.19e-7):
///   h_opt = (1.19e-7)^(1/3) ≈ 4.9e-3
///
/// The previously used step `(f32::EPSILON).sqrt() * 1e-2 ≈ 3.45e-6` is ~1400× too small,
/// causing catastrophic cancellation in the numerator `f(x+h) − f(x−h)`.
///
/// For second derivatives `f''(x) ≈ [f(x+h) − 2f(x) + f(x−h)] / h²` the optimal
/// step is h = ε^(1/4) ≈ 1.85e-2, but EPS_FD_F32 = 4.9e-3 is a safe conservative
/// choice that works for both first and second derivatives in f32.
///
/// Reference: Gill, P.E., Murray, W. & Wright, M.H. (1981).
/// *Practical Optimization*. Academic Press. §8.2, Eq. 8.6.
pub const EPS_FD_F32: f32 = 4.9e-3;
