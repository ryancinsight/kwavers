//! Module-level mathematical constants and linear-algebra helpers.
//!
//! These are pure functions and constants used by the rasterizer and transform
//! submodules. They carry no state and impose no grid or element dependencies.

/// Oversampling factor: integration points per grid cell along each surface dimension.
pub(super) const DISC_SAMPLE_UPSAMPLING_RATE: f64 = 10.0;
/// Azimuthal packing number: angular samples per radial ring index.
pub(super) const DISC_PACKING_NUMBER: f64 = 7.0;
/// BLI tolerance controlling stencil half-width via `ceil(1 / (π · tol))`.
pub(super) const DISC_BLI_TOLERANCE: f64 = 0.05;
/// Epsilon below which the disc-normal cross-product is considered degenerate.
pub(super) const DISC_AXIS_EPSILON: f64 = 1.0e-12;
/// Golden angle in radians: `2π(1 − 1/φ)` where `φ = (1+√5)/2`.
pub(super) const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_5_f64;

/// Build an intrinsic X-Y-Z Euler rotation matrix (`Rz · Ry · Rx`) from angles
/// in degrees. Matches the upstream k-wave-python
/// `rotation.rotate_rotation_matrix` contract used by `KWaveArray`.
///
/// # Theorem — Intrinsic X-Y-Z Euler composition
///
/// The total rotation is `R = Rz(γ) · Ry(β) · Rx(α)` (right-to-left), where
/// each elementary rotation rotates about its own (body-fixed) axis. This
/// corresponds to the extrinsic Z-Y-X rotation applied left-to-right.
///
/// Reference: Shuster, M.D. (1993). "A survey of attitude representations."
/// J. Astronaut. Sci. 41(4):439–517.
pub(super) fn euler_xyz_rotation_matrix(euler_deg: (f64, f64, f64)) -> [[f64; 3]; 3] {
    let rx = euler_deg.0.to_radians();
    let ry = euler_deg.1.to_radians();
    let rz = euler_deg.2.to_radians();
    let (cx, sx) = (rx.cos(), rx.sin());
    let (cy, sy) = (ry.cos(), ry.sin());
    let (cz, sz) = (rz.cos(), rz.sin());

    let mx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let my = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let mz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    matmul(&matmul(&mz, &my), &mx)
}

/// 3×3 matrix-matrix multiply (row-major, no BLAS dependency).
fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Apply a 3×3 rotation matrix to a 3-vector stored as a tuple.
pub(super) fn apply_matrix(m: &[[f64; 3]; 3], v: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        m[0][0] * v.0 + m[0][1] * v.1 + m[0][2] * v.2,
        m[1][0] * v.0 + m[1][1] * v.1 + m[1][2] * v.2,
        m[2][0] * v.0 + m[2][1] * v.1 + m[2][2] * v.2,
    )
}
