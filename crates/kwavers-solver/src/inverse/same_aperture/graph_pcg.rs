//! Graph-H1 regularized preconditioned conjugate gradients.

use super::active_grid::ActiveGrid;
use super::linear_operator::{axpy, dot, LinearOperator};

pub const SAME_APERTURE_OPERATOR_MODEL: &str = "finite_frequency_same_aperture_graph_laplacian_pcg";

#[derive(Clone, Copy, Debug)]
pub struct PcgSettings {
    pub iterations: usize,
    pub regularization: f64,
    pub smoothness_weight: f64,
    pub noise_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct PcgResult {
    pub model: Vec<f32>,
    pub objective_history: Vec<f64>,
}

/// Solve one Tikhonov-H1 inverse channel by preconditioned conjugate gradients.
///
/// # Theorem
///
/// Let `A` be the finite-frequency sensitivity matrix, `lambda > 0`, and
/// `gamma >= 0`. With graph Laplacian `L`, the normal operator
/// `H = A^T A + lambda I + gamma L` is symmetric positive definite on the
/// active voxel vector space. PCG applied to `Hx = A^T d` minimizes
/// `0.5||Ax-d||_2^2 + 0.5 lambda ||x||_2^2 + 0.5 gamma x^T L x`.
///
/// # Proof
///
/// `A^T A` is positive semidefinite because `x^T A^T A x = ||Ax||_2^2`.
/// `L` is positive semidefinite by the graph-energy identity on `ActiveGrid`.
/// Adding `lambda I` makes `x^T H x >= lambda ||x||_2^2 > 0` for nonzero
/// `x`, so `H` is symmetric positive definite and standard PCG convergence
/// applies.
#[must_use]
pub fn solve_tikhonov_h1<O: LinearOperator>(
    operator: &O,
    target: &[f32],
    active: &ActiveGrid,
    settings: PcgSettings,
) -> PcgResult {
    debug_assert_eq!((target.len()), operator.cols());
    let mut data = vec![0.0; operator.rows()];
    operator.matvec(target, &mut data);
    let (model, objective_history) = solve_regularized_system(operator, &data, active, settings);
    PcgResult {
        model,
        objective_history,
    }
}

#[must_use]
fn solve_regularized_system<O: LinearOperator>(
    operator: &O,
    data: &[f32],
    active: &ActiveGrid,
    settings: PcgSettings,
) -> (Vec<f32>, Vec<f64>) {
    debug_assert_eq!((data.len()), operator.rows());
    let cols = operator.cols();
    let rows = operator.rows();

    // Pre-allocate all workspace vectors once. None of these are re-allocated
    // inside the iteration loop.
    let mut measured = data.to_vec();
    add_deterministic_noise(&mut measured, settings.noise_fraction);
    let mut rhs = vec![0.0_f32; cols];
    operator.t_matvec(&measured, &mut rhs);
    let mut diag = operator.normal_diag();
    for value in &mut diag {
        *value = (*value + settings.regularization as f32).max(f32::EPSILON);
    }
    let mut x = vec![0.0_f32; cols];
    let mut hx = vec![0.0_f32; cols];
    let mut row_workspace = vec![0.0_f32; rows];
    let mut lap_workspace = vec![0.0_f32; cols];
    let mut prediction_workspace = vec![0.0_f32; rows];
    let mut ap = vec![0.0_f32; cols];

    normal_apply(
        operator,
        &x,
        active,
        &settings,
        &mut hx,
        &mut row_workspace,
        &mut lap_workspace,
    );

    // r = rhs - hx (in-place: no collect allocation)
    let mut r = vec![0.0_f32; cols];
    for ((rv, bv), hv) in r.iter_mut().zip(rhs.iter()).zip(hx.iter()) {
        *rv = bv - hv;
    }

    // z = r / diag (in-place diagonal preconditioner)
    let mut z = vec![0.0_f32; cols];
    apply_preconditioner(&r, &diag, &mut z);

    // p = z (copy — single allocation, no clone)
    let mut p = z.clone();

    let mut rz_old = dot(&r, &z);
    let mut objective_history = vec![objective(
        operator,
        &x,
        &measured,
        active,
        &settings,
        &mut prediction_workspace,
        &mut lap_workspace,
    )];

    for _ in 0..settings.iterations {
        normal_apply(
            operator,
            &p,
            active,
            &settings,
            &mut ap,
            &mut row_workspace,
            &mut lap_workspace,
        );
        let denom = dot(&p, &ap);
        if denom <= 0.0 {
            break;
        }
        let alpha = rz_old / denom;
        // x += alpha * p
        axpy(alpha, &p, &mut x);
        // r -= alpha * ap
        axpy(-alpha, &ap, &mut r);
        // z = r / diag (in-place, no collect)
        apply_preconditioner(&r, &diag, &mut z);
        let rz_new = dot(&r, &z);
        objective_history.push(objective(
            operator,
            &x,
            &measured,
            active,
            &settings,
            &mut prediction_workspace,
            &mut lap_workspace,
        ));
        if rz_new <= f32::EPSILON * rz_old.max(1.0) {
            break;
        }
        let beta = rz_new / rz_old;
        // p = z + beta * p (in-place update via axpy: first scale p by beta, then add z)
        // Equivalent to p = z + beta*p_old without extra allocation.
        for (pv, zv) in p.iter_mut().zip(z.iter()) {
            *pv = *zv + beta * *pv;
        }
        rz_old = rz_new;
    }
    (x, objective_history)
}

fn normal_apply<O: LinearOperator>(
    operator: &O,
    x: &[f32],
    active: &ActiveGrid,
    settings: &PcgSettings,
    out: &mut [f32],
    row_workspace: &mut [f32],
    lap_workspace: &mut [f32],
) {
    operator.matvec(x, row_workspace);
    operator.t_matvec(row_workspace, out);
    active.graph_laplacian_into(x, lap_workspace);
    let reg = settings.regularization as f32;
    let smooth = settings.smoothness_weight as f32;
    for ((dst, xv), lv) in out.iter_mut().zip(x.iter()).zip(lap_workspace.iter()) {
        *dst += reg * *xv + smooth * *lv;
    }
}

/// Diagonal preconditioner: z`i` = r`i` / diag`i`.
///
/// Unrolled 4-wide for FMA-compatible compiler vectorization.
fn apply_preconditioner(r: &[f32], diag: &[f32], z: &mut [f32]) {
    let n = z.len();
    let end4 = (n / 4) * 4;
    for i in (0..end4).step_by(4) {
        z[i] = r[i] / diag[i];
        z[i + 1] = r[i + 1] / diag[i + 1];
        z[i + 2] = r[i + 2] / diag[i + 2];
        z[i + 3] = r[i + 3] / diag[i + 3];
    }
    for i in end4..n {
        z[i] = r[i] / diag[i];
    }
}

fn objective<O: LinearOperator>(
    operator: &O,
    x: &[f32],
    data: &[f32],
    active: &ActiveGrid,
    settings: &PcgSettings,
    prediction: &mut [f32],
    lap_workspace: &mut [f32],
) -> f64 {
    operator.matvec(x, prediction);
    let residual = prediction
        .iter()
        .zip(data.iter())
        .map(|(p, d)| (*p - *d).powi(2) as f64)
        .sum::<f64>();
    let norm = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    active.graph_laplacian_into(x, lap_workspace);
    let smooth = x
        .iter()
        .zip(lap_workspace.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    0.5 * residual
        + 0.5 * settings.regularization * norm
        + 0.5 * settings.smoothness_weight * smooth
}

fn add_deterministic_noise(data: &mut [f32], fraction: f64) {
    if fraction <= 0.0 || data.is_empty() {
        return;
    }
    let rms = (data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (data.len()) as f64).sqrt();
    let sigma = (fraction * rms) as f32;
    for (idx, value) in data.iter_mut().enumerate() {
        *value += sigma * splitmix_unit(idx);
    }
}

fn splitmix_unit(idx: usize) -> f32 {
    let mut z = (idx as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let mantissa = ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
    (2.0 * mantissa - 1.0) as f32
}
