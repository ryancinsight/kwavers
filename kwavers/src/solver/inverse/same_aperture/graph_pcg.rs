//! Graph-H1 regularized preconditioned conjugate gradients.

use super::active_grid::ActiveGrid;
use super::linear_operator::{dot, LinearOperator};

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
    debug_assert_eq!(target.len(), operator.cols());
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
    debug_assert_eq!(data.len(), operator.rows());
    let mut measured = data.to_vec();
    add_deterministic_noise(&mut measured, settings.noise_fraction);
    let mut rhs = vec![0.0; operator.cols()];
    operator.t_matvec(&measured, &mut rhs);
    let mut diag = operator.normal_diag();
    for value in &mut diag {
        *value = (*value + settings.regularization as f32).max(f32::EPSILON);
    }
    let mut x = vec![0.0; operator.cols()];
    let mut hx = vec![0.0; operator.cols()];
    let mut row_workspace = vec![0.0; operator.rows()];
    let mut lap_workspace = vec![0.0; operator.cols()];
    let mut prediction_workspace = vec![0.0; operator.rows()];
    normal_apply(
        operator,
        &x,
        active,
        &settings,
        &mut hx,
        &mut row_workspace,
        &mut lap_workspace,
    );
    let mut r = rhs
        .iter()
        .zip(hx.iter())
        .map(|(b, h)| b - h)
        .collect::<Vec<_>>();
    let mut z = r
        .iter()
        .zip(diag.iter())
        .map(|(rv, dv)| rv / dv)
        .collect::<Vec<_>>();
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
    let mut ap = vec![0.0; operator.cols()];
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
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        z.iter_mut()
            .zip(r.iter().zip(diag.iter()))
            .for_each(|(zv, (rv, dv))| *zv = rv / dv);
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
    for ((dst, xv), lv) in out.iter_mut().zip(x.iter()).zip(lap_workspace.iter()) {
        *dst += settings.regularization as f32 * *xv + settings.smoothness_weight as f32 * *lv;
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
    let rms = (data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
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

fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yv, xv) in y.iter_mut().zip(x.iter()) {
        *yv += alpha * *xv;
    }
}
