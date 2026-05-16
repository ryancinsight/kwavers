use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};

use super::super::network::ParamFieldPINNNetwork;
use super::types::TrainingBatch;

/// Compute the **dimensionless** Helmholtz residual on a batch via
/// central finite differences. All seven forward passes share the
/// same autodiff graph so one `backward()` propagates gradients
/// through the network.
///
/// The physical residual `R_phys = ∇²p + k²·p` has units of
/// [Pa/m²] and magnitudes ~10¹⁴ for histotripsy fields — squaring it
/// overflows f32 precision and produces NaN under autodiff. We
/// instead return the dimensionless ratio
///
/// ```text
///   R̂ = R_phys / (k² · p_max_scale)
///      = [∇²p̂ + (k·eps_m)⁻² · (k_eps_m)² · p̂] / (k·eps_m)²
///      = (∑ p̂(±ε̂) − 6·p̂) / (k·eps_m)² + p̂
/// ```
///
/// Both terms are O(1) for Helmholtz-consistent predictions and
/// stay in f32-safe range under squaring + autodiff.
pub(super) fn helmholtz_residual_tensor<B: AutodiffBackend>(
    network: &ParamFieldPINNNetwork<B>,
    batch: &TrainingBatch<B>,
    eps_m: f32,
    c0: f32,
) -> Tensor<B, 1> {
    let device = batch.inputs.device();
    let n = batch.inputs.dims()[0];
    let center_out = network.forward(batch.inputs.clone());
    let p_center = center_out.slice([0..n, 1..2]).reshape([n]);

    let inv_hx = 1.0 / batch.coord_half_m.0;
    let inv_hy = 1.0 / batch.coord_half_m.1;
    let inv_hz = 1.0 / batch.coord_half_m.2;
    let one_hot = |axis: usize, sign: f32, inv_h: f32| -> Tensor<B, 2> {
        let mut data = vec![0.0_f32; n * 5];
        let perturb = sign * eps_m * inv_h;
        for i in 0..n {
            data[i * 5 + axis] = perturb;
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [n, 5]), &device)
    };

    let plus_x = batch.inputs.clone() + one_hot(0, 1.0, inv_hx);
    let minus_x = batch.inputs.clone() + one_hot(0, -1.0, inv_hx);
    let plus_y = batch.inputs.clone() + one_hot(1, 1.0, inv_hy);
    let minus_y = batch.inputs.clone() + one_hot(1, -1.0, inv_hy);
    let plus_z = batch.inputs.clone() + one_hot(2, 1.0, inv_hz);
    let minus_z = batch.inputs.clone() + one_hot(2, -1.0, inv_hz);

    let pick_pmax = |t: Tensor<B, 2>| -> Tensor<B, 1> { t.slice([0..n, 1..2]).reshape([n]) };
    let p_xp = pick_pmax(network.forward(plus_x));
    let p_xm = pick_pmax(network.forward(minus_x));
    let p_yp = pick_pmax(network.forward(plus_y));
    let p_ym = pick_pmax(network.forward(minus_y));
    let p_zp = pick_pmax(network.forward(plus_z));
    let p_zm = pick_pmax(network.forward(minus_z));

    // Sum of finite-difference second-difference contributions across
    // all three axes (still dimensionless, equal to eps_m² · ∇²p̂).
    let lap_sum = p_xp + p_xm + p_yp + p_ym + p_zp + p_zm - p_center.clone().mul_scalar(6.0);

    // Per-sample (k·eps_m)². Tensor of shape [n].
    let k_eps = batch
        .f0_phys_hz
        .clone()
        .mul_scalar(2.0 * std::f32::consts::PI * eps_m / c0);
    let k_eps_sq = k_eps.clone() * k_eps;

    // Dimensionless residual: lap_sum / (k·eps_m)² + p̂.
    // Small floor on (k·eps_m)² prevents divide-by-zero when f0 = 0.
    let safe_k_eps_sq = k_eps_sq.add_scalar(1.0e-12);
    lap_sum / safe_k_eps_sq + p_center
}
