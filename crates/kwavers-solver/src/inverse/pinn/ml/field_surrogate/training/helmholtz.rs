use coeus_autograd::Var;

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
pub(super) fn helmholtz_residual_tensor<
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
>(
    network: &ParamFieldPINNNetwork<B>,
    batch: &TrainingBatch<B>,
    eps_m: f32,
    c0: f32,
) -> Var<f32, B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let backend = B::default();
    let n = batch.inputs.tensor.shape()[0];
    let center_out = network.forward(&batch.inputs);
    let p_center = coeus_autograd::reshape(
        &coeus_autograd::slice(&center_out, &[(0, n), (1, 2)]),
        vec![n],
    );

    let inv_hx = 1.0 / batch.coord_half_m.0;
    let inv_hy = 1.0 / batch.coord_half_m.1;
    let inv_hz = 1.0 / batch.coord_half_m.2;
    let one_hot = |axis: usize, sign: f32, inv_h: f32| -> Var<f32, B> {
        let mut data = vec![0.0_f32; n * 5];
        let perturb = sign * eps_m * inv_h;
        for i in 0..n {
            data[i * 5 + axis] = perturb;
        }
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 5], &data, &backend),
            false,
        )
    };

    let plus_x = coeus_autograd::add(&batch.inputs, &one_hot(0, 1.0, inv_hx));
    let minus_x = coeus_autograd::add(&batch.inputs, &one_hot(0, -1.0, inv_hx));
    let plus_y = coeus_autograd::add(&batch.inputs, &one_hot(1, 1.0, inv_hy));
    let minus_y = coeus_autograd::add(&batch.inputs, &one_hot(1, -1.0, inv_hy));
    let plus_z = coeus_autograd::add(&batch.inputs, &one_hot(2, 1.0, inv_hz));
    let minus_z = coeus_autograd::add(&batch.inputs, &one_hot(2, -1.0, inv_hz));

    let pick_pmax = |v: &Var<f32, B>| -> Var<f32, B> {
        coeus_autograd::reshape(&coeus_autograd::slice(v, &[(0, n), (1, 2)]), vec![n])
    };
    let p_xp = pick_pmax(&network.forward(&plus_x));
    let p_xm = pick_pmax(&network.forward(&minus_x));
    let p_yp = pick_pmax(&network.forward(&plus_y));
    let p_ym = pick_pmax(&network.forward(&minus_y));
    let p_zp = pick_pmax(&network.forward(&plus_z));
    let p_zm = pick_pmax(&network.forward(&minus_z));

    // Sum of finite-difference second-difference contributions across
    // all three axes (still dimensionless, equal to eps_m² · ∇²p̂).
    let six_center = coeus_autograd::scalar_mul(&p_center, 6.0);
    let sum_neighbors = coeus_autograd::add(
        &coeus_autograd::add(&p_xp, &p_xm),
        &coeus_autograd::add(
            &coeus_autograd::add(&p_yp, &p_ym),
            &coeus_autograd::add(&p_zp, &p_zm),
        ),
    );
    let lap_sum = coeus_autograd::sub(&sum_neighbors, &six_center);

    // Per-sample (k·eps_m)². Tensor of shape [n].
    let k_eps =
        coeus_autograd::scalar_mul(&batch.f0_phys_hz, 2.0 * std::f32::consts::PI * eps_m / c0);
    let k_eps_sq = coeus_autograd::mul(&k_eps, &k_eps);

    // Dimensionless residual: lap_sum / (k·eps_m)² + p̂.
    // Small floor on (k·eps_m)² prevents divide-by-zero when f0 = 0.
    let safe_k_eps_sq = coeus_autograd::scalar_add(&k_eps_sq, 1.0e-12);
    coeus_autograd::add(&coeus_autograd::div(&lap_sum, &safe_k_eps_sq), &p_center)
}
