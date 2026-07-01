use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn build_absorption_kernels(
    has_absorption: bool,
    absorption: Option<&PyReadonlyArray3<f64>>,
    c0_flat: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    alpha_power: f64,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
    use std::f64::consts::PI;

    let total = nx * ny * nz;
    if !has_absorption {
        return Ok((
            vec![0.0f32; total],
            vec![0.0f32; total],
            vec![0.0f32; total],
            vec![0.0f32; total],
        ));
    }

    let ab_arr = absorption
        .expect("invariant: absorption array exists when has_absorption is true")
        .as_array();
    if (alpha_power - 1.0).abs() < 1e-12 && ab_arr.iter().any(|&v| v > 0.0) {
        return Err(PyValueError::new_err(
            "alpha_power must not be 1.0 for fractional Laplacian absorption",
        ));
    }

    let dk_x = 2.0 * PI / (nx as f64 * dx);
    let dk_y = 2.0 * PI / (ny as f64 * dy);
    let dk_z = 2.0 * PI / (nz as f64 * dz);
    let singularity_thresh: f64 = 1e-8;

    let mut n1 = vec![0.0f32; total];
    let mut n2 = vec![0.0f32; total];
    let mut tau_v = vec![0.0f32; total];
    let mut eta_v = vec![0.0f32; total];

    for flat in 0..total {
        let ix = flat / (ny * nz);
        let iy = (flat % (ny * nz)) / nz;
        let iz = flat % nz;

        let kix = if ix <= nx / 2 {
            ix as f64
        } else {
            (nx - ix) as f64
        } * dk_x;
        let kiy = if iy <= ny / 2 {
            iy as f64
        } else {
            (ny - iy) as f64
        } * dk_y;
        let kiz = if iz <= nz / 2 {
            iz as f64
        } else {
            (nz - iz) as f64
        } * dk_z;
        let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

        if k_mag > singularity_thresh {
            n1[flat] = k_mag.powf(alpha_power - 2.0) as f32;
            n2[flat] = k_mag.powf(alpha_power - 1.0) as f32;
        }

        let alpha_db_cm = ab_arr[[ix, iy, iz]];
        let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
        let c0_local = c0_flat[flat] as f64;
        tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(alpha_power - 1.0)) as f32;
        eta_v[flat] =
            (2.0 * alpha_0_si * c0_local.powf(alpha_power) * (PI * alpha_power / 2.0).tan()) as f32;
    }

    Ok((n1, n2, tau_v, eta_v))
}
