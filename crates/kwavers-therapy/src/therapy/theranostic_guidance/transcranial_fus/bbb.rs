use leto::{
    Array2,
    Array3,
};

/// Compute BBB opening dose and derived fields for a set of sonication subspots.
///
/// Dose convention (Ch.24): D = MI² × t_on, accumulated with Gaussian focal
/// weighting. Hill function maps dose to permeability.
///
/// # Returns
/// `(dose, permeability, stable_cavitation_prob, inertial_cavitation_risk)`
#[allow(clippy::too_many_arguments)]
pub fn bbb_opening_dose(
    tumor_mask: &Array3<bool>,
    subspot_indices: &Array2<usize>,
    spacing_m: [f64; 3],
    mechanical_index: f64,
    sonication_s: f64,
    duty_cycle: f64,
    focal_radius_m: f64,
    d50: f64,
    hill_n: f64,
) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) {
    let (nx, ny, nz) = tumor_mask.dim();
    let on_time = sonication_s * duty_cycle;
    let subspot_dose_val = mechanical_index * mechanical_index * on_time;
    let radius2 = focal_radius_m * focal_radius_m;
    let n_subspots = subspot_indices.nrows();

    let mut dose = Array3::<f64>::zeros((nx, ny, nz));
    for si in 0..n_subspots {
        let cx = subspot_indices[[si, 0]];
        let cy = subspot_indices[[si, 1]];
        let cz = subspot_indices[[si, 2]];
        for ((ix, iy, iz), d) in dose.indexed_iter_mut() {
            let dx = (ix as f64 - cx as f64) * spacing_m[0];
            let dy = (iy as f64 - cy as f64) * spacing_m[1];
            let dz = (iz as f64 - cz as f64) * spacing_m[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            *d += subspot_dose_val * (-0.5 * d2 / radius2).exp();
        }
    }

    let d50_n = d50.powf(hill_n);
    let stable_low = logistic((mechanical_index - 0.20) / 0.04);
    let stable_high = logistic((0.55 - mechanical_index) / 0.04);
    let inertial_gate = logistic((mechanical_index - 0.55) / 0.04);

    let permeability = dose.mapv(|d| {
        let dn = d.powf(hill_n);
        (dn / (d50_n + dn)) as f32
    });
    let dose_f32 = dose.mapv(|d| d as f32);
    let stable_cavitation = permeability.mapv(|p| (p as f64 * stable_low * stable_high) as f32);
    let inertial_risk = permeability.mapv(|p| (p as f64 * inertial_gate) as f32);

    (dose_f32, permeability, stable_cavitation, inertial_risk)
}

#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
