use anyhow::Result;
use leto::Array3;

use super::super::configuration::{alpha_np_m, ALPHA0_DB, F_REF, GAMMAS, N};
use super::super::measurement::{measure_attenuation, sensor_spectra};
use super::super::model::SweepRow;
use super::super::propagation::run_pulse;

pub(crate) fn homogeneous_sweep(dt: f64, reference: &[(f64, f64)]) -> Result<Vec<SweepRow>> {
    let mut rows = Vec::new();
    for &alpha0_db in &ALPHA0_DB {
        for &gamma in &GAMMAS {
            let alpha0 = alpha_np_m(alpha0_db);
            let alpha_field = Array3::from_elem([N, 1, 1], alpha0);
            let gamma_field = Array3::from_elem([N, 1, 1], gamma);
            let (near, far) = run_pulse(&alpha_field, &gamma_field, dt)?;
            let spectra = sensor_spectra(&near, &far, dt)?;
            for (frequency_hz, measured_np_m) in measure_attenuation(&spectra, reference)? {
                rows.push(SweepRow {
                    alpha0_db,
                    gamma,
                    frequency_hz,
                    prescribed_np_m: alpha0 * (frequency_hz / F_REF).powf(gamma),
                    measured_np_m,
                });
            }
        }
    }
    Ok(rows)
}
