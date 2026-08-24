use anyhow::Result;
use leto::Array3;

use super::super::configuration::{alpha_np_m, DX, F_REF, N, SENSOR_FAR, SENSOR_NEAR};
use super::super::measurement::{measure_attenuation, sensor_spectra};
use super::super::model::Layer;
use super::super::propagation::run_pulse;

/// Fat/muscle stack spanning the sensors. The distinct exponents make its
/// integrated attenuation irreducible to one uniform power law.
pub(crate) const LAYERS: [Layer; 4] = [
    Layer {
        name: "fat",
        cells: 150,
        alpha0_db: 0.4,
        gamma: 0.6,
    },
    Layer {
        name: "muscle",
        cells: 150,
        alpha0_db: 0.75,
        gamma: 1.1,
    },
    Layer {
        name: "fat",
        cells: 150,
        alpha0_db: 0.4,
        gamma: 0.6,
    },
    Layer {
        name: "muscle",
        cells: 150,
        alpha0_db: 0.75,
        gamma: 1.1,
    },
];

pub(crate) fn layered_medium(dt: f64, reference: &[(f64, f64)]) -> Result<Vec<(f64, f64, f64)>> {
    let mut alpha_field = Array3::<f64>::zeros([N, 1, 1]);
    let mut gamma_field = Array3::<f64>::from_elem([N, 1, 1], 1.0);
    let mut cursor = SENSOR_NEAR;
    for layer in &LAYERS {
        for index in cursor..(cursor + layer.cells).min(N) {
            alpha_field[[index, 0, 0]] = alpha_np_m(layer.alpha0_db);
            gamma_field[[index, 0, 0]] = layer.gamma;
        }
        cursor += layer.cells;
    }
    assert!(
        cursor <= SENSOR_FAR,
        "layer stack must terminate before the far sensor"
    );

    let (near, far) = run_pulse(&alpha_field, &gamma_field, dt)?;
    let spectra = sensor_spectra(&near, &far, dt)?;
    let separation = (SENSOR_FAR - SENSOR_NEAR) as f64 * DX;

    measure_attenuation(&spectra, reference)?
        .into_iter()
        .map(|(frequency_hz, measured)| {
            let integrated: f64 = LAYERS
                .iter()
                .map(|layer| {
                    alpha_np_m(layer.alpha0_db)
                        * (frequency_hz / F_REF).powf(layer.gamma)
                        * (layer.cells as f64 * DX)
                })
                .sum();
            (frequency_hz, integrated / separation, measured)
        })
        .map(Ok)
        .collect()
}
