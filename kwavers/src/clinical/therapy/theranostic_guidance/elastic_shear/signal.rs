//! Velocity source waveform construction for elastic shear propagation.

use crate::solver::forward::pstd::extensions::{ElasticPstdSourceMode, ElasticPstdVelocitySource};
use ndarray::{Array1, Array2, Array3};

const LINEAR_STRAIN_SOURCE_AMPLITUDE: f64 = 1.0e-5;
const TONE_BURST_CYCLES: f64 = 2.0;

pub(super) fn velocity_source(
    source_mask_2d: &Array2<bool>,
    shape_2d: (usize, usize),
    time_steps: usize,
    dt_s: f64,
    frequency_hz: f64,
    shear_speed_m_s: f64,
) -> ElasticPstdVelocitySource {
    let mut mask = Array3::<bool>::from_elem((shape_2d.0, shape_2d.1, 1), false);
    for ((ix, iy), active) in source_mask_2d.indexed_iter() {
        mask[[ix, iy, 0]] = *active;
    }
    let duration_s = TONE_BURST_CYCLES / frequency_hz;
    let amplitude = LINEAR_STRAIN_SOURCE_AMPLITUDE * shear_speed_m_s;
    let signal = Array1::from_iter((0..time_steps).map(|step| {
        let t = step as f64 * dt_s;
        if t > duration_s {
            0.0
        } else {
            let phase = std::f64::consts::TAU * frequency_hz * t;
            let window = 0.5 * (1.0 - (std::f64::consts::TAU * t / duration_s).cos());
            amplitude * window * phase.sin()
        }
    }));
    ElasticPstdVelocitySource {
        mask,
        ux: None,
        uy: None,
        uz: Some(signal),
        mode: ElasticPstdSourceMode::Additive,
    }
}
