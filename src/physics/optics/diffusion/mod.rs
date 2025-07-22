// physics/optics/diffusion/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::optics::{PolarizationModel, OpticalThermalModel, polarization::SimplePolarizationModel};
use crate::physics::scattering::optic::{OpticalScatteringModel, rayleigh::RayleighOpticalScatteringModel};
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, Zip};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::time::Instant;

pub const LIGHT_IDX: usize = 1;

#[derive(Debug)]
pub struct LightDiffusion {
    pub fluence_rate: Array4<f64>,
    pub emission_spectrum: Array3<f64>,
    polarization: Option<Box<dyn PolarizationModel>>,
    scattering: Option<Box<dyn OpticalScatteringModel>>,
    thermal: Option<OpticalThermalModel>,
    enable_polarization: bool,
    enable_scattering: bool,
    enable_thermal: bool,
    // Performance metrics
    update_time: f64,
    fft_time: f64,
    diffusion_time: f64,
    effect_time: f64,
    call_count: usize,
    // Precomputed arrays for better performance
    d_inv: Option<Array3<f64>>,
}