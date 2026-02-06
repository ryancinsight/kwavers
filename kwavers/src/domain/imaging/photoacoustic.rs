use crate::domain::medium::properties::OpticalPropertyData;
use ndarray::Array3;

#[derive(Debug)]
pub struct PhotoacousticResult {
    pub pressure_fields: Vec<Array3<f64>>,
    pub time: Vec<f64>,
    pub reconstructed_image: Array3<f64>,
    pub snr: f64,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticParameters {
    pub wavelengths: Vec<f64>,
    pub absorption_coefficients: Vec<f64>,
    pub scattering_coefficients: Vec<f64>,
    pub anisotropy_factors: Vec<f64>,
    pub gruneisen_parameters: Vec<f64>,
    pub pulse_duration: f64,
    pub laser_fluence: f64,
    pub speed_of_sound: f64,
    pub center_frequency: f64,
    pub dt: f64,
}

impl Default for PhotoacousticParameters {
    fn default() -> Self {
        Self {
            wavelengths: vec![532.0, 650.0, 750.0, 850.0],
            absorption_coefficients: vec![10.0, 5.0, 2.0, 1.0],
            scattering_coefficients: vec![100.0, 80.0, 60.0, 40.0],
            anisotropy_factors: vec![0.9, 0.85, 0.8, 0.75],
            gruneisen_parameters: vec![0.12, 0.12, 0.12, 0.12],
            pulse_duration: 10e-9,
            laser_fluence: 10.0,
            speed_of_sound: 1500.0,
            center_frequency: 5e6,
            dt: 1e-9,
        }
    }
}

#[derive(Debug)]
pub struct PhotoacousticOpticalProperties;

impl PhotoacousticOpticalProperties {
    pub fn blood(wavelength: f64) -> OpticalPropertyData {
        let absorption = if wavelength < 600.0 {
            100.0 + (wavelength - 400.0) * 0.5
        } else {
            50.0 + (wavelength - 600.0) * (-0.1)
        };
        let absorption = absorption.max(0.0);

        OpticalPropertyData {
            absorption_coefficient: absorption,
            scattering_coefficient: 150.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    pub fn soft_tissue(wavelength: f64) -> OpticalPropertyData {
        OpticalPropertyData {
            absorption_coefficient: (0.1 + wavelength * 0.001).max(0.0),
            scattering_coefficient: (100.0 + wavelength * 0.1).max(0.0),
            anisotropy: 0.8,
            refractive_index: 1.4,
        }
    }

    pub fn tumor(wavelength: f64) -> OpticalPropertyData {
        OpticalPropertyData {
            absorption_coefficient: (5.0 + wavelength * 0.01).max(0.0),
            scattering_coefficient: (120.0 + wavelength * 0.15).max(0.0),
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }
}

#[derive(Debug)]
pub struct InitialPressure {
    pub pressure: Array3<f64>,
    pub max_pressure: f64,
    pub fluence: Array3<f64>,
}
