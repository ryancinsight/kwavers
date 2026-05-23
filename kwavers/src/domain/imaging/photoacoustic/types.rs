use crate::core::constants::{GRUNEISEN_WATER_20C, SOUND_SPEED_WATER_SIM};
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::medium::properties::OpticalPropertyData;
use ndarray::{Array2, Array3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavelengthBand {
    pub wavelength_nm: f64,
}

impl WavelengthBand {
    #[must_use]
    pub fn new(wavelength_nm: f64) -> Self {
        Self { wavelength_nm }
    }
}

#[derive(Debug, Clone)]
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
    pub sensors: Option<usize>,
}

impl Default for PhotoacousticParameters {
    fn default() -> Self {
        Self {
            wavelengths: vec![532.0, 650.0, 750.0, 850.0],
            absorption_coefficients: vec![10.0, 5.0, 2.0, 1.0],
            scattering_coefficients: vec![100.0, 80.0, 60.0, 40.0],
            anisotropy_factors: vec![0.9, 0.85, 0.8, 0.75],
            gruneisen_parameters: vec![GRUNEISEN_WATER_20C; 4],
            pulse_duration: 10e-9,
            laser_fluence: 10.0,
            speed_of_sound: SOUND_SPEED_WATER_SIM,
            center_frequency: 5.0 * MHZ_TO_HZ,
            dt: 1e-9,
            sensors: None,
        }
    }
}

impl PhotoacousticParameters {
    #[must_use]
    pub fn detector_count(&self) -> usize {
        self.sensors.unwrap_or(72)
    }
}

#[derive(Debug)]
pub struct PhotoacousticOpticalProperties;

impl PhotoacousticOpticalProperties {
    #[must_use]
    pub fn blood(wavelength: f64) -> OpticalPropertyData {
        let absorption = if wavelength < 600.0 {
            (wavelength - 400.0).mul_add(0.5, 100.0)
        } else {
            (wavelength - 600.0).mul_add(-0.1, 50.0)
        };
        let absorption = absorption.max(0.0_f64);

        OpticalPropertyData {
            absorption_coefficient: absorption,
            scattering_coefficient: 150.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    #[must_use]
    pub fn soft_tissue(wavelength: f64) -> OpticalPropertyData {
        OpticalPropertyData {
            absorption_coefficient: wavelength.mul_add(0.001, 0.1).max(0.0_f64),
            scattering_coefficient: wavelength.mul_add(0.1, 100.0).max(0.0_f64),
            anisotropy: 0.8,
            refractive_index: 1.4,
        }
    }

    #[must_use]
    pub fn tumor(wavelength: f64) -> OpticalPropertyData {
        OpticalPropertyData {
            absorption_coefficient: wavelength.mul_add(0.01, 5.0).max(0.0_f64),
            scattering_coefficient: wavelength.mul_add(0.15, 120.0).max(0.0_f64),
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InitialPressure {
    pub pressure: Array3<f64>,
    pub max_pressure: f64,
    pub fluence: Array3<f64>,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticSignalSet {
    pub sensor_positions: Vec<[f64; 3]>,
    pub sensor_data: Array2<f64>,
    pub sampling_frequency_hz: f64,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticSimulation {
    pub optical_fluence: Array3<f64>,
    pub initial_pressure: InitialPressure,
    pub pressure_fields: Vec<Array3<f64>>,
    pub time_points: Vec<f64>,
    pub signals: PhotoacousticSignalSet,
    pub reconstruction: Array3<f64>,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticValidationReport {
    pub optical_model: String,
    pub wavelength_nm: f64,
    pub stress_confined: bool,
    pub thermal_confined: bool,
    pub total_optical_energy: f64,
    pub max_initial_pressure: f64,
    pub relative_pressure_balance_error: f64,
}
