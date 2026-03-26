//! Jones calculus polarization model implementation
//!
//! Applies Jones matrix transformations to 3D polarization fields.

use super::jones_matrix::JonesMatrix;
use super::jones_vector::JonesVector;
use super::PolarizationModel;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use log::debug;
use ndarray::{Array3, Array4};
use num_complex::Complex64;

/// Physically accurate polarization model using Jones calculus
#[derive(Debug)]
pub struct JonesPolarizationModel {
    /// Sequence of Jones matrices representing optical elements
    pub optical_elements: Vec<JonesMatrix>,
    /// Reference wavelength for phase calculations
    pub reference_wavelength: f64,
}

impl JonesPolarizationModel {
    /// Create a new Jones polarization model
    #[must_use]
    pub fn new(reference_wavelength: f64) -> Self {
        Self {
            optical_elements: Vec::new(),
            reference_wavelength,
        }
    }

    /// Add an optical element to the polarization model
    pub fn add_element(&mut self, element: JonesMatrix) {
        self.optical_elements.push(element);
    }

    /// Create a linear polarizer model
    #[must_use]
    pub fn linear_polarizer(axis_angle: f64, _extinction_ratio: f64) -> Self {
        let mut model = Self::new(500e-9);

        let rotation = JonesMatrix::rotation(axis_angle);
        let polarizer = JonesMatrix::horizontal_polarizer();
        let inverse_rotation = JonesMatrix::rotation(-axis_angle);

        let combined = inverse_rotation.multiply(&polarizer).multiply(&rotation);

        model.add_element(combined);
        model
    }

    /// Create a waveplate model
    #[must_use]
    pub fn waveplate(retardance: f64, axis_angle: f64) -> Self {
        let mut model = Self::new(500e-9);

        let rotation = JonesMatrix::rotation(axis_angle);

        let phase = retardance;
        let cos = Complex64::new(phase.cos(), 0.0);
        let sin = Complex64::new(0.0, phase.sin());
        let waveplate = JonesMatrix::new(cos, sin, sin, -cos);

        let inverse_rotation = JonesMatrix::rotation(-axis_angle);

        let combined = inverse_rotation.multiply(&waveplate).multiply(&rotation);

        model.add_element(combined);
        model
    }

    /// Calculate the total Jones matrix for all optical elements
    #[must_use]
    pub fn total_matrix(&self) -> JonesMatrix {
        let mut result = JonesMatrix::identity();

        for element in &self.optical_elements {
            result = element.multiply(&result);
        }

        result
    }
}

impl PolarizationModel for JonesPolarizationModel {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        polarization_state: &mut Array4<Complex64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("Applying Jones calculus polarization transformation");

        let total_matrix = self.total_matrix();

        for i in 0..fluence.shape()[0] {
            for j in 0..fluence.shape()[1] {
                for k in 0..fluence.shape()[2] {
                    let ex = polarization_state[[0, i, j, k]];
                    let ey = polarization_state[[1, i, j, k]];
                    let input_vector = JonesVector::new(ex, ey);

                    let output_vector = total_matrix.apply(&input_vector);

                    polarization_state[[0, i, j, k]] = output_vector.ex;
                    polarization_state[[1, i, j, k]] = output_vector.ey;

                    fluence[[i, j, k]] = output_vector.intensity();
                }
            }
        }
    }
}
