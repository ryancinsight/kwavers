use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::domain::source::array_2d::{
    ApodizationType as KwaversApodizationType, TransducerArray2D as KwaversTransducerArray2D,
    TransducerArray2DConfig,
};

fn parse_apodization_type(apodization: &str) -> PyResult<KwaversApodizationType> {
    match apodization {
        "Rectangular" => Ok(KwaversApodizationType::Rectangular),
        "Hanning" => Ok(KwaversApodizationType::Hanning),
        "Hamming" => Ok(KwaversApodizationType::Hamming),
        "Blackman" => Ok(KwaversApodizationType::Blackman),
        _ => Err(PyValueError::new_err(
            "Apodization must be one of: Rectangular, Hanning, Hamming, Blackman",
        )),
    }
}

fn apodization_to_string(apodization: &KwaversApodizationType) -> String {
    match apodization {
        KwaversApodizationType::Rectangular => "Rectangular".to_string(),
        KwaversApodizationType::Hanning => "Hanning".to_string(),
        KwaversApodizationType::Hamming => "Hamming".to_string(),
        KwaversApodizationType::Blackman => "Blackman".to_string(),
        KwaversApodizationType::Gaussian { sigma } => format!("Gaussian(sigma={})", sigma),
    }
}

#[pyclass]
#[derive(Clone)]
pub struct KWaveArray {
    pub(crate) inner: kwavers::domain::source::kwave_array::KWaveArray,
}

#[pymethods]
impl KWaveArray {
    #[new]
    fn new() -> Self {
        Self {
            inner: kwavers::domain::source::kwave_array::KWaveArray::new(),
        }
    }

    fn set_frequency(&mut self, frequency: f64) {
        self.inner = kwavers::domain::source::kwave_array::KWaveArray::with_params(
            frequency,
            self.inner.frequency(),
        );
    }

    fn set_sound_speed(&mut self, sound_speed: f64) {
        self.inner = kwavers::domain::source::kwave_array::KWaveArray::with_params(
            self.inner.frequency(),
            sound_speed,
        );
    }

    fn add_disc_element(&mut self, position: (f64, f64, f64), diameter: f64) {
        self.inner.add_disc_element(position, diameter);
    }

    #[pyo3(signature = (position, radius, diameter, start_angle=-45.0, end_angle=45.0))]
    fn add_arc_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        self.inner
            .add_arc_element_with_angles(position, radius, diameter, start_angle, end_angle);
    }

    fn add_rect_element(&mut self, position: (f64, f64, f64), dims: (f64, f64, f64)) {
        self.inner
            .add_rect_element(position, dims.0, dims.1, dims.2);
    }

    fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        dims: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.inner
            .add_rect_rot_element(position, dims.0, dims.1, dims.2, euler_xyz_deg);
    }

    fn add_bowl_element(&mut self, position: (f64, f64, f64), radius: f64, diameter: f64) {
        self.inner.add_bowl_element(position, radius, diameter);
    }

    #[getter]
    fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner.get_element_positions()
    }

    fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_focus_delays(focus_point)
    }

    fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_element_delays(focus_point)
    }

    fn get_apodization(&self, window: &str) -> PyResult<Vec<f64>> {
        use kwavers::domain::source::kwave_array::ApodizationWindow;

        let w = match window {
            "Rectangular" => ApodizationWindow::Rectangular,
            "Hann" => ApodizationWindow::Hann,
            "Hamming" => ApodizationWindow::Hamming,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown apodization window '{}'. Choose from: Rectangular, Hann, Hamming",
                    other
                )))
            }
        };
        Ok(self.inner.get_apodization(w))
    }

    fn __repr__(&self) -> String {
        format!("KWaveArray(num_elements={})", self.inner.num_elements())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TransducerArray2D {
    pub(crate) inner: KwaversTransducerArray2D,
    pub(crate) amplitude: f64,
    pub(crate) input_signal: Option<Array1<f64>>,
}

#[pymethods]
impl TransducerArray2D {
    #[new]
    #[pyo3(signature = (number_elements, element_width, element_length, element_spacing, sound_speed, frequency))]
    fn new(
        number_elements: usize,
        element_width: f64,
        element_length: f64,
        element_spacing: f64,
        sound_speed: f64,
        frequency: f64,
    ) -> PyResult<Self> {
        if number_elements == 0 {
            return Err(PyValueError::new_err("Number of elements must be positive"));
        }
        if element_width <= 0.0 {
            return Err(PyValueError::new_err("Element width must be positive"));
        }
        if element_length <= 0.0 {
            return Err(PyValueError::new_err("Element length must be positive"));
        }
        if element_spacing < element_width {
            return Err(PyValueError::new_err(
                "Element spacing must be >= element width",
            ));
        }
        if sound_speed <= 0.0 {
            return Err(PyValueError::new_err("Sound speed must be positive"));
        }
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }

        let config = TransducerArray2DConfig {
            number_elements,
            element_width,
            element_length,
            element_spacing,
            radius: f64::INFINITY,
            center_position: (0.0, 0.0, 0.0),
        };

        let inner = KwaversTransducerArray2D::new(config, sound_speed, frequency).map_err(|e| {
            PyValueError::new_err(format!("Failed to create transducer array: {}", e))
        })?;

        Ok(Self {
            inner,
            amplitude: 1.0,
            input_signal: None,
        })
    }

    fn set_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance <= 0.0 && !distance.is_infinite() {
            return Err(PyValueError::new_err("Focus distance must be positive"));
        }
        self.inner.set_focus_distance(distance);
        Ok(())
    }

    fn set_elevation_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance <= 0.0 && !distance.is_infinite() {
            return Err(PyValueError::new_err(
                "Elevation focus distance must be positive",
            ));
        }
        self.inner.set_elevation_focus_distance(distance);
        Ok(())
    }

    fn set_steering_angle(&mut self, angle: f64) {
        self.inner.set_steering_angle(angle);
    }

    fn set_transmit_apodization(&mut self, apodization: &str) -> PyResult<()> {
        self.inner
            .set_transmit_apodization(parse_apodization_type(apodization)?);
        Ok(())
    }

    fn set_receive_apodization(&mut self, apodization: &str) -> PyResult<()> {
        self.inner
            .set_receive_apodization(parse_apodization_type(apodization)?);
        Ok(())
    }

    fn set_active_elements(&mut self, mask: Vec<bool>) -> PyResult<()> {
        if mask.len() != self.inner.number_elements() {
            return Err(PyValueError::new_err(format!(
                "Mask length {} does not match number of elements {}",
                mask.len(),
                self.inner.number_elements()
            )));
        }
        self.inner
            .set_active_elements(&mask)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.inner.set_center_position((x, y, z));
    }

    fn set_input_signal(&mut self, signal: PyReadonlyArray1<f64>) -> PyResult<()> {
        let signal_arr = signal.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        self.input_signal = Some(signal_arr);
        Ok(())
    }

    #[getter]
    fn number_elements(&self) -> usize {
        self.inner.number_elements()
    }

    #[getter]
    fn element_spacing(&self) -> f64 {
        self.inner.element_spacing()
    }

    #[getter]
    fn aperture_width(&self) -> f64 {
        self.inner.aperture_width()
    }

    #[getter]
    fn element_width(&self) -> f64 {
        self.inner.element_width()
    }

    #[getter]
    fn element_length(&self) -> f64 {
        self.inner.element_length()
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }

    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency()
    }

    #[getter]
    fn focus_distance(&self) -> f64 {
        self.inner.focus_distance()
    }

    #[getter]
    fn steering_angle(&self) -> f64 {
        self.inner.steering_angle()
    }

    #[getter]
    fn transmit_apodization(&self) -> String {
        apodization_to_string(self.inner.transmit_apodization())
    }

    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude
    }

    fn __repr__(&self) -> String {
        format!(
            "TransducerArray2D(elements={}, width={:.2e}m, focus={:.2e}m, steering={:.1} deg)",
            self.inner.number_elements(),
            self.inner.aperture_width(),
            if self.inner.focus_distance().is_infinite() {
                0.0
            } else {
                self.inner.focus_distance()
            },
            self.inner.steering_angle()
        )
    }
}
