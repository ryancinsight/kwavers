use aequitas::systems::si::quantities::{Angle, Frequency, Length, Pressure, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, Pascal, Radian};
use leto::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::breast_fwi_bindings::complex_compat::nd_to_leto1;

use kwavers_transducer::array_2d::{
    ArrayCurvature, TransducerArray2D as KwaversTransducerArray2D, TransducerArray2DConfig,
};

use crate::source_py::helpers::{apodization_to_string, parse_apodization_type};

/// 2D transducer array with electronic beam control.
///
/// Mathematical Specification:
/// - Linear array geometry with configurable elements
/// - Electronic steering: time-delay beam steering in azimuthal direction
/// - Electronic focusing: focus at arbitrary depths
/// - Apodization: amplitude weighting (transmit and receive)
///
/// Equivalent to k-Wave's kWaveTransducerSimple and NotATransducer.
///
/// References:
/// - Treeby & Cox (2010) k-Wave toolbox
/// - Szabo (2014) Diagnostic Ultrasound Imaging
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct TransducerArray2D {
    /// Internal kwavers transducer array
    pub(crate) inner: KwaversTransducerArray2D,
    /// Amplitude `Pa` (not in kwavers, added for Python API)
    pub(crate) amplitude: Pressure<f64>,
    /// Input signal (optional, overrides sinusoidal)
    pub(crate) input_signal: Option<Array1<f64>>,
}

#[pymethods]
impl TransducerArray2D {
    /// Create a new 2D transducer array.
    ///
    /// Parameters
    /// ----------
    /// number_elements : int
    ///     Number of elements in the array
    /// element_width : float
    ///     Width of each element `m`
    /// element_length : float
    ///     Length of each element in elevation direction `m`
    /// element_spacing : float
    ///     Spacing between element centers `m`
    /// sound_speed : float
    ///     Speed of sound in medium [m/s]
    /// frequency : float
    ///     Operating frequency `Hz`
    ///
    /// Returns
    /// -------
    /// TransducerArray2D
    ///     Configured transducer array
    ///
    /// Examples
    /// --------
    /// >>> array = TransducerArray2D(
    /// ...     number_elements=32,
    /// ...     element_width=0.3e-3,
    /// ...     element_length=10e-3,
    /// ...     element_spacing=0.5e-3,
    /// ...     sound_speed=1540.0,
    /// ...     frequency=1e6
    /// ... )
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
        if !element_width.is_finite() || element_width <= 0.0 {
            return Err(PyValueError::new_err(
                "Element width must be finite and positive",
            ));
        }
        if !element_length.is_finite() || element_length <= 0.0 {
            return Err(PyValueError::new_err(
                "Element length must be finite and positive",
            ));
        }
        if !element_spacing.is_finite() || element_spacing < element_width {
            return Err(PyValueError::new_err(
                "Element spacing must be finite and >= element width",
            ));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(PyValueError::new_err(
                "Sound speed must be finite and positive",
            ));
        }
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(PyValueError::new_err(
                "Frequency must be finite and positive",
            ));
        }

        let config = TransducerArray2DConfig {
            number_elements,
            element_width: Length::from_unit::<Meter>(element_width),
            element_length: Length::from_unit::<Meter>(element_length),
            element_spacing: Length::from_unit::<Meter>(element_spacing),
            curvature: ArrayCurvature::Flat,
            center_position: [Length::from_unit::<Meter>(0.0); 3],
        };

        let inner = KwaversTransducerArray2D::new(
            config,
            Velocity::from_unit::<MeterPerSecond>(sound_speed),
            Frequency::from_unit::<Hertz>(frequency),
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to create transducer array: {}", e)))?;

        Ok(TransducerArray2D {
            inner,
            amplitude: Pressure::from_unit::<Pascal>(1.0),
            input_signal: None,
        })
    }

    /// Set focus distance `m`.
    ///
    /// Parameters
    /// ----------
    /// distance : float
    ///     Focus distance from array (INF for no focusing)
    #[pyo3(signature = (distance))]
    fn set_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance.is_infinite() {
            self.inner.clear_focus_distance();
        } else if !distance.is_finite() || distance <= 0.0 {
            return Err(PyValueError::new_err(
                "Focus distance must be finite and positive, or infinity to clear",
            ));
        } else {
            self.inner
                .set_focus_distance(Length::from_unit::<Meter>(distance))
                .map_err(PyValueError::new_err)?;
        }
        Ok(())
    }

    /// Set elevation focus distance `m`.
    #[pyo3(signature = (distance))]
    fn set_elevation_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance.is_infinite() {
            self.inner.clear_elevation_focus_distance();
        } else if !distance.is_finite() || distance <= 0.0 {
            return Err(PyValueError::new_err(
                "Elevation focus distance must be finite and positive, or infinity to clear",
            ));
        } else {
            self.inner
                .set_elevation_focus_distance(Length::from_unit::<Meter>(distance))
                .map_err(PyValueError::new_err)?;
        }
        Ok(())
    }

    /// Set steering angle [degrees].
    ///
    /// Parameters
    /// ----------
    /// angle : float
    ///     Steering angle in degrees (0 = straight ahead)
    #[pyo3(signature = (angle))]
    fn set_steering_angle(&mut self, angle: f64) -> PyResult<()> {
        if !angle.is_finite() {
            return Err(PyValueError::new_err("Steering angle must be finite"));
        }
        self.inner
            .set_steering_angle(Angle::from_unit::<Radian>(angle.to_radians()))
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Set transmit apodization type.
    ///
    /// Parameters
    /// ----------
    /// apodization : str
    ///     One of: "Rectangular", "Hanning", "Hamming", "Blackman"
    #[pyo3(signature = (apodization))]
    fn set_transmit_apodization(&mut self, apodization: &str) -> PyResult<()> {
        let apod_type = parse_apodization_type(apodization)?;
        self.inner.set_transmit_apodization(apod_type);
        Ok(())
    }

    /// Set receive apodization type.
    #[pyo3(signature = (apodization))]
    fn set_receive_apodization(&mut self, apodization: &str) -> PyResult<()> {
        let apod_type = parse_apodization_type(apodization)?;
        self.inner.set_receive_apodization(apod_type);
        Ok(())
    }

    /// Set active element mask.
    ///
    /// Parameters
    /// ----------
    /// mask : list[bool]
    ///     Boolean mask of length number_elements
    #[pyo3(signature = (mask))]
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

    /// Set center position.
    #[pyo3(signature = (x, y, z))]
    fn set_position(&mut self, x: f64, y: f64, z: f64) -> PyResult<()> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(PyValueError::new_err("Position coordinates must be finite"));
        }
        self.inner
            .set_center_position([
                Length::from_unit::<Meter>(x),
                Length::from_unit::<Meter>(y),
                Length::from_unit::<Meter>(z),
            ])
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Set input signal (overrides sinusoidal).
    #[pyo3(signature = (signal))]
    fn set_input_signal(&mut self, signal: PyReadonlyArray1<f64>) -> PyResult<()> {
        let signal_arr = signal.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        self.input_signal = Some(nd_to_leto1(signal_arr));
        Ok(())
    }

    /// Get number of elements.
    #[getter]
    fn number_elements(&self) -> usize {
        self.inner.number_elements()
    }

    /// Get element spacing.
    #[getter]
    fn element_spacing(&self) -> f64 {
        self.inner.element_spacing().in_unit::<Meter>()
    }

    /// Get total aperture width.
    #[getter]
    fn aperture_width(&self) -> f64 {
        self.inner.aperture_width().in_unit::<Meter>()
    }

    /// Element width `m`.
    #[getter]
    fn element_width(&self) -> f64 {
        self.inner.element_width().in_unit::<Meter>()
    }

    /// Element length (elevation) `m`.
    #[getter]
    fn element_length(&self) -> f64 {
        self.inner.element_length().in_unit::<Meter>()
    }

    /// Radius of curvature `m`.
    #[getter]
    fn radius(&self) -> f64 {
        self.inner
            .curvature()
            .radius()
            .map_or(f64::INFINITY, |radius| radius.in_unit::<Meter>())
    }

    /// Operating frequency `Hz`.
    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency().in_unit::<Hertz>()
    }

    /// Focus distance `m`.
    #[getter]
    fn focus_distance(&self) -> f64 {
        self.inner
            .focus_distance()
            .map_or(f64::INFINITY, |distance| distance.in_unit::<Meter>())
    }

    /// Steering angle [degrees].
    #[getter]
    fn steering_angle(&self) -> f64 {
        self.inner.steering_angle().in_unit::<Radian>().to_degrees()
    }

    /// Transmit apodization type.
    #[getter]
    fn transmit_apodization(&self) -> String {
        apodization_to_string(self.inner.transmit_apodization())
    }

    /// Amplitude `Pa`.
    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude.in_unit::<Pascal>()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "TransducerArray2D(elements={}, width={:.2e}m, focus={:.2e}m, steering={:.1} deg)",
            self.inner.number_elements(),
            self.inner.aperture_width().in_unit::<Meter>(),
            self.inner
                .focus_distance()
                .map_or(0.0, |distance| distance.in_unit::<Meter>()),
            self.inner.steering_angle().in_unit::<Radian>().to_degrees()
        )
    }
}
