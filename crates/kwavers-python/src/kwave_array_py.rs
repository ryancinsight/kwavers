use numpy::PyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid_py::Grid;

/// Custom transducer array with mixed element geometries.
///
/// Matches k-wave-python's `KWaveArray` API for building arbitrary transducer
/// arrays from arc, disc, rectangular, and bowl elements. For 3-D disc
/// elements, `focus_position` selects the beam-axis normal; `None` keeps the
/// canonical x-y plane.
///
/// Examples
/// --------
/// >>> arr = KWaveArray()
/// >>> arr.add_disc_element(position=(0.015, 0.015, 0.0), diameter=0.01)
/// >>> source = Source.from_kwave_array(arr, signal)
#[pyclass]
#[derive(Clone)]
pub struct KWaveArray {
    pub(crate) inner: kwavers_domain::source::kwave_array::KWaveArray,
}

#[pymethods]
impl KWaveArray {
    #[new]
    fn new() -> Self {
        KWaveArray {
            inner: kwavers_domain::source::kwave_array::KWaveArray::new(),
        }
    }

    /// Set the operating frequency [Hz].
    fn set_frequency(&mut self, frequency: f64) {
        self.inner.set_frequency(frequency);
    }

    /// Set the sound speed [m/s].
    fn set_sound_speed(&mut self, sound_speed: f64) {
        self.inner.set_sound_speed(sound_speed);
    }

    /// Add a disc-shaped element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Element center [x, y, z] in meters
    /// diameter : float
    ///     Disc diameter [m]
    /// focus_position : tuple[float, float, float], optional
    ///     Optional point on the beam axis defining the disc normal
    #[pyo3(signature = (position, diameter, focus_position=None))]
    fn add_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) -> PyResult<()> {
        if matches!(focus_position, Some(focus) if focus == position) {
            return Err(PyValueError::new_err(
                "focus_position must differ from position for a 3D disc",
            ));
        }
        self.inner
            .add_disc_element(position, diameter, focus_position);
        Ok(())
    }

    /// Generate a binary mask on a computational grid.
    fn get_array_binary_mask<'py>(
        &self,
        py: Python<'py>,
        grid: &Grid,
    ) -> PyResult<Py<PyArray3<bool>>> {
        Ok(PyArray3::from_owned_array(py, self.inner.get_array_binary_mask(&grid.inner)).into())
    }

    /// Generate a weighted mask on a computational grid.
    fn get_array_weighted_mask<'py>(
        &self,
        py: Python<'py>,
        grid: &Grid,
    ) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_owned_array(py, self.inner.get_array_weighted_mask(&grid.inner)).into())
    }

    /// Add an arc-shaped element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Arc center [x, y, z] in meters
    /// radius : float
    ///     Arc radius [m]
    /// diameter : float
    ///     Arc aperture diameter [m]
    /// start_angle : float, optional
    ///     Start angle in degrees (default: -45)
    /// end_angle : float, optional
    ///     End angle in degrees (default: 45)
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

    /// Add a rectangular element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Center position [x, y, z] in meters
    /// dims : tuple[float, float, float]
    ///     Dimensions [width, height, length] in meters
    fn add_rect_element(&mut self, position: (f64, f64, f64), dims: (f64, f64, f64)) {
        self.inner
            .add_rect_element(position, dims.0, dims.1, dims.2);
    }

    /// Add a rectangular element rotated about its center by intrinsic X-Y-Z
    /// Euler angles (degrees). Matches the upstream k-wave-python
    /// ``KWaveArray.add_rect_element`` rotation used by the
    /// ``at_linear_array_transducer`` example.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Center position [x, y, z] in meters.
    /// dims : tuple[float, float, float]
    ///     Dimensions [width, height, length] in meters.
    /// euler_xyz_deg : tuple[float, float, float]
    ///     Intrinsic X-Y-Z Euler angles in degrees.
    fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        dims: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.inner
            .add_rect_rot_element(position, dims.0, dims.1, dims.2, euler_xyz_deg);
    }

    /// Install a global translation + intrinsic X-Y-Z Euler rotation (degrees)
    /// applied to every element at rasterization time. Mirrors
    /// ``kWaveArray.set_array_position`` in k-wave-python.
    fn set_array_position(&mut self, translation: (f64, f64, f64), euler_xyz_deg: (f64, f64, f64)) {
        self.inner.set_array_position(translation, euler_xyz_deg);
    }

    /// Remove the global array transform if one was previously installed.
    fn clear_array_position(&mut self) {
        self.inner.clear_array_position();
    }

    /// Add a bowl-shaped element (focused transducer).
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Bowl center position [x, y, z] in meters
    /// radius : float
    ///     Radius of curvature [m]
    /// diameter : float
    ///     Bowl aperture diameter [m]
    fn add_bowl_element(&mut self, position: (f64, f64, f64), radius: f64, diameter: f64) {
        self.inner.add_bowl_element(position, radius, diameter);
    }

    /// Add a single annular (spherical-ring) element.
    ///
    /// Mirrors k-wave-python's `kWaveArray.add_annular_element`: a spherical
    /// cap between `inner_diameter` and `outer_diameter` apertures on a bowl
    /// of curvature `radius`.
    fn add_annular_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        self.inner
            .add_annular_element(position, radius, inner_diameter, outer_diameter);
    }

    /// Add a concentric annular array — one `ElementShape::Annulus` per
    /// `(inner_diameter, outer_diameter)` pair, all sharing `position` and
    /// `radius`. Mirrors `kWaveArray.add_annular_array`.
    fn add_annular_array(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameters: Vec<(f64, f64)>,
    ) {
        self.inner.add_annular_array(position, radius, &diameters);
    }

    /// Number of elements in the array.
    #[getter]
    fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    /// Get element centroid positions as list of (x, y, z) tuples.
    fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner.get_element_positions()
    }

    /// Compute focus delays [s] for each element to focus at a point.
    fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_focus_delays(focus_point)
    }

    /// Compute time delays [s] for electronic focusing at a point.
    ///
    /// Returns per-element delays such that `τᵢ = (d_max − dᵢ) / c`, where
    /// `dᵢ` is the distance from element `i` to `focus_point` and `d_max = max(dᵢ)`.
    /// The farthest element has delay 0; closer elements are delayed so all wavefronts
    /// arrive at the focus simultaneously.
    ///
    /// Parameters
    /// ----------
    /// focus_point : tuple[float, float, float]
    ///     Focus position (x, y, z) in metres
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Per-element delays in seconds. All values ≥ 0.
    ///
    /// Reference: Selfridge et al. (1980) Appl. Phys. Lett. 37(1):35–36.
    fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_element_delays(focus_point)
    }

    /// Compute per-element amplitude weights for array apodization.
    ///
    /// Parameters
    /// ----------
    /// window : str
    ///     Window type: ``"Rectangular"`` (uniform), ``"Hann"``, or ``"Hamming"``
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Apodization weights in ``[0, 1]``, one per element.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `window` is not one of the recognised strings.
    ///
    /// Reference: Harris (1978) Proc. IEEE 66(1):51–83.
    fn get_apodization(&self, window: &str) -> PyResult<Vec<f64>> {
        use kwavers_domain::source::kwave_array::KwaveApodizationWindow;
        let w = match window {
            "Rectangular" => KwaveApodizationWindow::Rectangular,
            "Hann" => KwaveApodizationWindow::Hann,
            "Hamming" => KwaveApodizationWindow::Hamming,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown apodization window '{}'. Choose from: Rectangular, Hann, Hamming",
                    other
                )));
            }
        };
        Ok(self.inner.get_apodization(w))
    }

    fn __repr__(&self) -> String {
        format!("KWaveArray(num_elements={})", self.inner.num_elements())
    }
}
