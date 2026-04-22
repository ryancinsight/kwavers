use crate::Grid;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::heterogeneous::HeterogeneousMedium;
use kwavers::domain::medium::traits::Medium as MediumTrait;
use kwavers::domain::medium::HomogeneousMedium;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Internal enum holding either homogeneous or heterogeneous medium.
#[derive(Clone, Debug)]
pub(crate) enum MediumInner {
    Homogeneous(Box<HomogeneousMedium>),
    Heterogeneous(Box<HeterogeneousMedium>),
}

impl MediumInner {
    /// Get a reference to the inner medium as a trait object.
    pub(crate) fn as_medium(&self) -> &dyn MediumTrait {
        match self {
            MediumInner::Homogeneous(h) => h.as_ref(),
            MediumInner::Heterogeneous(h) => h.as_ref(),
        }
    }
}

/// Acoustic medium with material properties.
///
/// Mathematical Specification:
/// - Sound speed: c(x, y, z) [m/s]
/// - Density: ρ(x, y, z) [kg/m³]
/// - Absorption: α(x, y, z) [dB/(MHz^y·cm)] where y ∈ [0, 3]
/// - Nonlinearity: B/A parameter (optional)
///
/// Supports both homogeneous (uniform) and heterogeneous (spatially varying)
/// acoustic media.
///
/// Equivalent to k-Wave medium struct:
/// ```python
/// # Homogeneous
/// medium = Medium.homogeneous(sound_speed=1500.0, density=1000.0)
///
/// # Heterogeneous
/// c = np.ones((32, 32, 32)) * 1500.0
/// c[16:, :, :] = 2000.0
/// rho = np.ones((32, 32, 32)) * 1000.0
/// medium = Medium(sound_speed=c, density=rho)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct Medium {
    /// Internal medium (homogeneous or heterogeneous)
    pub(crate) inner: MediumInner,
}

#[pymethods]
impl Medium {
    #[new]
    #[pyo3(signature = (sound_speed, density, absorption=None, nonlinearity=None))]
    fn new(
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        absorption: Option<PyReadonlyArray3<f64>>,
        nonlinearity: Option<PyReadonlyArray3<f64>>,
    ) -> PyResult<Self> {
        let c_arr = sound_speed.as_array().to_owned();
        let rho_arr = density.as_array().to_owned();

        let shape = c_arr.shape().to_vec();
        if shape.len() != 3 {
            return Err(PyValueError::new_err("sound_speed must be a 3D array"));
        }
        if rho_arr.shape() != shape.as_slice() {
            return Err(PyValueError::new_err(
                "density shape must match sound_speed shape",
            ));
        }

        if c_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err(
                "All sound_speed values must be positive",
            ));
        }
        if rho_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err("All density values must be positive"));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        // Use acoustic-only constructor: skips 22 non-acoustic zero arrays (~740 MB saved)
        let mut het = HeterogeneousMedium::new_acoustic_only(nx, ny, nz, true);
        het.sound_speed = c_arr;
        het.density = rho_arr;

        if let Some(abs) = absorption {
            let abs_arr = abs.as_array().to_owned();
            if abs_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "absorption shape must match sound_speed shape",
                ));
            }
            het.absorption = abs_arr;
        }

        if let Some(nl) = nonlinearity {
            let nl_arr = nl.as_array().to_owned();
            if nl_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "nonlinearity shape must match sound_speed shape",
                ));
            }
            het.nonlinearity = nl_arr;
        }

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(het)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (sound_speed, density, absorption=0.0, nonlinearity=0.0, alpha_power=1.0, grid=None))]
    fn homogeneous(
        sound_speed: f64,
        density: f64,
        absorption: f64,
        nonlinearity: f64,
        alpha_power: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        if sound_speed <= 0.0 {
            return Err(PyValueError::new_err("Sound speed must be positive"));
        }
        if density <= 0.0 {
            return Err(PyValueError::new_err("Density must be positive"));
        }
        if absorption < 0.0 {
            return Err(PyValueError::new_err("Absorption must be non-negative"));
        }

        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        let mut medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, grid_ref);
        if absorption > 0.0 || nonlinearity > 0.0 {
            medium
                .set_acoustic_properties(absorption, alpha_power, nonlinearity)
                .map_err(|e| {
                    PyValueError::new_err(format!("Invalid acoustic properties: {}", e))
                })?;
        }

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    #[getter]
    fn sound_speed(&self) -> f64 {
        self.inner.as_medium().max_sound_speed()
    }

    #[getter]
    fn density(&self) -> f64 {
        self.inner.as_medium().density(0, 0, 0)
    }

    #[getter]
    fn is_homogeneous(&self) -> bool {
        matches!(self.inner, MediumInner::Homogeneous(_))
    }

    pub(crate) fn __repr__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(h) => {
                let medium = h.as_ref() as &dyn MediumTrait;
                format!(
                    "Medium.homogeneous(sound_speed={:.1}, density={:.1})",
                    medium.max_sound_speed(),
                    medium.density(0, 0, 0)
                )
            }
            MediumInner::Heterogeneous(h) => {
                let medium = h.as_ref() as &dyn MediumTrait;
                let shape = h.sound_speed.shape();
                format!(
                    "Medium(heterogeneous, shape=({}, {}, {}), c_max={:.1})",
                    shape[0],
                    shape[1],
                    shape[2],
                    medium.max_sound_speed()
                )
            }
        }
    }

    fn __str__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(_) => "Homogeneous Medium".to_string(),
            MediumInner::Heterogeneous(h) => {
                let shape = h.sound_speed.shape();
                format!(
                    "Heterogeneous Medium ({}x{}x{})",
                    shape[0], shape[1], shape[2]
                )
            }
        }
    }
}
