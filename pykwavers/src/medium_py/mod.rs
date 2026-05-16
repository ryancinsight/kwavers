mod properties;

use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::heterogeneous::{HeterogeneousFactory, HeterogeneousMedium};
use kwavers::domain::medium::traits::Medium as MediumTrait;
use kwavers::domain::medium::HomogeneousMedium;

use crate::grid_py::Grid;

#[derive(Clone, Debug)]
pub(crate) enum MediumInner {
    Homogeneous(Box<HomogeneousMedium>),
    Heterogeneous(Box<HeterogeneousMedium>),
}

impl MediumInner {
    pub(crate) fn as_medium(&self) -> &dyn MediumTrait {
        match self {
            MediumInner::Homogeneous(h) => h.as_ref(),
            MediumInner::Heterogeneous(h) => h.as_ref(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Medium {
    pub(crate) inner: MediumInner,
}

#[pymethods]
impl Medium {
    /// Create a heterogeneous medium from 3D arrays.
    #[new]
    #[pyo3(signature = (sound_speed, density, absorption=None, alpha_power=None, nonlinearity=None))]
    fn new(
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        absorption: Option<PyReadonlyArray3<f64>>,
        alpha_power: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
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

        if let Some(py_ap) = alpha_power {
            use ndarray::Array3 as A3;
            if let Ok(scalar) = py_ap.extract::<f64>() {
                het.alpha_power = A3::from_elem((nx, ny, nz), scalar);
            } else if let Ok(arr) = py_ap.extract::<PyReadonlyArray3<f64>>() {
                let ap_arr = arr.as_array().to_owned();
                if ap_arr.shape() != [nx, ny, nz] {
                    return Err(PyValueError::new_err(
                        "alpha_power shape must match sound_speed shape",
                    ));
                }
                het.alpha_power = ap_arr;
            } else {
                return Err(PyValueError::new_err(
                    "alpha_power must be a float or a 3D ndarray matching sound_speed shape",
                ));
            }
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

    /// Create a homogeneous medium with uniform properties.
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

    /// Create a homogeneous elastic medium parameterised by physical wave speeds.
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, grid=None))]
    fn elastic(
        c_compression: f64,
        c_shear: f64,
        density: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        let medium =
            HomogeneousMedium::elastic_homogeneous(density, c_compression, c_shear, grid_ref)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "Invalid elastic parameters. Requirements: density > 0, \
                 c_compression > 0, c_shear ≥ 0, 2·c_shear² ≤ c_compression². \
                 (Stability bound: ν ≥ 0; recovers fluid medium when c_shear = 0.)",
                    )
                })?;

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    /// Create a heterogeneous elastic medium from per-voxel wave-speed and density arrays.
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, reference_frequency=1.0e6))]
    fn elastic_heterogeneous(
        c_compression: PyReadonlyArray3<f64>,
        c_shear: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        reference_frequency: f64,
    ) -> PyResult<Self> {
        let cp = c_compression.as_array();
        let cs = c_shear.as_array();
        let rho = density.as_array();

        let medium = HeterogeneousFactory::from_elastic_arrays(cp, cs, rho, reference_frequency)
            .map_err(PyValueError::new_err)?;

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(medium)),
        })
    }
}
