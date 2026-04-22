use kwavers::domain::grid::GridDimensions;
use kwavers::domain::imaging::photoacoustic::{
    IlluminationGeometry, MonteCarloModelConfig, OpticalModel, PhotoacousticAcousticConfig,
    PhotoacousticScenario as KwaversPhotoacousticScenario, PhotoacousticSolverConfig,
    ThermoelasticProperties,
};
use kwavers::domain::medium::optical_map::OpticalPropertyMap;
use kwavers::domain::medium::properties::OpticalPropertyData;
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::runner::PhotoacousticOpticalModel;
use crate::bindings::common::kwavers_error_to_py;
use crate::Grid;

#[pyclass]
#[derive(Clone)]
pub struct PhotoacousticScenario {
    pub(crate) inner: KwaversPhotoacousticScenario,
}

#[pymethods]
impl PhotoacousticScenario {
    #[new]
    #[pyo3(signature = (
        grid,
        wavelength_nm,
        sensor_positions_m,
        *,
        optical_model=PhotoacousticOpticalModel::Diffusion,
        incident_fluence_j_m2=10.0,
        pulse_duration_s=8e-9,
        speed_of_sound_m_s=1500.0,
        cfl_factor=0.2,
        num_time_steps=32,
        snapshot_interval=4,
        gruneisen=0.12,
        density_kg_m3=1000.0,
        specific_heat_j_kgk=4180.0,
        thermal_conductivity_w_mk=0.6,
        absorption_coefficient=0.5,
        scattering_coefficient=120.0,
        anisotropy=0.9,
        refractive_index=1.4,
        illumination_origin_m=(0.0, 0.0, 0.0),
        illumination_direction=(0.0, 0.0, 1.0),
        photon_count=5000
    ))]
    fn new(
        grid: Grid,
        wavelength_nm: f64,
        sensor_positions_m: Vec<(f64, f64, f64)>,
        optical_model: PhotoacousticOpticalModel,
        incident_fluence_j_m2: f64,
        pulse_duration_s: f64,
        speed_of_sound_m_s: f64,
        cfl_factor: f64,
        num_time_steps: usize,
        snapshot_interval: usize,
        gruneisen: f64,
        density_kg_m3: f64,
        specific_heat_j_kgk: f64,
        thermal_conductivity_w_mk: f64,
        absorption_coefficient: f64,
        scattering_coefficient: f64,
        anisotropy: f64,
        refractive_index: f64,
        illumination_origin_m: (f64, f64, f64),
        illumination_direction: (f64, f64, f64),
        photon_count: usize,
    ) -> PyResult<Self> {
        if sensor_positions_m.is_empty() {
            return Err(PyValueError::new_err(
                "sensor_positions_m must not be empty",
            ));
        }
        // `gruneisen` is accepted for API continuity; the kwavers core now
        // sources the Gruneisen coefficient internally via `GrueneisenModel`.
        let _ = gruneisen;

        let dims = GridDimensions::from_grid(&grid.inner);
        let optical_props = OpticalPropertyData {
            absorption_coefficient,
            scattering_coefficient,
            anisotropy,
            refractive_index,
        };
        let optical_map = OpticalPropertyMap::homogeneous(&optical_props, dims);
        let config = PhotoacousticSolverConfig {
            optical_model: match optical_model {
                PhotoacousticOpticalModel::Diffusion => OpticalModel::Diffusion,
                PhotoacousticOpticalModel::MonteCarlo => OpticalModel::MonteCarlo,
            },
            illumination: IlluminationGeometry::PencilBeam {
                origin_m: [
                    illumination_origin_m.0,
                    illumination_origin_m.1,
                    illumination_origin_m.2,
                ],
                direction: [
                    illumination_direction.0,
                    illumination_direction.1,
                    illumination_direction.2,
                ],
            },
            pulse_duration_s,
            incident_fluence_j_m2,
            acoustic: PhotoacousticAcousticConfig {
                speed_of_sound_m_s,
                cfl_factor,
                num_time_steps,
                snapshot_interval,
            },
            // NOTE: `gruneisen` is accepted as a kwarg for API continuity but is
            // no longer plumbed into the kwavers core. The Gruneisen coefficient
            // is sourced internally by the thermoelastic pressure model
            // (`GrueneisenModel::soft_tissue()` at the time of writing). The
            // kwarg is retained to avoid breaking existing Python callers.
            thermoelastic: ThermoelasticProperties {
                density_kg_m3,
                sound_speed_m_s: speed_of_sound_m_s,
                specific_heat_j_kgk,
                thermal_conductivity_w_mk,
            },
            monte_carlo: MonteCarloModelConfig {
                photon_count,
                ..MonteCarloModelConfig::default()
            },
        };
        let sensor_positions_m = sensor_positions_m
            .into_iter()
            .map(|(x, y, z)| [x, y, z])
            .collect();
        let inner = KwaversPhotoacousticScenario::new(
            grid.inner,
            wavelength_nm,
            optical_map,
            config,
            sensor_positions_m,
        )
        .map_err(kwavers_error_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn wavelength_nm(&self) -> f64 {
        self.inner.wavelength_nm
    }

    #[getter]
    fn optical_model(&self) -> PhotoacousticOpticalModel {
        match self.inner.config.optical_model {
            OpticalModel::Diffusion => PhotoacousticOpticalModel::Diffusion,
            OpticalModel::MonteCarlo => PhotoacousticOpticalModel::MonteCarlo,
        }
    }

    #[getter]
    fn sensor_positions_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.sensor_positions_m.len();
        let mut out = Array2::zeros((n, 3));
        for (idx, pos) in self.inner.sensor_positions_m.iter().enumerate() {
            out[[idx, 0]] = pos[0];
            out[[idx, 1]] = pos[1];
            out[[idx, 2]] = pos[2];
        }
        PyArray2::from_owned_array(py, out)
    }
}
