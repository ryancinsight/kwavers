use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

use ndarray::Array1;

use crate::breast_fwi_bindings::complex_compat::leto3_to_nd3;
use crate::simulation_result_py::{SimulationResult, SimulationRunResult};

use super::super::Simulation;

impl Simulation {
    /// Convert a [`SimulationRunResult`] into a [`SimulationResult`] exposed to Python.
    pub(crate) fn simulation_run_result_to_py(
        py: Python<'_>,
        result: SimulationRunResult,
        shape: (usize, usize, usize),
        time_steps: usize,
        dt_actual: f64,
    ) -> PyResult<SimulationResult> {
        let SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
            thermal_temperature,
            thermal_dose,
        } = result;

        let (p_max_3d, p_min_3d, p_rms_3d, p_final_3d) =
            if let Some((mx, mn, rm, fn_)) = full_grid_stats {
                (
                    Some(PyArray3::from_owned_array(py, leto3_to_nd3(mx)).into()),
                    Some(PyArray3::from_owned_array(py, leto3_to_nd3(mn)).into()),
                    Some(PyArray3::from_owned_array(py, leto3_to_nd3(rm)).into()),
                    Some(PyArray3::from_owned_array(py, leto3_to_nd3(fn_)).into()),
                )
            } else {
                (None, None, None, None)
            };

        let p_max = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_max.clone()).into());
        let p_min = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_min.clone()).into());
        let p_rms = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_rms.clone()).into());
        let p_final = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_final.clone()).into());

        let ux = ux_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let uy = uy_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let uz = uz_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let ix = ix_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let iy = iy_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let iz = iz_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let i_avg_x = i_avg_x.map(|d| PyArray1::from_owned_array(py, d).into());
        let i_avg_y = i_avg_y.map(|d| PyArray1::from_owned_array(py, d).into());
        let i_avg_z = i_avg_z.map(|d| PyArray1::from_owned_array(py, d).into());

        let (ux_max, ux_min, ux_rms, uy_max, uy_min, uy_rms, uz_max, uz_min, uz_rms) =
            if let Some(vs) = velocity_stats {
                (
                    Some(PyArray1::from_owned_array(py, vs.ux_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.ux_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.ux_rms).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_rms).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_rms).into()),
                )
            } else {
                (None, None, None, None, None, None, None, None, None)
            };

        let time_arr = PyArray1::from_owned_array(
            py,
            Array1::from_iter((0..time_steps).map(|i| i as f64 * dt_actual)),
        )
        .into();

        // K → °C conversion for thermal outputs at the Python boundary.
        let thermal_temp_py = thermal_temperature
            .map(|t| PyArray3::from_owned_array(py, t.mapv(|v| v - KELVIN_OFFSET_C)).into());
        let thermal_dose_py = thermal_dose.map(|d| PyArray3::from_owned_array(py, d).into());

        let n_sensors = sensor_data.nrows();
        if n_sensors <= 1 {
            let sensor_1d = sensor_data.row(0).to_owned();
            Ok(SimulationResult {
                sensor_data_1d: Some(PyArray1::from_owned_array(py, sensor_1d).into()),
                sensor_data_2d: None,
                time: time_arr,
                shape,
                sensor_data_shape: (1, time_steps),
                time_steps,
                dt: dt_actual,
                final_time: dt_actual * time_steps as f64,
                p_max,
                p_min,
                p_rms,
                p_final,
                p_max_field: p_max_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_min_field: p_min_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_rms_field: p_rms_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_final_field: p_final_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                ux,
                uy,
                uz,
                ix,
                iy,
                iz,
                i_avg_x,
                i_avg_y,
                i_avg_z,
                ux_max,
                ux_min,
                ux_rms,
                uy_max,
                uy_min,
                uy_rms,
                uz_max,
                uz_min,
                uz_rms,
                thermal_temperature: thermal_temp_py,
                thermal_dose: thermal_dose_py,
            })
        } else {
            Ok(SimulationResult {
                sensor_data_1d: None,
                sensor_data_2d: Some(PyArray2::from_owned_array(py, sensor_data).into()),
                time: time_arr,
                shape,
                sensor_data_shape: (n_sensors, time_steps),
                time_steps,
                dt: dt_actual,
                final_time: dt_actual * time_steps as f64,
                p_max,
                p_min,
                p_rms,
                p_final,
                p_max_field: p_max_3d,
                p_min_field: p_min_3d,
                p_rms_field: p_rms_3d,
                p_final_field: p_final_3d,
                ux,
                uy,
                uz,
                ix,
                iy,
                iz,
                i_avg_x,
                i_avg_y,
                i_avg_z,
                ux_max,
                ux_min,
                ux_rms,
                uy_max,
                uy_min,
                uy_rms,
                uz_max,
                uz_min,
                uz_rms,
                thermal_temperature: thermal_temp_py,
                thermal_dose: thermal_dose_py,
            })
        }
    }
}
