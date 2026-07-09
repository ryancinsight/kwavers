//! Phase 22 wrappers: PID controller, resampling, reconstruction, and bubble field.

use kwavers_core::error::KwaversError;
use kwavers_solver::inverse::reconstruction::photoacoustic::{
    kspace_line_recon as kwavers_kspace_line_recon, LineReconDataOrder, LineReconInterpolation,
};
use leto::{
    Array2,
    Array3,
};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use crate::grid_py::Grid;

// ============================================================================
// PID Controller
// ============================================================================

#[pyclass(name = "PIDController")]
pub struct PyPIDController {
    inner: kwavers_physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDController,
}

#[pymethods]
impl PyPIDController {
    #[new]
    #[pyo3(signature = (kp, ki, kd, setpoint, sample_time=0.001, output_min=0.0, output_max=1.0, integral_limit=100.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kp: f64,
        ki: f64,
        kd: f64,
        setpoint: f64,
        sample_time: f64,
        output_min: f64,
        output_max: f64,
        integral_limit: f64,
    ) -> Self {
        let gains = kwavers_physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDGains { kp, ki, kd };
        let config = kwavers_physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDConfig {
            gains,
            sample_time,
            output_min,
            output_max,
            integral_limit,
            ..kwavers_physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDConfig::default()
        };
        let mut controller = kwavers_physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDController::new(config);
        controller.set_setpoint(setpoint);
        Self { inner: controller }
    }

    fn update(&mut self, measurement: f64) -> (f64, f64, f64, f64) {
        let out = self.inner.update(measurement);
        (
            out.control_signal,
            out.proportional_term,
            out.integral_term,
            out.derivative_term,
        )
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ============================================================================
// Resampling and reconstruction functions
// ============================================================================

#[pyfunction]
fn resample_to_target_grid<'py>(
    py: Python<'py>,
    source_image: PyReadonlyArray3<f64>,
    transform: [f64; 16],
    target_dims: (usize, usize, usize),
) -> Py<PyArray3<f64>> {
    use kwavers_physics::acoustics::imaging::fusion::registration::resample_to_target_grid as kwavers_resample;
    let arr = source_image.as_array().to_owned();
    let shape = arr.shape();
    let source_leto = leto::Array3::from_shape_vec(
        [shape[0], shape[1], shape[2]],
        arr.iter().copied().collect(),
    )
    .expect("ndarray source image shape must match contiguous voxel payload");
    let target = [target_dims.0, target_dims.1, target_dims.2];

    let resampled = py.detach(|| kwavers_resample(&source_leto, &transform, target));
    let out = leto::Array3::from_shape_vec(
        (target[0], target[1], target[2]),
        resampled
            .as_slice_memory_order()
            .expect("Leto resample output should be contiguous")
            .to_vec(),
    )
    .expect("target dimensions must match resampled voxel payload");

    PyArray3::from_owned_array(py, out).into()
}

#[pyfunction]
#[pyo3(signature = (sensor_data, dy, dt, c, *, data_order = "ty", interp = "linear", pos_cond = false))]
#[allow(clippy::too_many_arguments)]
fn kspace_line_recon<'py>(
    py: Python<'py>,
    sensor_data: PyReadonlyArray2<f64>,
    dy: f64,
    dt: f64,
    c: f64,
    data_order: &str,
    interp: &str,
    pos_cond: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let data_order = match data_order.to_ascii_lowercase().as_str() {
        "ty" => LineReconDataOrder::Ty,
        "yt" => LineReconDataOrder::Yt,
        other => {
            return Err(PyValueError::new_err(format!(
                "data_order must be 'ty' or 'yt', got {other}"
            )))
        }
    };
    let interp = match interp.to_ascii_lowercase().as_str() {
        "linear" => LineReconInterpolation::Linear,
        "nearest" => LineReconInterpolation::Nearest,
        other => {
            return Err(PyValueError::new_err(format!(
                "interp must be 'linear' or 'nearest', got {other}"
            )))
        }
    };

    let input = sensor_data.as_array().to_owned();
    let recon = py
        .detach(|| kwavers_kspace_line_recon(input.view(), dy, dt, c, data_order, interp, pos_cond))
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers error: {}", err)))?;

    Ok(PyArray2::from_owned_array(py, recon).into())
}

#[pyfunction]
#[pyo3(signature = (sensor_data, sensor_positions, grid, sound_speed, sampling_frequency, pml_size=None))]
fn time_reversal_reconstruction<'py>(
    py: Python<'py>,
    sensor_data: PyReadonlyArray2<f64>,
    sensor_positions: PyReadonlyArray2<f64>,
    grid: &Grid,
    sound_speed: f64,
    sampling_frequency: f64,
    pml_size: Option<usize>,
) -> PyResult<Py<PyArray3<f64>>> {
    let sensor_data = sensor_data.as_array().to_owned();
    let sensor_positions = sensor_positions.as_array().to_owned();
    let grid_inner = grid.inner.clone();
    let reconstruction = py
        .detach(move || {
            time_reversal_reconstruction_impl(
                sensor_data,
                sensor_positions,
                &grid_inner,
                sound_speed,
                sampling_frequency,
                pml_size,
            )
        })
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers error: {}", err)))?;

    Ok(PyArray3::from_owned_array(py, reconstruction).into())
}

/// Reconstruct an initial pressure field by replaying time-reversed boundary data.
///
/// # Theorem
/// Let `u_n` denote the discrete pressure field after `n` forward FDTD steps,
/// and let `g[t, s]` be the recorded pressure at sensor `s` and time index `t`.
/// If the same grid, medium, timestep, boundary treatment, and source mask are
/// used for replay, then a Dirichlet source driven by `g` reversed in time
/// produces the same discrete time-reversal experiment as the vendored
/// k-Wave `TimeReversal` example. The returned field is the solver's final
/// pressure state cropped back to the physical domain when outer PML layers
/// are requested.
///
/// # Proof sketch
/// The helper constructs the same sensor mask, reverses the same discrete
/// trace matrix, applies the same Dirichlet boundary condition, and advances
/// the same FDTD update operator. Outer PML is represented by embedding the
/// physical domain in a larger computational grid and cropping the final field
/// back to the interior. This preserves the interior discrete operator on the
/// physical domain while matching the source replay used by k-Wave.
pub(crate) fn time_reversal_reconstruction_impl(
    sensor_data: Array2<f64>,
    sensor_positions: Array2<f64>,
    grid: &kwavers_grid::Grid,
    sound_speed: f64,
    sampling_frequency: f64,
    pml_size: Option<usize>,
) -> kwavers_core::error::KwaversResult<Array3<f64>> {
    use kwavers_boundary::cpml::CPMLConfig;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_solver::forward::pstd::config::{BoundaryConfig, CompatibilityMode, PSTDConfig};
    use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
    use kwavers_solver::interface::solver::Solver as SolverTrait;
    use kwavers_source::grid_source::SourceMode;
    use kwavers_source::GridSource;

    if sound_speed <= 0.0 || !sound_speed.is_finite() {
        return Err(KwaversError::Validation(
            kwavers_core::error::ValidationError::FieldValidation {
                field: "sound_speed".to_string(),
                value: sound_speed.to_string(),
                constraint: "must be a positive finite scalar".to_string(),
            },
        ));
    }
    if sampling_frequency <= 0.0 || !sampling_frequency.is_finite() {
        return Err(KwaversError::Validation(
            kwavers_core::error::ValidationError::FieldValidation {
                field: "sampling_frequency".to_string(),
                value: sampling_frequency.to_string(),
                constraint: "must be a positive finite scalar".to_string(),
            },
        ));
    }
    if sensor_positions.ncols() != 3 || sensor_positions.nrows() == 0 {
        return Err(KwaversError::Validation(
            kwavers_core::error::ValidationError::FieldValidation {
                field: "sensor_positions".to_string(),
                value: format!("{:?}", sensor_positions.dim()),
                constraint: "must have shape (n_sensors, 3) and contain at least one sensor"
                    .to_string(),
            },
        ));
    }

    let n_sensors = sensor_positions.nrows();
    let sensor_data = match sensor_data.dim() {
        (rows, _cols) if rows == n_sensors => sensor_data,
        (_rows, cols) if cols == n_sensors => sensor_data.reversed_axes().to_owned(),
        (rows, cols) => {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::FieldValidation {
                    field: "sensor_data".to_string(),
                    value: format!("shape=({rows}, {cols})"),
                    constraint: format!(
                        "must align with sensor_positions rows {} along one axis",
                        n_sensors
                    ),
                },
            ))
        }
    };

    if sensor_data.ncols() == 0 {
        return Err(KwaversError::Validation(
            kwavers_core::error::ValidationError::FieldValidation {
                field: "sensor_data".to_string(),
                value: "0 time samples".to_string(),
                constraint: "must contain at least one time sample".to_string(),
            },
        ));
    }
    let nt = sensor_data.ncols();

    let (default_thickness, max_allowed) =
        crate::Simulation::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
    let pml = pml_size.unwrap_or(default_thickness).min(max_allowed);

    // Expand the grid by `pml` cells on each active side so the TR sensor falls
    // on the first non-PML cell (sigma = 0) of the expanded domain.
    let expand_x = if grid.nx > 1 { pml } else { 0 };
    let expand_y = if grid.ny > 1 { pml } else { 0 };
    let expand_z = if grid.nz > 1 { pml } else { 0 };
    let expanded_grid = kwavers_grid::Grid::new(
        grid.nx + 2 * expand_x,
        grid.ny + 2 * expand_y,
        grid.nz + 2 * expand_z,
        grid.dx,
        grid.dy,
        grid.dz,
    )?;

    let mut p_mask = Array3::<f64>::zeros((expanded_grid.nx, expanded_grid.ny, expanded_grid.nz));
    for row in sensor_positions.outer_iter() {
        let x = row[0] + expand_x as f64 * grid.dx;
        let y = row[1] + expand_y as f64 * grid.dy;
        let z = row[2] + expand_z as f64 * grid.dz;
        let i = (x / grid.dx).round() as isize;
        let j = (y / grid.dy).round() as isize;
        let k = (z / grid.dz).round() as isize;
        if i < 0
            || j < 0
            || k < 0
            || i >= expanded_grid.nx as isize
            || j >= expanded_grid.ny as isize
            || k >= expanded_grid.nz as isize
        {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::FieldValidation {
                    field: "sensor_positions".to_string(),
                    value: format!("[{x}, {y}, {z}]"),
                    constraint: "must map to a grid node inside the expanded domain".to_string(),
                },
            ));
        }
        let (i, j, k) = (i as usize, j as usize, k as usize);
        if p_mask[[i, j, k]] != 0.0 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::FieldValidation {
                    field: "sensor_positions".to_string(),
                    value: format!("duplicate grid node ({i}, {j}, {k})"),
                    constraint: "sensor positions must map to unique grid nodes".to_string(),
                },
            ));
        }
        p_mask[[i, j, k]] = 1.0;
    }

    let mut reversed_signal = sensor_data;
    reversed_signal.invert_axis(Axis(1));

    let grid_source = GridSource {
        p_mask: Some(p_mask.into()),
        p_signal: Some(reversed_signal.into()),
        p_mode: SourceMode::Dirichlet,
        ..GridSource::new_empty()
    };

    let medium = HomogeneousMedium::from_minimal(1000.0, sound_speed, &expanded_grid);
    let dt = 1.0 / sampling_frequency;
    let boundary = if pml > 0 {
        BoundaryConfig::CPML(CPMLConfig::with_thickness(pml))
    } else {
        BoundaryConfig::None
    };

    let config = PSTDConfig {
        nt,
        dt,
        compatibility_mode: CompatibilityMode::Reference,
        boundary,
        sensor_mask: None,
        pml_inside: true,
        ..PSTDConfig::default()
    };

    let mut solver = PSTDSolver::new(config, expanded_grid, &medium, grid_source)?;

    SolverTrait::run(&mut solver, nt)?;
    let pressure = SolverTrait::pressure_field(&solver);
    let mut cropped = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
        pressure[[i + expand_x, j + expand_y, k + expand_z]]
    });
    // Apply the standard Dirichlet half-amplitude compensation (k-Wave
    // convention): the reverse-time enforced-pressure source radiates into
    // both half-spaces; only the inward-traveling wave focuses, so the recon
    // recovers half the original initial pressure and is scaled by 2 here.
    // Signed values are preserved — the non-negativity prior p₀ ≥ 0 is a
    // photoacoustic post-processing choice and is left to the caller because
    // clipping breaks signed-pattern parity against k-Wave's `p_final`.
    cropped.mapv_inplace(|value| 2.0 * value);

    Ok(cropped)
}

// ============================================================================
// Bubble Field
// ============================================================================

#[pyclass(name = "BubbleField")]
pub struct PyBubbleField {
    inner: kwavers_physics::acoustics::bubble_dynamics::bubble_field::BubbleField,
}

#[pymethods]
impl PyBubbleField {
    #[new]
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let params =
            kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default();
        Self {
            inner: kwavers_physics::acoustics::bubble_dynamics::bubble_field::BubbleField::new(
                (nx, ny, nz),
                params,
            ),
        }
    }

    fn add_center_bubble(&mut self) {
        let params =
            kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default();
        self.inner.add_center_bubble(&params);
    }

    fn num_bubbles(&self) -> usize {
        self.inner.bubbles.len()
    }
}

// ============================================================================
// Module registration
// ============================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPIDController>()?;
    m.add_class::<PyBubbleField>()?;
    m.add_function(wrap_pyfunction!(resample_to_target_grid, m)?)?;
    m.add_function(wrap_pyfunction!(kspace_line_recon, m)?)?;
    m.add_function(wrap_pyfunction!(time_reversal_reconstruction, m)?)?;
    Ok(())
}
