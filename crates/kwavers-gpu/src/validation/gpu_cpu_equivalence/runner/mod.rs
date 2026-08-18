use super::{EquivalenceReport, EquivalenceValidator};
use hephaestus_core::{ComputeDevice, Fdtd3dOps, Fdtd3dParams, FdtdMedium, FdtdVelocity};
use hephaestus_wgpu::{WgpuDevice, WgpuFdtd3dOps};
use kwavers_core::error::{KwaversError, SystemError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use std::time::Instant;

const CFL_SAFETY_FACTOR: f64 = 0.9;
const FDTD_ROUNDING_OPERATIONS_PER_STEP: f64 = 24.0;

/// Calculate a conservative timestep from the maximum medium speed.
///
/// The provider executes a three-dimensional centered gradient/divergence
/// pair. The 0.5 factor is the CFL margin before the explicit provider safety
/// factor is applied by the caller.
fn calculate_stable_dt(grid: &Grid, medium: &dyn Medium) -> Result<f64, KwaversError> {
    let c_max = medium.max_sound_speed();
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    if !c_max.is_finite() || c_max <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FDTD maximum sound speed must be finite and positive: {c_max}"
        )));
    }
    if !dx_min.is_finite() || dx_min <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FDTD minimum grid spacing must be finite and positive: {dx_min}"
        )));
    }
    Ok(0.5 * dx_min / c_max)
}

fn provider_f32(name: &str, value: f64) -> Result<f32, KwaversError> {
    let converted = value as f32;
    if !value.is_finite() || !converted.is_finite() || converted <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FDTD {name} must remain finite and positive in provider f32 precision: {value}"
        )));
    }
    Ok(converted)
}

fn provider_params(grid: &Grid, dt: f64) -> Result<Fdtd3dParams, KwaversError> {
    let nx = u32::try_from(grid.nx).map_err(|error| {
        KwaversError::InvalidInput(format!("FDTD nx does not fit provider u32: {error}"))
    })?;
    let ny = u32::try_from(grid.ny).map_err(|error| {
        KwaversError::InvalidInput(format!("FDTD ny does not fit provider u32: {error}"))
    })?;
    let nz = u32::try_from(grid.nz).map_err(|error| {
        KwaversError::InvalidInput(format!("FDTD nz does not fit provider u32: {error}"))
    })?;
    Fdtd3dParams::new(
        nx,
        ny,
        nz,
        provider_f32("dx", grid.dx)?,
        provider_f32("dy", grid.dy)?,
        provider_f32("dz", grid.dz)?,
        provider_f32("dt", dt)?,
    )
    .map_err(|error| KwaversError::InvalidInput(format!("FDTD provider parameters: {error}")))
}

fn medium_values(grid: &Grid, medium: &dyn Medium) -> Result<Vec<FdtdMedium>, KwaversError> {
    let len = grid
        .checked_size()
        .ok_or_else(|| KwaversError::InvalidInput("FDTD grid size overflows usize".to_owned()))?;
    let mut values = Vec::with_capacity(len);
    for z in 0..grid.nz {
        for y in 0..grid.ny {
            for x in 0..grid.nx {
                let density = provider_f32("density", medium.density(x, y, z))?;
                let sound_speed = provider_f32("sound speed", medium.sound_speed(x, y, z))?;
                let cell = FdtdMedium::new(density, sound_speed).map_err(|error| {
                    KwaversError::InvalidInput(format!(
                        "FDTD medium at ({x}, {y}, {z}) is invalid: {error}"
                    ))
                })?;
                values.push(cell);
            }
        }
    }
    Ok(values)
}

/// Build an input-sensitive pressure field for the source-free provider
/// contract. Source injection is a consumer concern and is not hidden inside
/// the provider operation.
fn initial_pressure(grid: &Grid) -> Result<Vec<f32>, KwaversError> {
    let len = grid
        .checked_size()
        .ok_or_else(|| KwaversError::InvalidInput("FDTD grid size overflows usize".to_owned()))?;
    let mut pressure = Vec::with_capacity(len);
    for z in 0..grid.nz {
        for y in 0..grid.ny {
            for x in 0..grid.nx {
                pressure
                    .push(0.25 * (x % 5) as f32 + 0.5 * (y % 7) as f32 + 0.75 * (z % 11) as f32);
            }
        }
    }
    Ok(pressure)
}

#[inline]
fn flat_index(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    x + y * nx + z * nx * ny
}

fn cpu_step(
    pressure: &mut [f32],
    velocity: &mut [FdtdVelocity],
    medium: &[FdtdMedium],
    params: Fdtd3dParams,
) {
    let nx = params.nx() as usize;
    let ny = params.ny() as usize;
    let nz = params.nz() as usize;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = flat_index(x, y, z, nx, ny);
                if x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz {
                    velocity[index] = FdtdVelocity {
                        components: [0.0; 3],
                        padding: 0.0,
                    };
                    continue;
                }
                let gradient = [
                    (pressure[index + 1] - pressure[index - 1]) / (2.0 * params.dx()),
                    (pressure[index + nx] - pressure[index - nx]) / (2.0 * params.dy()),
                    (pressure[index + nx * ny] - pressure[index - nx * ny]) / (2.0 * params.dz()),
                ];
                let cell = medium[index];
                let scale = params.dt() / cell.density();
                let previous = velocity[index];
                velocity[index] = FdtdVelocity {
                    components: [
                        previous.components[0] - scale * gradient[0],
                        previous.components[1] - scale * gradient[1],
                        previous.components[2] - scale * gradient[2],
                    ],
                    padding: 0.0,
                };
            }
        }
    }
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = flat_index(x, y, z, nx, ny);
                if x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz {
                    pressure[index] = 0.0;
                    continue;
                }
                let divergence = (velocity[index + 1].components[0]
                    - velocity[index - 1].components[0])
                    / (2.0 * params.dx())
                    + (velocity[index + nx].components[1] - velocity[index - nx].components[1])
                        / (2.0 * params.dy())
                    + (velocity[index + nx * ny].components[2]
                        - velocity[index - nx * ny].components[2])
                        / (2.0 * params.dz());
                let cell = medium[index];
                pressure[index] -= params.dt()
                    * cell.density()
                    * cell.sound_speed()
                    * cell.sound_speed()
                    * divergence;
            }
        }
    }
}

/// Run the independent f32 CPU reference for the provider contract.
///
/// The CPU path mirrors the mathematical stencil, not the provider's source
/// or dispatch implementation. It returns flat row-major storage so the
/// comparison remains f32-native until report metric accumulation.
fn run_simulation_cpu(
    medium: &[FdtdMedium],
    nt: usize,
    params: Fdtd3dParams,
    mut pressure: Vec<f32>,
) -> Result<Vec<f32>, KwaversError> {
    let expected = params
        .storage_len()
        .map_err(|error| KwaversError::InvalidInput(format!("FDTD provider storage: {error}")))?;
    if medium.len() != expected || pressure.len() != expected {
        return Err(KwaversError::InvalidInput(
            "FDTD CPU reference storage does not match provider geometry".to_owned(),
        ));
    }
    let mut velocity = vec![
        FdtdVelocity {
            components: [0.0; 3],
            padding: 0.0,
        };
        expected
    ];
    for _ in 0..nt {
        cpu_step(&mut pressure, &mut velocity, medium, params);
    }
    Ok(pressure)
}

/// Run the provider-owned WGPU FDTD simulation.
///
/// Device acquisition is the only unavailable-provider branch. Once acquired,
/// kernel compilation, upload, dispatch, synchronization, and download errors
/// remain provider failures and are never replaced with CPU output.
fn run_simulation_gpu(
    nt: usize,
    pressure: &[f32],
    medium: &[FdtdMedium],
    params: Fdtd3dParams,
) -> Result<Vec<f32>, KwaversError> {
    let device = WgpuDevice::try_default("kwavers-fdtd-equivalence").map_err(|error| {
        KwaversError::System(SystemError::FeatureNotAvailable {
            feature: "Hephaestus WGPU FDTD provider".to_owned(),
            reason: error.to_string(),
        })
    })?;
    let pressure_buffer = device
        .upload(pressure)
        .map_err(|error| KwaversError::GpuError(format!("FDTD pressure upload: {error}")))?;
    let velocity = vec![
        FdtdVelocity {
            components: [0.0; 3],
            padding: 0.0,
        };
        pressure.len()
    ];
    let velocity_buffer = device
        .upload(&velocity)
        .map_err(|error| KwaversError::GpuError(format!("FDTD velocity upload: {error}")))?;
    let medium_buffer = device
        .upload(medium)
        .map_err(|error| KwaversError::GpuError(format!("FDTD medium upload: {error}")))?;
    let provider = WgpuFdtd3dOps;
    let kernel = provider.prepare_fdtd_3d(&device).map_err(|error| {
        KwaversError::GpuError(format!("FDTD provider kernel compilation: {error}"))
    })?;
    for _ in 0..nt {
        provider
            .step_fdtd_3d(
                &device,
                &kernel,
                &pressure_buffer,
                &velocity_buffer,
                &medium_buffer,
                &params,
            )
            .map_err(|error| KwaversError::GpuError(format!("FDTD provider dispatch: {error}")))?;
    }
    device
        .synchronize()
        .map_err(|error| KwaversError::GpuError(format!("FDTD provider synchronize: {error}")))?;
    device
        .download_owned(&pressure_buffer)
        .map_err(|error| KwaversError::GpuError(format!("FDTD pressure download: {error}")))
}

/// Validate GPU/CPU equivalence for acoustic wave simulation.
///
/// Both paths execute the provider contract's f32 central-difference equations.
/// The tolerance grows with the number of stencil operations and is derived
/// from `f32::EPSILON`; it is not an f64 solver fallback.
///
/// # Errors
///
/// Returns [`ValidationError`] when grid, medium, or provider parameters are
/// invalid. Provider unavailability or runtime failure is returned in the
/// report so callers can distinguish an unexecuted GPU path from a mismatch.
pub fn validate_gpu_cpu_equivalence(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
) -> Result<EquivalenceReport, ValidationError> {
    let validator = EquivalenceValidator::default();
    validate_gpu_cpu_equivalence_with_config(grid, medium, nt, &validator)
}

/// Validate GPU/CPU equivalence with a caller-provided report configuration.
///
/// The FDTD f32 rounding bound is selected from the provider contract after
/// applying the caller's other report settings.
///
/// # Errors
///
/// Returns [`ValidationError`] when the input domain cannot be represented by
/// the provider contract.
pub fn validate_gpu_cpu_equivalence_with_config(
    grid: &Grid,
    medium: &dyn Medium,
    nt: usize,
    validator: &EquivalenceValidator,
) -> Result<EquivalenceReport, ValidationError> {
    if grid.nx < 3 || grid.ny < 3 || grid.nz < 3 {
        return Err(ValidationError::InvalidParameter {
            parameter: "grid dimensions".to_owned(),
            reason: "FDTD provider dimensions must be at least three".to_owned(),
        });
    }
    let dt = calculate_stable_dt(grid, medium).map_err(|error| {
        ValidationError::ConstraintViolation {
            message: format!("FDTD timestep calculation failed: {error}"),
        }
    })? * CFL_SAFETY_FACTOR;
    let params =
        provider_params(grid, dt).map_err(|error| ValidationError::ConstraintViolation {
            message: format!("FDTD provider parameter construction failed: {error}"),
        })?;
    let medium_values =
        medium_values(grid, medium).map_err(|error| ValidationError::ConstraintViolation {
            message: format!("FDTD medium construction failed: {error}"),
        })?;
    let initial = initial_pressure(grid).map_err(|error| ValidationError::ConstraintViolation {
        message: format!("FDTD initial pressure construction failed: {error}"),
    })?;
    let shape = [grid.nx, grid.ny, grid.nz];
    let tolerance =
        (FDTD_ROUNDING_OPERATIONS_PER_STEP * nt.max(1) as f64 + 1.0) * f32::EPSILON as f64;
    let mut provider_validator = *validator;
    provider_validator.tolerance_absolute = tolerance;
    provider_validator.tolerance_relative = tolerance;

    let cpu_start = Instant::now();
    let cpu_pressure =
        run_simulation_cpu(&medium_values, nt, params, initial.clone()).map_err(|error| {
            ValidationError::ConstraintViolation {
                message: format!("CPU FDTD reference failed: {error}"),
            }
        })?;
    let cpu_time_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    let gpu_start = Instant::now();
    let gpu_result = run_simulation_gpu(nt, &initial, &medium_values, params);
    let gpu_time_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_pressure = match gpu_result {
        Ok(pressure) => pressure,
        Err(error) => {
            let total_points =
                grid.checked_size()
                    .ok_or_else(|| ValidationError::ConstraintViolation {
                        message: "FDTD grid size overflows usize".to_owned(),
                    })?;
            let mut report =
                EquivalenceReport::new(provider_validator.tolerance_relative, total_points);
            report.cpu_time_ms = cpu_time_ms;
            report.gpu_time_ms = gpu_time_ms;
            report.failure_reason = Some(match error {
                KwaversError::System(SystemError::FeatureNotAvailable { .. }) => {
                    format!("GPU unavailable: {error}")
                }
                error => format!("GPU provider failed: {error}"),
            });
            return Ok(report);
        }
    };

    provider_validator.validate_f32(
        shape,
        &cpu_pressure,
        &gpu_pressure,
        cpu_time_ms,
        gpu_time_ms,
    )
}

/// Validate equivalence for a specific grid and homogeneous medium.
///
/// # Errors
///
/// Returns [`ValidationError`] when grid or provider construction fails.
pub fn validate_equivalence_config(
    grid_size: (usize, usize, usize),
    dx: f64,
    c0: f64,
    rho0: f64,
    nt: usize,
) -> Result<EquivalenceReport, ValidationError> {
    use kwavers_medium::HomogeneousMedium;

    let (nx, ny, nz) = grid_size;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).map_err(|error| {
        ValidationError::ConstraintViolation {
            message: format!("Grid creation failed: {error}"),
        }
    })?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    validate_gpu_cpu_equivalence(&grid, &medium, nt)
}

#[cfg(test)]
mod tests;
