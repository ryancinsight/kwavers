//! k-Wave Core Solver Implementation
//!
//! Main solver implementation following GRASP principles.
//! This module focuses solely on the core solving logic.

use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Axis, Zip};
use rustfft::{num_complex::Complex64, FftPlanner};

use super::config::{AbsorptionMode, KWaveConfig};
use super::data::{FieldArrays, KSpaceData};
use super::operators::kspace::compute_k_operators;
use super::sensors::{SensorData, SensorHandler};
use super::sources::{KWaveSource, SourceHandler};
use super::utils::compute_pml_operators;
use crate::utils::spectral::compute_kspace_correction_factors;

/// Core k-Wave solver implementing the k-space pseudospectral method
pub struct KWaveSolver {
    config: KWaveConfig,
    #[allow(dead_code)]
    grid: Grid,

    // Sensors
    sensor_handler: SensorHandler,

    // Sources
    source_handler: SourceHandler,
    time_step_index: usize,

    // FFT planning
    fft_planner: FftPlanner<f64>,

    // k-space operators
    kappa: Array3<f64>,                             // k-space correction
    k_vec: (Array3<f64>, Array3<f64>, Array3<f64>), // k-vectors
    k_max: f64,                                     // Maximum supported k
    c_ref: f64,

    // PML operators (sponge layer factors)
    pml: Array3<f64>,

    // Field variables (Physical Space)
    p: Array3<f64>,   // Pressure
    rho: Array3<f64>, // Density perturbation
    ux: Array3<f64>,  // Velocity x
    uy: Array3<f64>,  // Velocity y
    uz: Array3<f64>,  // Velocity z

    // Field variables (Spectral Space - Scratch)
    // Pre-allocated to avoid allocation in time loop
    p_k: Array3<Complex64>,
    ux_k: Array3<Complex64>,
    uy_k: Array3<Complex64>,
    uz_k: Array3<Complex64>,

    // Gradient Scratch Spaces
    grad_x_k: Array3<Complex64>,
    grad_y_k: Array3<Complex64>,
    grad_z_k: Array3<Complex64>,

    // Physical Space Scratch (Pre-allocated)
    dpx: Array3<f64>,
    dpy: Array3<f64>,
    dpz: Array3<f64>,
    div_u: Array3<f64>,

    // Material Properties (Pre-computed)
    rho0: Array3<f64>,
    c0: Array3<f64>,
    bon: Array3<f64>, // Nonlinearity parameter B/A

    // Material Gradients
    grad_rho0_x: Array3<f64>,
    grad_rho0_y: Array3<f64>,
    grad_rho0_z: Array3<f64>,

    // Absorption variables
    absorb_tau: Array3<f64>,
    absorb_eta: Array3<f64>,
}

impl std::fmt::Debug for KWaveSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KWaveSolver")
            .field("config", &self.config)
            .field("grid", &"Grid { ... }")
            .field("k_max", &self.k_max)
            .finish()
    }
}

impl KWaveSolver {
    /// Get access to the sensor handler
    pub fn sensor_handler(&self) -> &SensorHandler {
        &self.sensor_handler
    }

    /// Create a new k-Wave solver
    pub fn new<M: Medium>(
        config: KWaveConfig,
        grid: Grid,
        medium: &M,
        source: KWaveSource,
    ) -> KwaversResult<Self> {
        let (kspace_data, k_max, c_ref) =
            Self::initialize_kspace_operators(&config, &grid, medium)?;
        let pml_operators = Self::initialize_pml_operators(&config, &grid);
        let field_arrays = Self::initialize_field_arrays(&grid);
        let absorption_operators =
            Self::initialize_absorption_operators(&config, &grid, medium, k_max, c_ref)?;

        // Pre-compute material properties
        let mut rho0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut c0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut bon = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Use parallel iteration if possible, otherwise standard
        // For now, simple loop is fine during init
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    rho0[[i, j, k]] = crate::medium::density_at(medium, x, y, z, &grid);
                    c0[[i, j, k]] = crate::medium::sound_speed_at(medium, x, y, z, &grid);
                    bon[[i, j, k]] = crate::medium::nonlinearity_at(medium, x, y, z, &grid);
                }
            }
        }

        let shape = (grid.nx, grid.ny, grid.nz);

        let mut sensor_config = config.sensor_config.clone();
        if let Some(mask) = &config.sensor_mask {
            sensor_config.mask = mask.clone();
        }

        let mask_dim = sensor_config.mask.dim();
        if mask_dim != shape {
            if mask_dim == (1, 1, 1) && !sensor_config.mask[[0, 0, 0]] {
                sensor_config.mask = Array3::from_elem(shape, false);
            } else {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor mask shape mismatch: expected {:?}, got {:?}",
                            shape, mask_dim
                        ),
                    },
                ));
            }
        }

        let sensor_handler = SensorHandler::new(sensor_config, shape);

        // Initialize Source Handler
        let source_handler = SourceHandler::new(source, &grid);

        // Initialize solver partially to compute gradients of rho0
        let mut solver_partial = Self {
            config: config.clone(),
            grid: grid.clone(),
            sensor_handler,
            source_handler,
            time_step_index: 0,
            fft_planner: FftPlanner::new(),
            kappa: kspace_data.kappa,
            k_vec: kspace_data.k_vec,
            k_max,
            c_ref,
            pml: pml_operators,
            p: field_arrays.p,
            rho: Array3::zeros(shape),
            p_k: field_arrays.p_k, // Re-used as scratch
            ux: field_arrays.ux,
            uy: field_arrays.uy,
            uz: field_arrays.uz,
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            dpz: Array3::zeros(shape),
            div_u: Array3::zeros(shape),
            ux_k: Array3::zeros(shape),
            uy_k: Array3::zeros(shape),
            uz_k: Array3::zeros(shape),
            grad_x_k: Array3::zeros(shape),
            grad_y_k: Array3::zeros(shape),
            grad_z_k: Array3::zeros(shape),
            rho0: rho0.clone(),
            c0,
            bon,
            grad_rho0_x: Array3::zeros(shape),
            grad_rho0_y: Array3::zeros(shape),
            grad_rho0_z: Array3::zeros(shape),
            absorb_tau: absorption_operators.0,
            absorb_eta: absorption_operators.1,
        };

        // Apply initial conditions
        solver_partial.source_handler.apply_initial_conditions(
            &mut solver_partial.p,
            &mut solver_partial.rho,
            &solver_partial.c0,
            &mut solver_partial.ux,
            &mut solver_partial.uy,
            &mut solver_partial.uz,
        );

        // Compute gradients of rho0
        solver_partial.compute_rho0_gradients()?;

        Ok(solver_partial)
    }

    /// Compute gradients of rho0 using spectral method
    fn compute_rho0_gradients(&mut self) -> KwaversResult<()> {
        // Use p_k as scratch for rho0_k
        Self::fft_3d(&mut self.fft_planner, &self.rho0, &mut self.p_k);

        let i_img = Complex64::new(0.0, 1.0);

        // Compute gradients in k-space
        // grad_rho0_k = i * k * kappa * rho0_k

        // x
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // y
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(&self.k_vec.1)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // z
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(&self.k_vec.2)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // IFFT to get real gradients
        // Reuse ux_k as scratch
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_x_k,
            &mut self.grad_rho0_x,
            &mut self.ux_k,
        );
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_y_k,
            &mut self.grad_rho0_y,
            &mut self.ux_k,
        );
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_z_k,
            &mut self.grad_rho0_z,
            &mut self.ux_k,
        );

        Ok(())
    }

    /// Get reference to pressure field
    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.p
    }

    /// Get reference to velocity fields (ux, uy, uz)
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    /// Initialize k-space operators and correction factors
    fn initialize_kspace_operators<M: Medium>(
        config: &KWaveConfig,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<(KSpaceData, f64, f64)> {
        let (k_ops, k_max) = compute_k_operators(grid);
        let k_vec = (k_ops.kx.clone(), k_ops.ky.clone(), k_ops.kz.clone());

        let c_ref = medium.max_sound_speed();

        // Use Treeby2010 correction (k-Wave standard) which includes temporal dispersion correction
        let kappa = compute_kspace_correction_factors(
            &k_vec.0,
            &k_vec.1,
            &k_vec.2,
            grid,
            crate::utils::spectral::CorrectionType::Treeby2010,
            config.dt,
            c_ref,
        );

        Ok((KSpaceData { kappa, k_vec }, k_max, c_ref))
    }

    /// Initialize PML boundary operators
    fn initialize_pml_operators(config: &KWaveConfig, grid: &Grid) -> Array3<f64> {
        compute_pml_operators(grid, config.pml_size, config.pml_alpha)
    }

    /// Initialize field arrays
    fn initialize_field_arrays(grid: &Grid) -> FieldArrays {
        FieldArrays {
            p: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            p_k: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            ux: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            uy: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            uz: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    /// Initialize absorption operators
    fn initialize_absorption_operators<M: Medium>(
        config: &KWaveConfig,
        grid: &Grid,
        medium: &M,
        _k_max: f64,
        _c_ref: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut tau = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut eta = Array3::zeros((grid.nx, grid.ny, grid.nz));

        match &config.absorption_mode {
            AbsorptionMode::Lossless => {
                // No absorption - arrays remain zero
            }
            AbsorptionMode::Stokes => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "AbsorptionMode::Stokes is not supported by kwave_parity yet"
                            .to_string(),
                    },
                ));
            }
            AbsorptionMode::PowerLaw {
                alpha_coeff,
                alpha_power,
            } => {
                let y = *alpha_power;
                if (y - 1.0).abs() < 1e-12 {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message:
                                "alpha_power must not be 1.0 for fractional Laplacian formulation"
                                    .to_string(),
                        },
                    ));
                }
                // Calculate absorption terms from medium properties
                for k in 0..grid.nz {
                    for j in 0..grid.ny {
                        for i in 0..grid.nx {
                            let (x, y_coord, z) = grid.indices_to_coordinates(i, j, k);

                            // Use medium properties for spatially varying absorption
                            let alpha_0_medium = medium.alpha_coefficient(x, y_coord, z, grid);
                            let alpha_0 = if alpha_0_medium.abs() > 0.0 {
                                alpha_0_medium
                            } else {
                                *alpha_coeff
                            };
                            let c0_val = medium.sound_speed(i, j, k);

                            // Calculate tau and eta based on k-Wave formulation
                            // tau = -2 * alpha_0 * c0^(y-1)
                            // eta = 2 * alpha_0 * c0^y * tan(pi * y / 2)

                            // Note: We assume alpha_0 and c0 are in consistent units
                            // (e.g. Np and SI, or handled by the f_mhz scaling in apply_absorption)

                            let tau_val = -2.0 * alpha_0 * c0_val.powf(y - 1.0);
                            let eta_val = 2.0
                                * alpha_0
                                * c0_val.powf(y)
                                * (std::f64::consts::PI * y / 2.0).tan();

                            tau[[i, j, k]] = tau_val;
                            eta[[i, j, k]] = eta_val;
                        }
                    }
                }
            }
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by kwave_parity"
                            .to_string(),
                    },
                ));
            }
        }

        Ok((tau, eta))
    }

    /// Run the k-Wave simulation
    pub fn run(&mut self, steps: usize) -> KwaversResult<SensorData> {
        for _step in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_handler.extract_data())
    }

    /// Perform 3D FFT (in-place-ish: reads from input, writes to output)
    /// Optimized to reuse planner
    fn fft_3d(planner: &mut FftPlanner<f64>, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        // Copy real to complex
        Zip::from(&mut *output).and(input).for_each(|out, &in_val| {
            *out = Complex64::new(in_val, 0.0);
        });

        // FFT X
        let nx = output.shape()[0];
        let fft_x = planner.plan_fft_forward(nx);
        for mut lane in output.lanes_mut(Axis(0)) {
            let mut buf = lane.to_vec(); // Allocation per lane is hard to avoid without strided slice support in rustfft
            fft_x.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val;
            }
        }

        // FFT Y
        let ny = output.shape()[1];
        let fft_y = planner.plan_fft_forward(ny);
        for mut lane in output.lanes_mut(Axis(1)) {
            let mut buf = lane.to_vec();
            fft_y.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val;
            }
        }

        // FFT Z
        let nz = output.shape()[2];
        let fft_z = planner.plan_fft_forward(nz);
        for mut lane in output.lanes_mut(Axis(2)) {
            let mut buf = lane.to_vec();
            fft_z.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val;
            }
        }
    }

    // Optimized version avoiding full clone if we can write to a scratch buffer
    fn ifft_3d_to_real(
        planner: &mut FftPlanner<f64>,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        // Copy input to scratch
        scratch.assign(input);

        // IFFT X
        let nx = scratch.shape()[0];
        let fft_x = planner.plan_fft_inverse(nx);
        for mut lane in scratch.lanes_mut(Axis(0)) {
            let mut buf = lane.to_vec();
            fft_x.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val / (nx as f64);
            }
        }

        // IFFT Y
        let ny = scratch.shape()[1];
        let fft_y = planner.plan_fft_inverse(ny);
        for mut lane in scratch.lanes_mut(Axis(1)) {
            let mut buf = lane.to_vec();
            fft_y.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val / (ny as f64);
            }
        }

        // IFFT Z
        let nz = scratch.shape()[2];
        let fft_z = planner.plan_fft_inverse(nz);
        for mut lane in scratch.lanes_mut(Axis(2)) {
            let mut buf = lane.to_vec();
            fft_z.process(&mut buf);
            for (i, val) in buf.into_iter().enumerate() {
                lane[i] = val / (nz as f64);
            }
        }

        // Copy real part to output
        Zip::from(output).and(scratch).for_each(|out, c| {
            *out = c.re;
        });
    }

    /// Add a pressure source term to the mass equation (density perturbation)
    pub fn add_pressure_source(&mut self, source_term: &Array3<f64>) {
        // Source term in mass equation: S_m = source_p / c0^2
        Zip::from(&mut self.rho)
            .and(source_term)
            .and(&self.c0)
            .for_each(|rho, &s, &c| {
                *rho += s / (c * c);
            });
    }

    /// Perform a single time step using k-space pseudospectral method
    /// Implements:
    /// 1. Mass conservation: d(rho)/dt = -rho0 * div(u) - u.grad(rho0)
    /// 2. Momentum conservation: d(u)/dt = -1/rho0 * grad(p)
    /// 3. Equation of State: p = c0^2 * (rho + B/2A * rho^2/rho0 + ...)
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let i_img = Complex64::new(0.0, 1.0);

        // ----------------------------------------------------------------
        // 1. Calculate Pressure from Density (Equation of State)
        // ----------------------------------------------------------------
        // Linear: p = c0^2 * rho
        // Nonlinear: p = c0^2 * (rho + B/2A * rho^2 / rho0)

        if self.config.nonlinearity {
            Zip::from(&mut self.p)
                .and(&self.rho)
                .and(&self.c0)
                .and(&self.bon)
                .and(&self.rho0)
                .for_each(|p, &rho, &c, &bon, &rho0| {
                    let linear = rho;
                    // Taylor expansion for nonlinearity
                    // p = c0^2 * (rho + B/2A * rho^2 / rho0)
                    let nonlinear = (bon / (2.0 * rho0)) * rho * rho;
                    *p = c * c * (linear + nonlinear);
                });
        } else {
            Zip::from(&mut self.p)
                .and(&self.rho)
                .and(&self.c0)
                .for_each(|p, &rho, &c| {
                    *p = c * c * rho;
                });
        }

        // ----------------------------------------------------------------
        // 2. Momentum Conservation: Update Velocity
        // du/dt = -1/rho0 * grad(p)
        // ----------------------------------------------------------------

        // Transform pressure to k-space
        Self::fft_3d(&mut self.fft_planner, &self.p, &mut self.p_k);

        // Compute pressure gradients in k-space
        // grad_p_k = i * k * kappa * p_k

        // dx
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // dy
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(&self.k_vec.1)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // dz
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(&self.k_vec.2)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // Transform gradients back to physical space and update velocity

        // X-direction
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_x_k,
            &mut self.dpx,
            &mut self.ux_k,
        );
        Zip::from(&mut self.ux)
            .and(&self.dpx)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Y-direction
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_y_k,
            &mut self.dpy,
            &mut self.uy_k,
        );
        Zip::from(&mut self.uy)
            .and(&self.dpy)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Z-direction
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_z_k,
            &mut self.dpz,
            &mut self.uz_k,
        );
        Zip::from(&mut self.uz)
            .and(&self.dpz)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Inject force sources (velocity sources)
        self.source_handler.inject_force_source(
            self.time_step_index,
            &mut self.ux,
            &mut self.uy,
            &mut self.uz,
        );

        // Apply PML to velocities
        self.apply_pml_to_velocity();

        // ----------------------------------------------------------------
        // 3. Mass Conservation: Update Density
        // d(rho)/dt = -rho0 * div(u) - u.grad(rho0)
        // ----------------------------------------------------------------

        // Compute velocity divergence in k-space
        Self::fft_3d(&mut self.fft_planner, &self.ux, &mut self.ux_k);
        Self::fft_3d(&mut self.fft_planner, &self.uy, &mut self.uy_k);
        Self::fft_3d(&mut self.fft_planner, &self.uz, &mut self.uz_k);

        // div_u_k = i*kx*kappa*ux_k + ...
        // We use grad_x_k as scratch for div_u_k

        // 1. Term 1: kx * ux
        Zip::from(&mut self.grad_x_k)
            .and(&self.ux_k)
            .and(&self.k_vec.0)
            .for_each(|div, &ux, &kx| {
                *div = Complex64::new(kx, 0.0) * ux;
            });

        // 2. Term 2: + ky * uy
        Zip::from(&mut self.grad_x_k)
            .and(&self.uy_k)
            .and(&self.k_vec.1)
            .for_each(|div, &uy, &ky| {
                *div += Complex64::new(ky, 0.0) * uy;
            });

        // 3. Term 3: + kz * uz
        Zip::from(&mut self.grad_x_k)
            .and(&self.uz_k)
            .and(&self.k_vec.2)
            .for_each(|div, &uz, &kz| {
                *div += Complex64::new(kz, 0.0) * uz;
            });

        // 4. Apply i * kappa
        Zip::from(&mut self.grad_x_k)
            .and(&self.kappa)
            .for_each(|div, &kap| {
                *div *= i_img * kap;
            });

        // IFFT div_u_k -> div_u (physical)
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.grad_x_k,
            &mut self.div_u,
            &mut self.ux_k,
        ); // ux_k as scratch

        // Update density: rho -= dt * (rho0 * div_u + u.grad(rho0))

        // 1. Term: rho0 * div_u
        Zip::from(&mut self.rho)
            .and(&self.div_u)
            .and(&self.rho0)
            .for_each(|rho, &du, &rho0| {
                *rho -= dt * rho0 * du;
            });

        // 2. Term: ux * grad_rho0_x
        Zip::from(&mut self.rho)
            .and(&self.ux)
            .and(&self.grad_rho0_x)
            .for_each(|rho, &ux, &grx| {
                *rho -= dt * ux * grx;
            });

        // 3. Term: uy * grad_rho0_y
        Zip::from(&mut self.rho)
            .and(&self.uy)
            .and(&self.grad_rho0_y)
            .for_each(|rho, &uy, &gry| {
                *rho -= dt * uy * gry;
            });

        // 4. Term: uz * grad_rho0_z
        Zip::from(&mut self.rho)
            .and(&self.uz)
            .and(&self.grad_rho0_z)
            .for_each(|rho, &uz, &grz| {
                *rho -= dt * uz * grz;
            });

        // Inject mass sources (pressure sources)
        self.source_handler
            .inject_mass_source(self.time_step_index, &mut self.rho, &self.c0);

        self.apply_absorption(dt)?;

        // Apply PML to density
        self.apply_pml_to_density();

        // Record sensor data
        self.sensor_handler
            .record_step(&self.p, &self.ux, &self.uy, &self.uz);

        // Advance time step
        self.time_step_index += 1;

        Ok(())
    }

    /// Apply power law absorption using fractional Laplacian method
    ///
    /// Computes absorption terms:
    /// L1 = IFFT(f(k)^(y+1) * rho_k)
    /// L2 = IFFT(f(k)^(y+2) * rho_k)
    /// rho -= dt * (tau * L1 + eta * L2)
    ///
    /// This supports heterogeneous absorption coefficients (tau, eta).
    fn apply_absorption(&mut self, dt: f64) -> KwaversResult<()> {
        let (y, is_on) = match self.config.absorption_mode {
            AbsorptionMode::Lossless => (0.0, false),
            AbsorptionMode::Stokes => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "AbsorptionMode::Stokes is not supported by kwave_parity yet"
                            .to_string(),
                    },
                ));
            }
            AbsorptionMode::PowerLaw { alpha_power, .. } => (alpha_power, true),
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by kwave_parity"
                            .to_string(),
                    },
                ));
            }
        };

        if !is_on {
            return Ok(());
        }

        // Constants for unit conversion (matching k-Wave conventions)
        let c_ref = self.c_ref;
        let two_pi = 2.0 * std::f64::consts::PI;

        // 1. FFT rho -> p_k (using p_k as scratch for rho_k)
        Self::fft_3d(&mut self.fft_planner, &self.rho, &mut self.p_k);

        // 2. Compute L1 term: IFFT( f(k)^(y+1) * rho_k )
        // We compute f(k)^(y+1) * rho_k into ux_k
        Zip::from(&mut self.ux_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.k_vec.1)
            .and(&self.k_vec.2)
            .for_each(|out, &rho_k, &kx, &ky, &kz| {
                let k_sq = kx * kx + ky * ky + kz * kz;
                if k_sq > 1e-14 {
                    let k_mag = k_sq.sqrt();
                    // Convert k to f_MHz
                    let freq = k_mag * c_ref / two_pi;
                    let f_mhz = freq / 1e6;

                    let term = f_mhz.powf(y + 1.0);
                    *out = rho_k * Complex64::new(term, 0.0);
                } else {
                    *out = Complex64::new(0.0, 0.0);
                }
            });

        // IFFT ux_k -> dpx (using uy_k as scratch for IFFT)
        // dpx now contains L1
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.ux_k,
            &mut self.dpx,
            &mut self.uy_k,
        );

        // 3. Compute L2 term: IFFT( f(k)^(y+2) * rho_k )
        // We compute f(k)^(y+2) * rho_k into ux_k
        Zip::from(&mut self.ux_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.k_vec.1)
            .and(&self.k_vec.2)
            .for_each(|out, &rho_k, &kx, &ky, &kz| {
                let k_sq = kx * kx + ky * ky + kz * kz;
                if k_sq > 1e-14 {
                    let k_mag = k_sq.sqrt();
                    // Convert k to f_MHz
                    let freq = k_mag * c_ref / two_pi;
                    let f_mhz = freq / 1e6;

                    let term = f_mhz.powf(y + 2.0);
                    *out = rho_k * Complex64::new(term, 0.0);
                } else {
                    *out = Complex64::new(0.0, 0.0);
                }
            });

        // IFFT ux_k -> dpy (using uy_k as scratch)
        // dpy now contains L2
        Self::ifft_3d_to_real(
            &mut self.fft_planner,
            &self.ux_k,
            &mut self.dpy,
            &mut self.uy_k,
        );

        // 4. Update rho
        // rho -= dt * (tau * L1 + eta * L2)
        Zip::from(&mut self.rho)
            .and(&self.absorb_tau)
            .and(&self.absorb_eta)
            .and(&self.dpx) // L1
            .and(&self.dpy) // L2
            .for_each(|rho, &tau, &eta, &l1, &l2| {
                *rho -= dt * (tau * l1 + eta * l2);
            });

        Ok(())
    }

    /// Apply PML damping to density
    fn apply_pml_to_density(&mut self) {
        Zip::from(&mut self.rho)
            .and(&self.pml)
            .for_each(|rho, &pml| {
                *rho *= pml;
            });
    }

    /// Apply PML damping to velocity
    fn apply_pml_to_velocity(&mut self) {
        Zip::from(&mut self.ux)
            .and(&self.pml)
            .for_each(|u, &pml| *u *= pml);

        Zip::from(&mut self.uy)
            .and(&self.pml)
            .for_each(|u, &pml| *u *= pml);

        Zip::from(&mut self.uz)
            .and(&self.pml)
            .for_each(|u, &pml| *u *= pml);
    }
}
