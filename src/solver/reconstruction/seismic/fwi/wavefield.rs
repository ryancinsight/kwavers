//! Wavefield modeling for FWI
//! Based on Virieux (1986): "P-SV wave propagation in heterogeneous media"

use crate::error::{KwaversError, KwaversResult, PhysicsError};
use ndarray::{Array1, Array2, Array3, Zip};

/// Configuration for wavefield modeling
#[derive(Debug, Clone)]
pub struct WavefieldConfig {
    /// Grid spacing \[m\]
    pub dx: f64,
    /// Time step \[s\]
    pub dt: f64,
    /// Maximum simulation time \[s\]
    pub max_time: f64,
    /// Peak frequency for source wavelet \[Hz\]
    pub peak_frequency: f64,
    /// Source position (i, j, k)
    pub source_position: Option<(usize, usize, usize)>,
    /// Receiver positions
    pub receivers: Vec<(usize, usize, usize)>,
}

impl Default for WavefieldConfig {
    fn default() -> Self {
        Self {
            dx: 0.001,           // 1mm
            dt: 1e-6,            // 1 microsecond
            max_time: 0.01,      // 10ms
            peak_frequency: 1e6, // 1 MHz
            source_position: None,
            receivers: Vec::new(),
        }
    }
}

/// Wavefield modeling for forward and adjoint problems
#[derive(Debug)]
pub struct WavefieldModeler {
    /// Configuration
    config: WavefieldConfig,
    /// Stored forward wavefield for gradient computation
    forward_wavefield: Option<Array3<f64>>,
    /// PML boundary width
    pml_width: usize,
}

impl Default for WavefieldModeler {
    fn default() -> Self {
        Self::new()
    }
}

impl WavefieldModeler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WavefieldConfig::default(),
            forward_wavefield: None,
            pml_width: 20,
        }
    }

    #[must_use]
    pub fn with_config(config: WavefieldConfig) -> Self {
        Self {
            config,
            forward_wavefield: None,
            pml_width: 20,
        }
    }

    /// Forward wavefield modeling
    /// Solves: (1/v²)∂²u/∂t² - ∇²u = f
    pub fn forward_model(&mut self, velocity_model: &Array3<f64>) -> KwaversResult<Array2<f64>> {
        let (nx, ny, nz) = velocity_model.dim();
        let dt = self.calculate_stable_timestep(velocity_model);
        let nt = (self.config.max_time / dt) as usize;

        // Initialize wavefield arrays (pressure and particle velocity)
        let mut u_curr = Array3::zeros((nx, ny, nz));
        let mut u_prev = Array3::zeros((nx, ny, nz));
        let mut u_next = Array3::zeros((nx, ny, nz));

        // Initialize PML absorbing boundaries
        let pml_thickness = 10;
        let pml_damping = self.compute_pml_profile(pml_thickness, nx.max(ny).max(nz));

        // Precompute velocity squared for efficiency
        let v2 = velocity_model.mapv(|v| v * v);
        let dx2 = self.config.dx * self.config.dx;

        // Storage for receiver data
        let mut seismogram = Array2::zeros((self.config.receivers.len(), nt));

        // Time stepping loop with second-order finite difference
        for t_idx in 0..nt {
            let t = t_idx as f64 * dt;

            // Apply source wavelet at source position
            if let Some(src_pos) = self.config.source_position {
                let wavelet = self.ricker_wavelet(t, self.config.peak_frequency);
                u_curr[[src_pos.0, src_pos.1, src_pos.2]] += wavelet;
            }

            // Update wavefield using finite difference stencil
            // Cannot use Zip here as we need neighbor access for stencil
            for i in 2..nx - 2 {
                for j in 2..ny - 2 {
                    for k in 2..nz - 2 {
                        let laplacian = self.compute_laplacian_stencil_7pt(&u_curr, i, j, k, dx2);
                        let u_c = u_curr[[i, j, k]];
                        let u_p = u_prev[[i, j, k]];
                        let v2_local = v2[[i, j, k]];

                        u_next[[i, j, k]] = 2.0 * u_c - u_p + dt * dt * v2_local * laplacian;
                    }
                }
            }

            // Apply PML boundary conditions
            self.apply_pml_boundaries(&mut u_next, &pml_damping, nx, ny, nz, pml_thickness);

            // Record at receiver locations
            for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                seismogram[[r_idx, t_idx]] = u_curr[[rx, ry, rz]];
            }

            // Rotate arrays for next timestep
            std::mem::swap(&mut u_prev, &mut u_curr);
            std::mem::swap(&mut u_curr, &mut u_next);
        }

        // Store final wavefield for gradient computation
        self.forward_wavefield = Some(u_curr);

        Ok(seismogram)
    }

    /// Adjoint wavefield modeling
    /// Solves backward in time with residual as source
    pub fn adjoint_model(&self, adjoint_source: &Array2<f64>) -> KwaversResult<Array3<f64>> {
        let forward_field = self.forward_wavefield.as_ref().ok_or_else(|| {
            KwaversError::Physics(PhysicsError::InvalidState {
                field: "forward_wavefield".to_string(),
                value: "None".to_string(),
                reason: "Forward wavefield must be computed before adjoint modeling".to_string(),
            })
        })?;

        let (nx, ny, nz) = forward_field.dim();
        let dt = self.config.dt;
        let nt = adjoint_source.shape()[1];

        // Initialize adjoint wavefield
        let mut adj_curr = Array3::zeros((nx, ny, nz));
        let mut adj_prev = Array3::zeros((nx, ny, nz));
        let mut adj_next = Array3::zeros((nx, ny, nz));

        // Accumulate gradient
        let mut gradient = Array3::zeros((nx, ny, nz));

        // Time stepping backward
        for t_idx in (0..nt).rev() {
            // Inject adjoint source at receiver locations
            for (r_idx, &(rx, ry, rz)) in self.config.receivers.iter().enumerate() {
                adj_curr[[rx, ry, rz]] += adjoint_source[[r_idx, t_idx]];
            }

            // Update adjoint wavefield (same wave equation, backward in time)
            for i in 2..nx - 2 {
                for j in 2..ny - 2 {
                    for k in 2..nz - 2 {
                        let laplacian = self.compute_laplacian_stencil_7pt(
                            &adj_curr,
                            i,
                            j,
                            k,
                            self.config.dx * self.config.dx,
                        );
                        let a_c = adj_curr[[i, j, k]];
                        let a_p = adj_prev[[i, j, k]];

                        adj_next[[i, j, k]] = 2.0 * a_c - a_p + dt * dt * laplacian;
                    }
                }
            }

            // Apply PML boundaries (same as forward)
            let pml_thickness = 10;
            let pml_damping = self.compute_pml_profile(pml_thickness, nx.max(ny).max(nz));
            self.apply_pml_boundaries(&mut adj_next, &pml_damping, nx, ny, nz, pml_thickness);

            // Accumulate gradient: correlation of forward and adjoint fields
            Zip::from(&mut gradient)
                .and(forward_field)
                .and(&adj_curr)
                .for_each(|g, &f, &a| {
                    *g += f * a * dt;
                });

            // Rotate arrays
            std::mem::swap(&mut adj_prev, &mut adj_curr);
            std::mem::swap(&mut adj_curr, &mut adj_next);
        }

        Ok(gradient)
    }

    /// Get stored forward wavefield
    pub fn get_forward_wavefield(&self) -> KwaversResult<Array3<f64>> {
        self.forward_wavefield.clone().ok_or_else(|| {
            crate::error::KwaversError::InvalidInput("Forward wavefield not computed".to_string())
        })
    }

    /// Apply PML boundary conditions
    /// Based on Berenger (1994): "A perfectly matched layer for the absorption of electromagnetic waves"
    /// Journal of Computational Physics, 114(2), 185-200
    #[allow(dead_code)]
    fn apply_pml(&self, wavefield: &mut Array3<f64>) {
        let (nx, ny, nz) = wavefield.dim();
        let width = self.pml_width;

        // PML parameters following Collino & Tsogka (2001)
        let reflection_coeff: f64 = 1e-6; // Target reflection coefficient
        let pml_order = 2.0; // Polynomial order for damping profile
        let max_velocity = 4000.0; // Maximum velocity in model (m/s)

        // Maximum damping coefficient
        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * width as f64 * 0.001); // Assuming 1mm grid spacing

        // Apply damping in boundary regions with polynomial profile
        for i in 0..width {
            // Damping profile: d(x) = d_max * (x/L)^n
            let xi = (width - i) as f64 / width as f64;
            let damping = max_damping * xi.powf(pml_order);

            // X boundaries
            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= (-damping).exp();
                    wavefield[[nx - 1 - i, j, k]] *= (-damping).exp();
                }
            }

            // Y boundaries
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= (-damping).exp();
                    wavefield[[ii, ny - 1 - i, k]] *= (-damping).exp();
                }
            }

            // Z boundaries
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= (-damping).exp();
                    wavefield[[ii, j, nz - 1 - i]] *= (-damping).exp();
                }
            }
        }
    }

    /// Apply finite difference stencil for wave equation
    /// 4th order accurate in space, 2nd order in time
    #[allow(dead_code)]
    fn apply_fd_stencil(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
        velocity: &Array3<f64>,
        dt: f64,
        dx: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = current.dim();
        let mut next = Array3::zeros((nx, ny, nz));

        // Finite difference coefficients for 4th order
        const C0: f64 = -5.0 / 2.0;
        const C1: f64 = 4.0 / 3.0;
        const C2: f64 = -1.0 / 12.0;

        // Apply stencil avoiding boundaries (handled by PML)
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for k in 2..nz - 2 {
                    let laplacian = C0 * current[[i, j, k]]
                        + C1 * (current[[i + 1, j, k]]
                            + current[[i - 1, j, k]]
                            + current[[i, j + 1, k]]
                            + current[[i, j - 1, k]]
                            + current[[i, j, k + 1]]
                            + current[[i, j, k - 1]])
                        + C2 * (current[[i + 2, j, k]]
                            + current[[i - 2, j, k]]
                            + current[[i, j + 2, k]]
                            + current[[i, j - 2, k]]
                            + current[[i, j, k + 2]]
                            + current[[i, j, k - 2]]);

                    let v2 = velocity[[i, j, k]] * velocity[[i, j, k]];
                    next[[i, j, k]] = 2.0 * current[[i, j, k]] - previous[[i, j, k]]
                        + dt * dt * v2 * laplacian / (dx * dx);
                }
            }
        }

        next
    }

    /// Calculate stable timestep using CFL condition
    fn calculate_stable_timestep(&self, velocity_model: &Array3<f64>) -> f64 {
        let v_max = velocity_model
            .iter()
            .fold(0.0f64, |max, &v| max.max(v.abs()));
        let cfl = 0.5; // CFL number for stability
        cfl * self.config.dx / v_max
    }

    /// Compute PML damping profile
    fn compute_pml_profile(&self, thickness: usize, _max_dim: usize) -> Array1<f64> {
        let mut profile = Array1::zeros(thickness);
        let reflection_coeff: f64 = 1e-6;
        let pml_order = 2.0;
        let max_velocity = 4000.0;

        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * thickness as f64 * self.config.dx);

        for i in 0..thickness {
            let xi = (thickness - i) as f64 / thickness as f64;
            profile[i] = max_damping * xi.powf(pml_order);
        }

        profile
    }

    /// Apply PML boundaries to wavefield
    fn apply_pml_boundaries(
        &self,
        wavefield: &mut Array3<f64>,
        damping: &Array1<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        thickness: usize,
    ) {
        for i in 0..thickness.min(damping.len()) {
            let d = (-damping[i]).exp();

            // X boundaries
            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= d;
                    if nx > i {
                        wavefield[[nx - 1 - i, j, k]] *= d;
                    }
                }
            }

            // Y boundaries
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= d;
                    if ny > i {
                        wavefield[[ii, ny - 1 - i, k]] *= d;
                    }
                }
            }

            // Z boundaries
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= d;
                    if nz > i {
                        wavefield[[ii, j, nz - 1 - i]] *= d;
                    }
                }
            }
        }
    }

    /// Compute 7-point stencil Laplacian (2nd order accurate)
    /// ∇²u ≈ (u_{i+1} + u_{i-1} + u_{j+1} + u_{j-1} + u_{k+1} + u_{k-1} - 6u_{i,j,k}) / dx²
    ///
    /// References:
    /// - LeVeque (2007): "Finite Difference Methods for Ordinary and Partial Differential Equations"
    fn compute_laplacian_stencil_7pt(
        &self,
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        dx2: f64,
    ) -> f64 {
        let center = field[[i, j, k]];
        let neighbors_sum = field[[i + 1, j, k]]
            + field[[i - 1, j, k]]
            + field[[i, j + 1, k]]
            + field[[i, j - 1, k]]
            + field[[i, j, k + 1]]
            + field[[i, j, k - 1]];

        (neighbors_sum - 6.0 * center) / dx2
    }

    /// Generate Ricker wavelet source
    fn ricker_wavelet(&self, t: f64, f_peak: f64) -> f64 {
        let t0 = 1.0 / f_peak;
        let a = std::f64::consts::PI * f_peak * (t - t0);
        (1.0 - 2.0 * a * a) * (-a * a).exp()
    }
}
