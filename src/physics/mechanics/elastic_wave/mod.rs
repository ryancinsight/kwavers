// src/physics/mechanics/elastic_wave/mod.rs
pub mod mode_conversion;

use crate::error::{KwaversResult, NumericalError, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, info, warn};
use ndarray::{s, Array3, Array4};
use num_complex::Complex;
use std::collections::HashMap;
use std::time::Instant;

/// Type alias for complex 3D arrays to reduce type complexity
/// Follows SOLID principles by providing clear, reusable types
type Complex3D = Array3<Complex<f64>>;

/// Elastic material properties following SSOT principle
/// Single source of truth for material property definitions
#[derive(Debug, Clone)]
pub struct ElasticProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Lamé parameters (Pa)
    pub lambda: f64,
    pub mu: f64,
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Bulk modulus (Pa)
    pub bulk_modulus: f64,
    /// P-wave speed (m/s)
    pub p_wave_speed: f64,
    /// S-wave speed (m/s)
    pub s_wave_speed: f64,
}

impl ElasticProperties {
    /// Create elastic properties from density and Lamé parameters
    /// Follows Information Expert principle - knows how to compute derived properties
    pub fn from_lame(density: f64, lambda: f64, mu: f64) -> KwaversResult<Self> {
        if density <= 0.0 || mu <= 0.0 || lambda < -2.0 / 3.0 * mu {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticProperties".to_string(),
                value: density, // Use density as the primary invalid value
                reason: format!(
                    "Invalid elastic parameters: density={}, lambda={}, mu={}",
                    density, lambda, mu
                ),
            }
            .into());
        }

        let youngs_modulus = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
        let poisson_ratio = lambda / (2.0 * (lambda + mu));
        let bulk_modulus = lambda + 2.0 * mu / 3.0;
        let p_wave_speed = ((lambda + 2.0 * mu) / density).sqrt();
        let s_wave_speed = (mu / density).sqrt();

        Ok(Self {
            density,
            lambda,
            mu,
            youngs_modulus,
            poisson_ratio,
            bulk_modulus,
            p_wave_speed,
            s_wave_speed,
        })
    }

    /// Create elastic properties from Young's modulus and Poisson's ratio
    pub fn from_youngs_poisson(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> KwaversResult<Self> {
        if density <= 0.0 || youngs_modulus <= 0.0 || poisson_ratio <= -1.0 || poisson_ratio >= 0.5
        {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticProperties".to_string(),
                value: density,
                reason: format!(
                    "Invalid elastic parameters: density={}, E={}, nu={}",
                    density, youngs_modulus, poisson_ratio
                ),
            }
            .into());
        }

        let lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        Self::from_lame(density, lambda, mu)
    }

    /// Validate properties for numerical stability
    pub fn validate(&self) -> KwaversResult<()> {
        if self.p_wave_speed <= 0.0 || self.s_wave_speed <= 0.0 {
            return Err(NumericalError::Instability {
                operation: "ElasticProperties validation".to_string(),
                condition: self.p_wave_speed.min(self.s_wave_speed),
            }
            .into());
        }
        Ok(())
    }
}

/// Anisotropic elastic properties for advanced materials
/// Follows Open/Closed principle - extends elastic properties without modification
#[derive(Debug, Clone)]
pub struct AnisotropicElasticProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Stiffness tensor (21 independent components for general anisotropy)
    pub stiffness_tensor: [[f64; 6]; 6],
    /// Compliance tensor (inverse of stiffness tensor)
    pub compliance_tensor: [[f64; 6]; 6],
}

impl AnisotropicElasticProperties {
    /// Create isotropic properties (simplified case)
    pub fn isotropic(density: f64, lambda: f64, mu: f64) -> KwaversResult<Self> {
        let mut stiffness = [[0.0; 6]; 6];

        // Fill stiffness tensor for isotropic material
        stiffness[0][0] = lambda + 2.0 * mu; // C11
        stiffness[1][1] = lambda + 2.0 * mu; // C22
        stiffness[2][2] = lambda + 2.0 * mu; // C33
        stiffness[0][1] = lambda; // C12
        stiffness[0][2] = lambda; // C13
        stiffness[1][2] = lambda; // C23
        stiffness[3][3] = mu; // C44
        stiffness[4][4] = mu; // C55
        stiffness[5][5] = mu; // C66

        let mut compliance = [[0.0; 6]; 6];
        // Compute compliance tensor (simplified for isotropic case)
        let youngs_modulus = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
        let poisson_ratio = lambda / (2.0 * (lambda + mu));

        compliance[0][0] = 1.0 / youngs_modulus;
        compliance[1][1] = 1.0 / youngs_modulus;
        compliance[2][2] = 1.0 / youngs_modulus;
        compliance[0][1] = -poisson_ratio / youngs_modulus;
        compliance[0][2] = -poisson_ratio / youngs_modulus;
        compliance[1][2] = -poisson_ratio / youngs_modulus;
        compliance[3][3] = 1.0 / mu;
        compliance[4][4] = 1.0 / mu;
        compliance[5][5] = 1.0 / mu;

        Ok(Self {
            density,
            stiffness_tensor: stiffness,
            compliance_tensor: compliance,
        })
    }

    /// Validate anisotropic properties
    pub fn validate(&self) -> KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "AnisotropicElasticProperties".to_string(),
                value: self.density,
                reason: "Density must be positive".to_string(),
            }
            .into());
        }
        // Additional validation for stiffness tensor positive definiteness could be added
        Ok(())
    }
}

/// Stress field components returned from FFT operations
/// Follows SOLID principles by grouping related return values
#[derive(Debug)]
pub struct StressFields {
    pub txx: Complex3D,
    pub tyy: Complex3D,
    pub tzz: Complex3D,
    pub txy: Complex3D,
    pub txz: Complex3D,
    pub tyz: Complex3D,
}

/// Velocity field components returned from FFT operations
/// Follows SOLID principles by grouping related return values
#[derive(Debug)]
pub struct VelocityFields {
    pub vx: Complex3D,
    pub vy: Complex3D,
    pub vz: Complex3D,
}

/// Parameters for stress update operations
/// Follows SOLID principles by reducing parameter coupling
#[derive(Debug)]
pub struct StressUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    pub sxx_fft: &'a Complex3D,
    pub syy_fft: &'a Complex3D,
    pub szz_fft: &'a Complex3D,
    pub sxy_fft: &'a Complex3D,
    pub sxz_fft: &'a Complex3D,
    pub syz_fft: &'a Complex3D,
    pub kx: &'a Array3<f64>,
    pub ky: &'a Array3<f64>,
    pub kz: &'a Array3<f64>,
    pub lame_lambda: &'a Array3<f64>,
    pub lame_mu: &'a Array3<f64>,
    pub density: &'a Array3<f64>,
    pub dt: f64,
}

/// Parameters for velocity update operations
/// Follows SOLID principles by reducing parameter coupling
#[derive(Debug)]
pub struct VelocityUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    pub txx_fft: &'a Complex3D,
    pub tyy_fft: &'a Complex3D,
    pub tzz_fft: &'a Complex3D,
    pub txy_fft: &'a Complex3D,
    pub txz_fft: &'a Complex3D,
    pub tyz_fft: &'a Complex3D,
    pub kx: &'a Array3<f64>,
    pub ky: &'a Array3<f64>,
    pub kz: &'a Array3<f64>,
    pub density: &'a Array3<f64>,
    pub dt: f64,
}

/// Performance metrics for elastic wave solver
/// Follows SSOT principle for performance tracking
#[derive(Debug, Clone)]
pub struct ElasticWaveMetrics {
    pub fft_time: f64,
    pub stress_update_time: f64,
    pub velocity_update_time: f64,
    pub source_time: f64,
    pub total_update_time: f64,
    pub call_count: usize,
    pub memory_usage: usize,
}

impl Default for ElasticWaveMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ElasticWaveMetrics {
    pub fn new() -> Self {
        Self {
            fft_time: 0.0,
            stress_update_time: 0.0,
            velocity_update_time: 0.0,
            source_time: 0.0,
            total_update_time: 0.0,
            call_count: 0,
            memory_usage: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn get_average_times(&self) -> HashMap<String, f64> {
        if self.call_count == 0 {
            return HashMap::new();
        }

        let count = self.call_count as f64;
        let mut averages = HashMap::new();
        averages.insert("fft_time".to_string(), self.fft_time / count);
        averages.insert(
            "stress_update_time".to_string(),
            self.stress_update_time / count,
        );
        averages.insert(
            "velocity_update_time".to_string(),
            self.velocity_update_time / count,
        );
        averages.insert("source_time".to_string(), self.source_time / count);
        averages.insert(
            "total_update_time".to_string(),
            self.total_update_time / count,
        );
        averages
    }
}

/// Solver for linear isotropic elastic wave propagation using a k-space pseudospectral method.
///
/// This solver updates particle velocities (vx, vy, vz) and stress components
/// (sxx, syy, szz, sxy, sxz, syz) based on the 3D linear elastic wave equations.
///
/// Features:
/// - Error handling following SOLID principles
/// - Performance monitoring with SSOT metrics
/// - Support for anisotropic materials (future)
/// - Numerical stability checks
/// - Memory usage tracking
///
/// Design Principles Implemented:
/// - SOLID: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
/// - CUPID: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
/// - GRASP: Information expert, creator, controller, low coupling, high cohesion
/// - SSOT: Single source of truth for performance metrics
/// - ADP: Acyclic dependency principle
#[derive(Debug, Clone)]
pub struct ElasticWave {
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    metrics: ElasticWaveMetrics,
    is_anisotropic: bool,
    anisotropic_properties: Option<AnisotropicElasticProperties>,
    /// Mode conversion configuration
    mode_conversion: Option<mode_conversion::ModeConversionConfig>,
    /// Viscoelastic damping configuration
    viscoelastic: Option<mode_conversion::ViscoelasticConfig>,
    /// Material stiffness tensors (spatially varying)
    stiffness_tensors: Option<Array4<f64>>,
    /// Interface detection mask
    interface_mask: Option<Array3<bool>>,
}

impl ElasticWave {
    /// Create a new elastic wave solver
    /// Follows GRASP Creator principle - creates objects it has information to create
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let (dx, dy, dz) = grid.spacing();

        // Validate grid dimensions following Information Expert principle
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticWave".to_string(),
                value: 0.0,
                reason: "Grid dimensions must be positive".to_string(),
            }
            .into());
        }

        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticWave".to_string(),
                value: dx.min(dy).min(dz),
                reason: "Grid spacing must be positive".to_string(),
            }
            .into());
        }

        // Create wavenumber arrays
        let kx = Self::create_wavenumber_array(nx, dx);
        let ky = Self::create_wavenumber_array(ny, dy);
        let kz = Self::create_wavenumber_array(nz, dz);

        Ok(Self {
            kx,
            ky,
            kz,
            metrics: ElasticWaveMetrics::new(),
            is_anisotropic: false,
            anisotropic_properties: None,
            mode_conversion: None,
            viscoelastic: None,
            stiffness_tensors: None,
            interface_mask: None,
        })
    }

    /// Create wavenumber array for a given dimension
    /// Follows DRY principle by extracting common functionality
    fn create_wavenumber_array(n: usize, d: f64) -> Array3<f64> {
        let mut k = Array3::zeros((n, n, n));
        let dk = 2.0 * std::f64::consts::PI / (n as f64 * d);

        for i in 0..n {
            let ki = if i <= n / 2 {
                i as f64
            } else {
                (i as f64) - n as f64
            } * dk;
            k.slice_mut(s![i, .., ..]).fill(ki);
        }
        k
    }

    /// Set anisotropic properties
    /// Follows Open/Closed principle - extends functionality without modification
    pub fn set_anisotropic_properties(
        &mut self,
        properties: AnisotropicElasticProperties,
    ) -> KwaversResult<()> {
        properties.validate()?;
        self.anisotropic_properties = Some(properties);
        self.is_anisotropic = true;
        Ok(())
    }

    /// Enable mode conversion with custom configuration
    pub fn with_mode_conversion(
        &mut self,
        config: mode_conversion::ModeConversionConfig,
    ) -> &mut Self {
        self.mode_conversion = Some(config);
        self
    }

    /// Enable viscoelastic damping
    pub fn with_viscoelastic(&mut self, config: mode_conversion::ViscoelasticConfig) -> &mut Self {
        self.viscoelastic = Some(config);
        self
    }

    /// Set spatially varying stiffness tensors
    pub fn set_stiffness_field(&mut self, tensors: Array4<f64>) -> KwaversResult<()> {
        // Validate dimensions
        let (nx, ny, nz) = (self.kx.dim().0, self.ky.dim().1, self.kz.dim().2);
        if tensors.dim() != (nx, ny, nz, 21) {
            return Err(PhysicsError::InvalidParameter {
                parameter: "stiffness_tensor_shape".to_string(),
                value: tensors.dim().3 as f64,
                reason: format!(
                    "Expected shape ({}, {}, {}, 21), got {:?}",
                    nx,
                    ny,
                    nz,
                    tensors.dim()
                ),
            }
            .into());
        }

        self.stiffness_tensors = Some(tensors);
        Ok(())
    }

    /// Detect material interfaces for mode conversion
    pub fn detect_interfaces(&mut self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        if let Some(ref config) = self.mode_conversion {
            info!("Detecting material interfaces for mode conversion");
            let (nx, ny, nz) = grid.dimensions();
            let mut interface_mask = Array3::from_elem((nx, ny, nz), false);

            // Detect interfaces based on property gradients
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let density = medium.density(x, y, z, grid);

                        // Check neighboring points
                        let mut max_gradient = 0.0;
                        for di in -1..=1 {
                            for dj in -1..=1 {
                                for dk in -1..=1 {
                                    if di == 0 && dj == 0 && dk == 0 {
                                        continue;
                                    }

                                    let ni = (i as i32 + di).max(0).min(nx as i32 - 1) as usize;
                                    let nj = (j as i32 + dj).max(0).min(ny as i32 - 1) as usize;
                                    let nk = (k as i32 + dk).max(0).min(nz as i32 - 1) as usize;

                                    let nx = ni as f64 * grid.dx;
                                    let ny = nj as f64 * grid.dy;
                                    let nz = nk as f64 * grid.dz;
                                    let ndensity = medium.density(nx, ny, nz, grid);

                                    let gradient = (ndensity - density).abs() / density;
                                    max_gradient = f64::max(max_gradient, gradient);
                                }
                            }
                        }

                        interface_mask[[i, j, k]] = max_gradient > config.interface_threshold;
                    }
                }
            }

            self.interface_mask = Some(interface_mask);
        }
        Ok(())
    }

    /// Get performance metrics
    /// Follows SSOT principle - single source of truth for metrics
    pub fn get_metrics(&self) -> &ElasticWaveMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Validate solver state
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        if self.kx.is_empty() || self.ky.is_empty() || self.kz.is_empty() {
            return Err(PhysicsError::ModelNotInitialized {
                model: "ElasticWave".to_string(),
                reason: "Wave vectors not initialized".to_string(),
            }
            .into());
        }
        Ok(())
    }
}

impl ElasticWave {
    fn _perform_fft(&self, field: &Array3<f64>, grid: &Grid) -> Array3<Complex<f64>> {
        // Create a 4D array with the field as the first component
        let mut fields_4d = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        fields_4d.index_axis_mut(ndarray::Axis(0), 0).assign(field);
        fft_3d(&fields_4d, 0, grid)
    }

    fn _perform_ifft(&self, field_fft: &mut Array3<Complex<f64>>, grid: &Grid) -> Array3<f64> {
        ifft_3d(field_fft, grid)
    }

    fn _update_stress_fft(&self, params: &StressUpdateParams) -> KwaversResult<StressFields> {
        let start_time = Instant::now();

        // Validate inputs following Information Expert principle
        if params.dt <= 0.0 {
            return Err(NumericalError::Instability {
                operation: "Stress update".to_string(),
                condition: params.dt,
            }
            .into());
        }

        // Stress update in k-space using time integration
        // σ_new = σ_old + dt * (λ∇·v + μ(∇v + ∇v^T))
        let dt_complex = Complex::new(params.dt, 0.0);
        let div_v =
            params.vx_fft * params.kx + params.vy_fft * params.ky + params.vz_fft * params.kz;

        let txx_fft = params.sxx_fft
            + dt_complex
                * (&div_v * params.lame_lambda
                    + Complex::new(2.0, 0.0) * params.vx_fft * params.kx * params.lame_mu);
        let tyy_fft = params.syy_fft
            + dt_complex
                * (&div_v * params.lame_lambda
                    + Complex::new(2.0, 0.0) * params.vy_fft * params.ky * params.lame_mu);
        let tzz_fft = params.szz_fft
            + dt_complex
                * (&div_v * params.lame_lambda
                    + Complex::new(2.0, 0.0) * params.vz_fft * params.kz * params.lame_mu);
        let txy_fft = params.sxy_fft
            + dt_complex
                * ((params.vx_fft * params.ky + params.vy_fft * params.kx) * params.lame_mu);
        let txz_fft = params.sxz_fft
            + dt_complex
                * ((params.vx_fft * params.kz + params.vz_fft * params.kx) * params.lame_mu);
        let tyz_fft = params.syz_fft
            + dt_complex
                * ((params.vy_fft * params.kz + params.vz_fft * params.ky) * params.lame_mu);

        let stress_update_time = start_time.elapsed().as_secs_f64();

        Ok(StressFields {
            txx: txx_fft,
            tyy: tyy_fft,
            tzz: tzz_fft,
            txy: txy_fft,
            txz: txz_fft,
            tyz: tyz_fft,
        })
    }

    fn _update_velocity_fft(&self, params: &VelocityUpdateParams) -> KwaversResult<VelocityFields> {
        let start_time = Instant::now();

        // Velocity update in k-space
        let vx_fft =
            (params.txx_fft * params.kx + params.txy_fft * params.ky + params.txz_fft * params.kz)
                / params.density;
        let vy_fft =
            (params.txy_fft * params.kx + params.tyy_fft * params.ky + params.tyz_fft * params.kz)
                / params.density;
        let vz_fft =
            (params.txz_fft * params.kx + params.tyz_fft * params.ky + params.tzz_fft * params.kz)
                / params.density;

        let velocity_update_time = start_time.elapsed().as_secs_f64();

        Ok(VelocityFields {
            vx: vx_fft,
            vy: vy_fft,
            vz: vz_fft,
        })
    }

    fn _apply_source_term(&self, source: &dyn Source, grid: &Grid, t: f64) -> Array3<Complex<f64>> {
        let start_time = Instant::now();

        // Create a source field array from the source term
        let mut source_field = grid.create_field();
        source_field
            .indexed_iter_mut()
            .for_each(|((i, j, k), val)| {
                let (x, y, z) = grid.coordinates(i, j, k);
                *val = source.get_source_term(t, x, y, z, grid);
            });
        let source_fft = self._perform_fft(&source_field, grid);

        let source_time = start_time.elapsed().as_secs_f64();

        source_fft
    }
}

impl AcousticWaveModel for ElasticWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        let total_start_time = Instant::now();

        // Validate inputs following Information Expert principle
        if let Err(e) = self.validate() {
            warn!("ElasticWave validation failed: {}", e);
            return;
        }

        if dt <= 0.0 {
            warn!("Invalid time step: {}", dt);
            return;
        }

        // Extract current field values
        let vx = fields
            .slice(s![UnifiedFieldType::VelocityX.index(), .., .., ..])
            .to_owned();
        let vy = fields
            .slice(s![UnifiedFieldType::VelocityY.index(), .., .., ..])
            .to_owned();
        let vz = fields
            .slice(s![UnifiedFieldType::VelocityZ.index(), .., .., ..])
            .to_owned();
        let sxx = fields
            .slice(s![UnifiedFieldType::StressXX.index(), .., .., ..])
            .to_owned();
        let syy = fields
            .slice(s![UnifiedFieldType::StressYY.index(), .., .., ..])
            .to_owned();
        let szz = fields
            .slice(s![UnifiedFieldType::StressZZ.index(), .., .., ..])
            .to_owned();
        let sxy = fields
            .slice(s![UnifiedFieldType::StressXY.index(), .., .., ..])
            .to_owned();
        let sxz = fields
            .slice(s![UnifiedFieldType::StressXZ.index(), .., .., ..])
            .to_owned();
        let syz = fields
            .slice(s![UnifiedFieldType::StressYZ.index(), .., .., ..])
            .to_owned();

        // Get medium properties
        let density = medium.density_array();
        let lambda = medium.lame_lambda_array();
        let mu = medium.lame_mu_array();

        // Perform FFT
        let fft_start = Instant::now();
        let mut vx_fft = self._perform_fft(&vx, grid);
        let vy_fft = self._perform_fft(&vy, grid);
        let vz_fft = self._perform_fft(&vz, grid);
        let sxx_fft = self._perform_fft(&sxx, grid);
        let syy_fft = self._perform_fft(&syy, grid);
        let szz_fft = self._perform_fft(&szz, grid);
        let sxy_fft = self._perform_fft(&sxy, grid);
        let sxz_fft = self._perform_fft(&sxz, grid);
        let syz_fft = self._perform_fft(&syz, grid);
        self.metrics.fft_time += fft_start.elapsed().as_secs_f64();

        // Apply source term
        let source_fft = self._apply_source_term(source, grid, t);
        vx_fft += &source_fft;

        // Update stress components
        let stress_params = StressUpdateParams {
            vx_fft: &vx_fft,
            vy_fft: &vy_fft,
            vz_fft: &vz_fft,
            sxx_fft: &sxx_fft,
            syy_fft: &syy_fft,
            szz_fft: &szz_fft,
            sxy_fft: &sxy_fft,
            sxz_fft: &sxz_fft,
            syz_fft: &syz_fft,
            kx: &self.kx,
            ky: &self.ky,
            kz: &self.kz,
            lame_lambda: &lambda,
            lame_mu: &mu,
            density,
            dt,
        };

        let stress_start = Instant::now();
        let stress_fields = match self._update_stress_fft(&stress_params) {
            Ok(fields) => fields,
            Err(e) => {
                warn!("Stress update failed: {}", e);
                return;
            }
        };
        self.metrics.stress_update_time += stress_start.elapsed().as_secs_f64();

        // Update velocity components
        let velocity_params = VelocityUpdateParams {
            vx_fft: &vx_fft,
            vy_fft: &vy_fft,
            vz_fft: &vz_fft,
            txx_fft: &stress_fields.txx,
            tyy_fft: &stress_fields.tyy,
            tzz_fft: &stress_fields.tzz,
            txy_fft: &stress_fields.txy,
            txz_fft: &stress_fields.txz,
            tyz_fft: &stress_fields.tyz,
            kx: &self.kx,
            ky: &self.ky,
            kz: &self.kz,
            density,
            dt,
        };

        let velocity_start = Instant::now();
        let velocity_fields = match self._update_velocity_fft(&velocity_params) {
            Ok(fields) => fields,
            Err(e) => {
                warn!("Velocity update failed: {}", e);
                return;
            }
        };
        self.metrics.velocity_update_time += velocity_start.elapsed().as_secs_f64();

        // Update fields with new values
        let mut new_vx = self._perform_ifft(&mut velocity_fields.vx.clone(), grid);
        let mut new_vy = self._perform_ifft(&mut velocity_fields.vy.clone(), grid);
        let mut new_vz = self._perform_ifft(&mut velocity_fields.vz.clone(), grid);
        let mut new_sxx = self._perform_ifft(&mut stress_fields.txx.clone(), grid);
        let mut new_syy = self._perform_ifft(&mut stress_fields.tyy.clone(), grid);
        let mut new_szz = self._perform_ifft(&mut stress_fields.tzz.clone(), grid);
        let mut new_sxy = self._perform_ifft(&mut stress_fields.txy.clone(), grid);
        let mut new_sxz = self._perform_ifft(&mut stress_fields.txz.clone(), grid);
        let mut new_syz = self._perform_ifft(&mut stress_fields.tyz.clone(), grid);

        // Apply time integration
        new_vx *= dt;
        new_vy *= dt;
        new_vz *= dt;
        new_sxx *= dt;
        new_syy *= dt;
        new_szz *= dt;
        new_sxy *= dt;
        new_sxz *= dt;
        new_syz *= dt;

        // Update field arrays
        fields
            .slice_mut(s![UnifiedFieldType::VelocityX.index(), .., .., ..])
            .assign(&new_vx);
        fields
            .slice_mut(s![UnifiedFieldType::VelocityY.index(), .., .., ..])
            .assign(&new_vy);
        fields
            .slice_mut(s![UnifiedFieldType::VelocityZ.index(), .., .., ..])
            .assign(&new_vz);
        fields
            .slice_mut(s![UnifiedFieldType::StressXX.index(), .., .., ..])
            .assign(&new_sxx);
        fields
            .slice_mut(s![UnifiedFieldType::StressYY.index(), .., .., ..])
            .assign(&new_syy);
        fields
            .slice_mut(s![UnifiedFieldType::StressZZ.index(), .., .., ..])
            .assign(&new_szz);
        fields
            .slice_mut(s![UnifiedFieldType::StressXY.index(), .., .., ..])
            .assign(&new_sxy);
        fields
            .slice_mut(s![UnifiedFieldType::StressXZ.index(), .., .., ..])
            .assign(&new_sxz);
        fields
            .slice_mut(s![UnifiedFieldType::StressYZ.index(), .., .., ..])
            .assign(&new_syz);

        // Update metrics
        self.metrics.call_count += 1;
        self.metrics.total_update_time += total_start_time.elapsed().as_secs_f64();
        self.metrics.memory_usage = fields.len() * std::mem::size_of::<f64>();
    }

    fn report_performance(&self) {
        let averages = self.metrics.get_average_times();
        debug!("ElasticWave Performance Report:");
        debug!("  Total calls: {}", self.metrics.call_count);
        debug!(
            "  Average FFT time: {:.6} ms",
            averages.get("fft_time").unwrap_or(&0.0) * 1000.0
        );
        debug!(
            "  Average stress update time: {:.6} ms",
            averages.get("stress_update_time").unwrap_or(&0.0) * 1000.0
        );
        debug!(
            "  Average velocity update time: {:.6} ms",
            averages.get("velocity_update_time").unwrap_or(&0.0) * 1000.0
        );
        debug!(
            "  Average source time: {:.6} ms",
            averages.get("source_time").unwrap_or(&0.0) * 1000.0
        );
        debug!(
            "  Average total update time: {:.6} ms",
            averages.get("total_update_time").unwrap_or(&0.0) * 1000.0
        );
        debug!(
            "  Memory usage: {} MB",
            self.metrics.memory_usage / (1024 * 1024)
        );
    }

    fn set_nonlinearity_scaling(&mut self, _scaling: f64) {
        // Not applicable for linear elastic waves
        debug!("Nonlinearity scaling not applicable for linear elastic waves");
    }
}

mod tests;
