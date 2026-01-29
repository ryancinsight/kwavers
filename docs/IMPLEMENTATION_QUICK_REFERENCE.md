# Implementation Quick Reference
## Architectural Enhancements from Reference Library Analysis

**Quick Start**: This document provides copy-paste-ready code patterns for implementing recommendations from the architecture analysis.

---

## 1. Factory Methods for Auto-Configuration

### Pattern: Automatic CFL and Grid Spacing

```rust
// File: src/solver/forward/fdtd/config.rs

impl FdtdConfig {
    /// Create configuration automatically satisfying CFL condition
    /// 
    /// # Arguments
    /// * `medium` - Medium properties (determines max wave speed)
    /// * `grid` - Computational grid (determines spatial resolution)
    /// * `cfl` - CFL number (typically 0.95 for stability margin)
    /// * `duration` - Simulation duration in seconds
    ///
    /// # Returns
    /// Configuration with stable time step and appropriate settings
    ///
    /// # Example
    /// ```rust
    /// let config = FdtdConfig::from_medium_and_grid(
    ///     &medium,
    ///     &grid,
    ///     0.95,    // CFL number
    ///     1e-3     // 1 ms duration
    /// )?;
    /// ```
    pub fn from_medium_and_grid(
        medium: &dyn Medium,
        grid: &Grid,
        cfl: f64,
        duration: f64,
    ) -> KwaversResult<Self> {
        // 1. Determine maximum wave speed
        let c_max = medium.max_sound_speed();
        
        // 2. Calculate minimum spatial step
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        
        // 3. Apply CFL condition: dt <= CFL * dx / (c * sqrt(d))
        // where d is number of spatial dimensions
        let dt_stable = cfl * min_dx / (c_max * (3.0_f64).sqrt());
        
        // 4. Calculate number of time steps
        let num_steps = (duration / dt_stable).ceil() as usize;
        
        Ok(Self {
            dt: dt_stable,
            spatial_order: 4,              // Default to 4th-order
            cfl_number: cfl,
            num_steps,
            ..Default::default()
        })
    }
}
```

### Pattern: Grid from Constraints

```rust
// File: src/simulation/constraints.rs

pub struct SimulationConstraints {
    pub cfl_condition: f64,
    pub max_frequency: f64,
    pub points_per_wavelength: f64,
    pub max_memory_gb: Option<f64>,
}

impl SimulationConstraints {
    pub fn new() -> Self {
        Self {
            cfl_condition: 0.95,
            max_frequency: 5e6,
            points_per_wavelength: 4.0,
            max_memory_gb: None,
        }
    }
    
    pub fn with_max_frequency(mut self, freq: f64) -> Self {
        self.max_frequency = freq;
        self
    }
    
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.cfl_condition = cfl;
        self
    }
    
    pub fn with_points_per_wavelength(mut self, ppw: f64) -> Self {
        self.points_per_wavelength = ppw;
        self
    }
    
    /// Automatically determine optimal grid and time step
    ///
    /// # Arguments
    /// * `domain_size` - Physical domain size (Lx, Ly, Lz) in meters
    /// * `medium` - Medium properties
    /// * `duration` - Simulation duration in seconds
    ///
    /// # Returns
    /// Tuple of (Grid, dt) satisfying all constraints
    pub fn solve(
        &self,
        domain_size: (f64, f64, f64),
        medium: &dyn Medium,
        duration: f64,
    ) -> KwaversResult<(Grid, f64)> {
        let c_max = medium.max_sound_speed();
        let c_min = medium.min_sound_speed();
        
        // Determine spatial resolution from wavelength constraint
        let lambda_min = c_min / self.max_frequency;
        let dx_required = lambda_min / self.points_per_wavelength;
        
        let nx = (domain_size.0 / dx_required).ceil() as usize;
        let ny = (domain_size.1 / dx_required).ceil() as usize;
        let nz = (domain_size.2 / dx_required).ceil() as usize;
        
        // Check memory constraint
        if let Some(max_mem_gb) = self.max_memory_gb {
            let fields_per_point = 10;  // Estimate: p, vx, vy, vz, + intermediates
            let bytes_per_field = 8;    // f64
            let estimated_mem_gb = (nx * ny * nz * fields_per_point * bytes_per_field) as f64 / 1e9;
            
            if estimated_mem_gb > max_mem_gb {
                return Err(KwaversError::Config(
                    format!(
                        "Grid {}x{}x{} requires {:.2} GB, exceeds limit {:.2} GB",
                        nx, ny, nz, estimated_mem_gb, max_mem_gb
                    ).into()
                ));
            }
        }
        
        let grid = Grid::new(nx, ny, nz, dx_required, dx_required, dx_required)?;
        
        // Determine time step from CFL
        let dt = self.cfl_condition * dx_required / (c_max * (3.0_f64).sqrt());
        
        Ok((grid, dt))
    }
}

// Usage example:
// let constraints = SimulationConstraints::new()
//     .with_max_frequency(5e6)
//     .with_cfl(0.95);
// let (grid, dt) = constraints.solve((0.05, 0.05, 0.08), &medium, 1e-3)?;
```

---

## 2. Backend Abstraction

### Pattern: Compute Backend Trait

```rust
// File: src/solver/backend/mod.rs

use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex64;

pub trait ComputeBackend: Debug + Send + Sync {
    /// Backend name for logging
    fn name(&self) -> &str;
    
    /// Check if backend is available (e.g., GPU present)
    fn is_available(&self) -> bool;
    
    /// Priority for automatic selection (higher = preferred)
    fn priority(&self) -> usize;
    
    /// Apply spatial derivative operator
    fn spatial_derivative(
        &self,
        field: &Array3<f64>,
        axis: Axis,
        order: usize,
        dx: f64,
    ) -> KwaversResult<Array3<f64>>;
    
    /// FFT forward transform
    fn fft_forward(&self, field: &Array3<f64>) -> KwaversResult<Array3<Complex64>>;
    
    /// FFT inverse transform
    fn fft_inverse(&self, field_k: &Array3<Complex64>) -> KwaversResult<Array3<f64>>;
    
    /// Apply PML absorption layer
    fn apply_pml(
        &self,
        field: &mut Array3<f64>,
        pml_coefficient: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()>;
}

/// CPU backend using Rayon for parallelism
pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU (Rayon)"
    }
    
    fn is_available(&self) -> bool {
        true  // CPU always available
    }
    
    fn priority(&self) -> usize {
        10  // Baseline priority
    }
    
    fn spatial_derivative(
        &self,
        field: &Array3<f64>,
        axis: Axis,
        order: usize,
        dx: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Use existing FDTD derivative implementation
        crate::solver::forward::fdtd::derivatives::compute_derivative(
            field, axis, order, dx
        )
    }
    
    fn fft_forward(&self, field: &Array3<f64>) -> KwaversResult<Array3<Complex64>> {
        // Use existing FFT implementation
        crate::math::fft::fft_3d(field)
    }
    
    fn fft_inverse(&self, field_k: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        crate::math::fft::ifft_3d(field_k)
    }
    
    fn apply_pml(
        &self,
        field: &mut Array3<f64>,
        pml_coefficient: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        use ndarray::Zip;
        Zip::from(field)
            .and(pml_coefficient)
            .par_for_each(|f, &coeff| {
                *f *= (-coeff * dt).exp();
            });
        Ok(())
    }
}

/// GPU backend using WGPU
#[cfg(feature = "gpu")]
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // ... pipeline cache, etc.
}

#[cfg(feature = "gpu")]
impl ComputeBackend for GpuBackend {
    fn name(&self) -> &str {
        "GPU (WGPU)"
    }
    
    fn is_available(&self) -> bool {
        // Check at runtime if GPU is present
        true  // Already constructed, so available
    }
    
    fn priority(&self) -> usize {
        100  // Strongly prefer GPU
    }
    
    // GPU implementations...
}

/// Automatic backend selector
pub struct BackendSelector;

impl BackendSelector {
    /// Select best available backend
    pub fn select() -> Box<dyn ComputeBackend> {
        let mut candidates: Vec<Box<dyn ComputeBackend>> = vec![
            Box::new(CpuBackend::new()),
        ];
        
        #[cfg(feature = "gpu")]
        if let Ok(gpu) = GpuBackend::try_new() {
            candidates.push(Box::new(gpu));
        }
        
        // Sort by priority (descending)
        candidates.sort_by_key(|b| std::cmp::Reverse(b.priority()));
        
        // Return highest priority available backend
        candidates.into_iter().next().expect("At least CPU backend available")
    }
}

// Usage in solver:
// let backend = BackendSelector::select();
// let derivative = backend.spatial_derivative(&pressure, Axis::X, 4, grid.dx)?;
```

---

## 3. Tiered API Design

### Pattern: Simple/Standard/Advanced Tiers

```rust
// File: src/api/simple.rs

/// Simple API for beginners - automatic everything
pub mod simple {
    use crate::*;
    
    /// Run basic acoustic simulation with sensible defaults
    ///
    /// Automatically:
    /// - Creates optimal grid (4 points per wavelength)
    /// - Uses water medium
    /// - Applies PML boundaries
    /// - Selects best solver (FDTD or PSTD based on size)
    ///
    /// # Example
    /// ```rust
    /// let result = kwavers::simple::simulate_acoustic(
    ///     (0.05, 0.05, 0.08),  // 5x5x8 cm domain
    ///     5e6,                  // 5 MHz frequency
    ///     1e-3                  // 1 ms duration
    /// )?;
    /// println!("Max pressure: {:.1} Pa", result.max_pressure);
    /// ```
    pub fn simulate_acoustic(
        domain_size: (f64, f64, f64),
        frequency: f64,
        duration: f64,
    ) -> KwaversResult<SimulationResult> {
        // 1. Create grid automatically
        let constraints = crate::simulation::constraints::SimulationConstraints::new()
            .with_max_frequency(frequency)
            .with_points_per_wavelength(4.0)
            .with_cfl(0.95);
        
        let medium = crate::domain::medium::HomogeneousMedium::water();
        let (grid, dt) = constraints.solve(domain_size, &medium, duration)?;
        
        // 2. Create simple point source at center
        let center = (domain_size.0 / 2.0, domain_size.1 / 2.0, domain_size.2 / 4.0);
        let signal = std::sync::Arc::new(crate::domain::signal::SineWave::new(frequency, 1.0, 0.0));
        let source = crate::domain::source::PointSource::new(center, signal);
        
        // 3. Run simulation
        crate::api::standard::simulate(
            grid,
            medium,
            vec![std::sync::Arc::new(source)],
            duration,
        )
    }
}

// File: src/api/standard.rs

/// Standard API for common usage
pub mod standard {
    use crate::*;
    
    pub fn simulate(
        grid: Grid,
        medium: impl Medium,
        sources: Vec<Arc<dyn Source>>,
        duration: f64,
    ) -> KwaversResult<SimulationResult> {
        // Use builder with reasonable defaults
        let mut sim = crate::simulation::core::SimulationBuilder::new()
            .with_grid(grid)
            .with_medium(&medium)
            .build()?;
        
        for source in sources {
            sim.add_source(source)?;
        }
        
        let num_steps = (duration / sim.config.dt).ceil() as usize;
        sim.run(num_steps, sim.config.dt)
    }
}

// File: src/api/advanced.rs

/// Advanced API for expert users
pub mod advanced {
    use crate::*;
    
    pub struct AdvancedSimulationBuilder {
        grid: Option<Grid>,
        medium: Option<Box<dyn Medium>>,
        sources: Vec<Arc<dyn Source>>,
        sensors: Vec<GridSensorSet>,
        solver_type: Option<SolverType>,
        backend: Option<Box<dyn ComputeBackend>>,
        time_integrator: Option<TimeIntegrator>,
        boundary: Option<Box<dyn Boundary>>,
        plugins: Vec<Box<dyn Plugin>>,
        features: Vec<SolverFeature>,
    }
    
    impl AdvancedSimulationBuilder {
        pub fn new() -> Self {
            Self {
                grid: None,
                medium: None,
                sources: Vec::new(),
                sensors: Vec::new(),
                solver_type: None,
                backend: None,
                time_integrator: None,
                boundary: None,
                plugins: Vec::new(),
                features: Vec::new(),
            }
        }
        
        pub fn with_grid(mut self, grid: Grid) -> Self {
            self.grid = Some(grid);
            self
        }
        
        pub fn with_medium(mut self, medium: Box<dyn Medium>) -> Self {
            self.medium = Some(medium);
            self
        }
        
        pub fn with_solver(mut self, solver_type: SolverType) -> Self {
            self.solver_type = Some(solver_type);
            self
        }
        
        pub fn with_backend(mut self, backend: Box<dyn ComputeBackend>) -> Self {
            self.backend = Some(backend);
            self
        }
        
        pub fn add_plugin(mut self, plugin: Box<dyn Plugin>) -> Self {
            self.plugins.push(plugin);
            self
        }
        
        pub fn enable_feature(mut self, feature: SolverFeature) -> Self {
            self.features.push(feature);
            self
        }
        
        pub fn build(self) -> KwaversResult<AdvancedSimulation> {
            // Construct fully customized simulation
            todo!("Full advanced builder implementation")
        }
    }
}
```

---

## 4. Domain Builder for Clinical Applications

### Pattern: Anatomical Layer Construction

```rust
// File: src/clinical/domains/builder.rs

pub struct AnatomicalDomainBuilder {
    grid: Grid,
    layers: Vec<TissueLayer>,
    features: Vec<GeometricFeature>,
}

pub struct TissueLayer {
    pub name: String,
    pub thickness: f64,  // meters
    pub properties: TissueProperties,
}

#[derive(Debug, Clone)]
pub struct TissueProperties {
    pub sound_speed: f64,        // m/s
    pub density: f64,            // kg/m³
    pub attenuation_coeff: f64,  // dB/(cm·MHz)
    pub attenuation_power: f64,  // Power law exponent
    pub nonlinearity: f64,       // B/A parameter
}

impl TissueProperties {
    /// Human skin properties
    pub fn skin() -> Self {
        Self {
            sound_speed: 1540.0,
            density: 1100.0,
            attenuation_coeff: 0.62,
            attenuation_power: 1.0,
            nonlinearity: 6.0,
        }
    }
    
    /// Subcutaneous fat
    pub fn fat() -> Self {
        Self {
            sound_speed: 1450.0,
            density: 950.0,
            attenuation_coeff: 0.48,
            attenuation_power: 1.0,
            nonlinearity: 10.0,
        }
    }
    
    /// Skeletal muscle
    pub fn muscle() -> Self {
        Self {
            sound_speed: 1580.0,
            density: 1050.0,
            attenuation_coeff: 1.09,
            attenuation_power: 1.0,
            nonlinearity: 7.4,
        }
    }
    
    /// Blood
    pub fn blood() -> Self {
        Self {
            sound_speed: 1570.0,
            density: 1060.0,
            attenuation_coeff: 0.15,
            attenuation_power: 1.2,
            nonlinearity: 6.1,
        }
    }
}

impl AnatomicalDomainBuilder {
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            layers: Vec::new(),
            features: Vec::new(),
        }
    }
    
    /// Add a planar tissue layer (layers stacked in +z direction)
    pub fn add_layer(
        mut self,
        name: impl Into<String>,
        thickness: f64,
        properties: TissueProperties,
    ) -> Self {
        self.layers.push(TissueLayer {
            name: name.into(),
            thickness,
            properties,
        });
        self
    }
    
    /// Add a geometric feature (cylinder, sphere, etc.)
    pub fn add_feature(mut self, feature: GeometricFeature) -> Self {
        self.features.push(feature);
        self
    }
    
    /// Build the heterogeneous medium
    pub fn build(self) -> KwaversResult<HeterogeneousMedium> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let mut sound_speed = Array3::zeros((nx, ny, nz));
        let mut density = Array3::zeros((nx, ny, nz));
        let mut attenuation = Array3::zeros((nx, ny, nz));
        
        // Apply layers sequentially in z-direction
        let mut z_offset = 0.0;
        for layer in &self.layers {
            let z_start_idx = (z_offset / self.grid.dz) as usize;
            let z_end_idx = ((z_offset + layer.thickness) / self.grid.dz) as usize;
            
            for iz in z_start_idx..z_end_idx.min(nz) {
                for iy in 0..ny {
                    for ix in 0..nx {
                        sound_speed[[ix, iy, iz]] = layer.properties.sound_speed;
                        density[[ix, iy, iz]] = layer.properties.density;
                        attenuation[[ix, iy, iz]] = layer.properties.attenuation_coeff;
                    }
                }
            }
            z_offset += layer.thickness;
        }
        
        // Apply geometric features (override layers where applicable)
        for feature in &self.features {
            feature.apply_to_arrays(
                &mut sound_speed,
                &mut density,
                &mut attenuation,
                &self.grid,
            )?;
        }
        
        HeterogeneousMedium::new(&self.grid, sound_speed, density, attenuation)
    }
}

/// Geometric features for vessels, bones, etc.
#[derive(Debug, Clone)]
pub enum GeometricFeature {
    Cylinder {
        center: (f64, f64, f64),
        radius: f64,
        axis: Axis,
        properties: TissueProperties,
    },
    Sphere {
        center: (f64, f64, f64),
        radius: f64,
        properties: TissueProperties,
    },
}

impl GeometricFeature {
    pub fn cylinder_vessel(
        center: (f64, f64, f64),
        radius: f64,
        properties: TissueProperties,
    ) -> Self {
        Self::Cylinder {
            center,
            radius,
            axis: Axis::Z,  // Along z-axis by default
            properties,
        }
    }
    
    fn apply_to_arrays(
        &self,
        sound_speed: &mut Array3<f64>,
        density: &mut Array3<f64>,
        attenuation: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        match self {
            Self::Cylinder { center, radius, axis, properties } => {
                for iz in 0..grid.nz {
                    for iy in 0..grid.ny {
                        for ix in 0..grid.nx {
                            let (x, y, z) = grid.index_to_position(ix, iy, iz);
                            
                            // Check if point is inside cylinder
                            let dist = match axis {
                                Axis::Z => ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt(),
                                // ... other axes
                                _ => continue,
                            };
                            
                            if dist <= *radius {
                                sound_speed[[ix, iy, iz]] = properties.sound_speed;
                                density[[ix, iy, iz]] = properties.density;
                                attenuation[[ix, iy, iz]] = properties.attenuation_coeff;
                            }
                        }
                    }
                }
            }
            Self::Sphere { center, radius, properties } => {
                // Similar implementation for sphere
            }
        }
        Ok(())
    }
}

// Usage example:
// let medium = AnatomicalDomainBuilder::new(grid)
//     .add_layer("skin", 2e-3, TissueProperties::skin())
//     .add_layer("fat", 5e-3, TissueProperties::fat())
//     .add_feature(GeometricFeature::cylinder_vessel(
//         (0.025, 0.025, 0.010),
//         2e-3,
//         TissueProperties::blood()
//     ))
//     .build()?;
```

---

## 5. Configuration Validation

### Pattern: Pre-Simulation Checks

```rust
// File: src/simulation/validation.rs

pub trait Validatable {
    fn validate(&self) -> KwaversResult<()>;
    fn warnings(&self) -> Vec<String>;
}

impl Validatable for FdtdConfig {
    fn validate(&self) -> KwaversResult<()> {
        let mut errors = Vec::new();
        
        // Check time step is positive
        if self.dt <= 0.0 {
            errors.push("Time step must be positive".to_string());
        }
        
        // Check CFL condition (assuming known max speed and grid spacing)
        if let Some(cfl_max) = self.compute_cfl_number() {
            if cfl_max > 1.0 {
                errors.push(format!(
                    "CFL condition violated: {:.2} > 1.0 (simulation will be unstable)",
                    cfl_max
                ));
            }
        }
        
        // Check spatial order is valid
        if ![2, 4, 6, 8].contains(&self.spatial_order) {
            errors.push(format!(
                "Spatial order {} not supported (use 2, 4, 6, or 8)",
                self.spatial_order
            ));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(KwaversError::Config(
                format!("Configuration validation failed:\n{}", errors.join("\n")).into()
            ))
        }
    }
    
    fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // Warn if CFL is close to stability limit
        if let Some(cfl) = self.compute_cfl_number() {
            if cfl > 0.9 && cfl <= 1.0 {
                warnings.push(format!(
                    "CFL number {:.2} is close to stability limit. Consider reducing dt by 10-20%",
                    cfl
                ));
            }
        }
        
        // Warn if spatial resolution may be insufficient
        if let Some(ppw) = self.points_per_wavelength() {
            if ppw < 4.0 {
                warnings.push(format!(
                    "Grid resolution {:.1} points/wavelength is low. Recommend ≥4 for accuracy",
                    ppw
                ));
            }
        }
        
        warnings
    }
}

impl FdtdConfig {
    fn compute_cfl_number(&self) -> Option<f64> {
        // Would need grid and medium references to compute
        // This is a placeholder
        None
    }
    
    fn points_per_wavelength(&self) -> Option<f64> {
        // Would need frequency and grid spacing
        None
    }
}

// Usage:
// config.validate()?;  // Fails if errors
// for warning in config.warnings() {
//     eprintln!("Warning: {}", warning);
// }
```

---

## 6. Multi-Physics Coupling

### Pattern: Explicit Coupling Interface

```rust
// File: src/solver/multiphysics/coupling.rs

pub trait PhysicsCoupling: Debug + Send + Sync {
    /// Physics domains this coupling connects
    fn coupled_domains(&self) -> (PhysicsDomain, PhysicsDomain);
    
    /// Apply coupling effect from source to target
    fn apply_coupling(
        &self,
        source_field: &Array3<f64>,
        target_field: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()>;
    
    /// Coupling strength at position
    fn coupling_coefficient(&self, ix: usize, iy: usize, iz: usize) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsDomain {
    Acoustic,
    Thermal,
    Elastic,
    Electromagnetic,
}

/// Acoustic → Thermal coupling (ultrasound heating)
pub struct AcousticThermalCoupling {
    /// Absorption coefficient α [Np/m]
    absorption: Array3<f64>,
    /// Density ρ [kg/m³]
    density: Array3<f64>,
    /// Specific heat capacity c_p [J/(kg·K)]
    specific_heat: Array3<f64>,
}

impl AcousticThermalCoupling {
    pub fn new(medium: &dyn Medium, grid: &Grid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut absorption = Array3::zeros((nx, ny, nz));
        let mut density = Array3::zeros((nx, ny, nz));
        let mut specific_heat = Array3::zeros((nx, ny, nz));
        
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    absorption[[ix, iy, iz]] = medium.absorption_nepers(ix, iy, iz);
                    density[[ix, iy, iz]] = medium.density(ix, iy, iz);
                    specific_heat[[ix, iy, iz]] = medium.specific_heat(ix, iy, iz);
                }
            }
        }
        
        Self { absorption, density, specific_heat }
    }
}

impl PhysicsCoupling for AcousticThermalCoupling {
    fn coupled_domains(&self) -> (PhysicsDomain, PhysicsDomain) {
        (PhysicsDomain::Acoustic, PhysicsDomain::Thermal)
    }
    
    fn apply_coupling(
        &self,
        pressure: &Array3<f64>,
        temperature: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        use ndarray::Zip;
        
        // Heat deposition: dT/dt = α * I / (ρ * c_p)
        // where I ≈ p^2 / (2 * ρ * c) is acoustic intensity
        
        Zip::from(temperature)
            .and(pressure)
            .and(&self.absorption)
            .and(&self.density)
            .and(&self.specific_heat)
            .for_each(|temp, &p, &alpha, &rho, &cp| {
                // Simplified: actual intensity calculation needs velocity too
                let intensity_approx = p * p / (2.0 * rho);
                let heat_rate = alpha * intensity_approx / (rho * cp);
                *temp += heat_rate * dt;
            });
        
        Ok(())
    }
    
    fn coupling_coefficient(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        self.absorption[[ix, iy, iz]]
    }
}

/// Multi-physics solver orchestrator
pub struct MultiPhysicsSolver {
    solvers: HashMap<PhysicsDomain, Box<dyn Plugin>>,
    couplings: Vec<Box<dyn PhysicsCoupling>>,
    execution_order: Vec<PhysicsDomain>,
}

impl MultiPhysicsSolver {
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            couplings: Vec::new(),
            execution_order: Vec::new(),
        }
    }
    
    pub fn add_solver(
        mut self,
        domain: PhysicsDomain,
        solver: Box<dyn Plugin>,
    ) -> Self {
        self.solvers.insert(domain, solver);
        self.execution_order.push(domain);
        self
    }
    
    pub fn add_coupling(mut self, coupling: Box<dyn PhysicsCoupling>) -> Self {
        self.couplings.push(coupling);
        self
    }
    
    /// Operator splitting: solve each physics, then apply couplings
    pub fn step_operator_splitting(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // 1. Solve each domain independently
        for domain in &self.execution_order {
            let solver = self.solvers.get_mut(domain).unwrap();
            // solver.update(...)?;
        }
        
        // 2. Apply coupling terms
        for coupling in &self.couplings {
            // Get source and target fields from solver states
            // coupling.apply_coupling(source, target, grid, dt)?;
        }
        
        Ok(())
    }
}

// Usage:
// let multiphysics = MultiPhysicsSolver::new()
//     .add_solver(PhysicsDomain::Acoustic, Box::new(acoustic_solver))
//     .add_solver(PhysicsDomain::Thermal, Box::new(thermal_solver))
//     .add_coupling(Box::new(AcousticThermalCoupling::new(&medium, &grid)));
```

---

## 7. Testing Patterns

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factory_method_satisfies_cfl() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        
        let config = FdtdConfig::from_medium_and_grid(
            &medium,
            &grid,
            0.95,
            1e-3,
        ).unwrap();
        
        // Verify CFL condition: c*dt/dx <= CFL_target
        let c_max = medium.max_sound_speed();
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_actual = c_max * config.dt / min_dx * (3.0_f64).sqrt();
        
        assert!(cfl_actual <= 0.95);
        assert!(cfl_actual > 0.90);  // Should be close to target
    }
    
    #[test]
    fn test_domain_builder_creates_layers() {
        let grid = Grid::new(32, 32, 64, 1e-4, 1e-4, 1e-4).unwrap();
        
        let medium = AnatomicalDomainBuilder::new(grid.clone())
            .add_layer("skin", 2e-3, TissueProperties::skin())
            .add_layer("fat", 3e-3, TissueProperties::fat())
            .build()
            .unwrap();
        
        // Check skin layer (first 20 grid points in z)
        let c_skin = medium.sound_speed(16, 16, 10);
        assert!((c_skin - 1540.0).abs() < 1.0);
        
        // Check fat layer (next 30 grid points in z)
        let c_fat = medium.sound_speed(16, 16, 35);
        assert!((c_fat - 1450.0).abs() < 1.0);
    }
}
```

### Integration Test Template

```rust
// tests/integration/api_tiers.rs

#[test]
fn test_simple_api_runs_successfully() {
    let result = kwavers::simple::simulate_acoustic(
        (0.05, 0.05, 0.08),
        5e6,
        1e-4,
    );
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.max_pressure.is_finite());
    assert!(result.max_pressure > 0.0);
}

#[test]
fn test_backend_abstraction_produces_same_result() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let field = Array3::from_elem((32, 32, 32), 1.0);
    
    let cpu_backend = CpuBackend::new();
    let result_cpu = cpu_backend.spatial_derivative(
        &field,
        Axis::X,
        4,
        grid.dx,
    ).unwrap();
    
    #[cfg(feature = "gpu")]
    {
        let gpu_backend = GpuBackend::new().unwrap();
        let result_gpu = gpu_backend.spatial_derivative(
            &field,
            Axis::X,
            4,
            grid.dx,
        ).unwrap();
        
        // Results should match within numerical precision
        assert_arrays_close(&result_cpu, &result_gpu, 1e-10);
    }
}
```

---

## Quick Reference Checklist

When implementing a new feature from the analysis:

- [ ] **Factory Method**: Does it automatically satisfy constraints (CFL, memory, etc.)?
- [ ] **Backend Agnostic**: Can it work with both CPU and GPU backends?
- [ ] **Validation**: Does it check configuration before running?
- [ ] **Documentation**: Examples for simple/standard/advanced users?
- [ ] **Testing**: Unit tests, integration tests, validation against reference?
- [ ] **Error Messages**: Helpful guidance for common mistakes?
- [ ] **Backward Compatibility**: Can existing code continue to work?

---

## Additional Resources

- **Full Analysis**: `ULTRASOUND_SIMULATION_ARCHITECTURE_ANALYSIS.md`
- **Executive Summary**: `ARCHITECTURE_ANALYSIS_EXECUTIVE_SUMMARY.md`
- **Reference Libraries**:
  - [jWave](https://github.com/ucl-bug/jwave)
  - [k-Wave](https://github.com/ucl-bug/k-wave)
  - [Fullwave25](https://github.com/pinton-lab/fullwave25)

---

**Last Updated**: 2026-01-28
