# Ultrasound Simulation Architecture Analysis
## Reference Repository Patterns for Kwavers Enhancement

**Date**: 2026-01-28  
**Purpose**: Extract architectural patterns from leading ultrasound simulation libraries  
**Target**: Improve kwavers library structure and extensibility

---

## Executive Summary

This analysis examines six leading ultrasound and wave simulation platforms to extract architectural patterns, module organization strategies, and best practices applicable to kwavers:

1. **jWave** (JAX/Python) - Differentiable, GPU-accelerated, functional composition
2. **k-Wave** (MATLAB) - k-space pseudospectral methods, mature API design
3. **k-wave-python** - Python bindings, backward compatibility patterns
4. **OptimUS** - BEM frequency-domain, multi-domain coupling
5. **Fullwave 2.5** - Clinical workflows, high-order FDTD, GPU optimization
6. **SimSonic** - FDTD elastodynamics, heterogeneous media

### Key Findings

**Strong Architectural Patterns Identified:**
- Plugin/composition over inheritance for solver extensibility
- Functional parameter passing vs. stateful solver objects
- Factory methods for automatic constraint satisfaction (CFL, stability)
- Explicit separation: Domain → Medium → Source → Sensor → Solver
- Backend abstraction layers for CPU/GPU acceleration

**Kwavers Current State:**
- ✅ Strong layered architecture (8 layers, unidirectional dependencies)
- ✅ Plugin system already implemented (`domain::plugin::Plugin` trait)
- ✅ Trait-based extensibility (`Medium`, `Source`, `Boundary`)
- 🔄 Opportunity: Enhance solver backend abstraction patterns
- 🔄 Opportunity: Improve factory methods for configuration

---

## 1. Module Organization & Separation of Concerns

### 1.1 jWave: Functional Composition Pattern

**Architecture:**
```
jwave/
├── geometry.py          # Domain, Medium, TimeAxis (spatial/temporal specs)
├── acoustics/
│   ├── time_varying.py  # Simulation functions (pure, stateless)
│   └── operators.py     # Differential operators
├── utils/              # Utilities (FourierSeries, grids)
└── notebooks/          # Examples
```

**Key Principles:**
1. **Pure Functions over Classes**: Simulation logic as pure functions accepting domain objects
2. **Composition over Inheritance**: `FourierSeries` wraps initial conditions, not subclassing
3. **Explicit Parameter Passing**: `simulate_wave_propagation(medium, time_axis, p0=p0)`
4. **Domain as Single Source of Truth**: All components reference the same `Domain` object

**Code Pattern (JAX/Python):**
```python
# Domain specification
domain = Domain(shape=(128, 128, 128), dx=1e-3)

# Medium as parameter container
medium = Medium(
    domain=domain,
    sound_speed=1500.0,  # Accepts scalar, array, or Field
    density=1000.0,
    attenuation=0.5
)

# TimeAxis with factory method
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=1e-3)

# Simulation as pure function
@jit
def run_simulation(medium, time_axis, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)

p_final = run_simulation(medium, time_axis, initial_pressure)
```

**Benefits:**
- JAX auto-differentiation: gradients through entire simulation
- JIT compilation: GPU acceleration without code changes
- Testability: Pure functions, no hidden state
- Composability: Easy to chain simulations, wrap in optimizers

**Kwavers Current Approach:**
```rust
// Kwavers uses trait-based composition
let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3)?;
let medium = HomogeneousMedium::water(&grid);
let solver = FdtdSolver::new(config, &grid, &medium, source)?;

// Plugin-based execution
let mut executor = PluginExecutor::new(strategy);
executor.add_plugin(Box::new(FdtdPlugin::new(solver)));
executor.execute(&mut fields, &grid, &medium, dt, t, &mut context)?;
```

**Recommendation 1.1: Enhance Factory Methods**

Add factory methods that automatically satisfy constraints (similar to jWave's `TimeAxis.from_medium`):

```rust
// Proposed enhancement for kwavers
impl FdtdConfig {
    /// Create configuration automatically satisfying CFL condition
    pub fn from_medium_and_grid(
        medium: &dyn Medium,
        grid: &Grid,
        cfl: f64,
        duration: f64,
    ) -> KwaversResult<Self> {
        let c_max = medium.max_sound_speed();
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let dt_stable = cfl * min_dx / (c_max * (3.0_f64).sqrt());
        let num_steps = (duration / dt_stable).ceil() as usize;
        
        Ok(Self {
            dt: dt_stable,
            spatial_order: 4,
            cfl_number: cfl,
            ..Default::default()
        })
    }
}

// Usage
let config = FdtdConfig::from_medium_and_grid(&medium, &grid, 0.95, 1e-3)?;
```

**Recommendation 1.2: Functional Simulation API**

Provide a high-level functional API alongside the existing trait-based system:

```rust
// Proposed: Functional simulation interface
pub mod functional {
    pub fn simulate_acoustic_wave(
        grid: &Grid,
        medium: &dyn Medium,
        sources: &[&dyn Source],
        duration: f64,
        config: SimulationConfig,
    ) -> KwaversResult<SimulationResult> {
        // Internally uses plugin system
        // Automatically selects solver based on config
        // Returns immutable result
    }
}

// Usage
let result = functional::simulate_acoustic_wave(
    &grid,
    &medium,
    &[&point_source],
    1e-3,
    SimulationConfig::default_fdtd()
)?;
```

---

### 1.2 k-Wave: Modular Pseudospectral Architecture

**Architecture:**
```
k-Wave/
├── kspaceFirstOrder2D.m   # 2D k-space solver
├── kspaceFirstOrder3D.m   # 3D k-space solver
├── kWaveGrid.m            # Grid specification
├── kWaveMedium.m          # Material properties
├── makeSource.m           # Source creation utilities
└── utils/                 # Supporting functions
```

**Key Principles:**
1. **Function-Based API**: Each solver is a standalone function, not a class
2. **Struct-Based Configuration**: MATLAB structs for medium, source, sensor
3. **Backend Abstraction**: Transparent CPU/GPU selection via binary availability
4. **Dimension-Specific Implementations**: Separate 1D/2D/3D solvers, not generic

**Code Pattern (MATLAB):**
```matlab
% Grid definition
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% Medium properties (struct)
medium.sound_speed = 1500;     % scalar or matrix
medium.density = 1000;
medium.alpha_coeff = 0.75;
medium.alpha_power = 1.5;

% Source specification
source.p0 = initial_pressure;  % initial pressure distribution
source.p_mask = source_mask;   % source geometry

% Sensor specification
sensor.mask = sensor_mask;     % sensor positions

% Run simulation
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor);
```

**Solver Architecture:**
- **k-space Pseudospectral Method**: Spatial gradients via FFT
- **Temporal Integration**: k-space corrected finite-difference
- **Absorption Model**: Fractional Laplacian for power-law absorption
- **PML Boundaries**: Split-field perfectly matched layers

**Benefits:**
- High accuracy with fewer grid points (vs FDTD)
- Natural handling of power-law absorption
- Efficient for homogeneous and heterogeneous media
- Automatic backend selection (MATLAB → C++ → CUDA)

**Kwavers Current Approach:**

Kwavers has PSTD implementation but could enhance k-space methods:

```rust
// Current: src/solver/forward/pstd/mod.rs
pub struct PSTDSolver {
    config: PSTDConfig,
    // Implementation details
}

// Opportunity: Add k-space specific operators
```

**Recommendation 1.3: k-Space Operator Library**

Implement k-Wave's k-space correction methods as reusable operators:

```rust
// Proposed: src/solver/forward/pstd/kspace/operators.rs
pub mod kspace {
    use crate::math::fft::FFTPlan;
    
    /// k-space gradient operator with spectral correction
    pub struct KSpaceGradient {
        kx: Array3<f64>,  // Wavenumber grids
        ky: Array3<f64>,
        kz: Array3<f64>,
        fft_plan: FFTPlan,
    }
    
    impl KSpaceGradient {
        /// Compute gradient using Fourier collocation
        pub fn apply(&self, field: &Array3<f64>) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
            let field_k = self.fft_plan.forward(field);
            let grad_x = self.fft_plan.inverse(&(1i * &self.kx * &field_k));
            let grad_y = self.fft_plan.inverse(&(1i * &self.ky * &field_k));
            let grad_z = self.fft_plan.inverse(&(1i * &self.kz * &field_k));
            (grad_x, grad_y, grad_z)
        }
    }
    
    /// k-space corrected time integration
    pub struct KSpaceTimeCorrection {
        /// Dispersion correction factor: sinc(kc*dt/2)
        correction: Array3<f64>,
    }
    
    impl KSpaceTimeCorrection {
        pub fn new(grid: &Grid, c: f64, dt: f64) -> Self {
            // Compute correction factor for each k-mode
            // k-Wave Eq: correction = sinc(k*c*dt/2)
        }
        
        pub fn apply(&self, field_k: &Array3<Complex64>) -> Array3<Complex64> {
            field_k * &self.correction
        }
    }
}
```

**Recommendation 1.4: Power-Law Absorption via Fractional Laplacian**

k-Wave's absorption model using fractional Laplacian:

```rust
// Proposed: src/physics/acoustics/absorption/power_law.rs
pub struct PowerLawAbsorption {
    alpha_coeff: Array3<f64>,  // Absorption coefficient
    alpha_power: Array3<f64>,  // Power law exponent (typically 1.5)
}

impl PowerLawAbsorption {
    /// Apply absorption operator via fractional Laplacian
    /// k-Wave method: absorption ~ ω^y where y = alpha_power
    pub fn apply_frequency_domain(
        &self,
        pressure_k: &Array3<Complex64>,
        omega: f64,
        dt: f64,
    ) -> Array3<Complex64> {
        // Absorption factor: exp(-alpha * omega^y * dt)
        let absorption = self.alpha_coeff.mapv(|a| {
            (-a * omega.powf(self.alpha_power[(0,0,0)]) * dt).exp()
        });
        pressure_k * absorption
    }
}
```

---

### 1.3 Fullwave 2.5: Clinical Workflow Integration

**Architecture:**
```
fullwave25/
├── fullwave/
│   ├── simulation/        # Core simulation engine
│   ├── domains/           # Anatomical domain builders
│   ├── transducers/       # Ultrasound array models
│   └── reconstruction/    # Image formation
├── examples/
│   ├── linear_array/      # Clinical transducer configs
│   └── convex_array/
└── utils/
    └── MediumBuilder.py   # Geometric medium construction
```

**Key Principles:**
1. **Two-Phase Workflow**: 2D prototyping → 3D production
2. **Domain Builder Pattern**: Geometric operations for anatomical structures
3. **Transducer Abstraction**: Linear, convex, phased arrays as first-class objects
4. **Multi-GPU Scaling**: Depth-dimension domain decomposition

**Code Pattern (Python):**
```python
# Domain builder for anatomical structures
builder = MediumBuilder(grid)
builder.add_layer("skin", thickness_mm=2, sound_speed=1540, density=1100)
builder.add_layer("fat", thickness_mm=5, sound_speed=1450, density=950)
builder.add_layer("muscle", thickness_mm=20, sound_speed=1580, density=1050)
medium = builder.build()

# Transducer specification
transducer = LinearArray(
    num_elements=128,
    pitch=0.3e-3,
    element_width=0.28e-3,
    frequency=5e6
)

# Simulation with clinical parameters
sim = FullwaveSimulation(grid, medium, transducer)
sim.run_plane_wave(angle_deg=0)
sim.run_focused(focus_depth_mm=30)
```

**Solver Details:**
- **High-order FDTD**: 8th-order spatial, 4th-order temporal
- **Staggered Grid**: Pressure-velocity formulation
- **Attenuation**: Multiple relaxation processes (frequency-dependent)
- **GPU Optimization**: C-array ordering, depth-wise decomposition

**Kwavers Current Approach:**

Kwavers has clinical layer but could enhance workflow integration:

```rust
// Current: src/clinical/imaging/mod.rs
pub mod imaging;
pub mod therapy;
pub mod safety;
```

**Recommendation 1.5: Domain Builder Pattern for Clinical Applications**

```rust
// Proposed: src/clinical/domains/builder.rs
pub struct AnatomicalDomainBuilder<'a> {
    grid: &'a Grid,
    layers: Vec<TissueLayer>,
    geometric_features: Vec<GeometricFeature>,
}

pub struct TissueLayer {
    name: String,
    thickness: f64,
    properties: TissueProperties,
}

pub struct TissueProperties {
    sound_speed: f64,
    density: f64,
    attenuation_coeff: f64,
    attenuation_power: f64,
}

impl<'a> AnatomicalDomainBuilder<'a> {
    pub fn new(grid: &'a Grid) -> Self {
        Self {
            grid,
            layers: Vec::new(),
            geometric_features: Vec::new(),
        }
    }
    
    /// Add a planar tissue layer
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
    
    /// Add a geometric feature (e.g., rib, vessel)
    pub fn add_feature(mut self, feature: GeometricFeature) -> Self {
        self.geometric_features.push(feature);
        self
    }
    
    /// Build the heterogeneous medium
    pub fn build(self) -> KwaversResult<HeterogeneousMedium> {
        let mut sound_speed = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        let mut density = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        let mut attenuation = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        
        // Apply layers in sequence
        let mut z_offset = 0.0;
        for layer in &self.layers {
            let z_start = (z_offset / self.grid.dz) as usize;
            let z_end = ((z_offset + layer.thickness) / self.grid.dz) as usize;
            
            for iz in z_start..z_end.min(self.grid.nz) {
                for iy in 0..self.grid.ny {
                    for ix in 0..self.grid.nx {
                        sound_speed[[ix, iy, iz]] = layer.properties.sound_speed;
                        density[[ix, iy, iz]] = layer.properties.density;
                        attenuation[[ix, iy, iz]] = layer.properties.attenuation_coeff;
                    }
                }
            }
            z_offset += layer.thickness;
        }
        
        // Apply geometric features
        for feature in &self.geometric_features {
            feature.apply(&mut sound_speed, &mut density, &mut attenuation, self.grid)?;
        }
        
        HeterogeneousMedium::new(self.grid, sound_speed, density, attenuation)
    }
}

// Usage
let medium = AnatomicalDomainBuilder::new(&grid)
    .add_layer("skin", 2e-3, TissueProperties::skin())
    .add_layer("fat", 5e-3, TissueProperties::fat())
    .add_layer("muscle", 20e-3, TissueProperties::muscle())
    .add_feature(GeometricFeature::cylinder_vessel(
        center: (0.01, 0.01, 0.005),
        radius: 2e-3,
        properties: TissueProperties::blood()
    ))
    .build()?;
```

**Recommendation 1.6: Transducer Array Abstractions**

```rust
// Proposed: src/domain/source/arrays/mod.rs
pub mod arrays {
    pub struct LinearArray {
        pub num_elements: usize,
        pub pitch: f64,          // Element spacing
        pub element_width: f64,
        pub element_height: f64,
        pub frequency: f64,
    }
    
    impl LinearArray {
        /// Generate plane wave excitation
        pub fn plane_wave_delays(&self, angle_rad: f64) -> Vec<f64> {
            (0..self.num_elements)
                .map(|i| {
                    let x = i as f64 * self.pitch;
                    x * angle_rad.sin() / self.sound_speed
                })
                .collect()
        }
        
        /// Generate focused excitation
        pub fn focused_delays(&self, focus: (f64, f64, f64)) -> Vec<f64> {
            (0..self.num_elements)
                .map(|i| {
                    let x = i as f64 * self.pitch;
                    let dist = ((x - focus.0).powi(2) + focus.2.powi(2)).sqrt();
                    dist / self.sound_speed
                })
                .collect()
        }
        
        /// Convert to collection of point sources
        pub fn to_sources(&self, signal: Arc<dyn Signal>) -> Vec<TimeVaryingSource> {
            // Generate individual element sources
        }
    }
    
    pub struct ConvexArray {
        pub num_elements: usize,
        pub radius: f64,         // Curvature radius
        pub pitch_angle: f64,    // Angular pitch
        pub frequency: f64,
    }
    
    // Similar methods for curved geometry
}
```

---

## 2. Solver Architecture Patterns

### 2.1 Backend Abstraction: Strategy Pattern

**k-Wave Approach:**
- High-level API remains constant
- Backend selection: MATLAB → C++ → CUDA (transparent to user)
- Performance tiers: Reference, Optimized, GPU-accelerated

**Fullwave Approach:**
- Python wrapper over CUDA/C kernel
- Multi-GPU domain decomposition
- Automatic device selection

**Kwavers Current Implementation:**

Already has good foundation with plugin system:

```rust
// Current: src/domain/plugin/mod.rs
pub trait Plugin: Debug + Send + Sync {
    fn update(&mut self, fields: &mut Array4<f64>, ...) -> KwaversResult<()>;
}

// Current: src/solver/plugin/execution.rs
pub trait ExecutionStrategy {
    fn execute(&self, plugins: &[&mut dyn Plugin], ...) -> KwaversResult<()>;
}
```

**Recommendation 2.1: Explicit Backend Trait**

Add an explicit backend abstraction for computational kernels:

```rust
// Proposed: src/solver/backend/mod.rs
pub trait ComputeBackend: Debug + Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn preferred_order(&self) -> usize;  // For automatic selection
    
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
    
    /// Apply PML absorption
    fn apply_pml(
        &self,
        field: &mut Array3<f64>,
        pml_mask: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()>;
}

pub struct CpuBackend {
    num_threads: usize,
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub struct BackendSelector;

impl BackendSelector {
    /// Automatically select best available backend
    pub fn select() -> Box<dyn ComputeBackend> {
        #[cfg(feature = "gpu")]
        if GpuBackend::is_available() {
            return Box::new(GpuBackend::new());
        }
        
        Box::new(CpuBackend::new())
    }
}

// Usage in solver
impl FdtdSolver {
    pub fn new_with_backend(
        config: FdtdConfig,
        grid: &Grid,
        medium: &dyn Medium,
        backend: Box<dyn ComputeBackend>,
    ) -> KwaversResult<Self> {
        // Use backend for all computations
    }
}
```

---

### 2.2 State Management: Functional vs Object-Oriented

**jWave Pattern (Functional):**
```python
# State is explicit function parameter
def simulate(medium, time_axis, p0):
    def step(fields, n):
        p, u, rho = fields
        # Update equations
        return [p_new, u_new, rho_new], sensors(p, u, rho)
    
    # JAX scan handles loop
    final_fields, sensor_data = jax.lax.scan(step, initial_fields, time_steps)
    return final_fields
```

**k-Wave Pattern (Procedural):**
```matlab
% State is implicit in function variables
function sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor)
    % Initialize fields
    p = source.p0;
    ux = zeros(size); uy = zeros(size); uz = zeros(size);
    
    % Time loop
    for t = 1:Nt
        % Update equations
        p = p - dt * rho * c^2 * (duxdx + duydy + duzdz);
        ux = ux - dt / rho * dpdx;
        % ... etc
        
        % Record sensors
        sensor_data(t) = p(sensor.mask);
    end
end
```

**Kwavers Current Pattern (OO with Plugin):**
```rust
pub struct FdtdSolver {
    config: FdtdConfig,
    fields: Array4<f64>,  // State stored in solver
    // ...
}

impl FdtdSolver {
    pub fn step(&mut self, dt: f64) -> KwaversResult<()> {
        // Mutate internal state
    }
}
```

**Recommendation 2.2: Hybrid Approach**

Provide both stateful (OO) and stateless (functional) APIs:

```rust
// Stateful API (current, keep for compatibility)
pub struct FdtdSolver {
    state: SolverState,
    config: FdtdConfig,
}

impl FdtdSolver {
    pub fn step(&mut self) -> KwaversResult<()> {
        self.state = Self::step_pure(&self.state, &self.config)?;
        Ok(())
    }
    
    /// Pure function version for composability
    fn step_pure(state: &SolverState, config: &FdtdConfig) -> KwaversResult<SolverState> {
        // Immutable computation
        let new_pressure = Self::update_pressure(&state.pressure, &state.velocity, config);
        let new_velocity = Self::update_velocity(&state.velocity, &new_pressure, config);
        
        Ok(SolverState {
            pressure: new_pressure,
            velocity: new_velocity,
            time: state.time + config.dt,
        })
    }
}

// Functional API (new, for advanced use cases)
pub mod functional {
    pub fn fdtd_step(
        fields: &PhysicsState,
        grid: &Grid,
        medium: &dyn Medium,
        config: &FdtdConfig,
    ) -> KwaversResult<PhysicsState> {
        // Pure function, no mutation
        // Enables:
        // - Easy testing
        // - Parallel execution
        // - Optimization (future auto-diff)
    }
    
    pub fn simulate_n_steps(
        initial_state: PhysicsState,
        n: usize,
        grid: &Grid,
        medium: &dyn Medium,
        config: &FdtdConfig,
    ) -> KwaversResult<Vec<PhysicsState>> {
        // Iterate pure function
        (0..n).try_fold(vec![initial_state.clone()], |mut states, _| {
            let next = fdtd_step(states.last().unwrap(), grid, medium, config)?;
            states.push(next);
            Ok(states)
        })
    }
}
```

---

### 2.3 Multi-Physics Coupling Patterns

**OptimUS Approach (BEM Multi-Domain):**
- Each domain has homogeneous properties
- Coupling through boundary integral equations
- Interface conditions automatically enforced

**Fullwave Approach (Staggered Grid):**
- All physics on same grid
- Material properties vary spatially
- Explicit time stepping couples equations

**Kwavers Current Approach:**

Has multi-physics infrastructure but could enhance coupling:

```rust
// Current: src/physics/mod.rs
pub mod acoustics;
pub mod thermal;
pub mod optics;
pub mod electromagnetic;
```

**Recommendation 2.3: Explicit Coupling Interface**

```rust
// Proposed: src/solver/multiphysics/coupling.rs
pub trait PhysicsCoupling: Debug + Send + Sync {
    /// Get the physics domains this coupling connects
    fn coupled_domains(&self) -> (PhysicsDomain, PhysicsDomain);
    
    /// Apply coupling terms
    fn apply_coupling(
        &self,
        source_field: &Array3<f64>,
        target_field: &mut Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()>;
    
    /// Get coupling strength/coefficient
    fn coupling_coefficient(&self, position: (usize, usize, usize)) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub enum PhysicsDomain {
    Acoustic,
    Thermal,
    Elastic,
    Electromagnetic,
}

/// Acoustic-Thermal coupling (heating due to absorption)
pub struct AcousticThermalCoupling {
    /// Absorption coefficient α
    absorption: Array3<f64>,
    /// Heat capacity
    specific_heat: Array3<f64>,
}

impl PhysicsCoupling for AcousticThermalCoupling {
    fn coupled_domains(&self) -> (PhysicsDomain, PhysicsDomain) {
        (PhysicsDomain::Acoustic, PhysicsDomain::Thermal)
    }
    
    fn apply_coupling(
        &self,
        pressure: &Array3<f64>,
        temperature: &mut Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Heat deposition: dT/dt = α * I / (ρ * c_p)
        // where I = p^2 / (2 * ρ * c) is intensity
        Zip::from(temperature)
            .and(pressure)
            .and(&self.absorption)
            .and(&self.specific_heat)
            .for_each(|t, &p, &alpha, &cp| {
                let intensity = p * p / 2.0;  // Simplified
                let heating = alpha * intensity / cp * dt;
                *t += heating;
            });
        Ok(())
    }
    
    fn coupling_coefficient(&self, (ix, iy, iz): (usize, usize, usize)) -> f64 {
        self.absorption[(ix, iy, iz)]
    }
}

/// Multi-physics orchestrator
pub struct MultiPhysicsSolver {
    solvers: HashMap<PhysicsDomain, Box<dyn Plugin>>,
    couplings: Vec<Box<dyn PhysicsCoupling>>,
    execution_order: Vec<PhysicsDomain>,
}

impl MultiPhysicsSolver {
    /// Operator splitting: solve each physics independently, then apply couplings
    pub fn step_operator_splitting(&mut self, dt: f64) -> KwaversResult<()> {
        // 1. Solve each domain independently
        for domain in &self.execution_order {
            self.solvers.get_mut(domain).unwrap().update(...)?;
        }
        
        // 2. Apply coupling terms
        for coupling in &self.couplings {
            coupling.apply_coupling(...)?;
        }
        
        Ok(())
    }
    
    /// Strang splitting: improved accuracy O(dt^2)
    pub fn step_strang_splitting(&mut self, dt: f64) -> KwaversResult<()> {
        // Half step domain 1
        // Full step domain 2
        // Half step domain 1
    }
}
```

---

## 3. Domain/Geometry/Medium Specification Patterns

### 3.1 Type Flexibility: Scalar vs Array Properties

**jWave Pattern:**
```python
# Accepts scalar, array, or Field
medium = Medium(
    domain=domain,
    sound_speed=1500.0,              # Scalar for homogeneous
    # OR
    sound_speed=speed_array,         # Array for heterogeneous
    # OR
    sound_speed=FourierSeries(...)   # Field for spectral
)
```

**Validation:** Prevents mixing types (all scalars or all arrays/fields)

**Kwavers Current Pattern:**

Has separate types for homogeneous/heterogeneous:

```rust
pub struct HomogeneousMedium { ... }
pub struct HeterogeneousMedium { ... }

// Both implement Medium trait
```

**Recommendation 3.1: Unified Medium Type with Enum Properties**

```rust
// Proposed: More flexible property specification
#[derive(Debug, Clone)]
pub enum PropertyDistribution {
    Constant(f64),
    Spatially Varying(Array3<f64>),
    Functional(Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>),
}

pub struct UnifiedMedium {
    grid: Grid,
    sound_speed: PropertyDistribution,
    density: PropertyDistribution,
    attenuation: PropertyDistribution,
    // ... other properties
}

impl UnifiedMedium {
    /// Query property at specific location
    pub fn sound_speed_at(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        match &self.sound_speed {
            PropertyDistribution::Constant(c) => *c,
            PropertyDistribution::SpatiallyVarying(arr) => arr[(ix, iy, iz)],
            PropertyDistribution::Functional(f) => {
                let (x, y, z) = self.grid.index_to_position(ix, iy, iz);
                f(x, y, z)
            }
        }
    }
    
    /// Check if property is homogeneous
    pub fn is_homogeneous(&self) -> bool {
        matches!(self.sound_speed, PropertyDistribution::Constant(_)) &&
        matches!(self.density, PropertyDistribution::Constant(_))
    }
    
    /// Convert to array representation (for solver efficiency)
    pub fn materialize(&self) -> MaterializedMedium {
        let mut sound_speed_array = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        
        for iz in 0..self.grid.nz {
            for iy in 0..self.grid.ny {
                for ix in 0..self.grid.nx {
                    sound_speed_array[(ix, iy, iz)] = self.sound_speed_at(ix, iy, iz);
                }
            }
        }
        
        MaterializedMedium { sound_speed: sound_speed_array, ... }
    }
}

// Builder for ergonomic construction
impl UnifiedMedium {
    pub fn builder(grid: Grid) -> UnifiedMediumBuilder {
        UnifiedMediumBuilder::new(grid)
    }
}

pub struct UnifiedMediumBuilder {
    grid: Grid,
    sound_speed: Option<PropertyDistribution>,
    density: Option<PropertyDistribution>,
}

impl UnifiedMediumBuilder {
    pub fn sound_speed_constant(mut self, c: f64) -> Self {
        self.sound_speed = Some(PropertyDistribution::Constant(c));
        self
    }
    
    pub fn sound_speed_array(mut self, c: Array3<f64>) -> Self {
        self.sound_speed = Some(PropertyDistribution::SpatiallyVarying(c));
        self
    }
    
    pub fn sound_speed_function<F>(mut self, f: F) -> Self
    where
        F: Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    {
        self.sound_speed = Some(PropertyDistribution::Functional(Box::new(f)));
        self
    }
    
    pub fn build(self) -> KwaversResult<UnifiedMedium> {
        Ok(UnifiedMedium {
            grid: self.grid,
            sound_speed: self.sound_speed.ok_or("sound_speed required")?,
            density: self.density.ok_or("density required")?,
            // ...
        })
    }
}

// Usage examples
let medium1 = UnifiedMedium::builder(grid.clone())
    .sound_speed_constant(1500.0)
    .density_constant(1000.0)
    .build()?;

let medium2 = UnifiedMedium::builder(grid.clone())
    .sound_speed_array(speed_map)
    .density_constant(1000.0)
    .build()?;

let medium3 = UnifiedMedium::builder(grid.clone())
    .sound_speed_function(|x, y, z| 1500.0 + 50.0 * (x / 0.1))
    .density_constant(1000.0)
    .build()?;
```

---

### 3.2 Automatic Constraint Satisfaction

**jWave's `TimeAxis.from_medium`:**
```python
# Automatically computes stable dt from CFL condition
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=1e-3)
```

**Recommendation 3.2: Constraint Solver for Configuration**

```rust
// Proposed: src/simulation/constraints.rs
pub struct SimulationConstraints {
    cfl_condition: Option<f64>,
    max_frequency: Option<f64>,
    points_per_wavelength: Option<f64>,
    max_memory_gb: Option<f64>,
}

impl SimulationConstraints {
    pub fn new() -> Self {
        Self {
            cfl_condition: Some(0.95),
            max_frequency: None,
            points_per_wavelength: Some(4.0),
            max_memory_gb: None,
        }
    }
    
    /// Automatically determine optimal grid and time step
    pub fn solve(
        &self,
        domain_size: (f64, f64, f64),
        medium: &dyn Medium,
        duration: f64,
    ) -> KwaversResult<(Grid, f64)> {
        let c_max = medium.max_sound_speed();
        let c_min = medium.min_sound_speed();
        
        // Determine spatial resolution from wavelength constraint
        let lambda_min = c_min / self.max_frequency.unwrap_or(1e6);
        let dx_required = lambda_min / self.points_per_wavelength.unwrap_or(4.0);
        
        let nx = (domain_size.0 / dx_required).ceil() as usize;
        let ny = (domain_size.1 / dx_required).ceil() as usize;
        let nz = (domain_size.2 / dx_required).ceil() as usize;
        
        // Check memory constraint
        if let Some(max_mem_gb) = self.max_memory_gb {
            let estimated_mem_gb = (nx * ny * nz * 8 * 10) as f64 / 1e9;  // ~10 fields
            if estimated_mem_gb > max_mem_gb {
                return Err(format!(
                    "Grid requires {:.2} GB, exceeds limit {:.2} GB",
                    estimated_mem_gb, max_mem_gb
                ).into());
            }
        }
        
        let grid = Grid::new(nx, ny, nz, dx_required, dx_required, dx_required)?;
        
        // Determine time step from CFL
        let cfl = self.cfl_condition.unwrap_or(0.95);
        let dt = cfl * dx_required / (c_max * (3.0_f64).sqrt());
        
        Ok((grid, dt))
    }
}

// Usage
let constraints = SimulationConstraints::new()
    .with_max_frequency(5e6)
    .with_points_per_wavelength(4.0)
    .with_cfl(0.95)
    .with_max_memory_gb(16.0);

let (grid, dt) = constraints.solve(
    (0.05, 0.05, 0.08),  // 5cm x 5cm x 8cm domain
    &medium,
    1e-3  // 1 ms duration
)?;

println!("Optimal grid: {}x{}x{}", grid.nx, grid.ny, grid.nz);
println!("Time step: {:.2e} s", dt);
```

---

## 4. GPU Acceleration Patterns

### 4.1 jWave: JAX Compilation Model

**Pattern:**
```python
@jit  # Decorator for JIT compilation
def simulate(medium, time_axis, p0):
    # Pure Python/JAX code
    # Automatically compiled to GPU
    return result

# Transparent acceleration
result_cpu = simulate(medium, time_axis, p0)  # CPU
result_gpu = jax.device_put(simulate)(medium, time_axis, p0)  # GPU
```

**Benefits:**
- No separate GPU code
- Same source for CPU/GPU
- Auto-vectorization

### 4.2 k-Wave: Binary Backend Selection

**Pattern:**
```matlab
% User code unchanged
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor);

% Internal: checks for GPU binary
% Falls back to C++ or MATLAB if unavailable
```

### 4.3 Fullwave: Explicit CUDA with Python Wrapper

**Pattern:**
- Core: CUDA kernels in C++
- Wrapper: Python interface
- Multi-GPU: Domain decomposition

**Kwavers Current Approach:**

```rust
#[cfg(feature = "gpu")]
pub mod gpu {
    pub struct GpuAccelerator { ... }
}
```

**Recommendation 4.1: Compute Graph Abstraction**

Similar to JAX but in Rust:

```rust
// Proposed: src/gpu/graph/mod.rs
#[cfg(feature = "gpu")]
pub mod graph {
    /// Computation node in execution graph
    pub enum ComputeOp {
        FieldUpdate { field_id: usize, expr: Expression },
        FFT { input_id: usize, output_id: usize, direction: FFTDirection },
        Derivative { field_id: usize, axis: Axis, order: usize },
        BoundaryApply { field_id: usize, boundary_id: usize },
    }
    
    /// Expression for GPU kernel generation
    pub enum Expression {
        Field(usize),
        Constant(f64),
        Add(Box<Expression>, Box<Expression>),
        Mul(Box<Expression>, Box<Expression>),
        // ... other operations
    }
    
    pub struct ComputeGraph {
        operations: Vec<ComputeOp>,
        dependencies: HashMap<usize, Vec<usize>>,
    }
    
    impl ComputeGraph {
        /// Build graph from simulation configuration
        pub fn from_solver(solver: &dyn Solver) -> Self {
            // Analyze solver operations
            // Build dependency graph
        }
        
        /// Compile graph to GPU kernels
        pub fn compile(&self, device: &wgpu::Device) -> CompiledGraph {
            // Generate WGSL shaders
            // Create pipeline
            // Optimize execution order
        }
    }
    
    pub struct CompiledGraph {
        pipelines: Vec<wgpu::ComputePipeline>,
        execution_order: Vec<usize>,
    }
    
    impl CompiledGraph {
        /// Execute graph on GPU
        pub fn execute(&self, queue: &wgpu::Queue, buffers: &mut GpuBuffers) {
            for &op_idx in &self.execution_order {
                self.pipelines[op_idx].dispatch(queue, buffers);
            }
        }
    }
}

// Usage
let graph = ComputeGraph::from_solver(&fdtd_solver);
let compiled = graph.compile(&gpu_device);

// Execute many steps efficiently
for _ in 0..num_steps {
    compiled.execute(&gpu_queue, &mut buffers);
}
```

---

## 5. API Design & Extensibility

### 5.1 Progressive Complexity

**Observation:** All libraries provide multiple API levels:
1. **Simple/High-level**: Sensible defaults, minimal configuration
2. **Intermediate**: Common customization options
3. **Advanced**: Full control, expert features

**jWave Example:**
```python
# Level 1: Simplest
result = simulate_wave_propagation(medium, time_axis)

# Level 2: Common options
result = simulate_wave_propagation(
    medium, time_axis,
    p0=initial_pressure,
    sensors=sensor_mask
)

# Level 3: Full control
result = simulate_wave_propagation_custom(
    medium, time_axis,
    p0=initial_pressure,
    pml_size=20,
    absorbing_boundary=custom_pml,
    source_func=custom_source
)
```

**Recommendation 5.1: Tiered API for Kwavers**

```rust
// Proposed: src/api/mod.rs
pub mod simple;
pub mod standard;
pub mod advanced;

// Level 1: Simple API
pub mod simple {
    use super::*;
    
    /// Run basic acoustic simulation with defaults
    pub fn simulate_acoustic(
        domain_size: (f64, f64, f64),
        frequency: f64,
        duration: f64,
    ) -> KwaversResult<SimulationResult> {
        // Automatically:
        // - Create grid with optimal spacing
        // - Use water medium
        // - Apply default PML boundaries
        // - Select best solver (FDTD for small, PSTD for large)
        // - Return pressure field snapshots
        
        let constraints = SimulationConstraints::new()
            .with_max_frequency(frequency)
            .with_cfl(0.95);
        
        let medium = Medium::water();
        let (grid, dt) = constraints.solve(domain_size, &medium, duration)?;
        
        // ... rest of setup
        
        standard::simulate(grid, medium, sources, duration)
    }
}

// Level 2: Standard API (current kwavers level)
pub mod standard {
    pub fn simulate(
        grid: Grid,
        medium: impl Medium,
        sources: Vec<Box<dyn Source>>,
        duration: f64,
    ) -> KwaversResult<SimulationResult> {
        // Build with reasonable defaults
        SimulationBuilder::new()
            .with_grid(grid)
            .with_medium(&medium)
            .with_sources(sources)
            .with_duration(duration)
            .build()?
            .run()
    }
}

// Level 3: Advanced API (full control)
pub mod advanced {
    pub struct AdvancedSimulationBuilder {
        // All configuration options
        solver_type: Option<SolverType>,
        backend: Option<Box<dyn ComputeBackend>>,
        time_integrator: Option<TimeIntegrator>,
        boundary_conditions: Option<Box<dyn Boundary>>,
        plugins: Vec<Box<dyn Plugin>>,
        // ... many more options
    }
    
    impl AdvancedSimulationBuilder {
        pub fn with_custom_solver(mut self, solver: SolverType) -> Self { ... }
        pub fn with_backend(mut self, backend: Box<dyn ComputeBackend>) -> Self { ... }
        pub fn with_time_integrator(mut self, integrator: TimeIntegrator) -> Self { ... }
        pub fn add_plugin(mut self, plugin: Box<dyn Plugin>) -> Self { ... }
        // ... many more configuration methods
    }
}

// Usage examples:

// Beginner
let result = simple::simulate_acoustic(
    (0.05, 0.05, 0.08),  // 5x5x8 cm
    5e6,                  // 5 MHz
    1e-3                  // 1 ms
)?;

// Intermediate
let grid = Grid::new(128, 128, 256, 1e-4, 1e-4, 1e-4)?;
let medium = HomogeneousMedium::soft_tissue(&grid);
let source = PointSource::new((0.025, 0.025, 0.01), sine_wave);
let result = standard::simulate(grid, medium, vec![Box::new(source)], 1e-3)?;

// Expert
let result = advanced::AdvancedSimulationBuilder::new()
    .with_grid(grid)
    .with_medium(custom_medium)
    .with_custom_solver(SolverType::HybridFDTDPSTD)
    .with_backend(Box::new(GpuBackend::new(device)))
    .with_time_integrator(TimeIntegrator::ImexBDF(order: 2))
    .with_boundary_conditions(Box::new(CustomPML::new(...)))
    .add_plugin(Box::new(CustomPhysicsPlugin::new()))
    .enable_feature(SolverFeature::AdaptiveMeshRefinement)
    .build()?
    .run()?;
```

---

### 5.2 Backward Compatibility During Refactoring

**k-wave-python Note:**
> "will continue to diverge from the original k-Wave APIs to leverage pythonic practices"

**Pattern:** Explicit decision to break compatibility for better API

**Alternative Pattern (if compatibility required):**
1. **Deprecation warnings** with migration path
2. **Facade layer** for old API over new implementation
3. **Feature flags** for old/new behavior
4. **Version namespaces** (`kwavers::v2`, `kwavers::v3`)

**Recommendation 5.2: Compatibility Strategy for Kwavers**

```rust
// Proposed: src/compat/mod.rs
#[deprecated(
    since = "3.1.0",
    note = "Use `UnifiedMedium::builder()` instead. See migration guide: docs/migration_v3_to_v4.md"
)]
pub struct OldMediumAPI;

impl OldMediumAPI {
    pub fn create_medium(...) -> Medium {
        // Old API signature
        // Internally uses new implementation
        UnifiedMedium::builder(grid)
            .sound_speed_constant(c)
            .build()
            .unwrap()
    }
}

// Feature flag for strict mode
#[cfg(not(feature = "legacy-api"))]
compile_error!("Legacy API removed in v4.0. Disable 'legacy-api' feature or migrate.");

// Version namespaces (if major changes needed)
pub mod v3 {
    pub use crate::compat::*;
}

pub mod v4 {
    pub use crate::domain::medium::unified::*;
}

// Allow gradual migration
use kwavers::v3::Medium;  // Old code
use kwavers::v4::UnifiedMedium;  // New code
```

---

## 6. Specific Architectural Improvements for Kwavers

### 6.1 Plugin System Enhancement

**Current State:** Good foundation with `Plugin` trait

**Enhancement:** Add plugin lifecycle management and dependency resolution

```rust
// Proposed: Enhanced plugin metadata
pub struct PluginMetadata {
    pub name: String,
    pub version: semver::Version,
    pub dependencies: Vec<PluginDependency>,
    pub provides: Vec<Capability>,
    pub requires: Vec<Capability>,
}

pub struct PluginDependency {
    pub plugin_name: String,
    pub version_req: semver::VersionReq,
}

pub enum Capability {
    Field(UnifiedFieldType),
    Feature(String),
    Solver(String),
}

// Plugin registry with dependency resolution
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn Plugin>>,
}

impl PluginRegistry {
    /// Register plugin and check dependencies
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> KwaversResult<()> {
        let metadata = plugin.metadata();
        
        // Check dependencies
        for dep in &metadata.dependencies {
            if !self.has_plugin(&dep.plugin_name, &dep.version_req) {
                return Err(format!(
                    "Plugin {} requires {} {}",
                    metadata.name, dep.plugin_name, dep.version_req
                ).into());
            }
        }
        
        // Check capability conflicts
        for capability in &metadata.provides {
            if self.has_capability(capability) {
                return Err(format!(
                    "Capability {:?} already provided by another plugin",
                    capability
                ).into());
            }
        }
        
        self.plugins.insert(metadata.name.clone(), plugin);
        Ok(())
    }
    
    /// Resolve execution order based on dependencies
    pub fn resolve_execution_order(&self) -> KwaversResult<Vec<String>> {
        // Topological sort of dependency graph
    }
}
```

---

### 6.2 Solver Factory Pattern

**Pattern:** Automatic solver selection based on problem characteristics

```rust
// Proposed: src/solver/factory.rs
pub struct SolverFactory;

impl SolverFactory {
    /// Automatically select best solver for problem
    pub fn create_optimal(
        grid: &Grid,
        medium: &dyn Medium,
        config: &SimulationConfig,
    ) -> KwaversResult<Box<dyn Solver>> {
        let is_homogeneous = medium.is_homogeneous();
        let grid_size = grid.nx * grid.ny * grid.nz;
        let has_gpu = cfg!(feature = "gpu") && GpuBackend::is_available();
        
        // Decision tree
        if grid_size > 10_000_000 && has_gpu {
            // Large problem, use GPU-accelerated PSTD
            Ok(Box::new(PSTDSolver::new_with_gpu(grid, medium, config)?))
        } else if is_homogeneous && grid_size > 1_000_000 {
            // Medium problem, homogeneous medium → PSTD efficient
            Ok(Box::new(PSTDSolver::new(grid, medium, config)?))
        } else if config.requires_high_accuracy {
            // High accuracy needed → spectral methods
            Ok(Box::new(HybridSpectralDGSolver::new(grid, medium, config)?))
        } else {
            // Default: FDTD (most versatile)
            Ok(Box::new(FdtdSolver::new(grid, medium, config)?))
        }
    }
    
    /// Create solver with explicit type
    pub fn create(
        solver_type: SolverType,
        grid: &Grid,
        medium: &dyn Medium,
        config: &SimulationConfig,
    ) -> KwaversResult<Box<dyn Solver>> {
        match solver_type {
            SolverType::FDTD => Ok(Box::new(FdtdSolver::new(grid, medium, config)?)),
            SolverType::PSTD => Ok(Box::new(PSTDSolver::new(grid, medium, config)?)),
            SolverType::Hybrid => Ok(Box::new(HybridSolver::new(grid, medium, config)?)),
            SolverType::SpectralDG => Ok(Box::new(SpectralDGSolver::new(grid, medium, config)?)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SolverType {
    FDTD,
    PSTD,
    Hybrid,
    SpectralDG,
}
```

---

### 6.3 Configuration Validation & Defaults

**Pattern:** Validate configuration before simulation, provide sensible defaults

```rust
// Proposed: src/simulation/validation.rs
pub trait Validatable {
    fn validate(&self) -> KwaversResult<()>;
    fn warnings(&self) -> Vec<String>;
}

impl Validatable for SimulationConfig {
    fn validate(&self) -> KwaversResult<()> {
        let mut errors = Vec::new();
        
        // Check CFL condition
        if self.dt > self.compute_max_stable_dt() {
            errors.push(format!(
                "Time step {:.2e} exceeds CFL limit {:.2e}",
                self.dt,
                self.compute_max_stable_dt()
            ));
        }
        
        // Check grid resolution
        let ppw = self.points_per_wavelength();
        if ppw < 2.0 {
            errors.push(format!(
                "Grid too coarse: {:.1} points/wavelength (need ≥4 for accuracy)",
                ppw
            ));
        }
        
        // Check memory requirements
        let estimated_memory_gb = self.estimate_memory_usage();
        if estimated_memory_gb > self.max_memory_gb {
            errors.push(format!(
                "Estimated memory {:.1} GB exceeds limit {:.1} GB",
                estimated_memory_gb,
                self.max_memory_gb
            ));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("Configuration validation failed:\n{}", errors.join("\n")).into())
        }
    }
    
    fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        let ppw = self.points_per_wavelength();
        if ppw < 4.0 {
            warnings.push(format!(
                "Low grid resolution: {:.1} points/wavelength. Recommend ≥4 for accuracy.",
                ppw
            ));
        }
        
        if self.dt > 0.8 * self.compute_max_stable_dt() {
            warnings.push(
                "Time step close to stability limit. Consider reducing dt by 10-20%.".into()
            );
        }
        
        warnings
    }
}

// Usage
let config = SimulationConfig::new(...);

// Validate before running
config.validate()?;

// Show warnings
for warning in config.warnings() {
    eprintln!("Warning: {}", warning);
}
```

---

## 7. Summary of Key Recommendations

### Priority 1: High Impact, Low Effort

1. **Factory Methods for Auto-Configuration** (Rec 1.1)
   - `FdtdConfig::from_medium_and_grid(&medium, &grid, cfl, duration)`
   - `TimeAxis::from_constraints(...)`
   - Automatically satisfy CFL, wavelength resolution

2. **Backend Abstraction Trait** (Rec 2.1)
   - `trait ComputeBackend` for CPU/GPU transparency
   - Automatic backend selection
   - k-Wave pattern: same API, different implementations

3. **Configuration Validation** (Rec 6.3)
   - Pre-flight checks for CFL, memory, resolution
   - Warnings for suboptimal settings
   - Prevent silent simulation failures

### Priority 2: Moderate Impact, Moderate Effort

4. **Tiered API Design** (Rec 5.1)
   - Simple API for beginners
   - Standard API for common use
   - Advanced API for experts

5. **Unified Medium Type** (Rec 3.1)
   - `PropertyDistribution` enum: Constant/Array/Function
   - Builder pattern for ergonomic construction
   - jWave-style type flexibility

6. **Domain Builder for Clinical Apps** (Rec 1.5)
   - `AnatomicalDomainBuilder` for layered tissues
   - Geometric feature addition
   - Fullwave-style workflow integration

### Priority 3: High Impact, High Effort

7. **k-Space Operator Library** (Rec 1.3)
   - Fourier collocation spatial derivatives
   - k-space corrected time integration
   - Fractional Laplacian absorption

8. **Multi-Physics Coupling Interface** (Rec 2.3)
   - `trait PhysicsCoupling`
   - Operator splitting orchestration
   - Explicit coupling coefficients

9. **Compute Graph Abstraction** (Rec 4.1)
   - Build operation DAG from solver
   - Compile to optimized GPU kernels
   - JAX-style transparent acceleration

### Priority 4: Future Considerations

10. **Functional Simulation API** (Rec 1.2)
    - Pure function interface alongside OO
    - Enables future auto-differentiation
    - Better composability

11. **Plugin Dependency Resolution** (Rec 6.1)
    - Plugin registry with version management
    - Capability-based conflict detection
    - Automatic execution order resolution

12. **Solver Auto-Selection** (Rec 6.2)
    - `SolverFactory::create_optimal(...)`
    - Decision tree based on problem characteristics
    - Transparent performance optimization

---

## 8. Comparison Table: Kwavers vs Reference Libraries

| Feature | k-Wave | jWave | Fullwave | OptimUS | Kwavers | Recommendation |
|---------|--------|-------|----------|---------|---------|----------------|
| **Module Organization** |
| Layer separation | ✓ (implicit) | ✓ (explicit) | ✓ (packages) | ✓ (BEM modules) | ✅ (8 layers) | **Excellent** - Keep current |
| Plugin architecture | ✗ | ✓ (composition) | ✗ | ✗ | ✅ (trait-based) | **Excellent** - Enhance metadata |
| Factory methods | ✗ | ✅ (`from_medium`) | ✗ | ✗ | ⚠️ (limited) | **Add** auto-config factories |
| **Solver Architecture** |
| FDTD | ✗ | ✗ | ✅ (8th order) | ✗ | ✅ (2nd/4th order) | **Enhance** with higher orders |
| PSTD/k-space | ✅ (mature) | ✅ (JAX) | ✗ | ✗ | ✅ (basic) | **Add** k-space correction |
| Hybrid methods | ✗ | ✗ | ✗ | ✗ | ✅ (adaptive) | **Unique strength** - keep |
| Backend abstraction | ✅ (MATLAB/C++/CUDA) | ✅ (JAX backends) | ✓ (Python/CUDA) | ✓ (BEMPP) | ⚠️ (feature flags) | **Add** trait abstraction |
| **Domain Specification** |
| Flexible properties | ✓ (struct) | ✅ (Union types) | ✓ (arrays) | ✓ (homogeneous) | ⚠️ (split types) | **Add** unified medium |
| Domain builders | ✗ | ✗ | ✅ (MediumBuilder) | ✗ | ✗ | **Add** for clinical apps |
| Auto-constraints | ✗ | ✅ (CFL from medium) | ✗ | ✗ | ✗ | **Add** constraint solver |
| **GPU Acceleration** |
| Transparent | ✅ (binary select) | ✅ (JIT) | ✓ (wrapper) | ✗ | ⚠️ (manual) | **Add** auto-selection |
| Multi-GPU | ✗ | ✗ | ✅ (decomposition) | ✗ | ✗ | **Future** consideration |
| **API Design** |
| Tiered complexity | ✗ | ✓ | ✗ | ✗ | ⚠️ (single level) | **Add** simple/advanced APIs |
| Validation | ✗ | ✓ (type checks) | ✗ | ✗ | ⚠️ (limited) | **Add** config validation |
| Documentation | ✅ (excellent) | ✅ (excellent) | ✓ (good) | ✓ (papers) | ✅ (comprehensive) | **Excellent** - maintain |

**Legend:** ✅ Excellent, ✓ Good, ⚠️ Needs improvement, ✗ Not present

---

## 9. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Add factory methods for auto-configuration
- [ ] Implement configuration validation
- [ ] Create backend abstraction trait
- [ ] Add tiered API (simple/standard/advanced)

### Phase 2: Core Enhancements (1 month)
- [ ] Unified medium with PropertyDistribution
- [ ] k-space operator library
- [ ] Constraint solver for grid/dt selection
- [ ] Enhanced plugin metadata system

### Phase 3: Clinical Features (1 month)
- [ ] Domain builder for anatomical structures
- [ ] Transducer array abstractions
- [ ] Clinical workflow integration
- [ ] Tissue property database

### Phase 4: Advanced Features (2-3 months)
- [ ] Multi-physics coupling interface
- [ ] Compute graph for GPU optimization
- [ ] Functional simulation API
- [ ] Plugin dependency resolution

---

## 10. Code Examples: Before & After

### Example 1: Basic Simulation

**Before (Current Kwavers):**
```rust
let grid = Grid::new(128, 128, 256, 3.9e-4, 3.9e-4, 3.9e-4)?;  // Manual calculation
let medium = HomogeneousMedium::water(&grid);
let source = PointSource::new((0.025, 0.025, 0.01), sine_wave);
let config = FdtdConfig {
    dt: 1.3e-7,  // Manual CFL calculation
    spatial_order: 4,
    ..Default::default()
};
let mut solver = FdtdSolver::new(config, &grid, &medium, source)?;

for _ in 0..num_steps {
    solver.step()?;
}
let result = solver.pressure_field();
```

**After (With Recommendations):**
```rust
// Simple API - automatic everything
let result = kwavers::simple::simulate_acoustic(
    domain_size: (0.05, 0.05, 0.1),
    frequency: 5e6,
    duration: 1e-3,
)?;

// OR Standard API - some control
let (grid, dt) = SimulationConstraints::new()
    .with_max_frequency(5e6)
    .with_cfl(0.95)
    .solve((0.05, 0.05, 0.1), &medium, 1e-3)?;

let result = kwavers::standard::simulate(
    grid,
    Medium::water(),
    vec![PointSource::new((0.025, 0.025, 0.01), sine_wave)],
    1e-3,
)?;
```

### Example 2: Clinical Simulation

**Before (Manual Construction):**
```rust
let grid = Grid::new(256, 256, 512, 1e-4, 1e-4, 1e-4)?;
let mut sound_speed = Array3::zeros((256, 256, 512));
let mut density = Array3::zeros((256, 256, 512));

// Manual layer assignment
for iz in 0..20 {  // Skin layer
    for iy in 0..256 {
        for ix in 0..256 {
            sound_speed[[ix, iy, iz]] = 1540.0;
            density[[ix, iy, iz]] = 1100.0;
        }
    }
}
// ... repeat for each layer

let medium = HeterogeneousMedium::new(&grid, sound_speed, density, ...)?;
```

**After (Domain Builder):**
```rust
let medium = AnatomicalDomainBuilder::new(&grid)
    .add_layer("skin", 2e-3, TissueProperties::skin())
    .add_layer("fat", 5e-3, TissueProperties::fat())
    .add_layer("muscle", 20e-3, TissueProperties::muscle())
    .add_feature(GeometricFeature::cylinder_vessel(
        center: (0.025, 0.025, 0.010),
        radius: 2e-3,
        properties: TissueProperties::blood(),
    ))
    .build()?;
```

### Example 3: Multi-Physics Coupling

**Before (Manual Implementation):**
```rust
// User has to manually implement coupling
for step in 0..num_steps {
    acoustic_solver.step()?;
    
    // Manual coupling: heating from absorption
    let pressure = acoustic_solver.pressure_field();
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let p = pressure[[ix, iy, iz]];
                let alpha = medium.absorption(ix, iy, iz);
                let intensity = p * p / (2.0 * medium.density(ix, iy, iz) * medium.sound_speed(ix, iy, iz));
                let heating = alpha * intensity * dt;
                thermal_field[[ix, iy, iz]] += heating / medium.specific_heat(ix, iy, iz);
            }
        }
    }
    
    thermal_solver.step_with_source(&thermal_field)?;
}
```

**After (Coupling Interface):**
```rust
let mut multiphysics = MultiPhysicsSolver::new()
    .add_solver(PhysicsDomain::Acoustic, Box::new(acoustic_solver))
    .add_solver(PhysicsDomain::Thermal, Box::new(thermal_solver))
    .add_coupling(Box::new(AcousticThermalCoupling::new(&medium)))
    .build()?;

for _ in 0..num_steps {
    multiphysics.step_strang_splitting(dt)?;  // Automatic coupling
}
```

---

## 11. Testing Strategy for New Features

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factory_method_cfl_constraint() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        
        let config = FdtdConfig::from_medium_and_grid(&medium, &grid, 0.95, 1e-3).unwrap();
        
        // Verify CFL condition satisfied
        let c_max = medium.max_sound_speed();
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl_max = c_max * config.dt / min_dx * (3.0_f64).sqrt();
        assert!(cfl_max <= 0.95);
    }
    
    #[test]
    fn test_unified_medium_homogeneous_case() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = UnifiedMedium::builder(grid.clone())
            .sound_speed_constant(1500.0)
            .density_constant(1000.0)
            .build()
            .unwrap();
        
        assert!(medium.is_homogeneous());
        assert_eq!(medium.sound_speed_at(0, 0, 0), 1500.0);
        assert_eq!(medium.sound_speed_at(15, 15, 15), 1500.0);
    }
    
    #[test]
    fn test_domain_builder_layer_ordering() {
        let grid = Grid::new(64, 64, 128, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = AnatomicalDomainBuilder::new(&grid)
            .add_layer("skin", 2e-3, TissueProperties::skin())
            .add_layer("fat", 3e-3, TissueProperties::fat())
            .build()
            .unwrap();
        
        // Verify skin layer
        assert!((medium.sound_speed(32, 32, 5) - 1540.0).abs() < 1.0);
        
        // Verify fat layer
        assert!((medium.sound_speed(32, 32, 25) - 1450.0).abs() < 1.0);
    }
}
```

### Integration Tests
```rust
// tests/integration/api_tiers.rs
#[test]
fn test_simple_api_produces_valid_result() {
    let result = kwavers::simple::simulate_acoustic(
        (0.05, 0.05, 0.08),
        5e6,
        1e-4,
    ).unwrap();
    
    assert!(result.pressure_field.len() > 0);
    assert!(result.max_pressure.is_finite());
}

#[test]
fn test_api_tier_equivalence() {
    // Verify simple and standard APIs produce same result for same problem
    let simple_result = kwavers::simple::simulate_acoustic(...).unwrap();
    
    let (grid, dt) = /* derive from simple params */;
    let standard_result = kwavers::standard::simulate(...).unwrap();
    
    // Results should be numerically equivalent
    assert_arrays_close(&simple_result.pressure_field, &standard_result.pressure_field, 1e-6);
}
```

### Validation Against Reference
```rust
// tests/validation/kwave_comparison.rs
#[test]
#[ignore]  // Requires k-Wave installation
fn test_kspace_operators_match_kwave() {
    // Load k-Wave reference data
    let kwave_gradient = load_hdf5("tests/data/kwave_gradient_reference.h5")?;
    
    // Compute with kwavers k-space operator
    let kspace_grad = KSpaceGradient::new(&grid);
    let kwavers_gradient = kspace_grad.apply(&pressure_field);
    
    // Compare
    assert_arrays_close(&kwavers_gradient.0, &kwave_gradient.x, 1e-10);
    assert_arrays_close(&kwavers_gradient.1, &kwave_gradient.y, 1e-10);
    assert_arrays_close(&kwavers_gradient.2, &kwave_gradient.z, 1e-10);
}
```

---

## 12. Documentation Requirements

### API Documentation
- **Tiered examples**: Show simple → standard → advanced for each feature
- **Migration guides**: Clear path from current to enhanced APIs
- **Decision trees**: Help users choose solver, backend, configuration

### Architectural Documentation
- **Update ARCHITECTURE.md**: Reflect new patterns (factory methods, backend abstraction)
- **Plugin developer guide**: How to create compatible plugins with metadata
- **Multi-physics coupling guide**: Implementing custom coupling interfaces

### Scientific Validation
- **Benchmark suite**: Compare against k-Wave, jWave for standard problems
- **Convergence studies**: Demonstrate accuracy vs grid resolution
- **Performance profiles**: CPU vs GPU, FDTD vs PSTD for different problem sizes

---

## Conclusion

Kwavers already has a strong architectural foundation with clean layer separation and a plugin system. The key enhancements from reference repositories are:

1. **User Experience**: Factory methods, tiered APIs, auto-configuration (jWave patterns)
2. **Scientific Accuracy**: k-space operators, power-law absorption (k-Wave patterns)
3. **Clinical Applicability**: Domain builders, transducer abstractions (Fullwave patterns)
4. **Performance**: Backend abstraction, transparent GPU selection (all libraries)

Implementing Priority 1 recommendations would provide immediate value with minimal disruption to existing code. The modular architecture allows incremental adoption of these patterns while maintaining backward compatibility.

**Next Steps:**
1. Review recommendations with development team
2. Prioritize based on user needs and use cases
3. Create detailed implementation tickets for Phase 1
4. Set up validation infrastructure (comparison with k-Wave/jWave)
5. Begin implementation with comprehensive testing

---

**References:**
- [jWave GitHub](https://github.com/ucl-bug/jwave)
- [k-Wave GitHub](https://github.com/ucl-bug/k-wave)
- [k-wave-python Documentation](https://k-wave-python.readthedocs.io/)
- [OptimUS GitHub](https://github.com/optimuslib/optimus)
- [Fullwave25 GitHub](https://github.com/pinton-lab/fullwave25)
- [SimSonic](http://www.simsonic.fr/)
- [SIMUS: An open-source simulator for medical ultrasound imaging](https://arxiv.org/pdf/2102.02738)

