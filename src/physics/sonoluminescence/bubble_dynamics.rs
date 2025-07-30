//! Bubble dynamics for sonoluminescence
//! 
//! Implements the extended Rayleigh-Plesset equation with:
//! - Compressibility corrections (Keller-Miksis formulation)
//! - Thermal effects
//! - Mass transfer
//! - Non-equilibrium vapor dynamics

use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Physical constants
pub const WATER_DENSITY: f64 = 998.0; // kg/m³ at 20°C
pub const WATER_SOUND_SPEED: f64 = 1482.0; // m/s at 20°C
pub const WATER_VISCOSITY: f64 = 1.002e-3; // Pa·s at 20°C
pub const WATER_SURFACE_TENSION: f64 = 0.0728; // N/m at 20°C
pub const WATER_VAPOR_PRESSURE: f64 = 2.33e3; // Pa at 20°C
pub const GAS_CONSTANT: f64 = 8.314; // J/(mol·K)
pub const POLYTROPIC_INDEX_ISOTHERMAL: f64 = 1.0;
pub const POLYTROPIC_INDEX_ADIABATIC: f64 = 1.4; // for air

/// Bubble dynamics state
#[derive(Debug, Clone)]
pub struct BubbleState {
    pub radius: f64,           // Current radius [m]
    pub velocity: f64,         // Wall velocity dR/dt [m/s]
    pub acceleration: f64,     // Wall acceleration d²R/dt² [m/s²]
    pub temperature: f64,      // Internal temperature [K]
    pub pressure_internal: f64, // Internal pressure [Pa]
    pub pressure_liquid: f64,   // Liquid pressure at bubble wall [Pa]
    pub n_gas: f64,            // Number of gas molecules
    pub n_vapor: f64,          // Number of vapor molecules
}

/// Parameters for bubble dynamics
#[derive(Debug, Clone)]
pub struct BubbleParameters {
    pub r0: f64,              // Equilibrium radius [m]
    pub p0: f64,              // Ambient pressure [Pa]
    pub rho: f64,             // Liquid density [kg/m³]
    pub c: f64,               // Sound speed in liquid [m/s]
    pub sigma: f64,           // Surface tension [N/m]
    pub mu: f64,              // Dynamic viscosity [Pa·s]
    pub pv: f64,              // Vapor pressure [Pa]
    pub gamma: f64,           // Polytropic index
    pub thermal_conductivity: f64, // Thermal conductivity [W/(m·K)]
    pub diffusivity: f64,     // Mass diffusivity [m²/s]
    pub accommodation_coefficient: f64, // Evaporation/condensation coefficient
}

impl Default for BubbleParameters {
    fn default() -> Self {
        Self {
            r0: 5e-6,  // 5 μm typical for SBSL
            p0: 101325.0, // 1 atm
            rho: WATER_DENSITY,
            c: WATER_SOUND_SPEED,
            sigma: WATER_SURFACE_TENSION,
            mu: WATER_VISCOSITY,
            pv: WATER_VAPOR_PRESSURE,
            gamma: POLYTROPIC_INDEX_ADIABATIC,
            thermal_conductivity: 0.6, // W/(m·K) for water
            diffusivity: 2.5e-5, // m²/s for air in water
            accommodation_coefficient: 0.04, // typical for water
        }
    }
}

/// Rayleigh-Plesset solver with extensions
pub struct RayleighPlessetSolver {
    params: BubbleParameters,
    use_compressibility: bool,
    use_thermal_effects: bool,
    use_mass_transfer: bool,
}

impl RayleighPlessetSolver {
    pub fn new(params: BubbleParameters) -> Self {
        Self {
            params,
            use_compressibility: true,
            use_thermal_effects: true,
            use_mass_transfer: true,
        }
    }
    
    /// Calculate bubble wall acceleration using Keller-Miksis equation
    /// This is a compressible extension of Rayleigh-Plesset
    pub fn calculate_acceleration(
        &self,
        state: &BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let v = state.velocity;
        let rho = self.params.rho;
        let c = self.params.c;
        let mu = self.params.mu;
        let sigma = self.params.sigma;
        
        // Calculate internal pressure
        let pg = self.calculate_internal_pressure(state);
        
        // Liquid pressure at bubble wall
        let pl = self.params.p0 + p_acoustic - self.params.pv;
        
        // Viscous term
        let visc = 4.0 * mu * v / r;
        
        // Surface tension term
        let surf = 2.0 * sigma / r;
        
        // Compressibility factor (Keller-Miksis)
        let mach = v / c;
        let comp_factor = if self.use_compressibility {
            1.0 - mach
        } else {
            1.0
        };
        
        // Pressure difference
        let dp = pg - pl - visc - surf;
        
        // Keller-Miksis equation
        let numerator = dp / rho + r * self.calculate_pressure_derivative(state, p_acoustic, t) / (rho * c);
        let denominator = r * comp_factor + r * r * mach / c;
        
        let acceleration = (numerator - 1.5 * v * v * (1.0 - mach / 3.0)) / denominator;
        
        acceleration
    }
    
    /// Calculate internal gas pressure with thermal effects
    fn calculate_internal_pressure(&self, state: &BubbleState) -> f64 {
        let r0 = self.params.r0;
        let p0 = self.params.p0;
        let pv = self.params.pv;
        let sigma = self.params.sigma;
        
        // Van der Waals correction for hard core
        let b = 8.65e-6; // m³/mol for air
        let vh = 4.0 * PI * r0.powi(3) / 3.0;
        let v = 4.0 * PI * state.radius.powi(3) / 3.0;
        
        // Effective polytropic index depends on thermal conditions
        let gamma_eff = if self.use_thermal_effects {
            self.calculate_effective_polytropic_index(state)
        } else {
            self.params.gamma
        };
        
        // Modified polytropic relation with hard core
        let volume_ratio = (vh - state.n_gas * b) / (v - state.n_gas * b);
        let pg = (p0 + 2.0 * sigma / r0 - pv) * volume_ratio.powf(gamma_eff);
        
        pg + pv
    }
    
    /// Calculate effective polytropic index based on Peclet number
    fn calculate_effective_polytropic_index(&self, state: &BubbleState) -> f64 {
        // Thermal Peclet number
        let thermal_diffusivity = self.params.thermal_conductivity / (self.params.rho * 4200.0); // m²/s
        let pe_thermal = state.radius.abs() * state.velocity.abs() / thermal_diffusivity;
        
        // Interpolate between isothermal and adiabatic
        let gamma_iso = POLYTROPIC_INDEX_ISOTHERMAL;
        let gamma_adi = self.params.gamma;
        
        // Smooth transition based on Peclet number
        let transition = 1.0 / (1.0 + (pe_thermal / 10.0).powi(2));
        gamma_iso * transition + gamma_adi * (1.0 - transition)
    }
    
    /// Calculate time derivative of pressure for compressibility effects
    fn calculate_pressure_derivative(&self, state: &BubbleState, p_acoustic: f64, t: f64) -> f64 {
        // For now, assume sinusoidal driving
        // This should be replaced with actual pressure gradient in full implementation
        let omega = 2.0 * PI * 26.5e3; // 26.5 kHz typical for SBSL
        -p_acoustic * omega * omega * t.cos()
    }
    
    /// Update bubble temperature based on heat transfer
    pub fn update_temperature(&self, state: &mut BubbleState, dt: f64) {
        if !self.use_thermal_effects {
            return;
        }
        
        let r = state.radius;
        let v = state.velocity;
        let t_bulk = 293.15; // Ambient temperature [K]
        
        // Heat transfer coefficient (simplified)
        let h = self.params.thermal_conductivity / r;
        
        // Surface area
        let area = 4.0 * PI * r * r;
        
        // Volume
        let volume = 4.0 * PI * r.powi(3) / 3.0;
        
        // Specific heat capacity of gas
        let cv = 717.0; // J/(kg·K) for air
        let mass = state.n_gas * 0.029 / 6.022e23; // kg (air molecular mass)
        
        // PdV work
        let pdv_work = state.pressure_internal * 4.0 * PI * r * r * v;
        
        // Heat transfer
        let q_transfer = h * area * (t_bulk - state.temperature);
        
        // Temperature change
        let dt_temp = (pdv_work + q_transfer) / (mass * cv);
        state.temperature += dt_temp * dt;
    }
    
    /// Update vapor content based on evaporation/condensation
    pub fn update_vapor_content(&self, state: &mut BubbleState, dt: f64) {
        if !self.use_mass_transfer {
            return;
        }
        
        let t = state.temperature;
        let r = state.radius;
        
        // Clausius-Clapeyron equation for vapor pressure
        let pv_eq = 611.0 * ((17.27 * (t - 273.15)) / (t - 35.85)).exp(); // Pa
        
        // Current vapor pressure
        let pv_current = state.n_vapor * GAS_CONSTANT * t / (4.0 * PI * r.powi(3) / 3.0);
        
        // Mass flux (Hertz-Knudsen equation)
        let alpha = self.params.accommodation_coefficient;
        let m_water = 0.018; // kg/mol
        let flux = alpha * (pv_eq - pv_current) / (2.0 * PI * m_water * GAS_CONSTANT * t).sqrt();
        
        // Update vapor molecules
        let area = 4.0 * PI * r * r;
        let dn_vapor = flux * area * dt * 6.022e23 / m_water;
        state.n_vapor += dn_vapor;
        state.n_vapor = state.n_vapor.max(0.0);
    }
}

/// Main bubble dynamics interface
pub struct BubbleDynamics {
    solver: RayleighPlessetSolver,
    states: Array3<BubbleState>,
    time_history: Vec<(f64, BubbleState)>, // For tracking single bubble
}

impl BubbleDynamics {
    pub fn new(nx: usize, ny: usize, nz: usize, params: BubbleParameters) -> Self {
        let solver = RayleighPlessetSolver::new(params.clone());
        
        // Initialize bubble states
        let initial_state = BubbleState {
            radius: params.r0,
            velocity: 0.0,
            acceleration: 0.0,
            temperature: 293.15, // Room temperature
            pressure_internal: params.p0 + 2.0 * params.sigma / params.r0,
            pressure_liquid: params.p0,
            n_gas: 1e15, // Typical number of gas molecules
            n_vapor: 1e13, // Small initial vapor content
        };
        
        let states = Array3::from_elem((nx, ny, nz), initial_state);
        
        Self {
            solver,
            states,
            time_history: Vec::new(),
        }
    }
    
    /// Update bubble dynamics for all bubbles
    pub fn update(
        &mut self,
        pressure_field: &Array3<f64>,
        dt: f64,
        t: f64,
    ) {
        let shape = self.states.shape();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let p_acoustic = pressure_field[[i, j, k]];
                    let state = &mut self.states[[i, j, k]];
                    
                    // Calculate acceleration
                    let acceleration = self.solver.calculate_acceleration(state, p_acoustic, t);
                    
                    // Update velocity and radius (Verlet integration)
                    state.velocity += acceleration * dt;
                    state.radius += state.velocity * dt + 0.5 * acceleration * dt * dt;
                    state.acceleration = acceleration;
                    
                    // Ensure minimum radius
                    if state.radius < 0.1e-6 {
                        state.radius = 0.1e-6;
                        state.velocity = 0.0;
                    }
                    
                    // Update temperature
                    self.solver.update_temperature(state, dt);
                    
                    // Update vapor content
                    self.solver.update_vapor_content(state, dt);
                    
                    // Update pressures
                    state.pressure_internal = self.solver.calculate_internal_pressure(state);
                    state.pressure_liquid = self.solver.params.p0 + p_acoustic;
                }
            }
        }
        
        // Store history for center bubble (for SBSL tracking)
        if shape[0] > 0 && shape[1] > 0 && shape[2] > 0 {
            let center = [shape[0]/2, shape[1]/2, shape[2]/2];
            self.time_history.push((t, self.states[center].clone()));
        }
    }
    
    /// Get bubble states
    pub fn get_states(&self) -> &Array3<BubbleState> {
        &self.states
    }
    
    /// Get time history for analysis
    pub fn get_time_history(&self) -> &[(f64, BubbleState)] {
        &self.time_history
    }
    
    /// Calculate maximum compression ratio
    pub fn get_max_compression_ratio(&self) -> f64 {
        self.time_history.iter()
            .map(|(_, state)| self.solver.params.r0 / state.radius)
            .fold(0.0, f64::max)
    }
    
    /// Calculate maximum temperature
    pub fn get_max_temperature(&self) -> f64 {
        self.time_history.iter()
            .map(|(_, state)| state.temperature)
            .fold(0.0, f64::max)
    }
}