//! KZK Equation Solver Plugin
//! Based on Lee & Hamilton (1995): "Time-domain modeling of pulsed finite-amplitude sound beams"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::plugin::{PluginMetadata, PluginState};
use ndarray::Array3;

/// Frequency domain operator for KZK equation
#[derive(Debug, Clone)]
pub struct FrequencyOperator {
    /// Frequency grid points
    pub frequencies: Vec<f64>,
    /// Absorption operator in frequency domain
    pub absorption_operator: Array3<f64>,
    /// Diffraction operator in frequency domain
    pub diffraction_operator: Array3<f64>,
}

/// KZK Equation Solver Plugin
/// Implements the Khokhlov-Zabolotskaya-Kuznetsov equation for nonlinear beam propagation
#[derive(Debug)]
pub struct KzkSolverPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Frequency domain operators for efficient computation
    frequency_operators: Option<FrequencyOperator>,
    /// Retarded time frame for moving window
    retarded_time_window: Option<f64>,
}

impl Default for KzkSolverPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl KzkSolverPlugin {
    /// Create new KZK solver plugin
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "kzk_solver".to_string(),
                name: "KZK Equation Solver".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "Nonlinear beam propagation using KZK equation".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            frequency_operators: None,
            retarded_time_window: None,
        }
    }

    /// Initialize frequency domain operators
    /// Based on Aanonsen et al. (1984): "Distortion and harmonic generation in the nearfield"
    pub fn initialize_operators(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
        max_frequency: f64,
    ) -> KwaversResult<()> {
        use crate::medium::AcousticProperties;
        use std::f64::consts::PI;

        // Set up frequency grid (up to 10th harmonic typically)
        const NUM_HARMONICS: usize = 10;
        let fundamental = max_frequency / NUM_HARMONICS as f64;
        let mut frequencies = Vec::with_capacity(NUM_HARMONICS);
        for n in 1..=NUM_HARMONICS {
            frequencies.push(n as f64 * fundamental);
        }

        // Initialize operators
        let shape = (grid.nx, grid.ny, NUM_HARMONICS);
        let mut absorption_op = Array3::zeros(shape);
        let mut diffraction_op = Array3::zeros(shape);

        // Compute operators for each frequency
        for (f_idx, &freq) in frequencies.iter().enumerate() {
            let omega = 2.0 * PI * freq;
            const NOMINAL_SOUND_SPEED: f64 = 1500.0; // m/s in tissue
            let k = omega / NOMINAL_SOUND_SPEED;

            // Absorption operator: exp(-alpha * dz)
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = 0.0; // At source plane
                    let alpha =
                        AcousticProperties::absorption_coefficient(medium, x, y, z, grid, freq);
                    absorption_op[[i, j, f_idx]] = (-alpha * grid.dz).exp();

                    // Diffraction operator: exp(i * dz * (kx^2 + ky^2) / (2k))
                    // Simplified for real computation
                    let kx = 2.0 * PI * i as f64 / (grid.nx as f64 * grid.dx);
                    let ky = 2.0 * PI * j as f64 / (grid.ny as f64 * grid.dy);
                    diffraction_op[[i, j, f_idx]] = ((kx * kx + ky * ky) / (2.0 * k)).cos();
                }
            }
        }

        self.frequency_operators = Some(FrequencyOperator {
            frequencies,
            absorption_operator: absorption_op,
            diffraction_operator: diffraction_op,
        });

        self.state = PluginState::Running;
        Ok(())
    }

    /// Solve KZK equation using operator splitting
    /// Based on Tavakkoli et al. (1998): "A new algorithm for computational simulation"
    pub fn solve(
        &mut self,
        initial_field: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        time_steps: usize,
    ) -> KwaversResult<Array3<f64>> {
        use crate::medium::AcousticProperties;

        // Validate operators are initialized
        let operators =
            self.frequency_operators
                .as_ref()
                .ok_or(crate::error::KwaversError::Physics(
                    crate::error::PhysicsError::InvalidParameter {
                        parameter: "frequency_operators".to_string(),
                        value: 0.0,
                        reason: "KZK operators not initialized - call initialize_operators first"
                            .to_string(),
                    },
                ))?;

        // Working field (complex representation would be ideal, using real for now)
        let mut field = initial_field.clone();

        // Time step for operator splitting
        let dz = grid.dz;

        // Get medium properties at source plane
        let density = crate::medium::density_at(medium, 0.0, 0.0, 0.0, grid);
        let c0 = crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let beta = AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, grid);

        // Operator splitting: Strang splitting for second-order accuracy
        // P(dz) = L(dz/2) * N(dz) * L(dz/2)
        // where L = linear (diffraction + absorption), N = nonlinear

        for _step in 0..time_steps {
            // Step 1: Half-step linear propagation (diffraction + absorption)
            self.apply_linear_step(&mut field, operators, dz / 2.0)?;

            // Step 2: Full nonlinear step
            self.apply_nonlinear_step(&mut field, beta, density, c0, dz, grid)?;

            // Step 3: Half-step linear propagation
            self.apply_linear_step(&mut field, operators, dz / 2.0)?;
        }

        Ok(field)
    }

    /// Apply linear propagation (diffraction + absorption)
    fn apply_linear_step(
        &self,
        field: &mut Array3<f64>,
        operators: &FrequencyOperator,
        step_size: f64,
    ) -> KwaversResult<()> {
        // Apply absorption and diffraction in frequency domain
        // For each harmonic component
        for (f_idx, _freq) in operators.frequencies.iter().enumerate() {
            // Extract harmonic slice
            let nx = field.shape()[0];
            let ny = field.shape()[1];

            for i in 0..nx {
                for j in 0..ny {
                    if f_idx < field.shape()[2] {
                        // Apply absorption: multiply by exp(-alpha * dz)
                        let absorption = operators.absorption_operator[[i, j, f_idx]];
                        field[[i, j, f_idx]] *= absorption.powf(step_size);

                        // Apply diffraction (simplified - full implementation needs FFT)
                        let diffraction = operators.diffraction_operator[[i, j, f_idx]];
                        field[[i, j, f_idx]] *= diffraction;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply nonlinear step using Burgers equation solution
    fn apply_nonlinear_step(
        &self,
        field: &mut Array3<f64>,
        beta: f64,
        density: f64,
        c0: f64,
        step_size: f64,
        _grid: &Grid,
    ) -> KwaversResult<()> {
        use ndarray::Zip;

        // Nonlinear parameter
        let nonlinear_factor = beta / (2.0 * density * c0.powi(3));

        // Apply nonlinear distortion
        // Solution of inviscid Burgers equation: u_t + u*u_x = 0
        // Using implicit solution: u(z) = u0 / (1 - u0 * beta * z / (2 * rho * c^3))

        Zip::from(field.view_mut()).for_each(|p| {
            let p0 = *p;
            // Prevent shock singularity
            let denominator = 1.0 - nonlinear_factor * p0 * step_size;
            if denominator.abs() > 0.1 {
                // Avoid division by small numbers
                *p = p0 / denominator;
            } else {
                // Shock has formed - apply limiting
                *p = p0.signum() * p0.abs().min(1.0 / (nonlinear_factor * step_size));
            }
        });

        Ok(())
    }

    /// Calculate shock formation distance
    /// Based on Bacon (1984): "Finite amplitude distortion of the pulsed fields"
    pub fn shock_formation_distance(
        &self,
        source_pressure: f64,
        frequency: f64,
        medium: &dyn Medium,
    ) -> KwaversResult<f64> {
        use crate::medium::AcousticProperties;
        use std::f64::consts::PI;

        // Get medium properties at origin
        // Note: Minimal grid created only for API compatibility with point evaluation functions
        // Medium properties are evaluated at single point (origin), grid dimensions irrelevant
        let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0)?;
        let density = crate::medium::density_at(medium, 0.0, 0.0, 0.0, &grid);
        let sound_speed = crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, &grid);
        let beta = AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, &grid);

        // Shock formation distance: x_shock = ρc³/(βωp₀)
        let omega = 2.0 * PI * frequency;

        Ok(density * sound_speed.powi(3) / (beta * omega * source_pressure))
    }

    /// Apply retarded time transformation
    /// Based on Jing et al. (2012): "Verification of the Westervelt equation"
    pub fn apply_retarded_time(
        &mut self,
        field: &Array3<f64>,
        propagation_distance: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Retarded time: τ = t - z/c
        // This shifts the time window to follow the wave
        const SOUND_SPEED: f64 = 1500.0; // m/s nominal
        let time_shift = propagation_distance / SOUND_SPEED;

        // Store the shift for the moving window
        self.retarded_time_window = Some(time_shift);

        // For now, return the field as-is since actual shifting
        // requires interpolation in the time domain
        // In a full implementation, this would involve:
        // 1. FFT to frequency domain
        // 2. Apply phase shift exp(-i*omega*time_shift)
        // 3. IFFT back to time domain

        Ok(field.clone())
    }
}

// Plugin trait implementation
impl crate::physics::plugin::Plugin for KzkSolverPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        vec![crate::physics::field_mapping::UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        vec![crate::physics::field_mapping::UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &crate::physics::plugin::PluginContext,
    ) -> KwaversResult<()> {
        use crate::physics::field_mapping::UnifiedFieldType;

        // Extract pressure field
        let pressure_field =
            fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let mut pressure_array = pressure_field.to_owned();

        // Apply one KZK step
        if let Some(operators) = &self.frequency_operators {
            self.apply_linear_step(&mut pressure_array, operators, dt / 2.0)?;

            // Get medium properties for nonlinear step
            let density = crate::medium::density_at(medium, 0.0, 0.0, 0.0, grid);
            let c0 = crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
            let beta = crate::medium::AcousticProperties::nonlinearity_coefficient(
                medium, 0.0, 0.0, 0.0, grid,
            );

            self.apply_nonlinear_step(&mut pressure_array, beta, density, c0, dt, grid)?;
            self.apply_linear_step(&mut pressure_array, operators, dt / 2.0)?;

            // Update pressure field in the fields array
            let mut pressure_slice =
                fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
            pressure_slice.assign(&pressure_array);
        }

        Ok(())
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        // Use a default frequency for initialization
        let default_freq = 1e6; // 1 MHz
        self.initialize_operators(grid, medium, default_freq)?;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.frequency_operators = None;
        self.retarded_time_window = None;
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
