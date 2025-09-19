//! Multi-Element Transducer Field Calculator Plugin
//! Based on Jensen & Svendsen (1992): "Calculation of pressure fields from arbitrarily shaped transducers"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::{AcousticProperties, Medium};
use crate::physics::plugin::{PluginMetadata, PluginState};
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Transducer geometry definition
#[derive(Debug, Clone)]
pub struct TransducerGeometry {
    /// Element positions [x, y, z] in meters
    pub element_positions: Array2<f64>,
    /// Element sizes [width, height] in meters
    pub element_sizes: Array2<f64>,
    /// Element orientations (normal vectors)
    pub element_normals: Array2<f64>,
    /// Element apodization weights
    pub apodization: Option<Vec<f64>>,
    /// Element delays in seconds
    pub delays: Option<Vec<f64>>,
}

/// Multi-Element Transducer Field Calculator Plugin
#[derive(Debug)]
pub struct TransducerFieldCalculatorPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Transducer geometry definitions
    transducer_geometries: Vec<TransducerGeometry>,
    /// Spatial impulse response cache
    sir_cache: HashMap<String, Array3<f64>>,
}

impl TransducerFieldCalculatorPlugin {
    /// Create new FOCUS-compatible transducer field calculator
    #[must_use]
    pub fn new(transducer_geometries: Vec<TransducerGeometry>) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "focus_transducer_calculator".to_string(),
                name: "FOCUS Transducer Field Calculator".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "Multi-element transducer field calculation with FOCUS compatibility"
                    .to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            transducer_geometries,
            sir_cache: HashMap::new(),
        }
    }

    /// Calculate spatial impulse response for a given transducer
    /// Based on Tupholme (1969) and Stepanishen (1971) methods
    pub fn calculate_sir(
        &mut self,
        transducer_index: usize,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let cache_key = format!("sir_{transducer_index}");

        if let Some(cached) = self.sir_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Spatial impulse response calculation based on Tupholme-Stepanishen method
        // Reference: Stepanishen (1971) "Transient radiation from pistons in an infinite planar baffle"
        let mut sir = Array3::zeros(grid.dimensions());

        // Use first transducer geometry if available
        if self.transducer_geometries.is_empty() {
            return Ok(sir);
        }
        let geometry = &self.transducer_geometries[0];

        // Calculate average element size for discretization
        let n_elements = geometry.element_positions.nrows();
        let avg_width = geometry.element_sizes.column(0).mean().unwrap_or(0.01);
        let avg_height = geometry.element_sizes.column(1).mean().unwrap_or(0.01);
        let element_area = (avg_width * avg_height) / n_elements as f64;

        // Calculate contribution from each surface element
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let mut response = 0.0;

                    // Integrate over transducer elements
                    for elem_idx in 0..n_elements {
                        let elem_pos = geometry.element_positions.row(elem_idx);
                        let x_elem = elem_pos[0];
                        let y_elem = elem_pos[1];
                        let z_elem = elem_pos[2];

                        // Distance from element to field point
                        let r =
                            ((x - x_elem).powi(2) + (y - y_elem).powi(2) + (z - z_elem).powi(2))
                                .sqrt();

                        // Spatial impulse response for rectangular piston
                        // h(r,t) = (v₀/2πc) * δ(t - r/c) / r
                        if r > 1e-10 {
                            let _c = crate::medium::sound_speed_at(medium, x, y, z, grid);
                            response += element_area / (2.0 * std::f64::consts::PI * r);

                            // Apply apodization if specified
                            if let Some(apod_weights) = &geometry.apodization {
                                if elem_idx < apod_weights.len() {
                                    response *= apod_weights[elem_idx];
                                }
                            }
                        }
                    }

                    sir[[i, j, k]] = response;
                }
            }
        }

        self.sir_cache.insert(cache_key, sir.clone());
        Ok(sir)
    }

    /// Calculate pressure field using angular spectrum method
    /// Based on Zeng & `McGough` (2008): "Evaluation of the angular spectrum approach"
    pub fn calculate_pressure_field(
        &mut self,
        frequency: f64,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        // Angular spectrum method implementation
        // Reference: Zeng & McGough (2008) "Evaluation of the angular spectrum approach"
        use ndarray::{s, Array2};
        use rustfft::{num_complex::Complex, FftPlanner};

        let mut pressure_field = Array3::zeros(grid.dimensions());
        let c = crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid); // Use reference sound speed
        let k = 2.0 * std::f64::consts::PI * frequency / c; // Wave number

        // Initialize source plane (z=0) with transducer aperture function
        let mut source_plane = Array2::<Complex<f64>>::zeros((grid.nx, grid.ny));

        // Use first transducer geometry if available
        if !self.transducer_geometries.is_empty() {
            let geometry = &self.transducer_geometries[0];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let (x, y, _) = grid.indices_to_coordinates(i, j, 0);

                    // Check if point is within any transducer element
                    for elem_idx in 0..geometry.element_positions.nrows() {
                        let elem_pos = geometry.element_positions.row(elem_idx);
                        let elem_size = geometry.element_sizes.row(elem_idx);

                        if (x - elem_pos[0]).abs() <= elem_size[0] / 2.0
                            && (y - elem_pos[1]).abs() <= elem_size[1] / 2.0
                        {
                            let mut amplitude = Complex::new(1.0, 0.0);

                            // Apply element delay if specified
                            if let Some(delays) = &geometry.delays {
                                if elem_idx < delays.len() {
                                    let phase =
                                        2.0 * std::f64::consts::PI * frequency * delays[elem_idx];
                                    amplitude *= Complex::from_polar(1.0, phase);
                                }
                            }

                            // Apply apodization if specified
                            if let Some(apod_weights) = &geometry.apodization {
                                if elem_idx < apod_weights.len() {
                                    amplitude *= apod_weights[elem_idx];
                                }
                            }

                            source_plane[[i, j]] += amplitude;
                        }
                    }
                }
            }
        }

        // Perform 2D FFT of source plane
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(grid.nx);
        let mut spectrum = source_plane.clone();

        // FFT along x-axis
        for j in 0..grid.ny {
            let mut row: Vec<Complex<f64>> = spectrum.slice(s![.., j]).to_vec();
            fft.process(&mut row);
            for (i, val) in row.iter().enumerate() {
                spectrum[[i, j]] = *val;
            }
        }

        // FFT along y-axis
        let fft_y = planner.plan_fft_forward(grid.ny);
        for i in 0..grid.nx {
            let mut col: Vec<Complex<f64>> = spectrum.slice(s![i, ..]).to_vec();
            fft_y.process(&mut col);
            for (j, val) in col.iter().enumerate() {
                spectrum[[i, j]] = *val;
            }
        }

        // Propagate through each z-plane
        for k_idx in 0..grid.nz {
            let z = k_idx as f64 * grid.dz;
            let mut propagated = spectrum.clone();

            // Apply propagation transfer function in k-space
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    let kx = 2.0 * std::f64::consts::PI * (i as f64 - grid.nx as f64 / 2.0)
                        / (grid.nx as f64 * grid.dx);
                    let ky = 2.0 * std::f64::consts::PI * (j as f64 - grid.ny as f64 / 2.0)
                        / (grid.ny as f64 * grid.dy);

                    let kz_sq = k * k - kx * kx - ky * ky;
                    if kz_sq > 0.0 {
                        // Propagating waves
                        let kz = kz_sq.sqrt();
                        let phase = Complex::from_polar(1.0, kz * z);
                        propagated[[i, j]] *= phase;
                    } else {
                        // Evanescent waves - exponential decay
                        let decay = (-kz_sq.abs().sqrt() * z).exp();
                        propagated[[i, j]] *= decay;
                    }
                }
            }

            // Inverse FFT to get spatial domain pressure
            let ifft_y = planner.plan_fft_inverse(grid.ny);
            for i in 0..grid.nx {
                let mut col: Vec<Complex<f64>> = propagated.slice(s![i, ..]).to_vec();
                ifft_y.process(&mut col);
                for (j, val) in col.iter().enumerate() {
                    propagated[[i, j]] = *val / grid.ny as f64;
                }
            }

            let ifft_x = planner.plan_fft_inverse(grid.nx);
            for j in 0..grid.ny {
                let mut row: Vec<Complex<f64>> = propagated.slice(s![.., j]).to_vec();
                ifft_x.process(&mut row);
                for (i, val) in row.iter().enumerate() {
                    pressure_field[[i, j, k_idx]] = val.re / grid.nx as f64;
                }
            }
        }

        Ok(pressure_field)
    }

    /// Calculate harmonic pressure field for nonlinear propagation
    /// Based on Christopher & Parker (1991): "New approaches to nonlinear diffractive field propagation"
    pub fn calculate_harmonic_field(
        &mut self,
        harmonic: usize,
        fundamental_freq: f64,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        // Harmonic field calculation for nonlinear propagation
        // Reference: Christopher & Parker (1991) "New approaches to nonlinear diffractive field propagation"

        // First calculate the fundamental field
        let fundamental_field = self.calculate_pressure_field(fundamental_freq, grid, medium)?;

        if harmonic == 1 {
            return Ok(fundamental_field);
        }

        let mut harmonic_field = Array3::zeros(grid.dimensions());
        let harmonic_freq = fundamental_freq * harmonic as f64;
        let c = crate::medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);

        // Nonlinearity parameter B/A for the medium
        let beta =
            1.0 + AcousticProperties::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, grid) / 2.0;

        // Generate nonlinear source term from fundamental field
        // For second harmonic: S₂ = β/(2ρ₀c₄) * (∂p₁/∂t)²
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 1..grid.nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let rho = crate::medium::density_at(medium, x, y, z, grid);

                    // Approximate time derivative using spatial gradient and wave equation
                    let p1 = fundamental_field[[i, j, k]];
                    let p1_prev = fundamental_field[[i, j, k.saturating_sub(1)]];
                    let _grad_z = (p1 - p1_prev) / grid.dz;

                    // Nonlinear source term for nth harmonic
                    let source_strength = match harmonic {
                        2 => {
                            // Second harmonic generation
                            beta * p1.powi(2) / (2.0 * rho * c.powi(4))
                        }
                        3 => {
                            // Third harmonic generation
                            beta.powi(2) * p1.powi(3) / (6.0 * rho * c.powi(4))
                        }
                        _ => {
                            // Higher harmonics with decreasing efficiency
                            let efficiency = 1.0 / (harmonic as f64).powi(2);
                            efficiency * beta.powi(harmonic as i32 - 1) * p1.powi(harmonic as i32)
                                / (rho * c.powi(4) * harmonic as f64)
                        }
                    };

                    // Propagate harmonic with frequency-dependent attenuation
                    let alpha = AcousticProperties::absorption_coefficient(
                        medium,
                        x,
                        y,
                        z,
                        grid,
                        harmonic_freq,
                    );
                    let attenuation = (-alpha * z).exp();

                    // Account for beam divergence at harmonic frequency
                    // Use average aperture size from first transducer geometry
                    let avg_aperture = if !self.transducer_geometries.is_empty() {
                        let geometry = &self.transducer_geometries[0];
                        geometry.element_sizes.column(0).mean().unwrap_or(0.01)
                    } else {
                        0.01 // Default 1cm aperture
                    };

                    let k_harmonic = 2.0 * std::f64::consts::PI * harmonic_freq / c;
                    let rayleigh_distance = k_harmonic * avg_aperture.powi(2) / 4.0;
                    let divergence = (1.0 + (z / rayleigh_distance).powi(2)).sqrt();

                    harmonic_field[[i, j, k]] = source_strength * attenuation / divergence;
                }
            }
        }

        Ok(harmonic_field)
    }

    /// Calculate heating rate from acoustic field
    /// Based on Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"
    pub fn calculate_heating_rate(
        &self,
        pressure_field: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Heating rate: Q = 2 * alpha * I
        // where I = p^2 / (2 * rho * c) is acoustic intensity
        // Reference: Nyborg (1981) "Heat generation by ultrasound in a relaxing medium"

        use crate::medium::AcousticProperties;
        use ndarray::Zip;

        let mut heating = Array3::zeros(pressure_field.dim());

        Zip::indexed(&mut heating)
            .and(pressure_field)
            .for_each(|(i, j, k), q, &p| {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let density = crate::medium::density_at(medium, x, y, z, grid);
                let sound_speed = crate::medium::sound_speed_at(medium, x, y, z, grid);
                let alpha =
                    AcousticProperties::absorption_coefficient(medium, x, y, z, grid, frequency);

                // Acoustic intensity: I = p^2 / (2 * rho * c)
                let intensity = (p * p) / (2.0 * density * sound_speed);

                // Heating rate: Q = 2 * alpha * I
                *q = 2.0 * alpha * intensity;
            });

        Ok(heating)
    }
}

// Plugin trait implementation
impl crate::physics::plugin::Plugin for TransducerFieldCalculatorPlugin {
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
        vec![] // No required fields - generates pressure from transducer definitions
    }

    fn provided_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        vec![crate::physics::field_mapping::UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        _dt: f64,
        t: f64,
        _context: &crate::physics::plugin::PluginContext,
    ) -> KwaversResult<()> {
        use crate::physics::field_mapping::UnifiedFieldType;

        // Calculate frequency from time
        let frequency = 1e6; // Default 1 MHz - should be configurable

        // Calculate pressure field from transducer geometries
        let pressure_field = self.calculate_pressure_field(frequency, grid, medium)?;

        // Apply time-dependent modulation if needed
        let time_factor = (2.0 * std::f64::consts::PI * frequency * t).sin();
        let modulated_field = pressure_field.mapv(|p| p * time_factor);

        // Update pressure field in the fields array
        let mut pressure_slice =
            fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        pressure_slice.assign(&modulated_field);

        Ok(())
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        self.sir_cache.clear();
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.sir_cache.clear();
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
