// src/medium/homogeneous/mod.rs
use crate::grid::Grid;
use crate::medium::{absorption, power_law_absorption, Medium};
use log::debug;
use ndarray::{Array3, Zip};
use std::sync::{Mutex, OnceLock};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::f64;
use std::clone::Clone;

/// Custom wrapper for f64 that implements Eq and Hash properly for use as HashMap keys
#[derive(Debug, Clone, Copy)]
struct FloatKey(f64);

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        // Use an epsilon-based comparison for floats
        (self.0 - other.0).abs() < 1e-10
    }
}

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Quantize to avoid floating-point precision issues
        let quantized = (self.0 * 1e6).round() as i64;
        quantized.hash(state);
    }
}

/// Thread-safe wrapper for absorption cache
#[derive(Debug)]
struct AbsorptionCache {
    cache: Mutex<HashMap<FloatKey, f64>>,
}

impl AbsorptionCache {
    fn new() -> Self {
        AbsorptionCache {
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn get(&self, key: &FloatKey) -> Option<f64> {
        let guard = self.cache.lock().unwrap();
        guard.get(key).copied()
    }

    fn insert(&self, key: FloatKey, value: f64) {
        let mut guard = self.cache.lock().unwrap();
        guard.insert(key, value);
    }

    fn clear(&self) {
        let mut guard = self.cache.lock().unwrap();
        guard.clear();
    }
}

// Manual Clone implementation that doesn't try to clone the Mutex
impl Clone for AbsorptionCache {
    fn clone(&self) -> Self {
        // Create a new AbsorptionCache with an empty HashMap
        // This is safe because the cache is just for performance optimization
        AbsorptionCache::new()
    }
}

#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    pub density: f64,
    pub sound_speed: f64,
    pub viscosity: f64,
    pub surface_tension: f64,
    pub ambient_pressure: f64,
    pub vapor_pressure: f64,
    pub polytropic_index: f64,
    pub specific_heat: f64,
    pub thermal_conductivity: f64,
    pub thermal_expansion: f64,
    pub gas_diffusion_coeff: f64,
    pub thermal_diffusivity: f64,
    pub mu_a: f64,
    pub mu_s_prime: f64,
    pub temperature: Array3<f64>,
    pub bubble_radius: Array3<f64>,
    pub bubble_velocity: Array3<f64>,
    pub alpha0: f64,
    pub delta: f64,
    pub b_a: f64,
    pub reference_frequency: f64,
    // Caching for absorption coefficient calculations
    absorption_cache: AbsorptionCache,
    // Cached arrays for uniform properties
    density_array: OnceLock<Array3<f64>>,
    sound_speed_array: OnceLock<Array3<f64>>,
}

impl HomogeneousMedium {
    pub fn new(density: f64, sound_speed: f64, grid: &Grid, mu_a: f64, mu_s_prime: f64) -> Self {
        assert!(density > 0.0 && sound_speed > 0.0 && mu_a >= 0.0 && mu_s_prime >= 0.0);

        let viscosity = 1.002e-3; // Water at 20°C
        let surface_tension = 0.0728; // Water
        let ambient_pressure = 1.013e5; // Standard atmospheric pressure
        let vapor_pressure = 2.338e3; // Water at 20°C
        let polytropic_index = 1.4; // Typical for gases
        let specific_heat = 4182.0; // Water
        let thermal_conductivity = 0.598; // Water
        let thermal_expansion = 2.1e-4; // Water
        let gas_diffusion_coeff = 2e-9; // Typical value
        let thermal_diffusivity = 1.43e-7; // Water
        let b_a = 5.2; // Water
        let reference_frequency = 180000.0; // Default ultrasound frequency (180 kHz)

        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15); // 20°C
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10e-6);
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let alpha0 = 0.025; // Water attenuation coefficient
        let delta = 1.0; // Power law exponent

        debug!(
            "Initialized HomogeneousMedium: density = {:.2}, sound_speed = {:.2}, b_a = {:.2}, freq = {:.2e}",
            density, sound_speed, b_a, reference_frequency
        );
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension,
            ambient_pressure,
            vapor_pressure,
            polytropic_index,
            specific_heat,
            thermal_conductivity,
            thermal_expansion,
            gas_diffusion_coeff,
            thermal_diffusivity,
            mu_a,
            mu_s_prime,
            temperature,
            bubble_radius,
            bubble_velocity,
            alpha0,
            delta,
            b_a,
            reference_frequency,
            absorption_cache: AbsorptionCache::new(),
            density_array: OnceLock::new(),
            sound_speed_array: OnceLock::new(),
        }
    }

    pub fn water(grid: &Grid) -> Self {
        Self::new(998.0, 1500.0, grid, 1.0, 50.0)
    }

    // Clear caches when medium properties might change
    fn clear_caches(&mut self) {
        debug!("Clearing medium property caches");
        
        // Clear the absorption cache
        self.absorption_cache.clear();
        
        // Reset the array caches
        self.density_array = OnceLock::new();
        self.sound_speed_array = OnceLock::new();
    }
}

impl Medium for HomogeneousMedium {
    fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density }
    fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed }
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity }
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.surface_tension }
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.ambient_pressure }
    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.vapor_pressure }
    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.polytropic_index }
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.specific_heat }
    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_conductivity }
    
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        // Use cached value if available
        let cache_key = FloatKey(frequency);
        
        // Try to get the cached value
        if let Some(cached) = self.absorption_cache.get(&cache_key) {
            return cached;
        }
        
        // Calculate temperature at this point
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        let t = self.temperature[[ix.min(self.temperature.shape()[0]-1), 
                                     iy.min(self.temperature.shape()[1]-1), 
                                     iz.min(self.temperature.shape()[2]-1)]];
        
        let r = if let Some(indices) = grid.to_grid_indices(x, y, z) {
            self.bubble_radius[indices]
        } else {
            0.0
        };
        
        // Calculate absorption with temperature and bubble effects
        let alpha = if self.alpha0 > 0.0 && self.delta > 0.0 {
            // Power law absorption
            power_law_absorption::power_law_absorption_coefficient(frequency, self.alpha0, self.delta)
        } else {
            // Basic thermal absorption
            absorption::absorption_coefficient(frequency, t, Some(r))
        };
        
        // Cache the result
        self.absorption_cache.insert(cache_key, alpha);
        
        alpha
    }
    
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_expansion }
    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.gas_diffusion_coeff }
    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_diffusivity }
    fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.b_a }
    fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_a }
    fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_s_prime }
    fn reference_frequency(&self) -> f64 { self.reference_frequency }

    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        debug!("Updating temperature in homogeneous medium");
        self.temperature.assign(temperature);
        self.clear_caches();
    }
    
    fn temperature(&self) -> &Array3<f64> { &self.temperature }
    fn bubble_radius(&self) -> &Array3<f64> { &self.bubble_radius }
    fn bubble_velocity(&self) -> &Array3<f64> { &self.bubble_velocity }
    
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius.assign(radius);
        self.bubble_velocity.assign(velocity);
        self.clear_caches();
    }
    
    fn density_array(&self) -> Array3<f64> {
        self.density_array.get_or_init(|| {
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.density)
        }).clone()
    }
    
    fn sound_speed_array(&self) -> Array3<f64> {
        self.sound_speed_array.get_or_init(|| {
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.sound_speed)
        }).clone()
    }
    
    fn is_homogeneous(&self) -> bool { true }
}