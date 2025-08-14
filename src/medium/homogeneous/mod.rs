//! # Homogeneous Medium Module
//!
//! This module defines `HomogeneousMedium`, a struct representing a simulation medium
//! where physical properties are uniform throughout the spatial domain. It implements
//! the `Medium` trait, providing constant values for properties like density, sound speed, etc.,
//! irrespective of spatial coordinates.
//!
//! ## Key Components:
//! - `HomogeneousMedium`: The main struct holding the uniform properties of the medium.
//! - `FloatKey`: A wrapper around `f64` to enable its use as a key in `HashMap` by
//!   providing reliable `Eq` and `Hash` implementations.
//! - `AbsorptionCache`: A thread-safe cache for storing pre-calculated acoustic absorption
//!   coefficients to optimize repeated calculations for the same frequency.
//!
//! The module also includes constructors for `HomogeneousMedium`, including a general-purpose
//! `new()` and a convenience constructor `water()` for water-like properties.

use crate::grid::Grid;
use crate::medium::{absorption, absorption::power_law_absorption, Medium};
use log::debug;
use ndarray::Array3; // Zip might not be used directly in this file anymore
use std::sync::{Mutex, OnceLock};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::f64;
use std::clone::Clone;

// Default optical properties for water at typical wavelengths
/// Default absorption coefficient for water [1/m]
/// Based on typical values for near-infrared wavelengths
const DEFAULT_WATER_ABSORPTION_COEFFICIENT: f64 = 0.1;

/// Default reduced scattering coefficient for water [1/m]
/// Based on typical values for biological tissue imaging
const DEFAULT_WATER_SCATTERING_COEFFICIENT: f64 = 1.0;

/// A wrapper for `f64` to allow its use as a key in `HashMap`.
///
/// Standard `f64` values do not implement `Eq` and `Hash` in a way that is suitable
/// for direct use as hash map keys due to floating-point precision issues. `FloatKey`
/// addresses this by:
/// 1.  Implementing `PartialEq` and `Eq` using an epsilon-based comparison (`1e-10`)
///     to consider very close floating-point numbers as equal.
/// 2.  Implementing `Hash` by quantizing the `f64` value (multiplying by `1e6`, rounding,
///     and then hashing the resulting `i64`) to ensure that numerically close values
///     produce the same hash. This makes lookups in the cache more robust.
#[derive(Debug, Clone, Copy)]
struct FloatKey(f64);

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        // Equality is defined by whether the f64 values, when quantized
        // by multiplying by 1e6 and rounding, result in the same i64 value.
        // This ensures consistency with the Hash implementation.
        (self.0 * 1e6).round() == (other.0 * 1e6).round()
    }
}

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Quantize the f64 value to a certain precision before hashing.
        // This ensures that values that are very close (e.g., differing by less than 1e-6)
        // will produce the same hash value, which is important for cache lookups
        // where frequencies might have minor floating point variations.
        let quantized = (self.0 * 1e6).round() as i64;
        quantized.hash(state);
    }
}

/// A thread-safe cache for storing acoustic absorption coefficients.
///
/// This struct wraps a `HashMap` within a `Mutex` to allow concurrent access
/// for getting or inserting absorption values. `FloatKey` is used for the keys
/// to handle floating-point frequencies reliably.
///
/// The cache is primarily used by `HomogeneousMedium::absorption_coefficient` to store
/// results for specific frequencies, avoiding redundant calculations.
#[derive(Debug)]
struct AbsorptionCache {
    cache: Mutex<HashMap<FloatKey, f64>>,
}

impl AbsorptionCache {
    /// Creates a new, empty `AbsorptionCache`.
    fn new() -> Self {
        AbsorptionCache {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Retrieves a cached absorption coefficient for the given frequency key.
    /// Returns `None` if the frequency is not in the cache.
    ///
    /// # Panics
    /// Panics if the `Mutex` is poisoned.
    fn get(&self, key: &FloatKey) -> Option<f64> {
        let guard = self.cache.lock().unwrap(); 
        guard.get(key).copied()
    }

    /// Inserts an absorption coefficient into the cache for a given frequency key.
    ///
    /// # Panics
    /// Panics if the `Mutex` is poisoned.
    fn insert(&self, key: FloatKey, value: f64) {
        let mut guard = self.cache.lock().unwrap(); 
        guard.insert(key, value);
    }

    /// Clears all entries from the absorption cache.
    ///
    /// # Panics
    /// Panics if the `Mutex` is poisoned.
    fn clear(&self) {
        let mut guard = self.cache.lock().unwrap(); 
        guard.clear();
    }
}

/// Implements `Clone` for `AbsorptionCache` by creating a new, empty cache.
///
/// Cloning an `AbsorptionCache` does not duplicate the cached entries. Instead,
/// it provides a fresh cache. This behavior is acceptable because the cache is
/// an optimization, and a new medium instance (or a cloned one that might undergo
/// changes) should start with a clear cache to ensure correctness if properties
/// affecting absorption (like temperature or bubble radius) are modified.
impl Clone for AbsorptionCache {
    fn clone(&self) -> Self {
        AbsorptionCache::new()
    }
}

/// Represents a medium with spatially uniform (homogeneous) physical properties.
///
/// This struct implements the `Medium` trait. For a `HomogeneousMedium`, properties like
/// density, sound speed, viscosity, etc., are defined as single scalar values that apply
/// throughout the entire simulation domain.
///
/// It includes parameters for acoustic absorption (which can be frequency-dependent via
/// power-law or thermal/bubble-based models) and basic optical properties (`mu_a`, `mu_s_prime`).
/// Dynamic fields like `temperature`, `bubble_radius`, and `bubble_velocity` are stored as 3D arrays,
/// allowing them to vary spatially and temporally, even if the base medium properties are homogeneous.
///
/// For performance optimization, it features:
/// - An `AbsorptionCache` for frequency-dependent acoustic absorption coefficients.
/// - `OnceLock` fields (`density_array`, `sound_speed_array`) for lazily initializing and caching
///   full 3D arrays of density and sound speed when requested by the `Medium` trait.
#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    /// Density of the medium (kg/m³). Uniform throughout the medium.
    pub density: f64,
    /// Speed of sound in the medium (m/s). Uniform throughout the medium.
    pub sound_speed: f64,
    /// Dynamic viscosity of the medium (Pa·s). Uniform throughout the medium.
    pub viscosity: f64,
    /// Surface tension of the medium (N/m). Uniform, relevant for bubble dynamics.
    pub surface_tension: f64,
    /// Ambient pressure of the medium (Pa). Uniform.
    pub ambient_pressure: f64,
    /// Vapor pressure of the medium (Pa). Uniform, relevant for cavitation.
    pub vapor_pressure: f64,
    /// Polytropic index of gas within bubbles (dimensionless). Uniform, used in cavitation models.
    pub polytropic_index: f64,
    /// Specific heat capacity of the medium (J/kg·K). Uniform.
    pub specific_heat: f64,
    /// Thermal conductivity of the medium (W/m·K). Uniform.
    pub thermal_conductivity: f64,
    /// Volumetric thermal expansion coefficient (1/K). Uniform.
    pub thermal_expansion: f64,
    /// Gas diffusion coefficient within the medium (m²/s). Uniform, relevant for dissolved gases.
    pub gas_diffusion_coeff: f64,
    /// Thermal diffusivity of the medium (m²/s). Uniform.
    pub thermal_diffusivity: f64,
    /// Optical absorption coefficient (1/m) at the relevant wavelength. Uniform.
    pub mu_a: f64,
    /// Reduced optical scattering coefficient (1/m) at the relevant wavelength. Uniform.
    pub mu_s_prime: f64,
    /// 3D array representing the temperature at each grid point (Kelvin).
    /// This field can be dynamic and vary spatially, even in a homogeneous medium base.
    pub temperature: Array3<f64>,
    /// 3D array representing the radius of cavitation bubbles at each grid point (meters).
    /// This field can be dynamic if cavitation is modeled.
    pub bubble_radius: Array3<f64>,
    /// 3D array representing the radial velocity of bubble walls at each grid point (m/s).
    /// This field can be dynamic if cavitation is modeled.
    pub bubble_velocity: Array3<f64>,
    /// Coefficient `alpha_0` for power-law acoustic absorption (Np/m at `reference_frequency`).
    /// The units of `alpha0` depend on the units of `reference_frequency` and `delta`.
    /// Typically, if `reference_frequency` is in MHz and `delta` is `y`, then `alpha0` is in Np/m/MHz^y.
    /// If `alpha0 > 0` and `delta > 0`, power-law absorption is used.
    pub alpha0: f64,
    /// Exponent `delta` (often denoted as `y`) for power-law acoustic absorption.
    /// The absorption is proportional to `frequency^delta`.
    pub delta: f64,
    /// Acoustic nonlinearity parameter B/A (dimensionless). Uniform.
    pub b_a: f64,
    /// Reference frequency (Hz) at which `alpha0` is specified for power-law absorption.
    pub reference_frequency: f64,
    /// Internal cache for storing calculated absorption coefficients.
    /// This helps to avoid redundant computations for the same frequency.
    absorption_cache: AbsorptionCache,
    /// Lazily initialized 3D array for density, filled with `self.density`.
    /// Used by `Medium::density_array()`.
    density_array: OnceLock<Array3<f64>>,
    /// Lazily initialized 3D array for sound speed, filled with `self.sound_speed`.
    /// Used by `Medium::sound_speed_array()`.
    sound_speed_array: OnceLock<Array3<f64>>,
    // Removed shear_sound_speed_val, HomogeneousMedium will use trait default derived from Lame params
    /// Uniform shear viscosity coefficient value (Pa·s).
    pub shear_viscosity_coeff_val: f64,
    /// Uniform bulk viscosity coefficient value (Pa·s).
    pub bulk_viscosity_coeff_val: f64,
    // Removed shear_sound_speed_array cache, will use trait default
    /// Lazily initialized 3D array for shear viscosity coefficient.
    shear_viscosity_coeff_array: OnceLock<Array3<f64>>,
    /// Lazily initialized 3D array for bulk viscosity coefficient.
    bulk_viscosity_coeff_array: OnceLock<Array3<f64>>,
    /// Uniform Lamé's first parameter (lambda).
    pub lame_lambda_val: f64,
    /// Uniform Lamé's second parameter (mu, shear modulus).
    pub lame_mu_val: f64,
    /// Lazily initialized 3D array for Lamé's first parameter (lambda).
    lame_lambda_array: OnceLock<Array3<f64>>,
    /// Lazily initialized 3D array for Lamé's second parameter (mu).
    lame_mu_array: OnceLock<Array3<f64>>,
}

impl HomogeneousMedium {
    /// Creates a new `HomogeneousMedium` with specified primary properties and defaults for others.
    ///
    /// Initializes a medium with uniform `density` (kg/m³), `sound_speed` (m/s),
    /// optical `mu_a` (absorption coefficient, 1/m), and `mu_s_prime` (reduced optical scattering coefficient, 1/m).
    /// Other physical properties (viscosity, surface tension, etc.) are set to default values,
    /// often representative of water at approximately 20°C.
    /// Elastic properties (`lame_lambda_val`, `lame_mu_val`) are defaulted to 0.0 Pa, representing an ideal fluid
    /// from an elastic perspective unless explicitly set using builder methods like `with_lame_lambda` and `with_lame_mu`.
    ///
    /// The `temperature`, `bubble_radius`, and `bubble_velocity` fields are initialized as 3D arrays
    /// matching the `grid` dimensions, with default values:
    /// - Temperature: 293.15 K (20°C).
    /// - Bubble Radius: 10 µm.
    /// - Bubble Velocity: 0 m/s.
    ///
    /// Default acoustic absorption parameters (`alpha0`, `delta`, `reference_frequency`) are also set,
    /// typically corresponding to water.
    ///
    /// # Arguments
    ///
    /// * `density` - Density of the medium in kg/m³. Must be positive.
    /// * `sound_speed` - Speed of sound in the medium in m/s. Must be positive.
    /// * `grid` - A reference to the `Grid` defining the spatial dimensions for array fields.
    /// * `mu_a` - Optical absorption coefficient in 1/m. Must be non-negative.
    /// * `mu_s_prime` - Reduced optical scattering coefficient in 1/m. Must be non-negative.
    ///
    /// # Panics
    ///
    /// Panics if `density` or `sound_speed` are not positive, or if `mu_a` or `mu_s_prime` are negative.
    /// Creates a new `HomogeneousMedium` with specified primary properties and defaults for others.
    ///
    /// Initializes a medium with uniform `density` (kg/m³), `sound_speed` (m/s),
    /// optical `mu_a` (absorption coefficient, 1/m), and `mu_s_prime` (reduced optical scattering coefficient, 1/m).
    /// Other physical properties (viscosity, surface tension, etc.) are set to default values,
    /// often representative of water at approximately 20°C.
    ///
    /// The `temperature`, `bubble_radius`, and `bubble_velocity` fields are initialized as 3D arrays
    /// matching the `grid` dimensions, with default values:
    /// - Temperature: 293.15 K (20°C).
    /// - Bubble Radius: 10 µm.
    /// - Bubble Velocity: 0 m/s.
    ///
    /// Default acoustic absorption parameters (`alpha0`, `delta`, `reference_frequency`) are also set,
    /// typically corresponding to water.
    ///
    /// # Arguments
    ///
    /// * `density` - Density of the medium in kg/m³. Must be positive.
    /// * `sound_speed` - Speed of sound in the medium in m/s. Must be positive.
    /// * `grid` - A reference to the `Grid` defining the spatial dimensions for array fields.
    /// * `mu_a` - Optical absorption coefficient in 1/m. Must be non-negative.
    /// * `mu_s_prime` - Reduced optical scattering coefficient in 1/m. Must be non-negative.
    ///
    /// # Panics
    ///
    /// Panics if `density` or `sound_speed` are not positive, or if `mu_a` or `mu_s_prime` are negative.
    pub fn new(density: f64, sound_speed: f64, grid: &Grid, mu_a: f64, mu_s_prime: f64) -> Self {
        assert!(density > 0.0, "Density must be positive.");
        assert!(sound_speed > 0.0, "Sound speed must be positive.");
        assert!(mu_a >= 0.0, "Optical absorption mu_a must be non-negative.");
        assert!(mu_s_prime >= 0.0, "Reduced optical scattering mu_s_prime must be non-negative.");

        let viscosity = 1.002e-3;           // Pa·s (Water at 20°C)
        let surface_tension = 0.0728;       // N/m (Water at 20°C)
        let ambient_pressure = 1.013e5;     // Pa (Standard atmospheric pressure)
        let vapor_pressure = 2.338e3;       // Pa (Water at 20°C)
        let polytropic_index = 1.4;         // dimensionless (Typical for diatomic gases)
        let specific_heat = 4182.0;         // J/kg·K (Water)
        let thermal_conductivity = 0.598;   // W/m·K (Water at 20°C)
        let thermal_expansion = 2.1e-4;     // 1/K (Water at 20°C)
        let gas_diffusion_coeff = 2e-9;     // m²/s (Typical value for gas in water)
        let thermal_diffusivity = 1.43e-7;  // m²/s (Water at 20°C)
        let b_a = 5.2;                      // dimensionless (B/A parameter for water)
        let reference_frequency = 180000.0; // Hz (Example: 180 kHz)
        
        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15); // Kelvin (20°C)
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10e-6); // meters (10 µm default)
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz)); // m/s

        let alpha0 = 0.025; // Np/m @ reference_frequency (Example for water, ensure units match reference_frequency and delta)
        let delta = 1.0;    // Exponent for power law (e.g., linear dependence for water in some ranges)

        debug!(
            "Initialized HomogeneousMedium: density = {:.2e} kg/m³, sound_speed = {:.2e} m/s, B/A = {:.1}, optical mu_a = {:.2e} 1/m, mu_s' = {:.2e} 1/m",
            density, sound_speed, b_a, mu_a, mu_s_prime
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
            // shear_sound_speed_val: 0.0, // Removed
            shear_viscosity_coeff_val: 0.0,
            bulk_viscosity_coeff_val: 0.0,
            // shear_sound_speed_array: OnceLock::new(), // Removed
            shear_viscosity_coeff_array: OnceLock::new(),
            bulk_viscosity_coeff_array: OnceLock::new(),
            lame_lambda_val: 0.0, // Default for fluid
            lame_mu_val: 0.0,     // Default for fluid
            lame_lambda_array: OnceLock::new(),
            lame_mu_array: OnceLock::new(),
        }
    }

    /// Sets Lamé's first parameter (lambda) for the medium (Pa).
    ///
    /// This allows for configuring one of the fundamental elastic moduli.
    /// Setting this will invalidate the cached `lame_lambda_array`.
    ///
    /// # Arguments
    /// * `lambda` - The value for Lamé's first parameter (Pa).
    pub fn with_lame_lambda(mut self, lambda: f64) -> Self {
        self.lame_lambda_val = lambda;
        self.lame_lambda_array = OnceLock::new(); // Invalidate cache
        self
    }

    /// Sets Lamé's second parameter (mu, shear modulus) for the medium (Pa).
    ///
    /// This allows for configuring the shear modulus, crucial for shear wave propagation.
    /// Setting this will invalidate cached `lame_mu_array` and `shear_sound_speed_array`.
    ///
    /// # Arguments
    /// * `mu` - The value for Lamé's second parameter (shear modulus, Pa).
    pub fn with_lame_mu(mut self, mu: f64) -> Self {
        self.lame_mu_val = mu;
        self.lame_mu_array = OnceLock::new(); // Invalidate cache
        // self.shear_sound_speed_array = OnceLock::new(); // Removed, trait default will pick up change in lame_mu_array
        self
    }
    
    /// Sets the acoustic absorption coefficient parameters.
    ///
    /// # Arguments
    /// * `alpha0` - Absorption coefficient at reference frequency (Np/m)
    /// * `delta` - Power law exponent for frequency dependence
    pub fn with_acoustic_absorption(mut self, alpha0: f64, delta: f64) -> Self {
        self.alpha0 = alpha0;
        self.delta = delta;
        self.clear_caches();
        self
    }

    /// Creates a `HomogeneousMedium` instance with properties representative of water.
    ///
    /// Uses typical values for density (998 kg/m³) and sound speed (1500 m/s).
    /// Default optical properties are set using standard water values for near-infrared wavelengths.
    /// Other physical properties use the defaults from the `new()` constructor (e.g., viscosity of water at 20°C).
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `Grid` defining the spatial dimensions for array fields like temperature.
    pub fn water(grid: &Grid) -> Self {
        Self::new(998.0, 1500.0, grid, DEFAULT_WATER_ABSORPTION_COEFFICIENT, DEFAULT_WATER_SCATTERING_COEFFICIENT) 
    }

    /// Clears internal caches for absorption coefficient and property arrays.
    ///
    /// This method is called internally when properties that affect cached values
    /// (like temperature or bubble state) are updated, ensuring that subsequent
    /// requests for these values will trigger fresh calculations or re-initializations.
    fn clear_caches(&mut self) {
        debug!("Clearing medium property caches (absorption, density, sound_speed, shear_sound_speed, shear_viscosity, bulk_viscosity arrays)");
        self.absorption_cache.clear();
        self.density_array = OnceLock::new(); 
        self.sound_speed_array = OnceLock::new();
        // self.shear_sound_speed_array = OnceLock::new(); // Removed
        self.shear_viscosity_coeff_array = OnceLock::new();
        self.bulk_viscosity_coeff_array = OnceLock::new();
        self.lame_lambda_array = OnceLock::new();
        self.lame_mu_array = OnceLock::new();
    }
}

impl Medium for HomogeneousMedium {
    // Existing property getters...
    fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_lambda_val
    }

    fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_mu_val
    }

    fn lame_lambda_array(&self) -> Array3<f64> {
        self.lame_lambda_array.get_or_init(|| {
            debug!("Initializing lame_lambda_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.lame_lambda_val)
        }).clone()
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        self.lame_mu_array.get_or_init(|| {
            debug!("Initializing lame_mu_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.lame_mu_val)
        }).clone()
    }

    /// Returns the uniform density (kg/m³) of the medium.
    /// The spatial coordinates `_x`, `_y`, `_z` and `_grid` are ignored as the property is homogeneous.
    fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density }
    /// Returns the uniform speed of sound (m/s) in the medium.
    /// The spatial coordinates `_x`, `_y`, `_z` and `_grid` are ignored.
    fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed }
    /// Returns the uniform dynamic viscosity (Pa·s) of the medium.
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity }
    /// Returns the uniform surface tension (N/m) of the medium.
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.surface_tension }
    /// Returns the uniform ambient pressure (Pa) of the medium.
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.ambient_pressure }
    /// Returns the uniform vapor pressure (Pa) of the medium.
    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.vapor_pressure }
    /// Returns the uniform polytropic index (dimensionless) for gas within bubbles.
    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.polytropic_index }
    
    /// Returns the adiabatic index (gamma) - same as polytropic index
    fn gamma(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 
        self.polytropic_index 
    }

    /// Returns the uniform specific heat capacity (J/kg·K) of the medium.
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.specific_heat }
    /// Returns the uniform thermal conductivity (W/m·K) of the medium.
    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_conductivity }
    
    /// Calculates and returns the acoustic absorption coefficient (Np/m) at the specified `frequency`.
    ///
    /// The calculation depends on the medium's `alpha0` and `delta` fields:
    /// - If `alpha0 > 0` and `delta > 0`, power-law absorption is assumed: `alpha0 * (frequency / reference_frequency)^delta`.
    /// - Otherwise, a more complex model (`absorption::absorption_coefficient`) is used, which incorporates
    ///   thermal effects (based on temperature at `(x,y,z)`) and bubble effects (based on bubble radius at `(x,y,z)`).
    ///
    /// Results are cached in `self.absorption_cache` based on frequency to optimize performance
    /// for repeated calls with the same frequency. The cache is thread-safe.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Spatial coordinates (m) to determine local temperature and bubble radius if not using power-law.
    /// * `grid` - Reference to the grid for coordinate-to-index mapping.
    /// * `frequency` - Frequency (Hz) for which absorption is calculated.
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let cache_key = FloatKey(frequency);
        if let Some(cached) = self.absorption_cache.get(&cache_key) {
            return cached;
        }
        
        // Ensure indices are within bounds for temperature and bubble_radius arrays.
        let ix = grid.x_idx(x).min(self.temperature.shape()[0].saturating_sub(1));
        let iy = grid.y_idx(y).min(self.temperature.shape()[1].saturating_sub(1));
        let iz = grid.z_idx(z).min(self.temperature.shape()[2].saturating_sub(1));
        let t = self.temperature[[ix, iy, iz]];
        
        // Use the same clamped indices for bubble_radius for consistency.
        let r = self.bubble_radius[[ix, iy, iz]];
        
        let alpha = if self.alpha0 > 0.0 && self.delta > 0.0 {
            power_law_absorption::power_law_absorption_coefficient(frequency, self.alpha0, self.delta)
        } else {
            absorption::absorption_coefficient(frequency, t, Some(r))
        };
        
        self.absorption_cache.insert(cache_key, alpha);
        alpha
    }
    
    /// Returns the uniform volumetric thermal expansion coefficient (1/K) of the medium.
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_expansion }
    /// Returns the uniform gas diffusion coefficient (m²/s) in the medium.
    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.gas_diffusion_coeff }
    /// Returns the uniform thermal diffusivity (m²/s) of the medium.
    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_diffusivity }
    /// Returns the uniform acoustic nonlinearity parameter (B/A) of the medium.
    fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.b_a }
    /// Returns the uniform optical absorption coefficient (1/m) of the medium.
    fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_a }
    /// Returns the uniform reduced optical scattering coefficient (1/m) of the medium.
    fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_s_prime }
    /// Returns the reference frequency (Hz) used for power-law absorption calculations.
    fn reference_frequency(&self) -> f64 { self.reference_frequency }

    /// Updates the 3D temperature field of the medium with the provided `temperature` array.
    /// This action also clears internal caches (absorption, density array, sound speed array)
    /// as changes in temperature can affect these derived properties.
    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        debug!("Updating temperature in homogeneous medium and clearing caches.");
        self.temperature.assign(temperature);
        self.clear_caches();
    }
    
    /// Returns a reference to the 3D temperature field (Kelvin).
    fn temperature(&self) -> &Array3<f64> { &self.temperature }
    /// Returns a reference to the 3D bubble radius field (meters).
    fn bubble_radius(&self) -> &Array3<f64> { &self.bubble_radius }
    /// Returns a reference to the 3D bubble wall velocity field (m/s).
    fn bubble_velocity(&self) -> &Array3<f64> { &self.bubble_velocity }
    
    /// Updates the 3D bubble radius and velocity fields.
    /// This action also clears internal caches (absorption, density array, sound speed array)
    /// as changes in bubble properties can affect acoustic absorption.
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        debug!("Updating bubble state in homogeneous medium and clearing caches.");
        self.bubble_radius.assign(radius);
        self.bubble_velocity.assign(velocity);
        self.clear_caches();
    }
    
    /// Returns a 3D array of the medium's density, filled with the uniform `self.density` value.
    /// The array's shape matches the `self.temperature` field.
    /// The array is lazily initialized on first call and cached using `OnceLock`
    /// for subsequent accesses. The cache is cleared if relevant properties (like temperature, which dictates shape) change.
    fn density_array(&self) -> Array3<f64> {
        self.density_array.get_or_init(|| {
            debug!("Initializing density_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim(); 
            Array3::from_elem(shape, self.density)
        }).clone()
    }
    
    /// Returns a 3D array of the medium's speed of sound, filled with the uniform `self.sound_speed` value.
    /// The array's shape matches the `self.temperature` field.
    /// The array is lazily initialized on first call and cached using `OnceLock`
    /// for subsequent accesses. The cache is cleared if relevant properties change.
    fn sound_speed_array(&self) -> Array3<f64> {
        self.sound_speed_array.get_or_init(|| {
            debug!("Initializing sound_speed_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim(); 
            Array3::from_elem(shape, self.sound_speed)
        }).clone()
    }
    
    /// Returns `true` indicating that this medium's base properties are defined as homogeneous.
    fn is_homogeneous(&self) -> bool { true }

    // Removed override of shear_sound_speed_array to use trait default
    // fn shear_sound_speed_array(&self) -> Array3<f64> { ... }

    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        self.shear_viscosity_coeff_array.get_or_init(|| {
            debug!("Initializing shear_viscosity_coeff_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.shear_viscosity_coeff_val)
        }).clone()
    }

    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        self.bulk_viscosity_coeff_array.get_or_init(|| {
            debug!("Initializing bulk_viscosity_coeff_array cache for HomogeneousMedium.");
            let shape = self.temperature.dim();
            Array3::from_elem(shape, self.bulk_viscosity_coeff_val)
        }).clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use std::collections::HashSet; // For testing FloatKey hashing
    use ndarray::ShapeBuilder; // For .f() in Array3::from_elem

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    // --- FloatKey Tests ---
    #[test]
    fn test_float_key_equality() {
        // Helper to see the quantized value
        let _quantize = |f: f64| (f * 1e6).round() as i64; // Prefixed with _

        // Test case 1: Identical values
        assert_eq!(FloatKey(1.0), FloatKey(1.0));
        assert_eq!(FloatKey(0.0), FloatKey(0.0));
        assert_eq!(FloatKey(-1.0), FloatKey(-1.0));

        // Test case 2: Different values that should be EQUAL
        // (quantized values are the same)
        // 0.0 vs 0.4e-6: quantize(0.0) = 0, quantize(0.4e-6) = round(0.4) = 0
        assert_eq!(FloatKey(0.0), FloatKey(0.4e-6));
        // 0.0 vs -0.4e-6: quantize(-0.4e-6) = round(-0.4) = 0
        assert_eq!(FloatKey(0.0), FloatKey(-0.4e-6));
        // 1.0 vs 1.0 + 0.3e-6: quantize(1.0) = 1000000, quantize(1.0000003) = round(1000000.3) = 1000000
        assert_eq!(FloatKey(1.0), FloatKey(1.0 + 0.3e-6));
        // A value that rounds down vs one that rounds up to the same integer
        // 0.9999996 (rounds to 1.0 when scaled) vs 1.0000004 (rounds to 1.0 when scaled)
        // quantize(0.9999996) = round(999999.6) = 1000000
        // quantize(1.0000004) = round(1000000.4) = 1000000
        assert_eq!(FloatKey(0.9999996), FloatKey(1.0000004));


        // Test case 3: Values that should be DIFFERENT
        // (quantized values are different)
        // 0.0 vs 0.7e-6: quantize(0.0) = 0, quantize(0.7e-6) = round(0.7) = 1
        assert_ne!(FloatKey(0.0), FloatKey(0.7e-6));
        // 0.0 vs -0.7e-6: quantize(-0.7e-6) = round(-0.7) = -1
        assert_ne!(FloatKey(0.0), FloatKey(-0.7e-6));
        // 1.0 vs 1.0 + 0.8e-6: quantize(1.0) = 1000000, quantize(1.0000008) = round(1000000.8) = 1000001
        assert_ne!(FloatKey(1.0), FloatKey(1.0 + 0.8e-6));

        // Test case 4: Boundary conditions around 0.5 rounding
        // (0.5e-6 * 1e6).round() = (0.5).round() = 1 (in Rust, .5 rounds away from 0)
        // (0.49e-6 * 1e6).round() = (0.49).round() = 0
        assert_ne!(FloatKey(0.5e-6), FloatKey(0.49e-6)); // quantize(0.5e-6)=1, quantize(0.49e-6)=0
        assert_eq!(FloatKey(0.5e-6), FloatKey(5.000_000_000_000_001e-7)); // Both round to 1
        
        // (-0.5e-6 * 1e6).round() = (-0.5).round() = -1
        // (-0.49e-6 * 1e6).round() = (-0.49).round() = 0
        assert_ne!(FloatKey(-0.5e-6), FloatKey(-0.49e-6)); // quantize(-0.5e-6)=-1, quantize(-0.49e-6)=0

        // Test with a larger number
        // quantize(123.4567891) = round(123456789.1) = 123456789
        // quantize(123.4567894) = round(123456789.4) = 123456789
        assert_eq!(FloatKey(123.4567891), FloatKey(123.4567894));
        // quantize(123.4567896) = round(123456789.6) = 123456790
        assert_ne!(FloatKey(123.4567891), FloatKey(123.4567896));
    }

    #[test]
    fn test_float_key_hashing() {
        let mut set = HashSet::new();

        // Case 1: Values that are equal by new definition (quantize to same i64)
        // FloatKey(0.0) vs FloatKey(0.4e-6)
        // (0.0 * 1e6).round() = 0
        // (0.4e-6 * 1e6).round() = (0.4).round() = 0
        // These are equal and should hash to the same bucket and be one entry.
        set.insert(FloatKey(0.0));
        set.insert(FloatKey(0.4e-6)); // Should not add a new element
        set.insert(FloatKey(-0.3e-6)); // (-0.3).round() = 0. Should not add a new element
        assert_eq!(set.len(), 1, "Keys that quantize to the same i64 should result in one HashSet entry.");

        // Case 2: Add a value that is different (quantizes to a different i64)
        // FloatKey(0.7e-6)
        // (0.7e-6 * 1e6).round() = (0.7).round() = 1
        // This is different from FloatKey(0.0) and should be a new entry.
        set.insert(FloatKey(0.7e-6));
        assert_eq!(set.len(), 2, "Keys that quantize to different i64s should result in distinct HashSet entries.");

        // Case 3: Add another value equal to the first ones
        set.insert(FloatKey(0.1e-6)); // (0.1).round() = 0. Should not add a new element.
        assert_eq!(set.len(), 2, "Inserting a key equal to an existing one should not change set size.");

        // Case 4: Add another different value
        // FloatKey(-0.8e-6)
        // (-0.8e-6 * 1e6).round() = (-0.8).round() = -1
        set.insert(FloatKey(-0.8e-6));
        assert_eq!(set.len(), 3, "Adding another distinct key should increase set size.");

        // Case 5: Test with values from the original test that are still relevant
        // FloatKey(1.0)
        // (1.0 * 1e6).round() = 1000000
        set.insert(FloatKey(1.0)); // This is a new key, distinct from 0, 1, -1 (quantized)
        assert_eq!(set.len(), 4);

        // FloatKey(1.0 + 1e-7) = FloatKey(1.0000001)
        // (1.0000001 * 1e6).round() = (1000000.1).round() = 1000000
        // This is equal to FloatKey(1.0)
        set.insert(FloatKey(1.0 + 1e-7));
        assert_eq!(set.len(), 4, "Key equal to FloatKey(1.0) should not increase set size.");

        // FloatKey(1.00001)
        // (1.00001 * 1e6).round() = (1000010.0).round() = 1000010
        // This is different from 1000000.
        set.insert(FloatKey(1.00001));
        assert_eq!(set.len(), 5, "Key different from FloatKey(1.0) should increase set size.");
    }

    // --- AbsorptionCache Tests ---
    #[test]
    fn test_absorption_cache_empty() {
        let cache = AbsorptionCache::new();
        assert!(cache.cache.lock().unwrap().is_empty());
    }

    #[test]
    fn test_absorption_cache_insert_get() {
        let cache = AbsorptionCache::new();
        let key1 = FloatKey(100.0);
        let val1 = 0.5;
        cache.insert(key1, val1);

        assert_eq!(cache.get(&key1), Some(val1));
        assert_eq!(cache.get(&FloatKey(200.0)), None);
    }

    #[test]
    fn test_absorption_cache_clear() {
        let cache = AbsorptionCache::new();
        cache.insert(FloatKey(100.0), 0.5);
        cache.clear();
        assert!(cache.cache.lock().unwrap().is_empty());
    }
    
    #[test]
    fn test_absorption_cache_clone_is_independent() {
        let cache1 = AbsorptionCache::new();
        cache1.insert(FloatKey(100.0), 0.5);
        
        let cache2 = cache1.clone();
        assert!(cache2.cache.lock().unwrap().is_empty(), "Cloned cache should be empty");
        assert_eq!(cache1.get(&FloatKey(100.0)), Some(0.5), "Original cache should retain its values");
    }

    // --- HomogeneousMedium Constructor Tests ---
    #[test]
    fn test_homogeneous_medium_creation() {
        let grid_dims = (2,3,4);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let density = 1000.0;
        let sound_speed = 1500.0;
        let mu_a = 0.5;
        let mu_s_prime = 10.0;

        let medium = HomogeneousMedium::new(density, sound_speed, &test_grid, mu_a, mu_s_prime);

        assert_eq!(medium.density, density);
        assert_eq!(medium.sound_speed, sound_speed);
        assert_eq!(medium.mu_a, mu_a);
        assert_eq!(medium.mu_s_prime, mu_s_prime);

        // Check some default values (these are hardcoded in `new`)
        assert_eq!(medium.viscosity, 1.002e-3);
        assert_eq!(medium.surface_tension, 0.0728);
        assert_eq!(medium.ambient_pressure, 1.013e5);
        assert_eq!(medium.vapor_pressure, 2.338e3);
        assert_eq!(medium.polytropic_index, 1.4);
        assert_eq!(medium.specific_heat, 4182.0);
        assert_eq!(medium.thermal_conductivity, 0.598);
        assert_eq!(medium.thermal_expansion, 2.1e-4);
        assert_eq!(medium.gas_diffusion_coeff, 2e-9);
        assert_eq!(medium.thermal_diffusivity, 1.43e-7);
        assert_eq!(medium.b_a, 5.2);
        assert_eq!(medium.reference_frequency, 180000.0);
        assert_eq!(medium.alpha0, 0.025);
        assert_eq!(medium.delta, 1.0);
        
        let expected_dim = (grid_dims.0, grid_dims.1, grid_dims.2);
        assert_eq!(medium.temperature.dim(), expected_dim);
        assert!(medium.temperature.iter().all(|&t| (t - 293.15).abs() < 1e-9 ));
        assert_eq!(medium.bubble_radius.dim(), expected_dim);
        assert!(medium.bubble_radius.iter().all(|&r| (r - 10e-6).abs() < 1e-9 ));
        assert_eq!(medium.bubble_velocity.dim(), expected_dim);
        assert!(medium.bubble_velocity.iter().all(|&v| v == 0.0));

        assert!(medium.absorption_cache.cache.lock().unwrap().is_empty());
        assert!(medium.density_array.get().is_none());
        assert!(medium.sound_speed_array.get().is_none());
    }

    #[test]
    #[should_panic(expected = "Density must be positive.")]
    fn test_homogeneous_medium_panic_density() {
        let grid = create_test_grid(2,2,2);
        HomogeneousMedium::new(0.0, 1500.0, &grid, 0.5, 10.0);
    }

    #[test]
    #[should_panic(expected = "Sound speed must be positive.")]
    fn test_homogeneous_medium_panic_sound_speed() {
        let grid = create_test_grid(2,2,2);
        HomogeneousMedium::new(1000.0, 0.0, &grid, 0.5, 10.0);
    }
    
    #[test]
    #[should_panic(expected = "Optical absorption mu_a must be non-negative.")]
    fn test_homogeneous_medium_panic_mu_a() {
        let grid = create_test_grid(2,2,2);
        HomogeneousMedium::new(1000.0, 1500.0, &grid, -0.5, 10.0);
    }

    #[test]
    #[should_panic(expected = "Reduced optical scattering mu_s_prime must be non-negative.")]
    fn test_homogeneous_medium_panic_mu_s_prime() {
        let grid = create_test_grid(2,2,2);
        HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.5, -10.0);
    }


    #[test]
    fn test_homogeneous_medium_water() {
        let grid = create_test_grid(2,2,2);
        let medium = HomogeneousMedium::water(&grid);

        assert_eq!(medium.density, 998.0);
        assert_eq!(medium.sound_speed, 1500.0);
        assert_eq!(medium.mu_a, 0.1); 
        assert_eq!(medium.mu_s_prime, 1.0); 
        assert_eq!(medium.viscosity, 1.002e-3);
        assert_eq!(medium.b_a, 5.2);
    }

    // --- Medium Trait Implementation Tests ---
    #[test]
    fn test_property_getters() {
        let grid = create_test_grid(1,1,1);
        let medium = HomogeneousMedium::water(&grid); 
        
        assert_eq!(medium.density(0.0,0.0,0.0, &grid), medium.density);
        assert_eq!(medium.sound_speed(0.0,0.0,0.0, &grid), medium.sound_speed);
        assert_eq!(medium.viscosity(0.0,0.0,0.0, &grid), medium.viscosity);
        assert_eq!(medium.surface_tension(0.0,0.0,0.0, &grid), medium.surface_tension);
        assert_eq!(medium.ambient_pressure(0.0,0.0,0.0, &grid), medium.ambient_pressure);
        assert_eq!(medium.vapor_pressure(0.0,0.0,0.0, &grid), medium.vapor_pressure);
        assert_eq!(medium.polytropic_index(0.0,0.0,0.0, &grid), medium.polytropic_index);
        assert_eq!(medium.specific_heat(0.0,0.0,0.0, &grid), medium.specific_heat);
        assert_eq!(medium.thermal_conductivity(0.0,0.0,0.0, &grid), medium.thermal_conductivity);
        assert_eq!(medium.thermal_expansion(0.0,0.0,0.0, &grid), medium.thermal_expansion);
        assert_eq!(medium.gas_diffusion_coefficient(0.0,0.0,0.0, &grid), medium.gas_diffusion_coeff);
        assert_eq!(medium.thermal_diffusivity(0.0,0.0,0.0, &grid), medium.thermal_diffusivity);
        assert_eq!(medium.nonlinearity_coefficient(0.0,0.0,0.0, &grid), medium.b_a);
        assert_eq!(medium.absorption_coefficient_light(0.0,0.0,0.0, &grid), medium.mu_a);
        assert_eq!(medium.reduced_scattering_coefficient_light(0.0,0.0,0.0, &grid), medium.mu_s_prime);
        assert_eq!(medium.reference_frequency(), medium.reference_frequency);
    }

    #[test]
    fn test_absorption_coefficient_calculation_and_caching() {
        let grid_dims = (1,1,1);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut medium = HomogeneousMedium::water(&grid);
        medium.alpha0 = 0.1; 
        medium.delta = 1.1;  
        medium.reference_frequency = 1e6; 

        let freq1 = 2e6;
        let expected_alpha1 = medium.alpha0 * (freq1 / medium.reference_frequency).powf(medium.delta);
        
        let alpha1_call1 = medium.absorption_coefficient(0.0,0.0,0.0, &grid, freq1);
        assert!((alpha1_call1 - expected_alpha1).abs() < 1e-9, "Initial calculation for freq1 is incorrect. Got {}, expected {}", alpha1_call1, expected_alpha1);

        let alpha1_call2 = medium.absorption_coefficient(0.0,0.0,0.0, &grid, freq1);
        assert_eq!(alpha1_call1, alpha1_call2, "Expected cached result for absorption coefficient on freq1");

        medium.alpha0 = 0.0; 
        medium.absorption_cache.clear(); 
        let freq3 = 1.5e6;
        let t_val = medium.temperature[[0,0,0]];
        let r_val = medium.bubble_radius[[0,0,0]];
        let expected_alpha3_thermal = absorption::absorption_coefficient(freq3, t_val, Some(r_val));
        
        let alpha3_call1 = medium.absorption_coefficient(0.0,0.0,0.0, &grid, freq3);
        assert!((alpha3_call1 - expected_alpha3_thermal).abs() < 1e-9, "Thermal absorption calculation incorrect. Got {}, expected {}", alpha3_call1, expected_alpha3_thermal);
        
        let alpha3_call2 = medium.absorption_coefficient(0.0,0.0,0.0, &grid, freq3);
        assert_eq!(alpha3_call1, alpha3_call2, "Expected cached result for thermal absorption on freq3");
    }

    #[test]
    fn test_update_temperature_and_access() {
        let grid_dims = (2,2,2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut medium = HomogeneousMedium::water(&grid);
        let initial_temp_val = medium.temperature()[[0,0,0]];

        let new_temp_val = 300.0;
        let new_temp_array = Array3::from_elem(medium.temperature.dim().f(), new_temp_val);
        
        medium.absorption_coefficient(0.0,0.0,0.0, &grid, 1e6); 
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_some());
        let _ = medium.density_array(); 
        assert!(medium.density_array.get().is_some());

        medium.update_temperature(&new_temp_array);
        
        assert_eq!(medium.temperature()[[0,0,0]], new_temp_val);
        assert_ne!(medium.temperature()[[0,0,0]], initial_temp_val);
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_none(), "Absorption cache should be cleared");
        assert!(medium.density_array.get().is_none(), "Density array cache should be cleared");
        assert!(medium.sound_speed_array.get().is_none(), "Sound speed array cache should be cleared");
    }

    #[test]
    fn test_update_bubble_state_and_access() {
        let grid_dims = (2,2,2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut medium = HomogeneousMedium::water(&grid);
        let initial_radius_val = medium.bubble_radius()[[0,0,0]];
        let initial_velocity_val = medium.bubble_velocity()[[0,0,0]];

        let new_radius_val = 15e-6;
        let new_velocity_val = 1.0;
        let new_radius_array = Array3::from_elem(medium.bubble_radius.dim().f(), new_radius_val);
        let new_velocity_array = Array3::from_elem(medium.bubble_velocity.dim().f(), new_velocity_val);
        
        medium.absorption_coefficient(0.0,0.0,0.0, &grid, 1e6); 
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_some());

        medium.update_bubble_state(&new_radius_array, &new_velocity_array);
        
        assert_eq!(medium.bubble_radius()[[0,0,0]], new_radius_val);
        assert_ne!(medium.bubble_radius()[[0,0,0]], initial_radius_val);
        assert_eq!(medium.bubble_velocity()[[0,0,0]], new_velocity_val);
        assert_ne!(medium.bubble_velocity()[[0,0,0]], initial_velocity_val);
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_none(), "Cache should be cleared");
    }

    #[test]
    fn test_density_array_caching() {
        let grid_dims = (2,2,2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let medium = HomogeneousMedium::water(&grid);
        
        let arr1_ptr_val: *const f64 = medium.density_array().as_ptr();
        let arr1_clone = medium.density_array().clone();
        
        let arr2_ptr_val: *const f64 = medium.density_array().as_ptr();
        
        assert_eq!(arr1_ptr_val, arr2_ptr_val, "Expected density_array to return same pointer (cached)");
        assert_eq!(arr1_clone, medium.density_array(), "Expected density_array data to be consistent");
        assert_eq!(medium.density_array().dim(), grid_dims); 
        assert!(medium.density_array().iter().all(|&d| (d - medium.density).abs() < 1e-9 ));
    }

    #[test]
    fn test_sound_speed_array_caching() {
        let grid_dims = (2,2,2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let medium = HomogeneousMedium::water(&grid);
        
        let arr1_ptr_val: *const f64 = medium.sound_speed_array().as_ptr();
        let arr1_clone = medium.sound_speed_array().clone();
        
        let arr2_ptr_val: *const f64 = medium.sound_speed_array().as_ptr();
        
        assert_eq!(arr1_ptr_val, arr2_ptr_val, "Expected sound_speed_array to return same pointer (cached)");
        assert_eq!(arr1_clone, medium.sound_speed_array(), "Expected sound_speed_array data to be consistent");
        assert_eq!(medium.sound_speed_array().dim(), grid_dims); 
        assert!(medium.sound_speed_array().iter().all(|&c| (c - medium.sound_speed).abs() < 1e-9 ));
    }

    #[test]
    fn test_is_homogeneous() {
        let grid = create_test_grid(1,1,1);
        let medium = HomogeneousMedium::water(&grid);
        assert!(medium.is_homogeneous());
    }

    #[test]
    fn test_shear_fields_initialization() {
        let grid = create_test_grid(2, 2, 2);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
        // assert_eq!(medium.shear_sound_speed_val, 0.0); // Field removed
        assert_eq!(medium.shear_viscosity_coeff_val, 0.0);
        assert_eq!(medium.bulk_viscosity_coeff_val, 0.0);
        // assert!(medium.shear_sound_speed_array.get().is_none()); // Field removed
        assert!(medium.shear_viscosity_coeff_array.get().is_none());
        assert!(medium.bulk_viscosity_coeff_array.get().is_none());

        let water_medium = HomogeneousMedium::water(&grid);
        // assert_eq!(water_medium.shear_sound_speed_val, 0.0); // Field removed
        assert_eq!(water_medium.shear_viscosity_coeff_val, 0.0);
        assert_eq!(water_medium.bulk_viscosity_coeff_val, 0.0);
    }

    #[test]
    fn test_shear_property_array_methods() {
        let grid_dims = (2, 3, 4);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);

        // medium.shear_sound_speed_val = 10.0; // Field removed, shear speed derived from Lame params
        medium.lame_mu_val = 10.0 * 10.0 * 1000.0; // cs = 10, rho = 1000 => mu = cs^2 * rho
        medium.shear_viscosity_coeff_val = 0.1;
        medium.bulk_viscosity_coeff_val = 0.2;

        let sss_arr = medium.shear_sound_speed_array(); // Should use trait default
        assert_eq!(sss_arr.dim(), grid_dims);
        assert!(sss_arr.iter().all(|&x| (x - 10.0).abs() < 1e-9));
        // assert!(medium.shear_sound_speed_array.get().is_some()); // Cache field removed

        let svc_arr = medium.shear_viscosity_coeff_array();
        assert_eq!(svc_arr.dim(), grid_dims);
        assert!(svc_arr.iter().all(|&x| (x - 0.1).abs() < 1e-9));
        assert!(medium.shear_viscosity_coeff_array.get().is_some());

        let bvc_arr = medium.bulk_viscosity_coeff_array();
        assert_eq!(bvc_arr.dim(), grid_dims);
        assert!(bvc_arr.iter().all(|&x| (x - 0.2).abs() < 1e-9));
        assert!(medium.bulk_viscosity_coeff_array.get().is_some());
    }

    #[test]
    fn test_clear_caches_includes_shear_properties() {
        let grid = create_test_grid(2, 2, 2);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);

        // Populate caches
        let _ = medium.density_array();
        // let _ = medium.shear_sound_speed_array(); // No longer directly cached on struct
        let _ = medium.shear_viscosity_coeff_array();
        let _ = medium.bulk_viscosity_coeff_array();
        medium.absorption_coefficient(0.0,0.0,0.0, &grid, 1e6);


        assert!(medium.density_array.get().is_some());
        // assert!(medium.shear_sound_speed_array.get().is_none()); // No longer a field
        assert!(medium.shear_viscosity_coeff_array.get().is_some());
        assert!(medium.bulk_viscosity_coeff_array.get().is_some());
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_some());

        medium.clear_caches();

        assert!(medium.density_array.get().is_none());
        assert!(medium.sound_speed_array.get().is_none()); // Also check this standard one
        // assert!(medium.shear_sound_speed_array.get().is_none()); // No longer a field
        assert!(medium.shear_viscosity_coeff_array.get().is_none());
        assert!(medium.bulk_viscosity_coeff_array.get().is_none());
        assert!(medium.absorption_cache.get(&FloatKey(1e6)).is_none());
        assert!(medium.lame_lambda_array.get().is_none(), "Lame lambda array cache should be cleared");
        assert!(medium.lame_mu_array.get().is_none(), "Lame mu array cache should be cleared");
    }

    #[test]
    fn test_lame_parameter_getters_and_builders() {
        let grid = create_test_grid(1,1,1);
        let lambda_val = 2.25e9; // Water bulk modulus
        let mu_val = 0.0;       // Water shear modulus

        let medium = HomogeneousMedium::water(&grid)
            .with_lame_lambda(lambda_val)
            .with_lame_mu(mu_val);

        assert_eq!(medium.lame_lambda(0.0,0.0,0.0, &grid), lambda_val);
        assert_eq!(medium.lame_mu(0.0,0.0,0.0, &grid), mu_val);
        assert_eq!(medium.lame_lambda_val, lambda_val);
        assert_eq!(medium.lame_mu_val, mu_val);
    }

    #[test]
    fn test_lame_array_caching_and_values() {
        let grid_dims = (2,2,2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let lambda_val = 2.25e9;
        let mu_val = 1e6; // Some non-zero mu

        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0)
            .with_lame_lambda(lambda_val)
            .with_lame_mu(mu_val);

        // Test lambda array - method returns clones, so we test data consistency and caching behavior
        let lambda_arr1 = medium.lame_lambda_array();
        let lambda_arr2 = medium.lame_lambda_array();
        assert_eq!(lambda_arr1, lambda_arr2, "Expected lame_lambda_array to return consistent data (cached)");
        assert_eq!(lambda_arr1.dim(), grid_dims);
        assert!(lambda_arr1.iter().all(|&val| (val - lambda_val).abs() < 1e-9 ));
        
        // Verify that the cache is populated (internal state check)
        assert!(medium.lame_lambda_array.get().is_some(), "Expected lame_lambda_array cache to be populated");

        // Test mu array - since method returns clone, we test data consistency instead of pointer equality
        let mu_arr1 = medium.lame_mu_array();
        let mu_arr2 = medium.lame_mu_array();
        assert_eq!(mu_arr1, mu_arr2, "Expected lame_mu_array to return consistent data (cached)");
        assert_eq!(mu_arr1.dim(), grid_dims);
        assert!(mu_arr1.iter().all(|&val| (val - mu_val).abs() < 1e-9 ));
    }

    #[test]
    fn test_derived_wave_speeds_elastic() {
        let grid = create_test_grid(1,1,1);
        let density = 2000.0;
        // For steel: E ~ 200 GPa, nu ~ 0.3
        // mu = E / (2*(1+nu)) = 200e9 / (2 * 1.3) = 200e9 / 2.6 ~ 76.92e9 Pa
        // lambda = E*nu / ((1+nu)(1-2*nu)) = 200e9 * 0.3 / (1.3 * 0.4) = 60e9 / 0.52 ~ 115.38e9 Pa
        let mu_val = 76.92e9;
        let lambda_val = 115.38e9;

        let mut medium = HomogeneousMedium::new(density, 1500.0, &grid, 0.1, 1.0) // sound_speed is for acoustic model
            .with_lame_lambda(lambda_val)
            .with_lame_mu(mu_val);
        medium.density = density; // Ensure density is set

        let expected_cp = ((lambda_val + 2.0 * mu_val) / density).sqrt(); // Compressional wave speed
        let expected_cs = (mu_val / density).sqrt(); // Shear wave speed

        assert!((medium.compressional_wave_speed(0.0,0.0,0.0, &grid) - expected_cp).abs() < 1.0); // Allow small tolerance
        assert!((medium.shear_wave_speed(0.0,0.0,0.0, &grid) - expected_cs).abs() < 1.0);

        // Test array versions implicitly via default trait methods if Medium trait test covers them
        // Or test shear_sound_speed_array directly as it's overridden in Medium trait for Homogeneous
        let cs_array = medium.shear_sound_speed_array();
        assert!(cs_array.iter().all(|&cs_val| (cs_val - expected_cs).abs() < 1.0 ));

        // Check default shear_sound_speed_array (which uses lame_mu_array and density_array)
        let sss_array_from_trait = Medium::shear_sound_speed_array(&medium); // Explicitly call trait method
        assert!(sss_array_from_trait.iter().all(|&val| (val - expected_cs).abs() < 1.0 ));

    }

    #[test]
    fn test_water_elastic_properties_default_to_fluid() {
        let grid = create_test_grid(1,1,1);
        let medium = HomogeneousMedium::water(&grid); // Should have mu=0, lambda can be K if set

        assert_eq!(medium.lame_mu_val, 0.0, "Water (default) should have mu=0");
        // lambda for water is its bulk modulus K ~ 2.25 GPa.
        // The default HomogeneousMedium::new sets lambda to 0. water() constructor calls new().
        // If we want water to have its K as lambda, we'd need a .with_lame_lambda(K_water)
        assert_eq!(medium.lame_lambda_val, 0.0, "Default water lambda from new() is 0, not K_water unless explicitly set");

        assert_eq!(medium.shear_wave_speed(0.0,0.0,0.0, &grid), 0.0);

        // Test with explicit setting for water as an elastic fluid
        let k_water = 2.25e9;
        let water_elastic_fluid = HomogeneousMedium::water(&grid)
            .with_lame_lambda(k_water)
            .with_lame_mu(0.0);

        assert_eq!(water_elastic_fluid.lame_mu(0.0,0.0,0.0, &grid), 0.0);
        assert_eq!(water_elastic_fluid.lame_lambda(0.0,0.0,0.0, &grid), k_water);
        assert_eq!(water_elastic_fluid.shear_wave_speed(0.0,0.0,0.0, &grid), 0.0);

        let expected_cp_water = (k_water / water_elastic_fluid.density).sqrt();
        assert!((water_elastic_fluid.compressional_wave_speed(0.0,0.0,0.0, &grid) - expected_cp_water).abs() < 1.0);
        // Also check that sound_speed (original field) is different from this calculated one,
        // as sound_speed is set independently in new().
        assert!((water_elastic_fluid.sound_speed - expected_cp_water).abs() > 1.0);


    }
}
