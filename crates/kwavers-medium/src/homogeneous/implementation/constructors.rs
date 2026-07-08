use kwavers_core::constants::acoustic_parameters::{
    AIR_ABSORPTION_ALPHA_0, AIR_ABSORPTION_POWER, AIR_POLYTROPIC_INDEX, AIR_SPECIFIC_HEAT_CP,
    REFERENCE_FREQUENCY_HZ, TISSUE_NONLINEARITY_B_A,
};
use kwavers_core::constants::cavitation::{
    GAS_DIFFUSION_COEFFICIENT_AIR, VISCOSITY_AIR, VISCOSITY_WATER,
};
use kwavers_core::constants::fundamental::ACOUSTIC_ABSORPTION_TISSUE;
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, DENSITY_TISSUE, DENSITY_WATER, SOUND_SPEED_AIR, SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER,
};
use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_K, ROOM_TEMPERATURE_K, THERMAL_CONDUCTIVITY_AIR, THERMAL_EXPANSION_AIR_20C,
};
use kwavers_core::constants::tissue_acoustics::{
    B_OVER_A_AIR, DENSITY_AIR, DENSITY_BLOOD, SOUND_SPEED_BLOOD,
};
use kwavers_core::constants::BLOOD_VISCOSITY_37C;
use kwavers_core::constants::MHZ_TO_HZ;
use kwavers_grid::Grid;
use leto::Array3;

use super::HomogeneousMedium;

impl HomogeneousMedium {
    /// Create a tissue medium with standard properties
    pub fn tissue(grid: &Grid) -> Self {
        use kwavers_core::constants::SOUND_SPEED_TISSUE;
        let mut medium = Self::new(DENSITY_TISSUE, SOUND_SPEED_TISSUE, 0.75, 15.0, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, BODY_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / MHZ_TO_HZ).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a water medium with standard properties at 20°C
    pub fn water(grid: &Grid) -> Self {
        let mut medium = Self::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.01, 0.1, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, ROOM_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / MHZ_TO_HZ).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a blood medium with standard properties at 37°C
    pub fn blood(grid: &Grid) -> Self {
        let mut medium = Self::new(DENSITY_BLOOD, SOUND_SPEED_BLOOD, 0.15, 0.5, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, BODY_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        medium.viscosity = BLOOD_VISCOSITY_37C;
        medium.shear_viscosity = BLOOD_VISCOSITY_37C;
        medium.bulk_viscosity = 2.5 * BLOOD_VISCOSITY_37C;
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / MHZ_TO_HZ).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create an air medium with standard properties at 20°C
    pub fn air(grid: &Grid) -> Self {
        Self {
            density: DENSITY_AIR,
            sound_speed: SOUND_SPEED_AIR,
            viscosity: VISCOSITY_AIR,
            surface_tension: 0.0,
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: 0.0,
            polytropic_index: AIR_POLYTROPIC_INDEX,
            specific_heat: AIR_SPECIFIC_HEAT_CP, // 1005 J/(kg·K)
            thermal_conductivity: THERMAL_CONDUCTIVITY_AIR,
            shear_viscosity: VISCOSITY_AIR,
            bulk_viscosity: 0.0,
            absorption_alpha: AIR_ABSORPTION_ALPHA_0,
            absorption_power: AIR_ABSORPTION_POWER,
            thermal_expansion: THERMAL_EXPANSION_AIR_20C,
            gas_diffusion: GAS_DIFFUSION_COEFFICIENT_AIR,
            nonlinearity: B_OVER_A_AIR,
            optical_absorption: 0.0,
            optical_scattering: 0.0,
            reference_frequency: REFERENCE_FREQUENCY_HZ,
            temperature: Array3::from_elem([grid.nx, grid.ny, grid.nz], ROOM_TEMPERATURE_K),
            bubble_radius: Array3::zeros([grid.nx, grid.ny, grid.nz]),
            bubble_velocity: Array3::zeros([grid.nx, grid.ny, grid.nz]),
            density_cache: Array3::from_elem([grid.nx, grid.ny, grid.nz], DENSITY_AIR),
            sound_speed_cache: Array3::from_elem([grid.nx, grid.ny, grid.nz], SOUND_SPEED_AIR),
            absorption_cache: Array3::from_elem(
                [grid.nx, grid.ny, grid.nz],
                AIR_ABSORPTION_ALPHA_0 * 1.0_f64.powi(2),
            ),
            nonlinearity_cache: Array3::from_elem([grid.nx, grid.ny, grid.nz], B_OVER_A_AIR),
            lame_lambda: DENSITY_AIR * SOUND_SPEED_AIR * SOUND_SPEED_AIR,
            lame_mu: 0.0,
            grid_shape: [grid.nx, grid.ny, grid.nz],
        }
    }

    /// Create from minimal parameters
    pub fn from_minimal(density: f64, sound_speed: f64, grid: &Grid) -> Self {
        let mut medium = Self::new(density, sound_speed, 0.01, 0.1, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, ROOM_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / MHZ_TO_HZ).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a soft tissue medium for elastography simulations
    ///
    /// λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
    pub fn soft_tissue(youngs_modulus: f64, poisson_ratio: f64, grid: &Grid) -> Self {
        let density = DENSITY_TISSUE;
        let sound_speed = SOUND_SPEED_TISSUE;

        let mut medium = Self::new(density, sound_speed, 0.01, 0.1, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, BODY_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, sound_speed);

        let nu = poisson_ratio;
        medium.lame_lambda = youngs_modulus * nu / ((1.0 + nu) * 2.0f64.mul_add(-nu, 1.0));
        medium.lame_mu = youngs_modulus / (2.0 * (1.0 + nu));

        // Approximate soft tissue as water viscosity (1.002e-3 Pa·s) — SSOT VISCOSITY_WATER.
        medium.viscosity = VISCOSITY_WATER;
        medium.shear_viscosity = VISCOSITY_WATER;
        medium.bulk_viscosity = 2.5 * VISCOSITY_WATER;

        let alpha = ACOUSTIC_ABSORPTION_TISSUE * (medium.reference_frequency / MHZ_TO_HZ).powf(1.1);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, TISSUE_NONLINEARITY_B_A);

        medium
    }

    /// Create liver tissue medium for SWE simulations
    pub fn liver_tissue(fibrosis_stage: u8, grid: &Grid) -> Self {
        let youngs_modulus_kpa = match fibrosis_stage {
            0 => 5.0,
            1 => 6.5,
            2 => 8.0,
            3 => 11.0,
            4 => 18.0,
            _ => 8.0,
        };
        Self::soft_tissue(youngs_modulus_kpa * 1000.0, 0.49, grid)
    }

    /// Create a homogeneous elastic medium parameterised by **physical wave
    /// speeds** rather than Lamé parameters.
    ///
    /// This is the natural API for replicating k-Wave's
    /// `medium.sound_speed_compression` / `medium.sound_speed_shear` inputs to
    /// `pstdElastic2D` / `pstdElastic3D`. The Lamé parameters are derived from
    /// the wave-speed identities for an isotropic linear elastic medium:
    ///
    /// ```text
    /// μ  = ρ · c_s²                  (shear modulus)
    /// λ  = ρ · (c_p² − 2·c_s²)        (first Lamé parameter)
    /// ```
    ///
    /// where `c_p` (compressional) and `c_s` (shear) are the longitudinal and
    /// transverse wave speeds. These follow directly from the dispersion
    /// relations of the elastic wave equation
    /// `ρ·∂²u/∂t² = (λ+μ)·∇(∇·u) + μ·∇²u`:
    /// the longitudinal mode propagates at `√((λ+2μ)/ρ)` and the transverse
    /// mode at `√(μ/ρ)`.
    ///
    /// # Physical constraints
    ///
    /// - `c_p > 0`, `c_s ≥ 0`, `density > 0` are required.
    /// - `c_s ≤ c_p / √2` is required so that `λ ≥ 0`. This is the standard
    ///   thermodynamic-stability bound (Poisson ratio ν ≥ 0); for soft tissue
    ///   `c_s ≪ c_p` so the bound is satisfied with large margin.
    /// - `c_s = 0` recovers a fluid (μ = 0, no shear support); `λ = ρ·c_p²`
    ///   then matches the acoustic-wave bulk modulus `K = ρ·c_p²` of a fluid.
    ///
    /// # Returns
    ///
    /// `Some(medium)` on success, `None` if any constraint is violated.
    pub fn elastic_homogeneous(
        density: f64,
        c_compression: f64,
        c_shear: f64,
        grid: &Grid,
    ) -> Option<Self> {
        if !density.is_finite() || density <= 0.0 {
            return None;
        }
        if !c_compression.is_finite() || c_compression <= 0.0 {
            return None;
        }
        if !c_shear.is_finite() || c_shear < 0.0 {
            return None;
        }
        // Thermodynamic stability: λ ≥ 0 ⇔ c_p² ≥ 2·c_s²
        if c_shear * c_shear * 2.0 > c_compression * c_compression {
            return None;
        }

        let (lambda, mu) = crate::elastic::lame_from_speeds(c_compression, c_shear, density);

        // Build a baseline acoustic-only homogeneous medium at c_p, then
        // overwrite Lamé parameters with the elastic-derived values.
        let mut medium = Self::new(density, c_compression, 0.0, 0.0, grid);
        let shape = [grid.nx, grid.ny, grid.nz];
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, BODY_TEMPERATURE_K);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, c_compression);

        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / MHZ_TO_HZ).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);

        medium.lame_lambda = lambda;
        medium.lame_mu = mu;

        Some(medium)
    }

    /// Override the Lamé parameters on an existing homogeneous medium.
    ///
    /// Returns `Err` if either value is non-finite or negative; positivity is
    /// enforced for `μ` (shear modulus) and non-negativity for `λ`. Both are
    /// physically required for a passive isotropic elastic solid.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_lame_parameters(
        &mut self,
        lame_lambda: f64,
        lame_mu: f64,
    ) -> Result<(), &'static str> {
        if !lame_lambda.is_finite() {
            return Err("lame_lambda must be finite");
        }
        if !lame_mu.is_finite() || lame_mu < 0.0 {
            return Err("lame_mu must be finite and non-negative");
        }
        // λ may be slightly negative for some auxetic materials, but in
        // standard tissue/water/typical elastography we enforce λ ≥ 0.
        if lame_lambda < 0.0 {
            return Err("lame_lambda must be non-negative for non-auxetic media");
        }
        self.lame_lambda = lame_lambda;
        self.lame_mu = lame_mu;
        Ok(())
    }

    /// Read the first Lamé parameter λ (Pa) currently stored on this medium.
    #[must_use]
    pub fn lame_lambda_value(&self) -> f64 {
        self.lame_lambda
    }

    /// Read the second Lamé parameter μ (shear modulus, Pa) currently stored.
    #[must_use]
    pub fn lame_mu_value(&self) -> f64 {
        self.lame_mu
    }
}
