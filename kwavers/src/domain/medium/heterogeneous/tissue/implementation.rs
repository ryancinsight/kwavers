//! `HeterogeneousTissueMedium` implementation

use super::{TissueMap, TissueRegion};
use crate::core::error::{KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::absorption::TissueType;
use crate::domain::medium::{
    absorption::{tissue::TissueProperties, TISSUE_PROPERTIES},
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium, MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use ndarray::{Array3, ArrayView3, ArrayViewMut3};

/// Heterogeneous tissue medium with spatial tissue type variations
/// TODO_AUDIT: P2 - Tissue Attenuation - Add realistic tissue attenuation models (frequency-dependent, temperature-dependent) for accurate ultrasound propagation, replacing simplified constant absorption
/// DEPENDS ON: physics/acoustics/attenuation/biot.rs, physics/temperature/thermoelastic.rs
/// MISSING: Frequency power law: α(f) = α₀ fᵇ where b ∈ [1.0, 2.0] for different tissues
/// MISSING: Temperature dependence: α(T) = α₀ (1 + γ(T-T₀)) for thermal expansion effects
/// MISSING: Multiple relaxation mechanisms: α(f) = ∑ αᵢ / (1 + (f/fᵢ)²) for viscoelastic tissues
/// MISSING: Nonlinear parameter B/A with temperature dependence for high-intensity applications
#[derive(Debug, Clone)]
pub struct HeterogeneousTissueMedium {
    /// 3D map of tissue types
    tissue_map: TissueMap,
    /// Grid structure
    grid: Grid,

    // Cached property arrays (required for ArrayAccess)
    density: Array3<f64>,
    sound_speed: Array3<f64>,
    absorption: Array3<f64>,
    nonlinearity: Array3<f64>,

    // Thermal state
    temperature: Array3<f64>,

    // Bubble state
    bubble_radius: Array3<f64>,
    bubble_velocity: Array3<f64>,
}

impl HeterogeneousTissueMedium {
    /// Create a new heterogeneous tissue medium
    pub fn new(grid: Grid, default_tissue: TissueType) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        let tissue_map = Array3::from_elem(shape, default_tissue);

        // Initialize arrays with default tissue properties
        let props = TISSUE_PROPERTIES
            .get(&default_tissue)
            .expect("Default tissue properties not found");

        let density = Array3::from_elem(shape, props.density.max(MIN_PHYSICAL_DENSITY));
        let sound_speed = Array3::from_elem(shape, props.sound_speed.max(MIN_PHYSICAL_SOUND_SPEED));
        let absorption = Array3::from_elem(shape, props.alpha_0);
        let nonlinearity = Array3::from_elem(shape, props.nonlinearity);

        let temperature = Array3::from_elem(shape, 310.15); // 37°C default
        let bubble_radius = Array3::zeros(shape);
        let bubble_velocity = Array3::zeros(shape);

        Self {
            tissue_map,
            grid,
            density,
            sound_speed,
            absorption,
            nonlinearity,
            temperature,
            bubble_radius,
            bubble_velocity,
        }
    }

    /// Set tissue type for a region
    pub fn set_tissue_region(&mut self, region: &TissueRegion) -> KwaversResult<()> {
        region.validate()?;

        let mut changed = false;
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    if region.contains(x, y, z) {
                        self.tissue_map[[i, j, k]] = region.tissue_type;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            self.update_properties();
        }
        Ok(())
    }

    /// Update cached property arrays from tissue map
    fn update_properties(&mut self) {
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                self.density[[i, j, k]] = props.density.max(MIN_PHYSICAL_DENSITY);
                self.sound_speed[[i, j, k]] = props.sound_speed.max(MIN_PHYSICAL_SOUND_SPEED);
                self.absorption[[i, j, k]] = props.alpha_0;
                self.nonlinearity[[i, j, k]] = props.nonlinearity;
            }
        }
    }

    /// Get tissue properties at an index
    fn get_tissue_properties(&self, i: usize, j: usize, k: usize) -> &'static TissueProperties {
        let tissue_type = &self.tissue_map[[i, j, k]];
        TISSUE_PROPERTIES
            .get(tissue_type)
            .expect("Tissue type should have properties")
    }
}

impl CoreMedium for HeterogeneousTissueMedium {
    fn density(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]]
    }

    fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        self.sound_speed[[i, j, k]]
    }

    fn absorption(&self, i: usize, j: usize, k: usize) -> f64 {
        self.absorption[[i, j, k]]
    }

    fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.nonlinearity[[i, j, k]]
    }

    fn max_sound_speed(&self) -> f64 {
        self.sound_speed
            .iter()
            .cloned()
            .fold(f64::NAN, f64::max)
            .max(MIN_PHYSICAL_SOUND_SPEED)
    }

    fn is_homogeneous(&self) -> bool {
        let first = &self.tissue_map[[0, 0, 0]];
        self.tissue_map.iter().all(|t| t == first)
    }

    fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        if self.tissue_map.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("[{}, {}, {}]", grid.nx, grid.ny, grid.nz),
                actual: format!("{:?}", self.tissue_map.shape()),
            }
            .into());
        }
        Ok(())
    }
}

impl ArrayAccess for HeterogeneousTissueMedium {
    fn sound_speed_array(&self) -> ArrayView3<'_, f64> {
        self.sound_speed.view()
    }

    fn density_array(&self) -> ArrayView3<'_, f64> {
        self.density.view()
    }

    fn absorption_array(&self) -> ArrayView3<'_, f64> {
        self.absorption.view()
    }

    fn nonlinearity_array(&self) -> ArrayView3<'_, f64> {
        self.nonlinearity.view()
    }

    fn sound_speed_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.sound_speed.view_mut())
    }

    fn density_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.density.view_mut())
    }

    fn absorption_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.absorption.view_mut())
    }

    fn nonlinearity_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.nonlinearity.view_mut())
    }
}

impl AcousticProperties for HeterogeneousTissueMedium {
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // alpha = alpha_0 * (f / 1MHz)^y
        props.alpha_0 * (frequency / 1e6).powf(props.y)
    }

    fn alpha_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).alpha_0
    }

    fn alpha_power(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).y
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).nonlinearity
    }

    fn acoustic_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0 // Default/Placeholder
    }

    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        Some(self.tissue_map[[i, j, k]])
    }
}

impl ElasticProperties for HeterogeneousTissueMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).lame_lambda
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).lame_mu
    }
}

impl ElasticArrayAccess for HeterogeneousTissueMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                arr[[i, j, k]] = props.lame_lambda;
            }
        }
        arr
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                arr[[i, j, k]] = props.lame_mu;
            }
        }
        arr
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        // Mathematical specification: c_s = sqrt(μ / ρ)
        // Compute shear wave speed from tissue properties at each grid point
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                // Get density from tissue properties
                let density = props.density;
                let mu = props.lame_mu;

                // Compute shear wave speed: c_s = sqrt(μ / ρ)
                arr[[i, j, k]] = if density > 0.0 {
                    (mu / density).sqrt()
                } else {
                    0.0
                };
            }
        }
        arr
    }
}

impl ThermalProperties for HeterogeneousTissueMedium {
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).specific_heat
    }

    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.specific_heat(x, y, z, grid)
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).thermal_conductivity
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // alpha = k / (rho * Cp)
        props.thermal_conductivity / (props.density * props.specific_heat)
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).thermal_expansion
    }
}

impl ThermalField for HeterogeneousTissueMedium {
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}

impl OpticalProperties for HeterogeneousTissueMedium {
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).optical_absorption_coeff
    }

    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).optical_scattering_coeff
    }
}

impl ViscousProperties for HeterogeneousTissueMedium {
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).viscosity
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Assume viscosity is shear viscosity for tissues
        self.viscosity(x, y, z, grid)
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Approximation for tissue
        self.viscosity(x, y, z, grid) * 2.5
    }
}

impl BubbleProperties for HeterogeneousTissueMedium {
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        101325.0 // 1 atm
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2330.0 // Water vapor pressure at 20C approx
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4 // Air
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k)
            .gas_diffusion_coefficient
    }
}

impl BubbleState for HeterogeneousTissueMedium {
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}

// Finally implement the marker trait - Removed due to blanket implementation in medium/traits.rs
// impl Medium for HeterogeneousTissueMedium {}
