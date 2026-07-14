//! `HeterogeneousTissueMedium` implementation

use super::{DomainTissueRegion, TissueMap};
use crate::absorption::AbsorptionTissueType;
use crate::{
    absorption::{tissue::AbsorptionTissueProperties, TISSUE_PROPERTIES},
    core::{ArrayAccess, CoreMedium, MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED},
};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::error::{KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array3, ArrayView3, ArrayViewMut3};

mod acoustic;
mod bubble;
mod elastic;
mod optical;
mod thermal;
mod viscous;

/// Heterogeneous tissue medium with spatial tissue type variations.
///
/// ## Implemented
/// - Per-voxel tissue type mapping with spatial region assignment
/// - Frequency-dependent absorption: `absorption_coefficient(f)` returns α(f) in Np/m
///   via the power law α(f) = α₀·(f/1 MHz)^y (Szabo 1994 eq. 2)
/// - `alpha_coefficient()` returns the raw α₀ prefactor [dB/(MHz^y·cm)] for use by
///   the PSTD fractional Laplacian operator (which handles unit conversion internally)
///
/// ## Not yet implemented
/// - Temperature-dependent attenuation: α(T) = α₀·(1 + γ(T−T₀))
/// - Multiple relaxation mechanisms: α(f) = Σ αᵢ/(1 + (f/fᵢ)²) for viscoelastic tissue
/// - Temperature-dependent nonlinearity parameter B/A for high-intensity applications
///
/// ## References
/// - Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500. DOI:10.1121/1.410434
/// - Duck FA (1990). Physical Properties of Tissue. Academic Press.
/// - Bamber JC (1998). In: Ultrasound in Medicine (ed. Duck et al.), pp. 57–88.
#[derive(Debug, Clone)]
pub struct HeterogeneousTissueMedium {
    /// 3D map of tissue types
    pub(super) tissue_map: TissueMap,
    /// Grid structure
    pub(super) grid: Grid,

    // Cached property arrays (required for ArrayAccess)
    pub(super) density: Array3<f64>,
    pub(super) sound_speed: Array3<f64>,
    pub(super) absorption: Array3<f64>,
    pub(super) nonlinearity: Array3<f64>,

    // Thermal state
    pub(super) temperature: Array3<f64>,

    // Bubble state
    pub(super) bubble_radius: Array3<f64>,
    pub(super) bubble_velocity: Array3<f64>,
}

impl HeterogeneousTissueMedium {
    /// Create a new heterogeneous tissue medium
    /// # Panics
    /// - Panics if `Default tissue properties not found`.
    ///
    pub fn new(grid: Grid, default_tissue: AbsorptionTissueType) -> Self {
        let shape = [grid.nx, grid.ny, grid.nz];
        let tissue_map = Array3::from_elem(shape, default_tissue);

        // Initialize arrays with default tissue properties
        let props = TISSUE_PROPERTIES
            .get(&default_tissue)
            .expect("Default tissue properties not found");

        let density = Array3::from_elem(shape, props.density.max(MIN_PHYSICAL_DENSITY));
        let sound_speed = Array3::from_elem(shape, props.sound_speed.max(MIN_PHYSICAL_SOUND_SPEED));
        let absorption = Array3::from_elem(shape, props.alpha_0);
        let nonlinearity = Array3::from_elem(shape, props.nonlinearity);

        let temperature = Array3::from_elem(shape, BODY_TEMPERATURE_K); // 37°C default
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
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn set_tissue_region(&mut self, region: &DomainTissueRegion) -> KwaversResult<()> {
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
    pub(super) fn update_properties(&mut self) {
        for ([i, j, k], tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                self.density[[i, j, k]] = props.density.max(MIN_PHYSICAL_DENSITY);
                self.sound_speed[[i, j, k]] = props.sound_speed.max(MIN_PHYSICAL_SOUND_SPEED);
                self.absorption[[i, j, k]] = props.alpha_0;
                self.nonlinearity[[i, j, k]] = props.nonlinearity;
            }
        }
    }

    /// Get tissue properties at an index
    /// # Panics
    /// - Panics if `Tissue type should have properties`.
    ///
    pub(super) fn get_tissue_properties(
        &self,
        i: usize,
        j: usize,
        k: usize,
    ) -> &'static AbsorptionTissueProperties {
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
            .copied()
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
