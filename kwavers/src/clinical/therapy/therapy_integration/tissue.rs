//! Tissue Modeling for Clinical Therapy Planning
//!
//! This module provides tissue property mapping for spatially-varying acoustic properties
//! in clinical therapy applications. It follows the composition pattern established in
//! Phase 7.6, composing canonical `AcousticPropertyData` from the domain SSOT.
//!
//! ## Architecture
//!
//! `TissuePropertyMap` represents acoustic properties as 3D arrays for spatially-varying
//! tissue characteristics. It provides bidirectional composition with domain types:
//!
//! - **Domain → Physics**: Use constructors to initialize arrays from canonical properties
//! - **Physics → Domain**: Use `at()` to extract validated properties at specific locations
//!
//! ## Clinical Context
//!
//! This type is used in treatment planning where patient-specific tissue properties are
//! obtained from pre-treatment imaging (CT, MRI, ultrasound) and mapped to computational grids.

use crate::domain::medium::properties::AcousticPropertyData;
use ndarray::Array3;

/// Tissue property map for spatially-varying acoustic properties
///
/// # Architecture
///
/// `TissuePropertyMap` represents acoustic properties as 3D arrays for spatially-varying
/// tissue characteristics in clinical therapy planning. It composes canonical
/// `AcousticPropertyData` from the domain SSOT, following the composition pattern
/// established in Phase 7.6.
///
/// ## Composition Pattern
///
/// - **Domain → Physics**: Use `uniform()` or tissue-specific constructors (`water()`, `liver()`, etc.)
///   to initialize arrays from canonical point properties
/// - **Physics → Domain**: Use `at(index)` to extract validated `AcousticPropertyData` at specific locations
/// - **Validation**: Shape consistency is enforced; derived quantities (impedance, wavelength) computed on demand
///
/// ## Clinical Context
///
/// This type is used in treatment planning where patient-specific tissue properties are obtained
/// from pre-treatment imaging (CT, MRI, ultrasound) and mapped to computational grids.
///
/// # Example
///
/// ```
/// use kwavers::clinical::therapy::therapy_integration::TissuePropertyMap;
/// use kwavers::domain::medium::properties::AcousticPropertyData;
///
/// // Create uniform liver tissue properties
/// let liver_props = AcousticPropertyData::liver();
/// let tissue_map = TissuePropertyMap::uniform((64, 64, 64), liver_props);
///
/// // Extract properties at a specific voxel
/// let props_at_tumor = tissue_map.at((32, 32, 32)).expect("valid index");
/// assert_eq!(props_at_tumor.density, liver_props.density);
/// ```
#[derive(Debug, Clone)]
pub struct TissuePropertyMap {
    /// Speed of sound (m/s)
    pub speed_of_sound: Array3<f64>,
    /// Density (kg/m³)
    pub density: Array3<f64>,
    /// Attenuation (Np/m)
    pub attenuation: Array3<f64>,
    /// Nonlinearity parameter B/A
    pub nonlinearity: Array3<f64>,
}

impl TissuePropertyMap {
    /// Create uniform tissue property map from canonical domain properties
    ///
    /// # Arguments
    ///
    /// - `shape`: 3D grid dimensions (nx, ny, nz)
    /// - `props`: Canonical acoustic properties from domain SSOT
    ///
    /// # Returns
    ///
    /// Uniform tissue property map with all voxels having the same properties.
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::clinical::therapy::therapy_integration::TissuePropertyMap;
    /// use kwavers::domain::medium::properties::AcousticPropertyData;
    ///
    /// let water = AcousticPropertyData::water();
    /// let tissue = TissuePropertyMap::uniform((32, 32, 32), water);
    /// assert_eq!(tissue.shape(), (32, 32, 32));
    /// ```
    pub fn uniform(shape: (usize, usize, usize), props: AcousticPropertyData) -> Self {
        let (nx, ny, nz) = shape;
        Self {
            speed_of_sound: Array3::from_elem((nx, ny, nz), props.sound_speed),
            density: Array3::from_elem((nx, ny, nz), props.density),
            attenuation: Array3::from_elem((nx, ny, nz), props.absorption_coefficient),
            nonlinearity: Array3::from_elem((nx, ny, nz), props.nonlinearity),
        }
    }

    /// Create water tissue property map
    ///
    /// Convenience constructor for homogeneous water medium, commonly used
    /// in acoustic coupling and reference measurements.
    pub fn water(shape: (usize, usize, usize)) -> Self {
        let props = AcousticPropertyData::water();
        Self::uniform(shape, props)
    }

    /// Create liver tissue property map
    ///
    /// Typical acoustic properties for liver tissue, used in treatment planning
    /// for liver tumor ablation and imaging protocols.
    pub fn liver(shape: (usize, usize, usize)) -> Self {
        let props = AcousticPropertyData::liver();
        Self::uniform(shape, props)
    }

    /// Create brain tissue property map
    ///
    /// Typical acoustic properties for brain tissue, used in transcranial
    /// ultrasound therapy planning and neuromodulation protocols.
    pub fn brain(shape: (usize, usize, usize)) -> Self {
        let props = AcousticPropertyData::brain();
        Self::uniform(shape, props)
    }

    /// Create kidney tissue property map
    ///
    /// Typical acoustic properties for kidney tissue, used in lithotripsy
    /// and renal tumor treatment planning.
    pub fn kidney(shape: (usize, usize, usize)) -> Self {
        let props = AcousticPropertyData::kidney();
        Self::uniform(shape, props)
    }

    /// Create muscle tissue property map
    ///
    /// Typical acoustic properties for skeletal muscle, used in musculoskeletal
    /// therapy planning and sports medicine applications.
    pub fn muscle(shape: (usize, usize, usize)) -> Self {
        let props = AcousticPropertyData::muscle();
        Self::uniform(shape, props)
    }

    /// Extract canonical acoustic properties at a specific voxel
    ///
    /// # Arguments
    ///
    /// - `index`: 3D voxel coordinates (i, j, k)
    ///
    /// # Returns
    ///
    /// `AcousticPropertyData` at the specified location, with full validation.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Index out of bounds
    /// - Extracted properties violate physical constraints
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::clinical::therapy::therapy_integration::TissuePropertyMap;
    ///
    /// let tissue = TissuePropertyMap::liver((16, 16, 16));
    /// let props = tissue.at((8, 8, 8)).expect("valid index");
    /// assert!(props.impedance() > 0.0);
    /// ```
    pub fn at(&self, index: (usize, usize, usize)) -> Result<AcousticPropertyData, String> {
        let (i, j, k) = index;
        let shape = self.speed_of_sound.dim();

        // Bounds checking
        if i >= shape.0 || j >= shape.1 || k >= shape.2 {
            return Err(format!(
                "Index ({}, {}, {}) out of bounds for shape {:?}",
                i, j, k, shape
            ));
        }

        // Extract properties and construct canonical type with validation
        AcousticPropertyData::new(
            self.density[[i, j, k]],
            self.speed_of_sound[[i, j, k]],
            self.attenuation[[i, j, k]],
            1.0, // Default absorption power for clinical tissues
            self.nonlinearity[[i, j, k]],
        )
    }

    /// Get the shape of the tissue property map
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        self.speed_of_sound.dim()
    }

    /// Get the number of dimensions (always 3 for tissue maps)
    #[inline]
    pub fn ndim(&self) -> usize {
        3
    }

    /// Validate that all property arrays have consistent shapes
    ///
    /// # Returns
    ///
    /// `Ok(())` if all arrays have the same shape, error otherwise.
    pub fn validate_shape_consistency(&self) -> Result<(), String> {
        let shape = self.speed_of_sound.dim();

        if self.density.dim() != shape {
            return Err(format!(
                "Density shape {:?} does not match sound speed shape {:?}",
                self.density.dim(),
                shape
            ));
        }

        if self.attenuation.dim() != shape {
            return Err(format!(
                "Attenuation shape {:?} does not match sound speed shape {:?}",
                self.attenuation.dim(),
                shape
            ));
        }

        if self.nonlinearity.dim() != shape {
            return Err(format!(
                "Nonlinearity shape {:?} does not match sound speed shape {:?}",
                self.nonlinearity.dim(),
                shape
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TissuePropertyMap Composition Tests
    // ========================================================================

    #[test]
    fn test_tissue_property_map_uniform_composition() {
        // Create canonical liver properties
        let liver = AcousticPropertyData::liver();
        let shape = (8, 8, 8);

        // Compose tissue map from canonical type
        let tissue_map = TissuePropertyMap::uniform(shape, liver);

        // Verify shape
        assert_eq!(tissue_map.shape(), shape);
        assert_eq!(tissue_map.ndim(), 3);

        // Verify all elements match source properties
        assert_eq!(tissue_map.speed_of_sound[[0, 0, 0]], liver.sound_speed);
        assert_eq!(tissue_map.density[[4, 4, 4]], liver.density);
        assert_eq!(
            tissue_map.attenuation[[7, 7, 7]],
            liver.absorption_coefficient
        );
        assert_eq!(tissue_map.nonlinearity[[3, 2, 1]], liver.nonlinearity);
    }

    #[test]
    fn test_tissue_property_map_extraction() {
        // Create canonical water properties
        let water = AcousticPropertyData::water();
        let shape = (16, 16, 16);
        let tissue_map = TissuePropertyMap::uniform(shape, water);

        // Extract properties at various locations
        let props_center = tissue_map.at((8, 8, 8)).expect("valid index");
        let props_corner = tissue_map.at((0, 0, 0)).expect("valid index");
        let props_edge = tissue_map.at((15, 15, 15)).expect("valid index");

        // Verify extracted properties match source
        assert_eq!(props_center.density, water.density);
        assert_eq!(props_corner.sound_speed, water.sound_speed);
        assert_eq!(props_edge.nonlinearity, water.nonlinearity);

        // Verify derived quantities are available
        assert!(props_center.impedance() > 0.0);
        // Wavelength = c / f
        let wavelength_1mhz = props_center.sound_speed / 1e6;
        assert!(wavelength_1mhz > 0.0);
    }

    #[test]
    fn test_tissue_property_map_bounds_checking() {
        let tissue_map = TissuePropertyMap::water((10, 10, 10));

        // Valid indices should succeed
        assert!(tissue_map.at((0, 0, 0)).is_ok());
        assert!(tissue_map.at((9, 9, 9)).is_ok());
        assert!(tissue_map.at((5, 5, 5)).is_ok());

        // Out-of-bounds indices should fail
        assert!(tissue_map.at((10, 5, 5)).is_err());
        assert!(tissue_map.at((5, 10, 5)).is_err());
        assert!(tissue_map.at((5, 5, 10)).is_err());
        assert!(tissue_map.at((10, 10, 10)).is_err());
    }

    #[test]
    fn test_tissue_property_map_convenience_constructors() {
        let shape = (12, 12, 12);

        // Test all tissue-specific constructors
        let water_map = TissuePropertyMap::water(shape);
        let liver_map = TissuePropertyMap::liver(shape);
        let brain_map = TissuePropertyMap::brain(shape);
        let kidney_map = TissuePropertyMap::kidney(shape);
        let muscle_map = TissuePropertyMap::muscle(shape);

        // Verify shapes
        assert_eq!(water_map.shape(), shape);
        assert_eq!(liver_map.shape(), shape);
        assert_eq!(brain_map.shape(), shape);
        assert_eq!(kidney_map.shape(), shape);
        assert_eq!(muscle_map.shape(), shape);

        // Verify properties are distinct for different tissues
        let water_props = water_map.at((0, 0, 0)).unwrap();
        let liver_props = liver_map.at((0, 0, 0)).unwrap();
        let brain_props = brain_map.at((0, 0, 0)).unwrap();

        // Different tissues should have different properties
        assert_ne!(water_props.density, liver_props.density);
        assert_ne!(liver_props.sound_speed, brain_props.sound_speed);
    }

    #[test]
    fn test_tissue_property_map_shape_consistency() {
        let shape = (8, 8, 8);
        let tissue_map = TissuePropertyMap::liver(shape);

        // Validation should pass for consistent shapes
        assert!(tissue_map.validate_shape_consistency().is_ok());
    }

    #[test]
    fn test_tissue_property_map_round_trip() {
        // Create canonical properties
        let kidney = AcousticPropertyData::kidney();
        let shape = (10, 10, 10);

        // Domain → Physics: construct map
        let tissue_map = TissuePropertyMap::uniform(shape, kidney);

        // Physics → Domain: extract at multiple locations
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let extracted = tissue_map.at((i, j, k)).expect("valid index");

                    // Verify round-trip preserves properties
                    assert_eq!(extracted.density, kidney.density);
                    assert_eq!(extracted.sound_speed, kidney.sound_speed);
                    assert_eq!(extracted.nonlinearity, kidney.nonlinearity);

                    // Verify derived quantities are consistent
                    assert!((extracted.impedance() - kidney.impedance()).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_tissue_property_map_heterogeneous_simulation() {
        // Simulate a heterogeneous tissue structure: water background with liver inclusion
        let shape = (16, 16, 16);
        let water = AcousticPropertyData::water();
        let liver = AcousticPropertyData::liver();

        // Start with water background
        let mut tissue_map = TissuePropertyMap::uniform(shape, water);

        // Insert liver inclusion in center region (8x8x8 cube)
        for i in 4..12 {
            for j in 4..12 {
                for k in 4..12 {
                    tissue_map.speed_of_sound[[i, j, k]] = liver.sound_speed;
                    tissue_map.density[[i, j, k]] = liver.density;
                    tissue_map.attenuation[[i, j, k]] = liver.absorption_coefficient;
                    tissue_map.nonlinearity[[i, j, k]] = liver.nonlinearity;
                }
            }
        }

        // Verify background (water) properties
        let bg_props = tissue_map.at((0, 0, 0)).expect("valid");
        assert_eq!(bg_props.density, water.density);

        // Verify inclusion (liver) properties
        let inclusion_props = tissue_map.at((8, 8, 8)).expect("valid");
        assert_eq!(inclusion_props.density, liver.density);
        assert_eq!(inclusion_props.sound_speed, liver.sound_speed);

        // Verify shape consistency
        assert!(tissue_map.validate_shape_consistency().is_ok());
    }

    #[test]
    fn test_tissue_property_map_clinical_workflow() {
        // Simulate clinical workflow: patient imaging → treatment planning
        let grid_shape = (32, 32, 32);

        // Step 1: Initialize with background tissue (muscle)
        let muscle = AcousticPropertyData::muscle();
        let patient_tissue = TissuePropertyMap::uniform(grid_shape, muscle);

        // Step 2: Extract properties at treatment target
        let target_location = (16, 16, 16);
        let target_props = patient_tissue.at(target_location).expect("valid");

        // Step 3: Calculate treatment parameters using canonical properties
        let acoustic_impedance = target_props.impedance();
        // Wavelength = c / f
        let wavelength_at_1mhz = target_props.sound_speed / 1e6;

        // Verify clinical parameters are physically reasonable
        assert!(acoustic_impedance > 1e6); // Typical tissue impedance > 1 MRayl
        assert!(wavelength_at_1mhz > 1e-3); // Wavelength > 1 mm at 1 MHz
        assert!(wavelength_at_1mhz < 2e-3); // Wavelength < 2 mm for typical tissue
    }
}
