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

#[cfg(test)]
mod tests;

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
