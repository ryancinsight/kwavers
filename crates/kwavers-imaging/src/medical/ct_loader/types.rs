//! CT image metadata type.

/// Metadata extracted from CT NIFTI header.
#[derive(Debug, Clone)]
pub struct CTMetadata {
    /// Image dimensions (nx, ny, nz)
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in mm (dx, dy, dz)
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Voxel spacing in meters (dx, dy, dz)
    pub voxel_spacing_m: (f64, f64, f64),
    /// Affine transformation matrix (4×4)
    pub affine: [[f64; 4]; 4],
    /// Data type description
    pub data_type: String,
    /// Min/Max HU values in volume
    pub hu_range: (f64, f64),
}
