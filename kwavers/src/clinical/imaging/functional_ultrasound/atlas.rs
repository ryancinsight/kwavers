//! Brain Atlas Integration
//!
//! Provides stereotactic coordinate systems and anatomical reference data
//! for functional ultrasound imaging.
//!
//! References:
//! - Allen Brain Atlas: http://mouse.brain-map.org
//! - Paxinos & Watson (2013). "The Rat Brain in Stereotaxic Coordinates"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Brain atlas reference data
#[derive(Debug, Clone)]
pub struct BrainAtlas {
    /// Reference image (template)
    reference_image: Array3<f64>,

    /// Brain region annotations
    annotation: Array3<u32>,

    /// Voxel size [mm]
    voxel_size: [f64; 3],

    /// Brain center coordinates [mm]
    brain_center: [f64; 3],

    /// Atlas resolution
    shape: (usize, usize, usize),
}

impl BrainAtlas {
    /// Create brain atlas from reference image
    pub fn new(
        reference_image: Array3<f64>,
        voxel_size: [f64; 3],
        brain_center: [f64; 3],
    ) -> KwaversResult<Self> {
        let shape = reference_image.dim();

        if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
            return Err(KwaversError::InvalidInput(
                "Invalid atlas dimensions".to_string(),
            ));
        }

        let annotation = Array3::zeros(shape);

        Ok(Self {
            reference_image,
            annotation,
            voxel_size,
            brain_center,
            shape,
        })
    }

    /// Load default mouse brain atlas (Allen Brain Atlas)
    ///
    /// **Test-only placeholder**: Returns a uniform-intensity reference volume
    /// (80×100×80 at 100 μm resolution) with no anatomical annotation.
    /// Production use requires loading the Allen Common Coordinate Framework
    /// (CCFv3) from NIfTI or HDF5 files, which are ~5 GB at full 10 μm resolution.
    pub fn load_default() -> KwaversResult<Self> {
        // Standard mouse brain: ~8mm × 10mm × 8mm at reduced 100μm resolution for testing
        // Full resolution (10μm) would be 800x1000x800 = 5.12GB
        let reference_image = Array3::ones((80, 100, 80)); // Uniform — no real anatomy
        let voxel_size = [0.1, 0.1, 0.1]; // 100 μm voxels (reduced for testing)
        let brain_center = [4.0, 5.0, 4.0]; // Center in mm

        Self::new(reference_image, voxel_size, brain_center)
    }

    /// Get reference image
    pub fn reference_image(&self) -> Array3<f64> {
        self.reference_image.clone()
    }

    /// Get brain region at coordinates
    pub fn get_region(&self, coords: &[f64; 3]) -> KwaversResult<u32> {
        // Convert mm coordinates to voxel indices
        let voxel_x = ((coords[0] - self.brain_center[0]) / self.voxel_size[0]) as usize;
        let voxel_y = ((coords[1] - self.brain_center[1]) / self.voxel_size[1]) as usize;
        let voxel_z = ((coords[2] - self.brain_center[2]) / self.voxel_size[2]) as usize;

        if voxel_x >= self.shape.0 || voxel_y >= self.shape.1 || voxel_z >= self.shape.2 {
            return Err(KwaversError::InvalidInput(
                "Coordinates outside brain atlas".to_string(),
            ));
        }

        Ok(self.annotation[[voxel_x, voxel_y, voxel_z]])
    }

    /// Get anatomical region name from ID
    pub fn get_region_name(&self, region_id: u32) -> &'static str {
        match region_id {
            1 => "Prefrontal Cortex",
            2 => "Motor Cortex",
            3 => "Somatosensory Cortex",
            4 => "Visual Cortex",
            5 => "Auditory Cortex",
            6 => "Hippocampus",
            7 => "Amygdala",
            8 => "Cerebellum",
            9 => "Thalamus",
            10 => "Hypothalamus",
            _ => "Unknown Region",
        }
    }

    /// Get voxel size
    pub fn voxel_size(&self) -> [f64; 3] {
        self.voxel_size
    }

    /// Get brain center in mm
    pub fn brain_center(&self) -> [f64; 3] {
        self.brain_center
    }

    /// Convert atlas voxel to mm coordinates
    pub fn voxel_to_mm(&self, voxel: &[usize; 3]) -> [f64; 3] {
        [
            self.brain_center[0] + voxel[0] as f64 * self.voxel_size[0],
            self.brain_center[1] + voxel[1] as f64 * self.voxel_size[1],
            self.brain_center[2] + voxel[2] as f64 * self.voxel_size[2],
        ]
    }

    /// Convert mm coordinates to atlas voxel
    pub fn mm_to_voxel(&self, mm: &[f64; 3]) -> KwaversResult<[usize; 3]> {
        let voxel_x = ((mm[0] - self.brain_center[0]) / self.voxel_size[0]) as usize;
        let voxel_y = ((mm[1] - self.brain_center[1]) / self.voxel_size[1]) as usize;
        let voxel_z = ((mm[2] - self.brain_center[2]) / self.voxel_size[2]) as usize;

        if voxel_x >= self.shape.0 || voxel_y >= self.shape.1 || voxel_z >= self.shape.2 {
            return Err(KwaversError::InvalidInput(
                "Coordinates outside atlas bounds".to_string(),
            ));
        }

        Ok([voxel_x, voxel_y, voxel_z])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_atlas_creation() {
        let image = Array3::ones((100, 100, 100));
        let result = BrainAtlas::new(image, [0.01, 0.01, 0.01], [5.0, 5.0, 5.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_brain_atlas_default() {
        let result = BrainAtlas::load_default();
        assert!(result.is_ok());

        let atlas = result.unwrap();
        assert_eq!(atlas.voxel_size(), [0.1, 0.1, 0.1]); // 100μm resolution for testing
    }

    #[test]
    fn test_voxel_to_mm_conversion() {
        let atlas = BrainAtlas::load_default().unwrap();
        let mm = atlas.voxel_to_mm(&[100, 200, 150]);

        assert!(mm[0].is_finite());
        assert!(mm[1].is_finite());
        assert!(mm[2].is_finite());
    }

    #[test]
    fn test_mm_to_voxel_conversion() {
        let atlas = BrainAtlas::load_default().unwrap();
        let mm = [5.0, 5.0, 5.0];
        let result = atlas.mm_to_voxel(&mm);
        assert!(result.is_ok());
    }

    #[test]
    fn test_region_name_lookup() {
        let atlas = BrainAtlas::load_default().unwrap();
        assert_eq!(atlas.get_region_name(1), "Prefrontal Cortex");
        assert_eq!(atlas.get_region_name(6), "Hippocampus");
        assert_eq!(atlas.get_region_name(999), "Unknown Region");
    }
}
