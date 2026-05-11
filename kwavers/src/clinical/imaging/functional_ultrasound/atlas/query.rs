//! Coordinate conversions and field/region accessor methods.

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::BrainAtlas;

impl BrainAtlas {
    /// Clone the reference image (template).
    #[must_use]
    pub fn reference_image(&self) -> Array3<f64> {
        self.reference_image.clone()
    }

    /// Borrow the reference image without allocating.
    #[must_use]
    pub fn reference_image_ref(&self) -> &Array3<f64> {
        &self.reference_image
    }

    /// Return the annotation region ID at physical coordinates `coords` (mm).
    ///
    /// # Errors
    /// Returns `Err` when coordinates are non-finite or outside the atlas bounds.
    pub fn get_region(&self, coords: &[f64; 3]) -> KwaversResult<u32> {
        let [voxel_x, voxel_y, voxel_z] = self.mm_to_voxel(coords)?;
        Ok(self.annotation[[voxel_x, voxel_y, voxel_z]])
    }

    /// Return the anatomical name for a region ID.
    #[must_use]
    pub fn get_region_name(&self, region_id: u32) -> &'static str {
        match region_id {
            1  => "Prefrontal Cortex",
            2  => "Motor Cortex",
            3  => "Somatosensory Cortex",
            4  => "Visual Cortex",
            5  => "Auditory Cortex",
            6  => "Hippocampus",
            7  => "Amygdala",
            8  => "Cerebellum",
            9  => "Thalamus",
            10 => "Hypothalamus",
            _  => "Unknown Region",
        }
    }

    /// Return the voxel size in mm.
    #[must_use]
    pub fn voxel_size(&self) -> [f64; 3] {
        self.voxel_size
    }

    /// Return the brain center in mm.
    #[must_use]
    pub fn brain_center(&self) -> [f64; 3] {
        self.brain_center
    }

    /// Convert voxel indices to mm coordinates.
    ///
    /// The AP and ML origins are centred on the atlas; DV is measured
    /// downward from `brain_center[2]`.
    #[must_use]
    pub fn voxel_to_mm(&self, voxel: &[usize; 3]) -> [f64; 3] {
        let ap_origin =
            (0.5 * self.shape.0 as f64).mul_add(-self.voxel_size[0], self.brain_center[0]);
        let ml_origin =
            (0.5 * self.shape.1 as f64).mul_add(-self.voxel_size[1], self.brain_center[1]);
        [
            (voxel[0] as f64).mul_add(self.voxel_size[0], ap_origin),
            (voxel[1] as f64).mul_add(self.voxel_size[1], ml_origin),
            (voxel[2] as f64).mul_add(-self.voxel_size[2], self.brain_center[2]),
        ]
    }

    /// Convert mm coordinates to voxel indices.
    ///
    /// # Errors
    /// Returns `Err` when `mm` contains non-finite values or when the resulting
    /// voxel falls outside the atlas grid.
    pub fn mm_to_voxel(&self, mm: &[f64; 3]) -> KwaversResult<[usize; 3]> {
        if !mm.iter().all(|v| v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Coordinates outside atlas bounds".to_owned(),
            ));
        }

        let ap_origin =
            (0.5 * self.shape.0 as f64).mul_add(-self.voxel_size[0], self.brain_center[0]);
        let ml_origin =
            (0.5 * self.shape.1 as f64).mul_add(-self.voxel_size[1], self.brain_center[1]);
        let voxel = [
            (mm[0] - ap_origin) / self.voxel_size[0],
            (mm[1] - ml_origin) / self.voxel_size[1],
            (self.brain_center[2] - mm[2]) / self.voxel_size[2],
        ];

        if voxel[0] < 0.0
            || voxel[1] < 0.0
            || voxel[2] < 0.0
            || voxel[0] >= self.shape.0 as f64
            || voxel[1] >= self.shape.1 as f64
            || voxel[2] >= self.shape.2 as f64
        {
            return Err(KwaversError::InvalidInput(
                "Coordinates outside atlas bounds".to_owned(),
            ));
        }

        Ok([
            voxel[0].floor() as usize,
            voxel[1].floor() as usize,
            voxel[2].floor() as usize,
        ])
    }
}
