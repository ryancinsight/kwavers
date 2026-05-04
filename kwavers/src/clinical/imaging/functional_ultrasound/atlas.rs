//! Brain Atlas Integration
//!
//! Provides stereotactic coordinate systems and anatomical reference data
//! for functional ultrasound imaging.
//!
//! References:
//! - Allen Brain Atlas: http://mouse.brain-map.org
//! - Paxinos & Watson (2013). "The Rat Brain in Stereotaxic Coordinates"
//! - Franklin & Paxinos (2008). *The Mouse Brain in Stereotaxic Coordinates*

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

const DEFAULT_SHAPE: (usize, usize, usize) = (80, 100, 80);
const DEFAULT_VOXEL_SIZE_MM: [f64; 3] = [0.1, 0.1, 0.1];
const DEFAULT_BREGMA_MM: [f64; 3] = [4.0, 5.0, 4.0];

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
        Self::with_annotation(
            reference_image,
            Array3::zeros((0, 0, 0)),
            voxel_size,
            brain_center,
        )
    }

    /// Create brain atlas from a reference image and annotation volume.
    ///
    /// # Mathematical specification
    ///
    /// The image and annotation arrays define two scalar fields on the same
    /// Cartesian lattice. The atlas is valid iff both fields have identical
    /// nonzero extents and the affine voxel spacing is finite and strictly
    /// positive in every axis. Region lookup is then a total function over
    /// in-bounds physical coordinates and a structured error outside the domain.
    pub fn with_annotation(
        reference_image: Array3<f64>,
        annotation: Array3<u32>,
        voxel_size: [f64; 3],
        brain_center: [f64; 3],
    ) -> KwaversResult<Self> {
        let shape = reference_image.dim();

        if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
            return Err(KwaversError::InvalidInput(
                "Invalid atlas dimensions".to_string(),
            ));
        }

        if !voxel_size
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "Atlas voxel size must be finite and positive".to_string(),
            ));
        }

        if !brain_center.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Atlas brain center must be finite".to_string(),
            ));
        }

        if !reference_image.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Atlas reference image must contain only finite values".to_string(),
            ));
        }

        let annotation = if annotation.is_empty() {
            Array3::zeros(shape)
        } else if annotation.dim() == shape {
            annotation
        } else {
            return Err(KwaversError::InvalidInput(
                "Atlas annotation shape must match reference image shape".to_string(),
            ));
        };

        Ok(Self {
            reference_image,
            annotation,
            voxel_size,
            brain_center,
            shape,
        })
    }

    /// Build the embedded analytical mouse stereotactic reference.
    ///
    /// # Theorem
    ///
    /// Let the brain support be the ellipsoid
    /// `(ap/a)^2 + (ml/b)^2 + ((dv-c)/d)^2 <= 1`. The generated reference
    /// intensity is zero outside this compact support and strictly positive
    /// inside it, while each annotation is assigned by a deterministic
    /// stereotactic partition. Therefore the default atlas is input-sensitive
    /// for registration and region lookup; replacing it with a constant field
    /// changes both image variance and region IDs.
    ///
    /// # Algorithm
    ///
    /// The embedded model uses a 100 um lattice over the standard mouse
    /// stereotactic envelope. It is an analytical reference phantom for tests,
    /// demos, and registration smoke checks. It is not Allen CCF voxel data.
    /// Production Allen CCF use must load external template and annotation
    /// volumes with `with_annotation`.
    pub fn load_default() -> KwaversResult<Self> {
        let (nx, ny, nz) = DEFAULT_SHAPE;
        let mut reference_image = Array3::zeros(DEFAULT_SHAPE);
        let mut annotation = Array3::zeros(DEFAULT_SHAPE);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let [ap, ml, dv] =
                        Self::voxel_to_stereotactic(DEFAULT_SHAPE, DEFAULT_VOXEL_SIZE_MM, i, j, k);

                    let ellipsoid =
                        (ap / 4.0).powi(2) + (ml / 5.0).powi(2) + ((dv - 4.0) / 4.0).powi(2);
                    if ellipsoid > 1.0 {
                        continue;
                    }

                    let cortex_shell = (dv <= 2.0) as u32;
                    let region = if dv > 6.2 && ap < -1.5 {
                        8
                    } else if dv > 3.2 && ap.abs() < 1.5 && ml.abs() < 2.0 {
                        9
                    } else if dv > 4.8 && ap > -1.0 && ml.abs() < 1.2 {
                        10
                    } else if ap < -1.0 && dv > 2.0 && dv <= 5.5 && ml.abs() > 1.0 {
                        6
                    } else if ap > 0.8 && dv > 2.5 && ml.abs() > 2.0 {
                        7
                    } else if cortex_shell == 1 && ap > 1.0 {
                        1
                    } else if cortex_shell == 1 && ap >= -0.5 {
                        2
                    } else if cortex_shell == 1 && ap >= -2.2 {
                        3
                    } else if cortex_shell == 1 && ap < -2.2 && ml.abs() <= 2.5 {
                        4
                    } else {
                        5
                    };

                    annotation[[i, j, k]] = region;
                    reference_image[[i, j, k]] = Self::region_intensity(region, ap, ml, dv);
                }
            }
        }

        Self::with_annotation(
            reference_image,
            annotation,
            DEFAULT_VOXEL_SIZE_MM,
            DEFAULT_BREGMA_MM,
        )
    }

    /// Get reference image
    pub fn reference_image(&self) -> Array3<f64> {
        self.reference_image.clone()
    }

    /// Borrow the reference image without allocating.
    pub fn reference_image_ref(&self) -> &Array3<f64> {
        &self.reference_image
    }

    /// Get brain region at coordinates
    pub fn get_region(&self, coords: &[f64; 3]) -> KwaversResult<u32> {
        let [voxel_x, voxel_y, voxel_z] = self.mm_to_voxel(coords)?;
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
        let ap_origin = self.brain_center[0] - 0.5 * self.shape.0 as f64 * self.voxel_size[0];
        let ml_origin = self.brain_center[1] - 0.5 * self.shape.1 as f64 * self.voxel_size[1];
        [
            ap_origin + voxel[0] as f64 * self.voxel_size[0],
            ml_origin + voxel[1] as f64 * self.voxel_size[1],
            self.brain_center[2] - voxel[2] as f64 * self.voxel_size[2],
        ]
    }

    /// Convert mm coordinates to atlas voxel
    pub fn mm_to_voxel(&self, mm: &[f64; 3]) -> KwaversResult<[usize; 3]> {
        if !mm.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Coordinates outside atlas bounds".to_string(),
            ));
        }

        let ap_origin = self.brain_center[0] - 0.5 * self.shape.0 as f64 * self.voxel_size[0];
        let ml_origin = self.brain_center[1] - 0.5 * self.shape.1 as f64 * self.voxel_size[1];
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
                "Coordinates outside atlas bounds".to_string(),
            ));
        }

        Ok([
            voxel[0].floor() as usize,
            voxel[1].floor() as usize,
            voxel[2].floor() as usize,
        ])
    }

    fn voxel_to_stereotactic(
        shape: (usize, usize, usize),
        voxel_size: [f64; 3],
        i: usize,
        j: usize,
        k: usize,
    ) -> [f64; 3] {
        [
            (i as f64 - 0.5 * shape.0 as f64) * voxel_size[0],
            (j as f64 - 0.5 * shape.1 as f64) * voxel_size[1],
            k as f64 * voxel_size[2],
        ]
    }

    fn region_intensity(region: u32, ap: f64, ml: f64, dv: f64) -> f64 {
        let base = 0.15 + 0.07 * region as f64;
        let vascular_prior = (-(ml * ml) / 2.0).exp() * (1.0 - (ap / 4.0).abs()).max(0.0);
        let depth_attenuation = (-0.08 * dv).exp();
        base * depth_attenuation + 0.2 * vascular_prior
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_atlas_creation() {
        let image = Array3::ones((100, 100, 100));
        let result = BrainAtlas::new(image, [0.01, 0.01, 0.01], [5.0, 5.0, 5.0]);
        let atlas = result.unwrap();
        assert_eq!(atlas.voxel_size(), [0.01, 0.01, 0.01]);
        assert_eq!(atlas.brain_center(), [5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_brain_atlas_default() {
        let atlas = BrainAtlas::load_default().unwrap();
        assert_eq!(atlas.voxel_size(), [0.1, 0.1, 0.1]); // 100μm resolution for testing
        assert_eq!(atlas.reference_image_ref().dim(), (80, 100, 80));
        assert!(atlas
            .reference_image_ref()
            .iter()
            .any(|value| *value == 0.0));
        assert!(atlas.reference_image_ref().iter().any(|value| *value > 0.0));
        assert_ne!(atlas.get_region(&[4.0, 5.0, 3.8]).unwrap(), 0);
    }

    #[test]
    fn test_voxel_to_mm_conversion() {
        let atlas = BrainAtlas::load_default().unwrap();
        let mm = atlas.voxel_to_mm(&[40, 50, 20]);

        assert_eq!(mm, [4.0, 5.0, 2.0]);
    }

    #[test]
    fn test_mm_to_voxel_conversion() {
        let atlas = BrainAtlas::load_default().unwrap();
        let mm = [5.0, 5.0, 5.0];
        let result = atlas.mm_to_voxel(&mm);
        assert!(result.is_err());

        let voxel = atlas.mm_to_voxel(&[4.0, 5.0, 2.0]).unwrap();
        assert_eq!(voxel, [40, 50, 20]);
    }

    #[test]
    fn test_invalid_annotation_shape_is_rejected() {
        let image = Array3::zeros((4, 4, 4));
        let annotation = Array3::zeros((4, 4, 3));
        let result = BrainAtlas::with_annotation(image, annotation, [0.1, 0.1, 0.1], [0.0; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_region_name_lookup() {
        let atlas = BrainAtlas::load_default().unwrap();
        assert_eq!(atlas.get_region_name(1), "Prefrontal Cortex");
        assert_eq!(atlas.get_region_name(6), "Hippocampus");
        assert_eq!(atlas.get_region_name(999), "Unknown Region");
    }
}
