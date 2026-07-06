//! `BrainAtlas` constructors and analytical default phantom.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::BrainAtlas;

const DEFAULT_SHAPE: (usize, usize, usize) = (80, 100, 80);
const DEFAULT_VOXEL_SIZE_MM: [f64; 3] = [0.1, 0.1, 0.1];
const DEFAULT_BREGMA_MM: [f64; 3] = [4.0, 5.0, 4.0];

impl BrainAtlas {
    /// Create a brain atlas from a reference image.
    ///
    /// # Errors
    /// Propagates validation errors from `with_annotation`.
    pub fn new(
        reference_image: LetoArray3<f64>,
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

    /// Create a brain atlas from a reference image and annotation volume.
    ///
    /// # Mathematical specification
    ///
    /// The image and annotation arrays define two scalar fields on the same
    /// Cartesian lattice. The atlas is valid iff both fields have identical
    /// nonzero extents and the affine voxel spacing is finite and strictly
    /// positive in every axis. Region lookup is then a total function over
    /// in-bounds physical coordinates and a structured error outside the domain.
    ///
    /// # Errors
    /// Returns `Err` when dimensions are zero, when the annotation shape does
    /// not match the reference shape, when voxel size is non-positive, or when
    /// any reference image value is non-finite.
    pub fn with_annotation(
        reference_image: LetoArray3<f64>,
        annotation: Array3<u32>,
        voxel_size: [f64; 3],
        brain_center: [f64; 3],
    ) -> KwaversResult<Self> {
        let reference_shape = reference_image.shape();
        let shape = (reference_shape[0], reference_shape[1], reference_shape[2]);

        if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
            return Err(KwaversError::InvalidInput(
                "Invalid atlas dimensions".to_owned(),
            ));
        }
        if !voxel_size.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(KwaversError::InvalidInput(
                "Atlas voxel size must be finite and positive".to_owned(),
            ));
        }
        if !brain_center.iter().all(|v| v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Atlas brain center must be finite".to_owned(),
            ));
        }
        if !reference_image.iter().all(|v| v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Atlas reference image must contain only finite values".to_owned(),
            ));
        }

        let annotation = if annotation.is_empty() {
            Array3::zeros(shape)
        } else if annotation.dim() == shape {
            annotation
        } else {
            return Err(KwaversError::InvalidInput(
                "Atlas annotation shape must match reference image shape".to_owned(),
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
    /// `(ap/a)^2 + (ml/b)^2 + ((dv−c)/d)^2 ≤ 1`. The generated reference
    /// intensity is zero outside this compact support and strictly positive
    /// inside it, while each annotation is assigned by a deterministic
    /// stereotactic partition. Therefore the default atlas is input-sensitive
    /// for registration and region lookup; replacing it with a constant field
    /// changes both image variance and region IDs.
    ///
    /// # Notes
    ///
    /// The embedded model uses a 100 μm lattice over the standard mouse
    /// stereotactic envelope. It is an analytical reference phantom for tests,
    /// demos, and registration smoke checks — not Allen CCF voxel data.
    /// Production Allen CCF use must load external volumes with `with_annotation`.
    ///
    /// # Errors
    /// Propagates validation errors from `with_annotation`.
    pub fn load_default() -> KwaversResult<Self> {
        let (nx, ny, nz) = DEFAULT_SHAPE;
        let mut reference_image =
            LetoArray3::zeros([DEFAULT_SHAPE.0, DEFAULT_SHAPE.1, DEFAULT_SHAPE.2]);
        let mut annotation = Array3::zeros(DEFAULT_SHAPE);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let [ap, ml, dv] =
                        Self::voxel_to_stereotactic(DEFAULT_SHAPE, DEFAULT_VOXEL_SIZE_MM, i, j, k);

                    let ellipsoid = ((dv - 4.0) / 4.0).mul_add(
                        (dv - 4.0) / 4.0,
                        (ml / 5.0).mul_add(ml / 5.0, (ap / 4.0).powi(2)),
                    );
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

    /// Convert voxel indices to stereotactic (AP, ML, DV) coordinates in mm.
    pub(super) fn voxel_to_stereotactic(
        shape: (usize, usize, usize),
        voxel_size: [f64; 3],
        i: usize,
        j: usize,
        k: usize,
    ) -> [f64; 3] {
        [
            0.5f64.mul_add(-(shape.0 as f64), i as f64) * voxel_size[0],
            0.5f64.mul_add(-(shape.1 as f64), j as f64) * voxel_size[1],
            k as f64 * voxel_size[2],
        ]
    }

    /// Compute synthetic MRI-like intensity for a brain region.
    ///
    /// Intensity = base_level · depth_attenuation + 0.2 · vascular_prior
    #[must_use]
    pub(super) fn region_intensity(region: u32, ap: f64, ml: f64, dv: f64) -> f64 {
        let base = 0.07f64.mul_add(region as f64, 0.15);
        let vascular_prior = (-(ml * ml) / 2.0).exp() * (1.0 - (ap / 4.0).abs()).max(0.0);
        let depth_attenuation = (-0.08 * dv).exp();
        base * depth_attenuation + 0.2 * vascular_prior
    }
}
