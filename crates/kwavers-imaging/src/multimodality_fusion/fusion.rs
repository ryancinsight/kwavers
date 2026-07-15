use super::{FusionParameters, ImageData, MultimodalityFusionMethod, RegistrationTransform};
use kwavers_core::error::KwaversResult;
use leto::Array3;

/// Image fusion engine
#[derive(Debug)]
pub struct FusionEngine {
    /// Fusion parameters
    params: FusionParameters,
}

impl FusionEngine {
    /// Create new fusion engine
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(params: FusionParameters) -> Self {
        Self { params }
    }

    /// Perform image fusion
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn fuse(
        &self,
        reference: &ImageData,
        floating: &ImageData,
        transform: &RegistrationTransform,
    ) -> KwaversResult<Array3<f64>> {
        // Apply transformation to floating image
        let transformed = self.apply_transform(floating, transform)?;

        // Perform fusion based on method
        let fused = match self.params.method {
            MultimodalityFusionMethod::Overlay => {
                self.fusion_overlay(&reference.data, &transformed)
            }
            MultimodalityFusionMethod::Checkerboard => {
                self.fusion_checkerboard(&reference.data, &transformed)
            }
            MultimodalityFusionMethod::Difference => {
                self.fusion_difference(&reference.data, &transformed)
            }
            MultimodalityFusionMethod::FalseColor => {
                self.fusion_false_color(&reference.data, &transformed)
            }
            MultimodalityFusionMethod::MultiChannel => {
                self.fusion_multi_channel(&reference.data, &transformed)
            }
        };

        Ok(fused)
    }

    /// Apply transformation to floating image (nearest neighbor interpolation)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_transform(
        &self,
        floating: &ImageData,
        transform: &RegistrationTransform,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = floating.dimensions;
        let mut transformed = Array3::from_shape_vec([nx, ny, nz], vec![0.0; nx * ny * nz])
            .expect("invariant: dimensions match zero-filled transform buffer length");

        // Apply inverse transform to avoid holes
        let inv_transform = transform.invert()?;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Physical coordinates
                    let phys_x = i as f64 * floating.voxel_spacing_mm.0;
                    let phys_y = j as f64 * floating.voxel_spacing_mm.1;
                    let phys_z = k as f64 * floating.voxel_spacing_mm.2;

                    // Transform
                    let (tx, ty, tz) = inv_transform.apply_to_point((phys_x, phys_y, phys_z));

                    // Convert back to indices
                    let vi = (tx / floating.voxel_spacing_mm.0).round() as usize;
                    let vj = (ty / floating.voxel_spacing_mm.1).round() as usize;
                    let vk = (tz / floating.voxel_spacing_mm.2).round() as usize;

                    // Nearest neighbor interpolation with bounds check
                    if vi < nx && vj < ny && vk < nz {
                        transformed[[i, j, k]] = floating.data[[vi, vj, vk]];
                    }
                }
            }
        }

        Ok(transformed)
    }

    /// Overlay fusion: weighted blend
    fn fusion_overlay(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        reference.zip_map(floating, |reference_value, floating_value| {
            reference_value * (1.0 - self.params.blend_weight)
                + floating_value * self.params.blend_weight
        })
    }

    /// Checkerboard fusion: alternating tiles
    fn fusion_checkerboard(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        let [nx, ny, nz] = reference.shape();
        let tile_size = 32; // 32×32 voxel tiles

        let mut result = reference.clone();
        for i in (0..nx).step_by(tile_size) {
            for j in (0..ny).step_by(tile_size) {
                // Alternate tiles
                if ((i / tile_size) + (j / tile_size)) % 2 == 0 {
                    let i_end = (i + tile_size).min(nx);
                    let j_end = (j + tile_size).min(ny);
                    for ii in i..i_end {
                        for jj in j..j_end {
                            for kk in 0..nz {
                                result[[ii, jj, kk]] = floating[[ii, jj, kk]];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Difference fusion: subtraction (change detection)
    fn fusion_difference(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        floating.zip_map(reference, |floating_value, reference_value| {
            (floating_value - reference_value).abs()
        })
    }

    /// False color fusion: color-coded composite
    ///
    /// Proper false-color fusion requires mapping each modality to a separate
    /// color channel (e.g., R=CT, G=PET, B=MRI) and returning a 4D RGBA array.
    /// Current 3D output format cannot represent color information.
    fn fusion_false_color(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        // Approximate: intensity-weighted blend (true color requires 4D output)
        let ref_scale = reference
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-10);
        let flt_scale = floating
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-10);
        reference.zip_map(floating, |reference_value, floating_value| {
            let ref_norm = reference_value / ref_scale;
            let flt_norm = floating_value / flt_scale;
            ref_norm * (1.0 - self.params.blend_weight) + flt_norm * self.params.blend_weight
        })
    }

    /// Multi-channel fusion: R=ref, G=float, B=diff
    ///
    /// True multi-channel output requires a 4D array [nx, ny, nz, 3].
    /// This returns a 3D weighted combination as an approximation.
    fn fusion_multi_channel(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        // Weighted combination (true RGB requires 4D output format)
        reference.zip_map(floating, |reference_value, floating_value| {
            reference_value * 0.4
                + floating_value * 0.4
                + (floating_value - reference_value).abs() * 0.2
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        let params = FusionParameters::default();
        let _engine = FusionEngine::new(params);
        // Engine created successfully
    }
}
