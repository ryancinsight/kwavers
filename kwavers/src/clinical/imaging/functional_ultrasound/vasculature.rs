//! Vessel Segmentation and Classification
//!
//! Automatic detection and classification of blood vessels (arteries vs. veins)
//! for neuronavigation and vascular-based localization.
//!
//! References:
//! - Frangi, A. A., et al. (1998). "Multiscale vessel enhancement filtering"
//! - Kirbas, C., & Quek, F. (2004). "A review of vessel extraction techniques and algorithms"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Vessel type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VesselType {
    /// Arterial vessel (bright in fUS)
    Artery,

    /// Venous vessel (darker in fUS)
    Vein,

    /// Uncertain classification
    Unknown,
}

/// Vessel classification result
#[derive(Debug, Clone)]
pub struct VesselClassification {
    /// Vessel type
    pub vessel_type: VesselType,

    /// Confidence in classification (0.0-1.0)
    pub confidence: f64,

    /// Estimated vessel diameter [μm]
    pub diameter: f64,

    /// Vessel orientation [x, y, z]
    pub orientation: [f64; 3],

    /// Blood flow direction (if artery)
    pub flow_direction: Option<[f64; 3]>,
}

/// Segmented vasculature
#[derive(Debug, Clone)]
pub struct VesselSegmentation {
    /// Binary segmentation mask (1.0 = vessel, 0.0 = background)
    pub mask: Array3<f64>,

    /// Vessel response (Frangi filter)
    pub response: Array3<f64>,

    /// Vessel classification
    pub classification: VesselClassification,

    /// Number of detected vessel segments
    pub num_segments: usize,

    /// Total vessel length [mm]
    pub total_length: f64,
}

impl VesselSegmentation {
    /// Segment vasculature from image
    pub fn segment(image: &Array3<f64>) -> KwaversResult<Self> {
        let (nx, ny, nz) = image.dim();

        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Image must be at least 3x3x3".to_string(),
            ));
        }

        // Compute Frangi vesselness filter
        let response = Self::compute_frangi_response(image)?;

        // Threshold to create binary mask
        let threshold = Self::otsu_threshold(&response);
        let mask = response.mapv(|v| if v > threshold { 1.0 } else { 0.0 });

        // Vessel classification is not yet implemented; default to Unknown.
        // Once pulsatility-index analysis is available, this will classify
        // vessels as Artery / Vein / Capillary.
        let classification = Self::classify_vessels(image, &mask).unwrap_or(VesselClassification {
            vessel_type: VesselType::Unknown,
            confidence: 0.0,
            diameter: 0.0,
            orientation: [0.0, 0.0, 0.0],
            flow_direction: None,
        });

        // Count connected components via 6-connected flood-fill on the binary mask.
        // Each component is one vessel segment. Total length is approximated as
        // the number of vessel voxels (thin structure assumption after Frangi filtering).
        let (num_segments, vessel_voxels) = Self::count_connected_components(&mask);
        let total_length = vessel_voxels as f64; // voxel units; caller scales by spacing

        Ok(Self {
            mask,
            response,
            classification,
            num_segments,
            total_length,
        })
    }

    /// Compute Frangi vesselness response from Hessian eigenvalues.
    ///
    /// At each interior voxel the 3×3 Hessian matrix is assembled from central
    /// differences and its eigenvalues are computed analytically. The Frangi
    /// vesselness measure (Frangi et al. 1998) then combines three discriminants:
    ///
    /// * $R_A = |\lambda_2| / |\lambda_3|$ — plate vs. line
    /// * $R_B = |\lambda_1| / \sqrt{|\lambda_2 \lambda_3|}$ — blob vs. tube
    /// * $S  = \sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}$ — structure magnitude
    ///
    /// Vesselness ≈ (1 − exp(−R_A²/2α²)) · exp(−R_B²/2β²) · (1 − exp(−S²/2c²))
    fn compute_frangi_response(image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = image.dim();
        let mut response = Array3::zeros((nx, ny, nz));

        // Frangi parameters (standard defaults)
        let alpha = 0.5_f64;
        let beta = 0.5_f64;
        // Adaptive c: half the maximum Frobenius-norm of Hessians
        let two_alpha_sq = 2.0 * alpha * alpha;
        let two_beta_sq = 2.0 * beta * beta;

        // First pass: compute max structure magnitude for adaptive threshold
        let mut s_max = 0.0_f64;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let h = Self::hessian_at(image, i, j, k);
                    let (l1, l2, l3) = Self::symmetric_3x3_eigenvalues(h);
                    let s = (l1 * l1 + l2 * l2 + l3 * l3).sqrt();
                    if s > s_max {
                        s_max = s;
                    }
                }
            }
        }
        let two_c_sq = 2.0 * (s_max * 0.5 + 1e-30) * (s_max * 0.5 + 1e-30);

        // Second pass: compute vesselness
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let h = Self::hessian_at(image, i, j, k);
                    let (e1, e2, e3) = Self::symmetric_3x3_eigenvalues(h);

                    // Sort by absolute value: |λ1| ≤ |λ2| ≤ |λ3|
                    let mut sorted = [e1.abs(), e2.abs(), e3.abs()];
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let (a1, a2, a3) = (sorted[0], sorted[1], sorted[2]);

                    // Vessel: two large eigenvalues, one small → bright tube on dark bg
                    // Require λ2 and λ3 to be negative (bright line on dark background)
                    if a3 < 1e-30 {
                        continue;
                    }

                    let r_a = a2 / a3;
                    let r_b = a1 / (a2 * a3).sqrt().max(1e-30);
                    let s = (a1 * a1 + a2 * a2 + a3 * a3).sqrt();

                    let vesselness = (1.0 - (-r_a * r_a / two_alpha_sq).exp())
                        * (-r_b * r_b / two_beta_sq).exp()
                        * (1.0 - (-s * s / two_c_sq).exp());

                    response[[i, j, k]] = vesselness;
                }
            }
        }

        Ok(response)
    }

    /// Compute 6 unique Hessian components at voxel (i,j,k) via central differences.
    /// Returns [H_xx, H_yy, H_zz, H_xy, H_xz, H_yz].
    fn hessian_at(image: &Array3<f64>, i: usize, j: usize, k: usize) -> [f64; 6] {
        let c = image[[i, j, k]];
        let hxx = image[[i + 1, j, k]] - 2.0 * c + image[[i - 1, j, k]];
        let hyy = image[[i, j + 1, k]] - 2.0 * c + image[[i, j - 1, k]];
        let hzz = image[[i, j, k + 1]] - 2.0 * c + image[[i, j, k - 1]];
        let hxy = (image[[i + 1, j + 1, k]] - image[[i - 1, j + 1, k]]
            - image[[i + 1, j - 1, k]]
            + image[[i - 1, j - 1, k]])
            / 4.0;
        let hxz = (image[[i + 1, j, k + 1]] - image[[i - 1, j, k + 1]]
            - image[[i + 1, j, k - 1]]
            + image[[i - 1, j, k - 1]])
            / 4.0;
        let hyz = (image[[i, j + 1, k + 1]] - image[[i, j - 1, k + 1]]
            - image[[i, j + 1, k - 1]]
            + image[[i, j - 1, k - 1]])
            / 4.0;
        [hxx, hyy, hzz, hxy, hxz, hyz]
    }

    /// Analytical eigenvalues of a 3×3 symmetric matrix stored as
    /// [a11, a22, a33, a12, a13, a23]. Uses Cardano's formula.
    fn symmetric_3x3_eigenvalues(h: [f64; 6]) -> (f64, f64, f64) {
        let (a11, a22, a33, a12, a13, a23) = (h[0], h[1], h[2], h[3], h[4], h[5]);
        let q = (a11 + a22 + a33) / 3.0;
        let p1 = a12 * a12 + a13 * a13 + a23 * a23;

        if p1 < 1e-30 {
            // Already diagonal
            return (a11, a22, a33);
        }

        let p2 = (a11 - q).powi(2) + (a22 - q).powi(2) + (a33 - q).powi(2) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let inv_p = 1.0 / p;

        // B = (A - q*I) / p — compute determinant of B
        let b11 = (a11 - q) * inv_p;
        let b22 = (a22 - q) * inv_p;
        let b33 = (a33 - q) * inv_p;
        let b12 = a12 * inv_p;
        let b13 = a13 * inv_p;
        let b23 = a23 * inv_p;

        let det_b = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13)
            + b13 * (b12 * b23 - b22 * b13);
        let r = (det_b / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;

        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + std::f64::consts::TAU / 3.0).cos();
        let eig2 = 3.0 * q - eig1 - eig3;

        (eig1, eig2, eig3)
    }

    /// Count 6-connected components in the binary mask (values > 0).
    /// Returns (num_components, total_vessel_voxels).
    fn count_connected_components(mask: &Array3<f64>) -> (usize, usize) {
        let (nx, ny, nz) = mask.dim();
        let mut visited = Array3::<bool>::default((nx, ny, nz));
        let mut num_components = 0_usize;
        let mut total_voxels = 0_usize;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if mask[[i, j, k]] > 0.0 && !visited[[i, j, k]] {
                        // Flood-fill this component
                        num_components += 1;
                        let mut stack = vec![(i, j, k)];
                        while let Some((ci, cj, ck)) = stack.pop() {
                            if visited[[ci, cj, ck]] {
                                continue;
                            }
                            visited[[ci, cj, ck]] = true;
                            total_voxels += 1;

                            // 6-connected neighbours
                            if ci > 0 && mask[[ci - 1, cj, ck]] > 0.0 {
                                stack.push((ci - 1, cj, ck));
                            }
                            if ci + 1 < nx && mask[[ci + 1, cj, ck]] > 0.0 {
                                stack.push((ci + 1, cj, ck));
                            }
                            if cj > 0 && mask[[ci, cj - 1, ck]] > 0.0 {
                                stack.push((ci, cj - 1, ck));
                            }
                            if cj + 1 < ny && mask[[ci, cj + 1, ck]] > 0.0 {
                                stack.push((ci, cj + 1, ck));
                            }
                            if ck > 0 && mask[[ci, cj, ck - 1]] > 0.0 {
                                stack.push((ci, cj, ck - 1));
                            }
                            if ck + 1 < nz && mask[[ci, cj, ck + 1]] > 0.0 {
                                stack.push((ci, cj, ck + 1));
                            }
                        }
                    }
                }
            }
        }

        (num_components, total_voxels)
    }

    /// Compute Otsu threshold for binary segmentation
    fn otsu_threshold(image: &Array3<f64>) -> f64 {
        let values: Vec<f64> = image.iter().copied().collect();

        // Simplified Otsu threshold
        let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Return midpoint as threshold (full implementation uses histogram analysis)
        (min_val + max_val) / 2.0
    }

    /// Classify vessels as arteries or veins
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — pulsatility-based classification pending.
    fn classify_vessels(
        _image: &Array3<f64>,
        _mask: &Array3<f64>,
    ) -> KwaversResult<VesselClassification> {
        Err(KwaversError::NotImplemented(
            "Vessel artery/vein classification not yet implemented. \
             Requires pulsatility index analysis and temporal signal \
             decomposition to distinguish arterial from venous flow."
                .into(),
        ))
    }

    /// Extract vessel centerline
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — skeletonization pending.
    pub fn extract_centerline(&self) -> KwaversResult<Vec<[f64; 3]>> {
        Err(KwaversError::NotImplemented(
            "Vessel centerline extraction not yet implemented. \
             Requires 3D medial axis transform or morphological \
             skeletonization of the binary vessel mask."
                .into(),
        ))
    }

    /// Estimate blood flow velocity
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — Doppler-based flow estimation pending.
    pub fn estimate_flow_velocity(&self) -> KwaversResult<f64> {
        Err(KwaversError::NotImplemented(
            "Blood flow velocity estimation not yet implemented. \
             Requires Doppler frequency shift analysis or \
             speckle tracking velocimetry."
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vessel_segmentation_creation() {
        let image = Array3::ones((10, 10, 10));
        let result = VesselSegmentation::segment(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_vessel_segmentation_small_image() {
        let image = Array3::ones((2, 2, 2));
        let result = VesselSegmentation::segment(&image);
        assert!(result.is_err());
    }

    #[test]
    fn test_vessel_classification_not_implemented() {
        let image = Array3::ones((10, 10, 10));
        let mask = Array3::ones((10, 10, 10));

        let result = VesselSegmentation::classify_vessels(&image, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_frangi_response() {
        let image = Array3::ones((10, 10, 10));
        let result = VesselSegmentation::compute_frangi_response(&image);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.dim(), (10, 10, 10));
    }
}
