use super::super::types::{FeatureMap, LesionDetection};
use super::ClinicalDecisionSupport;
use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3};

impl ClinicalDecisionSupport {
    /// Detect lesions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn detect_lesions(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
        uncertainty: ArrayView3<f32>,
        confidence: ArrayView3<f32>,
    ) -> KwaversResult<Vec<LesionDetection>> {
        let mut lesions = Vec::new();
        let (nx, ny, nz) = volume.dim();
        let margin = 10;

        for z in margin..nz.saturating_sub(margin) {
            for y in margin..ny.saturating_sub(margin) {
                for x in margin..nx.saturating_sub(margin) {
                    let vol_val = volume[[x, y, z]];
                    let conf_val = confidence[[x, y, z]];
                    let uncert_val = uncertainty[[x, y, z]];

                    let gradient_mag = features
                        .morphological
                        .get("gradient_magnitude")
                        .map_or(0.0, |arr| arr[[x, y, z]]);

                    let speckle_var = features
                        .texture
                        .get("speckle_variance")
                        .map_or(0.0, |arr| arr[[x, y, z]]);

                    let is_high_contrast = vol_val > self.config.contrast_abnormality_threshold;
                    let is_high_confidence = conf_val > self.config.lesion_confidence_threshold;
                    let is_low_uncertainty = uncert_val < self.config.tissue_uncertainty_threshold;
                    let is_anomalous_speckle = speckle_var > self.config.speckle_anomaly_threshold;
                    let is_strong_boundary = gradient_mag > 0.5;

                    if is_high_contrast
                        && is_high_confidence
                        && is_low_uncertainty
                        && is_anomalous_speckle
                        && is_strong_boundary
                    {
                        lesions.push(LesionDetection {
                            center: (x, y, z),
                            size_mm: self.estimate_lesion_size(volume, features, x, y, z),
                            confidence: conf_val,
                            lesion_type: self.classify_lesion_type(vol_val, features, x, y, z),
                            clinical_significance: self
                                .assess_clinical_significance(conf_val, vol_val),
                        });
                    }
                }
            }
        }

        Ok(lesions)
    }

    /// Estimate lesion size using 3D connected component analysis (26-connectivity).
    ///
    /// 1. Compute adaptive threshold: local_mean + k·σ
    /// 2. Flood-fill from seed point
    /// 3. Compute equivalent spherical diameter: d = 2·[(3V)/(4π)]^(1/3)
    ///
    /// Reference: Gonzalez & Woods (2008), "Digital Image Processing", connected components.
    pub(super) fn estimate_lesion_size(
        &self,
        volume: ArrayView3<f32>,
        _features: &FeatureMap,
        seed_x: usize,
        seed_y: usize,
        seed_z: usize,
    ) -> f32 {
        let (dim_x, dim_y, dim_z) = volume.dim();

        let local_mean = self.compute_local_statistics(&volume, seed_x, seed_y, seed_z);
        let threshold = 2.0f32.mul_add(self.config.segmentation_sensitivity, local_mean);

        let mut visited = Array3::<bool>::from_elem((dim_x, dim_y, dim_z), false);
        let mut component_voxels = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((seed_x, seed_y, seed_z));
        visited[[seed_x, seed_y, seed_z]] = true;

        let offsets: [(isize, isize, isize); 26] = [
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, 1, 0),
            (-1, 1, 1),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ];

        while let Some((x, y, z)) = queue.pop_front() {
            component_voxels.push((x, y, z));

            for (dx, dy, dz) in &offsets {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;

                if nx >= 0
                    && nx < dim_x as isize
                    && ny >= 0
                    && ny < dim_y as isize
                    && nz >= 0
                    && nz < dim_z as isize
                {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;

                    if !visited[[nx, ny, nz]] && volume[[nx, ny, nz]] > threshold {
                        visited[[nx, ny, nz]] = true;
                        queue.push_back((nx, ny, nz));
                    }
                }
            }
        }

        let voxel_volume_mm3 = self.config.voxel_size_mm.powi(3);
        let lesion_volume_mm3 = component_voxels.len() as f32 * voxel_volume_mm3;
        let equivalent_radius_mm =
            (3.0 * lesion_volume_mm3 / (4.0 * std::f32::consts::PI)).cbrt();
        2.0 * equivalent_radius_mm
    }

    /// Compute local mean intensity in a 5×5×5 window for adaptive thresholding.
    pub(super) fn compute_local_statistics(
        &self,
        volume: &ArrayView3<f32>,
        x: usize,
        y: usize,
        z: usize,
    ) -> f32 {
        let (nx, ny, nz) = volume.dim();
        let window_size = 5;

        let mut sum = 0.0;
        let mut count = 0;

        for dx in -(window_size as isize)..=(window_size as isize) {
            for dy in -(window_size as isize)..=(window_size as isize) {
                for dz in -(window_size as isize)..=(window_size as isize) {
                    let neighbor_x = x as isize + dx;
                    let neighbor_y = y as isize + dy;
                    let neighbor_z = z as isize + dz;

                    if neighbor_x >= 0
                        && neighbor_x < nx as isize
                        && neighbor_y >= 0
                        && neighbor_y < ny as isize
                        && neighbor_z >= 0
                        && neighbor_z < nz as isize
                    {
                        sum += volume[[
                            neighbor_x as usize,
                            neighbor_y as usize,
                            neighbor_z as usize,
                        ]];
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            volume[[x, y, z]]
        }
    }

    /// Classify lesion echogenicity: Hyperechoic (>3.0), Hypoechoic (<0.5), Isoechoic otherwise.
    ///
    /// Reference: Stavros et al. (1995), "Solid breast nodules: use of sonography".
    pub(super) fn classify_lesion_type(
        &self,
        intensity: f32,
        _features: &FeatureMap,
        _x: usize,
        _y: usize,
        _z: usize,
    ) -> String {
        if intensity > 3.0 {
            "Hyperechoic Lesion".to_owned()
        } else if intensity < 0.5 {
            "Hypoechoic Lesion".to_owned()
        } else {
            "Isoechoic Lesion".to_owned()
        }
    }

    /// Compute clinical significance score [0, 1] from detection confidence and intensity.
    pub(super) fn assess_clinical_significance(&self, confidence: f32, intensity: f32) -> f32 {
        let confidence_score = confidence;
        let intensity_score = intensity.abs().min(1.0);
        (confidence_score + intensity_score) / 2.0
    }
}
