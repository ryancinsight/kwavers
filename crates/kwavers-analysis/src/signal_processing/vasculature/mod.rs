//! Vessel segmentation and classification for functional ultrasound.
//!
//! # Module layout
//!
//! | Sub-module  | Responsibility                                      |
//! |-------------|-----------------------------------------------------|
//! | `frangi`    | Frangi vesselness filter (Hessian + eigenvalues)    |
//! | `analysis`  | RITK Otsu adapter, 6-connected component labelling |
//! | `classify`  | Vessel geometry, orientation, classification        |
//!
//! # References
//! - Frangi et al. (1998). "Multiscale vessel enhancement filtering". MICCAI.
//! - Kirbas & Quek (2004). "A review of vessel extraction techniques". CSUR.
//! - Jensen (1996). *Estimation of Blood Velocities Using Ultrasound*.

use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

mod analysis;
mod classify;
mod frangi;

#[cfg(test)]
mod tests;

// ── Public types ─────────────────────────────────────────────────────────────

/// Physical spacing between adjacent voxels on each image axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelSpacing {
    components: [Length<f64>; 3],
}

impl VoxelSpacing {
    /// Validate and construct anisotropic voxel spacing in SI metres.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when any component is non-finite
    /// or not strictly positive.
    pub fn from_lengths(components: [Length<f64>; 3]) -> KwaversResult<Self> {
        if components
            .iter()
            .any(|component| !component.as_base().is_finite() || *component.as_base() <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "voxel spacing must be finite and positive in every axis".to_owned(),
            ));
        }
        Ok(Self { components })
    }

    /// Return the validated spacing components.
    #[must_use]
    pub fn components(self) -> [Length<f64>; 3] {
        self.components
    }

    fn base_values(self) -> [f64; 3] {
        self.components.map(Length::into_base)
    }

    fn scale_index(self, index: [usize; 3]) -> [Length<f64>; 3] {
        let spacing = self.base_values();
        [
            Length::from_base(index[0] as f64 * spacing[0]),
            Length::from_base(index[1] as f64 * spacing[1]),
            Length::from_base(index[2] as f64 * spacing[2]),
        ]
    }

    fn step_along(self, direction: [f64; 3]) -> f64 {
        let spacing = self.base_values();
        let x = direction[0] * spacing[0];
        let y = direction[1] * spacing[1];
        let z = direction[2] * spacing[2];
        x.mul_add(x, y.mul_add(y, z * z)).sqrt()
    }
}

/// Artery / vein classification label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VascularVesselType {
    /// Arterial vessel (bright in fUS).
    Artery,
    /// Venous vessel (darker in fUS).
    Vein,
    /// Insufficient evidence to classify.
    Unknown,
}

/// Result of vessel classification.
#[derive(Debug, Clone)]
pub struct VesselClassification {
    /// Vessel type label.
    pub vessel_type: VascularVesselType,
    /// Classification confidence ∈ [0, 0.95].
    pub confidence: f64,
    /// Estimated physical vessel diameter.
    pub diameter: Length<f64>,
    /// Principal axis direction (unit vector).
    pub orientation: [f64; 3],
    /// Estimated flow direction for arteries; `None` for veins or unknown.
    pub flow_direction: Option<[f64; 3]>,
}

/// Output of [`VesselSegmentation::segment`].
#[derive(Debug, Clone)]
pub struct VesselSegmentation {
    /// Binary vessel mask (1.0 = vessel, 0.0 = background).
    pub mask: Array3<f64>,
    /// Raw Frangi vesselness response.
    pub response: Array3<f64>,
    /// Vessel classification derived from the segmented region.
    pub classification: VesselClassification,
    /// Number of 6-connected vessel segments.
    pub num_segments: usize,
    /// Total physical vessel length along the extracted centerline.
    pub total_length: Length<f64>,
    /// Validated physical spacing of the input image.
    pub voxel_spacing: VoxelSpacing,
}

// ── Public API ────────────────────────────────────────────────────────────────

impl VesselSegmentation {
    /// Segment vasculature from a 3-D fUS image with physical voxel spacing.
    ///
    /// Steps:
    /// 1. Compute multi-scale Frangi vesselness response.
    /// 2. Threshold with Otsu's method → binary mask.
    /// 3. Classify vessels by intensity contrast and geometric axis.
    /// 4. Count 6-connected components.
    ///
    /// # Errors
    /// Returns `InvalidInput` when any image dimension is < 3.
    pub fn segment(image: &Array3<f64>, voxel_spacing: VoxelSpacing) -> KwaversResult<Self> {
        let [nx, ny, nz] = image.shape();
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "image must be at least 3×3×3".to_owned(),
            ));
        }

        let response = frangi::compute_frangi_response(image)?;
        let threshold = analysis::otsu_threshold(&response);
        let mask = response.mapv(|v| if v > threshold { 1.0 } else { 0.0 });

        let classification = classify::classify_vessels(image, &mask, voxel_spacing)?;
        let (num_segments, vessel_voxels) = analysis::count_connected_components(&mask);
        let centerline = classify::centerline_from_points(&mask, &classify::masked_points(&mask));
        let total_length =
            classify::centerline_length(&centerline, classification.orientation, voxel_spacing);

        Ok(Self {
            mask,
            response,
            classification,
            num_segments,
            total_length: if vessel_voxels == 0 {
                Length::from_base(0.0)
            } else {
                total_length
            },
            voxel_spacing,
        })
    }

    /// Extract the vessel centerline as physical coordinates.
    ///
    /// Returns voxels that are local maxima in the 6-neighbour topology —
    /// a deterministic one-pass medial-axis approximation for thin Frangi masks.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn extract_centerline(&self) -> KwaversResult<Vec<[Length<f64>; 3]>> {
        Ok(
            classify::centerline_from_points(&self.mask, &classify::masked_points(&self.mask))
                .into_iter()
                .map(|index| self.voxel_spacing.scale_index(index))
                .collect(),
        )
    }

    /// Estimate axial blood speed from a measured Doppler shift.
    ///
    /// # Algorithm
    ///
    /// Pulsed Doppler gives `f_d = 2 f₀ v cos(θ) / c`, hence
    ///
    /// ```text
    ///   v = f_d · c / (2 f₀ · cos(θ))
    /// ```
    ///
    /// # Errors
    /// - `InvalidInput` when any argument is non-finite or out of range.
    /// - `InvalidInput` when `cos(θ) < 1e-6` (beam nearly perpendicular).
    pub fn estimate_flow_velocity_from_doppler(
        doppler_shift: Frequency<f64>,
        transmit_frequency: Frequency<f64>,
        sound_speed: Velocity<f64>,
        beam_angle_rad: f64,
    ) -> KwaversResult<Velocity<f64>> {
        let doppler_shift_hz = doppler_shift.into_base();
        let transmit_frequency_hz = transmit_frequency.into_base();
        let sound_speed_m_s = sound_speed.into_base();
        if !doppler_shift_hz.is_finite()
            || !transmit_frequency_hz.is_finite()
            || transmit_frequency_hz <= 0.0
            || !sound_speed_m_s.is_finite()
            || sound_speed_m_s <= 0.0
            || !beam_angle_rad.is_finite()
        {
            return Err(KwaversError::InvalidInput(
                "Doppler velocity inputs must be finite with positive frequency and sound speed"
                    .to_owned(),
            ));
        }

        let cos_theta = beam_angle_rad.cos();
        if cos_theta.abs() < 1e-6 {
            return Err(KwaversError::InvalidInput(
                "Doppler beam angle is too close to 90 degrees".to_owned(),
            ));
        }

        Ok(Velocity::from_base(
            doppler_shift_hz * sound_speed_m_s / (2.0 * transmit_frequency_hz * cos_theta),
        ))
    }

    /// Static segmentation does not carry Doppler or tracking data.
    ///
    /// Use [`estimate_flow_velocity_from_doppler`](Self::estimate_flow_velocity_from_doppler)
    /// with measured Doppler inputs instead.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn estimate_flow_velocity(&self) -> KwaversResult<Velocity<f64>> {
        Err(KwaversError::InvalidInput(
            "static vessel segmentation does not contain Doppler or tracking data".to_owned(),
        ))
    }
}
