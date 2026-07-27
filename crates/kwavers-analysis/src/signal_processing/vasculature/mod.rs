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

use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

mod analysis;
mod classify;
mod frangi;

#[cfg(test)]
mod tests;

// ── Public types ─────────────────────────────────────────────────────────────

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
    pub diameter: Length,
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
    /// Estimated physical vessel length.
    pub total_length: Length,
    /// Physical voxel spacing along the image axes.
    pub voxel_spacing: [Length; 3],
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
    /// Returns `InvalidInput` when any image dimension is < 3 or voxel spacing
    /// is non-finite or non-positive.
    pub fn segment(image: &Array3<f64>, voxel_spacing: [Length; 3]) -> KwaversResult<Self> {
        validate_voxel_spacing(voxel_spacing)?;
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
        let (num_segments, _) = analysis::count_connected_components(&mask);
        let points = classify::masked_points(&mask);
        let total_length = classify::physical_length(&mask, &points, voxel_spacing);

        Ok(Self {
            mask,
            response,
            classification,
            num_segments,
            total_length,
            voxel_spacing,
        })
    }

    /// Extract the vessel centerline as physical coordinates in metres.
    ///
    /// Returns voxels that are local maxima in the 6-neighbour topology —
    /// a deterministic one-pass medial-axis approximation for thin Frangi masks.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn extract_centerline(&self) -> KwaversResult<Vec<[Length; 3]>> {
        Ok(
            classify::centerline_from_points(&self.mask, &classify::masked_points(&self.mask))
                .into_iter()
                .map(|[i, j, k]| {
                    [
                        Length::from_base(i as f64 * self.voxel_spacing[0].into_base()),
                        Length::from_base(j as f64 * self.voxel_spacing[1].into_base()),
                        Length::from_base(k as f64 * self.voxel_spacing[2].into_base()),
                    ]
                })
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
    /// Frequencies are in hertz, sound speed and the returned value are in
    /// metres per second, and the angle is in radians.
    ///
    /// # Errors
    /// - `InvalidInput` when any argument is non-finite or out of range.
    /// - `InvalidInput` when `cos(θ) < 1e-6` (beam nearly perpendicular).
    pub fn estimate_flow_velocity_from_doppler(
        doppler_shift: Frequency,
        transmit_frequency: Frequency,
        sound_speed: Velocity,
        beam_angle: Angle,
    ) -> KwaversResult<Velocity> {
        let doppler_shift_hz = doppler_shift.into_base();
        let transmit_frequency_hz = transmit_frequency.into_base();
        let sound_speed_m_s = sound_speed.into_base();
        let beam_angle_rad = beam_angle.into_base();
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
    pub fn estimate_flow_velocity(&self) -> KwaversResult<Velocity> {
        Err(KwaversError::InvalidInput(
            "static vessel segmentation does not contain Doppler or tracking data".to_owned(),
        ))
    }
}

fn validate_voxel_spacing(voxel_spacing: [Length; 3]) -> KwaversResult<()> {
    if voxel_spacing
        .iter()
        .map(|spacing| spacing.into_base())
        .all(|spacing| spacing.is_finite() && spacing > 0.0)
    {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(
            "voxel spacing must contain finite positive lengths".to_owned(),
        ))
    }
}
