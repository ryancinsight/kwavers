use super::types::{AblationTarget, FocalSpot, FocalSpotDoseEstimate};
use crate::therapy::parameters::ClinicalTherapyParameters;
use kwavers_core::constants::medical::THERMAL_DOSE_THRESHOLD;
use kwavers_core::error::{KwaversError, KwaversResult};

/// One planned HIFU focal dwell location.
#[derive(Debug, Clone)]
pub struct SonicationSubspot {
    pub index: usize,
    pub location_mm: (f64, f64, f64),
    pub dwell_time_s: f64,
    pub expected_cem43: f64,
    pub peak_temperature_c: f64,
}

/// Deterministic subspot schedule for volumetric HIFU ablation.
#[derive(Debug, Clone)]
pub struct SonicationSchedule {
    pub subspots: Vec<SonicationSubspot>,
    pub pitch_mm: (f64, f64, f64),
    pub expanded_target_dimensions_mm: (f64, f64, f64),
    pub coverage_guaranteed: bool,
    pub per_spot_dwell_time_s: f64,
    pub total_dwell_time_s: f64,
    pub minimum_subspot_cem43: f64,
    pub minimum_peak_temperature_c: f64,
}

impl SonicationSchedule {
    /// Plan a target-covering schedule from the focal FWHM ellipsoid.
    ///
    /// For an ellipsoid with semi-axes `a`, `a`, `c`, selecting grid pitch
    /// `(2a/sqrt(3), 2a/sqrt(3), 2c/sqrt(3))` bounds the normalized distance
    /// from any point in a grid cell to its nearest center by 1.0:
    ///
    /// `(hx/a)^2 + (hy/a)^2 + (hz/c)^2 <= 1`.
    ///
    /// The schedule therefore gives deterministic target-box coverage under the
    /// FWHM ellipsoid model rather than relying on a one-focus adequacy heuristic.
    pub fn plan(
        target: &AblationTarget,
        focal_spot: &FocalSpot,
        therapy_params: &ClinicalTherapyParameters,
        frequency_hz: f64,
    ) -> KwaversResult<Self> {
        validate_positive_finite("target.dimensions_mm.0", target.dimensions_mm.0)?;
        validate_positive_finite("target.dimensions_mm.1", target.dimensions_mm.1)?;
        validate_positive_finite("target.dimensions_mm.2", target.dimensions_mm.2)?;
        validate_positive_finite("target.safety_margin_mm", target.safety_margin_mm)?;
        validate_positive_finite("focal_spot.lateral_width_mm", focal_spot.lateral_width_mm)?;
        validate_positive_finite("focal_spot.axial_width_mm", focal_spot.axial_width_mm)?;
        validate_positive_finite(
            "therapy_params.treatment_duration",
            therapy_params.treatment_duration,
        )?;

        let sqrt3 = 3.0_f64.sqrt();
        let pitch = (
            focal_spot.lateral_width_mm / sqrt3,
            focal_spot.lateral_width_mm / sqrt3,
            focal_spot.axial_width_mm / sqrt3,
        );
        let expanded = (
            target.dimensions_mm.0 + 2.0 * target.safety_margin_mm,
            target.dimensions_mm.1 + 2.0 * target.safety_margin_mm,
            target.dimensions_mm.2 + 2.0 * target.safety_margin_mm,
        );

        let x = axis_centers(target.location_mm.0, expanded.0, pitch.0);
        let y = axis_centers(target.location_mm.1, expanded.1, pitch.1);
        let z = axis_centers(target.location_mm.2, expanded.2, pitch.2);
        let n = x.len() * y.len() * z.len();
        let per_spot_dwell = therapy_params.treatment_duration / n as f64;
        let dose = FocalSpotDoseEstimate::estimate_from_focal_spot(
            focal_spot,
            frequency_hz,
            therapy_params.duty_cycle,
            per_spot_dwell,
        )?;

        let mut subspots = Vec::with_capacity(n);
        for &x_mm in &x {
            for &y_mm in &y {
                for &z_mm in &z {
                    subspots.push(SonicationSubspot {
                        index: subspots.len(),
                        location_mm: (x_mm, y_mm, z_mm),
                        dwell_time_s: per_spot_dwell,
                        expected_cem43: dose.cem43,
                        peak_temperature_c: dose.peak_temperature_c,
                    });
                }
            }
        }

        let corner_bound = coverage_corner_bound(&x, &y, &z, expanded, focal_spot);
        let coverage_guaranteed = corner_bound <= 1.0 + 64.0 * f64::EPSILON;

        Ok(Self {
            subspots,
            pitch_mm: pitch,
            expanded_target_dimensions_mm: expanded,
            coverage_guaranteed,
            per_spot_dwell_time_s: per_spot_dwell,
            total_dwell_time_s: therapy_params.treatment_duration,
            minimum_subspot_cem43: dose.cem43,
            minimum_peak_temperature_c: dose.peak_temperature_c,
        })
    }

    #[must_use]
    pub fn subspot_count(&self) -> usize {
        self.subspots.len()
    }

    #[must_use]
    pub fn all_subspots_reach_ablation(&self) -> bool {
        self.minimum_subspot_cem43 >= THERMAL_DOSE_THRESHOLD
    }
}

fn validate_positive_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

fn axis_centers(center: f64, extent: f64, max_pitch: f64) -> Vec<f64> {
    let count = if extent <= max_pitch {
        1
    } else {
        (extent / max_pitch).ceil() as usize + 1
    };
    if count == 1 {
        return vec![center];
    }
    let start = center - 0.5 * extent;
    let step = extent / (count - 1) as f64;
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

fn coverage_corner_bound(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    expanded: (f64, f64, f64),
    focal_spot: &FocalSpot,
) -> f64 {
    let hx = 0.5 * max_cell_width(x, expanded.0);
    let hy = 0.5 * max_cell_width(y, expanded.1);
    let hz = 0.5 * max_cell_width(z, expanded.2);
    let lateral_semi = 0.5 * focal_spot.lateral_width_mm;
    let axial_semi = 0.5 * focal_spot.axial_width_mm;
    (hx / lateral_semi).powi(2) + (hy / lateral_semi).powi(2) + (hz / axial_semi).powi(2)
}

fn max_cell_width(axis: &[f64], extent: f64) -> f64 {
    if axis.len() <= 1 {
        extent
    } else {
        axis.windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(0.0, f64::max)
    }
}
