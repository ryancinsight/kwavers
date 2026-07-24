use super::types::{AblationTarget, FocalSpot, FocalSpotDoseEstimate};
use crate::therapy::domain_types::ClinicalTherapyParameters;
use aequitas::systems::si::quantities::{Frequency, Length, ThermodynamicTemperature, Time};
use kwavers_core::constants::medical::THERMAL_DOSE_THRESHOLD;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::thermal::CumulativeEquivalentMinutes;
use kwavers_transducer::transducers::physics::CartesianPosition;

/// One planned HIFU focal dwell location.
#[derive(Debug, Clone)]
pub struct SonicationSubspot {
    pub index: usize,
    pub location: CartesianPosition,
    pub dwell_time: Time<f64>,
    pub expected_cem43: CumulativeEquivalentMinutes,
    pub peak_temperature: ThermodynamicTemperature<f64>,
}

/// Deterministic subspot schedule for volumetric HIFU ablation.
#[derive(Debug, Clone)]
pub struct SonicationSchedule {
    pub subspots: Vec<SonicationSubspot>,
    pub pitch: [Length<f64>; 3],
    pub expanded_target_dimensions: [Length<f64>; 3],
    pub coverage_guaranteed: bool,
    pub per_spot_dwell: Time<f64>,
    pub total_dwell: Time<f64>,
    pub minimum_subspot_cem43: CumulativeEquivalentMinutes,
    pub minimum_peak_temperature: ThermodynamicTemperature<f64>,
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
        frequency: Frequency<f64>,
    ) -> KwaversResult<Self> {
        for (index, dimension) in target.dimensions.iter().enumerate() {
            validate_positive_finite(
                &format!("target.dimensions[{index}]"),
                dimension.into_base(),
            )?;
        }
        validate_positive_finite("target.safety_margin", target.safety_margin.into_base())?;
        validate_positive_finite(
            "focal_spot.lateral_width",
            focal_spot.lateral_width.into_base(),
        )?;
        validate_positive_finite("focal_spot.axial_width", focal_spot.axial_width.into_base())?;
        validate_positive_finite(
            "therapy_params.treatment_duration",
            therapy_params.treatment_duration,
        )?;

        let sqrt3 = 3.0_f64.sqrt();
        let pitch = [
            Length::from_base(focal_spot.lateral_width.into_base() / sqrt3),
            Length::from_base(focal_spot.lateral_width.into_base() / sqrt3),
            Length::from_base(focal_spot.axial_width.into_base() / sqrt3),
        ];
        let margin = target.safety_margin.into_base();
        let expanded = target
            .dimensions
            .map(|dimension| Length::from_base(2.0f64.mul_add(margin, dimension.into_base())));
        let target_location = target.location.into_base();

        let x = axis_centers(Length::from_base(target_location[0]), expanded[0], pitch[0]);
        let y = axis_centers(Length::from_base(target_location[1]), expanded[1], pitch[1]);
        let z = axis_centers(Length::from_base(target_location[2]), expanded[2], pitch[2]);
        let n = x.len() * y.len() * z.len();
        let per_spot_dwell = Time::from_base(therapy_params.treatment_duration / n as f64);
        let dose = FocalSpotDoseEstimate::estimate_from_focal_spot(
            focal_spot,
            frequency,
            therapy_params.duty_cycle,
            per_spot_dwell,
        )?;

        let mut subspots = Vec::with_capacity(n);
        for &x_position in &x {
            for &y_position in &y {
                for &z_position in &z {
                    subspots.push(SonicationSubspot {
                        index: subspots.len(),
                        location: CartesianPosition::from_base([
                            x_position.into_base(),
                            y_position.into_base(),
                            z_position.into_base(),
                        ])?,
                        dwell_time: per_spot_dwell,
                        expected_cem43: dose.cem43,
                        peak_temperature: dose.peak_temperature,
                    });
                }
            }
        }

        let corner_bound = coverage_corner_bound(&x, &y, &z, expanded, focal_spot);
        let coverage_guaranteed = corner_bound <= 1.0 + 64.0 * f64::EPSILON;

        Ok(Self {
            subspots,
            pitch,
            expanded_target_dimensions: expanded,
            coverage_guaranteed,
            per_spot_dwell,
            total_dwell: Time::from_base(therapy_params.treatment_duration),
            minimum_subspot_cem43: dose.cem43,
            minimum_peak_temperature: dose.peak_temperature,
        })
    }

    #[must_use]
    pub fn subspot_count(&self) -> usize {
        self.subspots.len()
    }

    #[must_use]
    pub fn all_subspots_reach_ablation(&self) -> bool {
        let threshold = CumulativeEquivalentMinutes::try_from_minutes(THERMAL_DOSE_THRESHOLD)
            .expect("invariant: canonical CEM43 threshold is finite and non-negative");
        self.minimum_subspot_cem43 >= threshold
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

fn axis_centers(
    center: Length<f64>,
    extent: Length<f64>,
    max_pitch: Length<f64>,
) -> Vec<Length<f64>> {
    let center = center.into_base();
    let extent = extent.into_base();
    let max_pitch = max_pitch.into_base();
    let count = if extent <= max_pitch {
        1
    } else {
        (extent / max_pitch).ceil() as usize + 1
    };
    if count == 1 {
        return vec![Length::from_base(center)];
    }
    let start = center - 0.5 * extent;
    let step = extent / (count - 1) as f64;
    (0..count)
        .map(|idx| Length::from_base(start + idx as f64 * step))
        .collect()
}

fn coverage_corner_bound(
    x: &[Length<f64>],
    y: &[Length<f64>],
    z: &[Length<f64>],
    expanded: [Length<f64>; 3],
    focal_spot: &FocalSpot,
) -> f64 {
    let hx = 0.5 * max_cell_width(x, expanded[0].into_base());
    let hy = 0.5 * max_cell_width(y, expanded[1].into_base());
    let hz = 0.5 * max_cell_width(z, expanded[2].into_base());
    let lateral_semi = 0.5 * focal_spot.lateral_width.into_base();
    let axial_semi = 0.5 * focal_spot.axial_width.into_base();
    (hx / lateral_semi).powi(2) + (hy / lateral_semi).powi(2) + (hz / axial_semi).powi(2)
}

fn max_cell_width(axis: &[Length<f64>], extent: f64) -> f64 {
    if axis.len() <= 1 {
        extent
    } else {
        axis.windows(2)
            .map(|pair| pair[1].into_base() - pair[0].into_base())
            .fold(0.0, f64::max)
    }
}
