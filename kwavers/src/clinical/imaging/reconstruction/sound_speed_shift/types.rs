//! Public types for ultrasonic speed-of-sound shift imaging.

use ndarray::Array2;

use crate::solver::inverse::same_aperture::PlanarPoint;

/// Linearized straight-ray speed-of-sound shift model identifier.
pub const SOUND_SPEED_SHIFT_MODEL: &str = "straight_ray_sound_speed_shift_tomography";
/// Curved-ray speed-of-sound shift model identifier.
pub const CURVED_RAY_SOUND_SPEED_SHIFT_MODEL: &str = "curved_ray_sound_speed_shift_tomography";
/// Finite-frequency speed-of-sound shift model identifier.
pub const FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL: &str =
    "finite_frequency_sound_speed_shift_tomography";

/// Propagation path used to assemble each measurement row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftPropagation {
    /// Straight transmitter-to-receiver chord.
    StraightRay,
    /// Circular arc through the transmitter and receiver.
    ///
    /// `sagitta_m` is the signed midpoint displacement from the chord along
    /// the chord-left normal. `segments` is the number of exact straight
    /// subsegments used to represent the circular arc in the voxelized model.
    CircularArc { sagitta_m: f64, segments: usize },
}

impl ShiftPropagation {
    pub(super) fn validate(self) -> Result<(), String> {
        match self {
            Self::StraightRay => Ok(()),
            Self::CircularArc {
                sagitta_m,
                segments,
            } => {
                if !sagitta_m.is_finite() || sagitta_m.abs() <= f64::EPSILON {
                    return Err(format!(
                        "Circular-arc propagation requires finite nonzero sagitta, got {sagitta_m}"
                    ));
                }
                if segments < 2 {
                    return Err(format!(
                        "Circular-arc propagation requires at least 2 segments, got {segments}"
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Sensitivity kernel used to convert a propagation path into voxel weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftSensitivity {
    /// Geometric ray length only.
    GeometricRay,
    /// Compact finite-frequency Fresnel tube around the propagation path.
    ///
    /// `wavelength_m` defines the local first-Fresnel-zone scale and
    /// `support_radius_m` truncates the transverse kernel.
    FiniteFrequency {
        wavelength_m: f64,
        support_radius_m: f64,
    },
}

impl ShiftSensitivity {
    pub(super) fn validate(self, spacing_m: f64) -> Result<(), String> {
        match self {
            Self::GeometricRay => Ok(()),
            Self::FiniteFrequency {
                wavelength_m,
                support_radius_m,
            } => {
                if !wavelength_m.is_finite() || wavelength_m <= 0.0 {
                    return Err(format!(
                        "Finite-frequency sensitivity requires positive wavelength, got {wavelength_m}"
                    ));
                }
                if !support_radius_m.is_finite() || support_radius_m < 0.5 * spacing_m {
                    return Err(format!(
                        "Finite-frequency support radius must be finite and at least half a pixel, got {support_radius_m}"
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Acquisition row selection policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftSampling {
    /// Use every supplied transmit/receive travel-time shift.
    Dense,
    /// Use a deterministic subset `row % stride == offset`.
    Sparse { stride: usize, offset: usize },
}

impl ShiftSampling {
    pub(super) fn accepts(self, row: usize) -> bool {
        match self {
            Self::Dense => true,
            Self::Sparse { stride, offset } => row % stride == offset,
        }
    }

    pub(super) fn validate(self) -> Result<(), String> {
        match self {
            Self::Dense => Ok(()),
            Self::Sparse { stride, offset } if stride > 0 && offset < stride => Ok(()),
            Self::Sparse { stride, offset } => Err(format!(
                "Sparse sampling requires stride > 0 and offset < stride, got stride={stride}, offset={offset}"
            )),
        }
    }
}

/// Image prior used by the inverse solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftPrior {
    /// Dense H1/Tikhonov speed-shift field.
    Dense,
    /// Sparse perturbation field solved with an L1 proximal step.
    Sparse,
}

/// Configuration for speed-of-sound shift reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct SoundSpeedShiftConfig {
    /// Reference homogeneous sound speed c0 [m/s].
    pub reference_sound_speed_m_s: f64,
    /// Square pixel spacing [m].
    pub spacing_m: f64,
    /// Iteration count for PCG or proximal gradient.
    pub iterations: usize,
    /// L2 Tikhonov weight on the speed shift [squared path units].
    pub tikhonov_weight: f64,
    /// Graph-H1 smoothness weight on active neighboring pixels.
    pub smoothness_weight: f64,
    /// L1 weight for [`ShiftPrior::Sparse`].
    pub sparsity_weight: f64,
    /// Deterministic measurement-row selection.
    pub sampling: ShiftSampling,
    /// Dense or sparse image prior.
    pub prior: ShiftPrior,
    /// Propagation path model used for measurement rows.
    pub propagation: ShiftPropagation,
    /// Voxel sensitivity kernel around the propagation path.
    pub sensitivity: ShiftSensitivity,
}

/// Reusable scratch buffers for speed-of-sound shift reconstruction.
///
/// A workspace owns all PCG/ISTA work vectors used by dense and sparse
/// reconstructions. Reusing one workspace across calls with compatible or
/// smaller geometry preserves vector capacities and avoids repeated heap
/// allocation in iterative solves.
#[derive(Clone, Debug, Default)]
pub struct SoundSpeedShiftWorkspace {
    pub(super) rhs: Vec<f64>,
    pub(super) diagonal: Vec<f64>,
    pub(super) solution: Vec<f64>,
    pub(super) normal_solution: Vec<f64>,
    pub(super) residual: Vec<f64>,
    pub(super) preconditioned: Vec<f64>,
    pub(super) direction: Vec<f64>,
    pub(super) normal_direction: Vec<f64>,
    pub(super) row: Vec<f64>,
    pub(super) laplacian: Vec<f64>,
    pub(super) prediction: Vec<f64>,
    pub(super) previous_solution: Vec<f64>,
    pub(super) power_vector: Vec<f64>,
    pub(super) power_normal: Vec<f64>,
    pub(super) objective_history: Vec<f64>,
}

impl Default for SoundSpeedShiftConfig {
    fn default() -> Self {
        Self {
            reference_sound_speed_m_s: 1540.0,
            spacing_m: 5.0e-4,
            iterations: 64,
            tikhonov_weight: 1.0e-8,
            smoothness_weight: 0.0,
            sparsity_weight: 0.0,
            sampling: ShiftSampling::Dense,
            prior: ShiftPrior::Dense,
            propagation: ShiftPropagation::StraightRay,
            sensitivity: ShiftSensitivity::GeometricRay,
        }
    }
}

impl SoundSpeedShiftConfig {
    /// Select acquisition-row sampling.
    #[must_use]
    pub fn with_sampling(mut self, sampling: ShiftSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Select dense or sparse image prior.
    #[must_use]
    pub fn with_prior(mut self, prior: ShiftPrior) -> Self {
        self.prior = prior;
        self
    }

    /// Select iteration count.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Select propagation path model.
    #[must_use]
    pub fn with_propagation(mut self, propagation: ShiftPropagation) -> Self {
        self.propagation = propagation;
        self
    }

    /// Select voxel sensitivity kernel.
    #[must_use]
    pub fn with_sensitivity(mut self, sensitivity: ShiftSensitivity) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    pub(super) fn model_family(self) -> &'static str {
        match (self.propagation, self.sensitivity) {
            (ShiftPropagation::StraightRay, ShiftSensitivity::GeometricRay) => {
                SOUND_SPEED_SHIFT_MODEL
            }
            (_, ShiftSensitivity::FiniteFrequency { .. }) => {
                FINITE_FREQUENCY_SOUND_SPEED_SHIFT_MODEL
            }
            (ShiftPropagation::CircularArc { .. }, ShiftSensitivity::GeometricRay) => {
                CURVED_RAY_SOUND_SPEED_SHIFT_MODEL
            }
        }
    }
}

/// One measured differential travel-time shift.
#[derive(Clone, Copy, Debug)]
pub struct SoundSpeedShiftSample {
    /// Transmit point in the imaging plane.
    pub transmitter: PlanarPoint,
    /// Receive point in the imaging plane.
    pub receiver: PlanarPoint,
    /// Observed minus reference travel time [s].
    pub time_shift_s: f64,
}

impl SoundSpeedShiftSample {
    /// Construct a measured shift sample.
    #[must_use]
    pub fn new(transmitter: PlanarPoint, receiver: PlanarPoint, time_shift_s: f64) -> Self {
        Self {
            transmitter,
            receiver,
            time_shift_s,
        }
    }
}

/// Reconstructed speed-of-sound shift image.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftImage {
    /// Estimated `delta c = c - c0` [m/s] on the input mask grid.
    pub sound_speed_shift_m_s: Array2<f64>,
    /// Objective value after each solver iteration, including the initial state.
    pub objective_history: Vec<f64>,
    /// Number of selected measurement rows used by the inverse solve.
    pub rows_used: usize,
    /// Number of supplied measurement rows before sampling.
    pub rows_available: usize,
    /// Number of active image pixels in the reconstruction support.
    pub active_voxels: usize,
    /// Model identifier for audit trails.
    pub model_family: &'static str,
    /// Measurement-row policy used for this image.
    pub sampling: ShiftSampling,
    /// Image prior used for this image.
    pub prior: ShiftPrior,
}
