//! Speed-shift reconstruction configuration and model identifiers.

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
    pub(in super::super) fn validate(self) -> Result<(), String> {
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
    pub(in super::super) fn validate(self, spacing_m: f64) -> Result<(), String> {
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
    pub(in super::super) fn accepts(self, row: usize) -> bool {
        match self {
            Self::Dense => true,
            Self::Sparse { stride, offset } => row % stride == offset,
        }
    }

    pub(in super::super) fn validate(self) -> Result<(), String> {
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftPrior {
    /// Dense H1/Tikhonov speed-shift field solved with PCG.
    Dense,
    /// Sparse perturbation field solved with an L1 proximal step (ISTA).
    Sparse,
    /// Tikhonov-regularised least-squares solved with matrix-free LSQR
    /// (Paige & Saunders 1982).  Better conditioned than PCG for overdetermined
    /// systems and does not require a precomputed normal-equation diagonal.
    ///
    /// `damping` is the LSQR λ parameter: minimise ‖Ax − b‖² + λ²‖x‖².
    /// Set `damping = tikhonov_weight.sqrt()` for equivalence with the Dense
    /// Tikhonov regularisation.
    Lsqr { damping: f64 },
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

impl Default for SoundSpeedShiftConfig {
    fn default() -> Self {
        Self {
            reference_sound_speed_m_s: kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE,
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

    pub(in super::super) fn model_family(self) -> &'static str {
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
