//! RTM struct definition and constructor.
//!
//! # Architecture Note
//!
//! [`ReverseTimeMigration`] does **not** implement the [`Reconstructor`] trait.
//! RTM is a stateful, multi-shot accumulator: each call to [`migrate_shot`] runs
//! full forward and backward finite-difference propagation (see
//! `inherent/propagation.rs`) and accumulates the imaging condition into
//! `self.image`.  The `Reconstructor` contract — `fn reconstruct(&self, …)` — is
//! incompatible with this stateful design for two reasons:
//!
//! 1. RTM requires `&mut self` to accumulate cross-correlation and illumination.
//! 2. The contract accepts sensor data as a flat `Array2<f64>` and a grid, but
//!    RTM internally generates the forward wavefield from the source signature;
//!    pre-computed sensor data alone is insufficient to drive the algorithm.
//!
//! Callers must use the explicit API:
//!
//! ```ignore
//! let mut rtm = ReverseTimeMigration::new(config, velocity_model);
//! for shot in &survey {
//!     rtm.migrate_shot(&shot.data, shot.source_pos, &shot.receiver_positions, &grid)?;
//! }
//! rtm.post_process_image()?;
//! let image: &Array3<f64> = rtm.get_image();
//! ```
//!
//! Reference: Baysal et al. (1983), *Geophysics* **48**(11), 1514–1524.

use ndarray::Array3;

use super::super::config::SeismicImagingConfig;

/// Reverse Time Migration image accumulator.
///
/// Stateful per-survey accumulator.  Initialised once per velocity model;
/// each call to [`migrate_shot`] propagates one source–receiver gather and
/// accumulates the zero-lag cross-correlation (or any of the six supported
/// imaging conditions) into `image` and the source illumination `Φ` into
/// `source_illumination`.  After all shots are processed, call
/// [`post_process_image`] to apply illumination normalisation and optional
/// Laplacian filtering.
///
/// [`migrate_shot`]: ReverseTimeMigration::migrate_shot
/// [`post_process_image`]: ReverseTimeMigration::post_process_image
#[derive(Debug)]
pub struct ReverseTimeMigration {
    pub(super) config: SeismicImagingConfig,
    /// Velocity model used by the 4th-order FD propagator (m/s).
    pub(super) velocity_model: Array3<f64>,
    /// Accumulated imaging-condition image: `I(x) = Σ_{shots} Σ_t S(x,t)·R(x,t)`.
    pub(super) image: Array3<f64>,
    /// Source illumination: `Φ(x) = Σ_{shots} Σ_t S²(x,t)`.
    ///
    /// Used by [`post_process_image`] to normalise `image` by `√Φ`.
    ///
    /// [`post_process_image`]: ReverseTimeMigration::post_process_image
    pub(super) source_illumination: Array3<f64>,
}

impl ReverseTimeMigration {
    /// Construct an RTM accumulator for the given velocity model.
    ///
    /// Both `image` and `source_illumination` are zero-initialised.
    /// Call [`reset`] to reuse the same instance for a new survey.
    ///
    /// [`reset`]: ReverseTimeMigration::reset
    #[must_use]
    pub fn new(config: SeismicImagingConfig, velocity_model: Array3<f64>) -> Self {
        let image = Array3::zeros(velocity_model.dim());
        let source_illumination = Array3::zeros(velocity_model.dim());

        Self {
            config,
            velocity_model,
            image,
            source_illumination,
        }
    }

    /// Return a shared reference to the current accumulated image.
    #[must_use]
    pub fn get_image(&self) -> &Array3<f64> {
        &self.image
    }

    /// Reset all accumulators to zero so the instance can be reused for a new
    /// survey without re-allocating.
    pub fn reset(&mut self) {
        self.image.fill(0.0);
        self.source_illumination.fill(0.0);
    }
}
