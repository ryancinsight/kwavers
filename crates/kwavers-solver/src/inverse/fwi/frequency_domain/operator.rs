//! Helmholtz forward-operator abstraction for frequency-domain FWI.
//!
//! # Why a trait
//!
//! The inversion loop is structurally independent of which Helmholtz forward
//! solver maps a slowness volume to predicted receiver pressure. Today three
//! impls exist: single-scatter (first) Born, dense convergent Born series,
//! and spectral (periodic FFT) convergent Born series. Future impls — e.g.
//! BiCGSTAB-preconditioned Helmholtz on a Cartesian grid, MUMPS sparse-direct
//! Helmholtz, FEM Helmholtz on a tetrahedral mesh — slot in by implementing
//! [`HelmholtzForwardOperator`]; the inversion loop and gradient kernel do
//! not change. This mirrors how the time-domain `Solver` trait
//! (`solver::interface::Solver`) lets `SimulationSolverFactory` dispatch
//! FDTD / PSTD / Hybrid behind one boxed-trait surface.
//!
//! # T17a scope (this commit)
//!
//! Trait + three impls landed as a new module. [`Config::propagation_model`]
//! (enum) is preserved temporarily so the existing match-based dispatch in
//! [`super::forward`] and [`super::gradient`] continues to compile unchanged.
//! T17b will replace the enum field with `forward_operator: Arc<dyn
//! HelmholtzForwardOperator>` and convert the match blocks to virtual
//! dispatch, removing the enum.

use std::fmt::Debug;

use leto::{Array2, Array3};
use kwavers_math::fft::Complex64;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;

use super::cbs::{
    AbsorbingBoundary, CbsConfig, GreenOperatorKind, GridSpec, PstdTemporalTransferConfig,
};
use super::Config;

/// Helmholtz forward operator: predicts complex receiver pressure rows for one
/// frequency given a slowness model and a multi-row ring geometry.
///
/// Implementations are stateless value types stored on [`Config`] (eventually
/// boxed); they hold only configuration, not workspace.
pub trait HelmholtzForwardOperator: Debug + Send + Sync {
    /// Predict complex receiver pressure for `transmissions` cylindrical-wave
    /// transmits at `frequency_hz`.
    ///
    /// # Errors
    /// Returns an error when grid, geometry, frequency, or config values
    /// violate the discrete forward contract.
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>>;

    /// Whether this operator drives the volume-field adjoint gradient kernel
    /// (CBS-style) instead of the explicit single-scatter sensitivity formula.
    /// Used by [`super::gradient`] to pick the matching gradient accumulator.
    fn uses_volume_field_adjoint(&self) -> bool;

    /// CBS solver descriptor — `Some((cbs_config, kernel))` for CBS impls,
    /// `None` otherwise. The volume-field adjoint gradient needs these to
    /// reconstruct the matching forward and adjoint solves.
    fn cbs_descriptor(
        &self,
        _config: &Config,
        _frequency_hz: f64,
    ) -> KwaversResult<Option<(CbsConfig, GreenOperatorKind)>> {
        Ok(None)
    }

    /// Validate operator-specific configuration (e.g. CBS iteration count,
    /// tolerance, absorbing-boundary parameters). Default: no-op.
    ///
    /// # Errors
    /// Returns an error if any operator-level invariant is violated.
    fn validate(&self) -> KwaversResult<()> {
        Ok(())
    }

    /// Validate operator-specific invariants that depend on the inversion
    /// grid (e.g. spectral CBS absorbing-layer thickness vs grid size).
    /// Default: no-op.
    ///
    /// # Errors
    /// Returns an error if any operator-grid invariant is violated.
    fn validate_for_grid(&self, _grid: GridSpec) -> KwaversResult<()> {
        Ok(())
    }

    /// Identifier for audit trails (`"single_scatter_born"`,
    /// `"dense_convergent_born"`, `"spectral_convergent_born"`).
    fn model_id(&self) -> &'static str;

    /// Whether this operator drives the finite-window PSTD Born adjoint
    /// gradient kernel instead of the CBS or explicit sensitivity paths.
    /// Default: false.
    fn uses_finite_window_adjoint(&self) -> bool {
        false
    }

    /// Finite-window Born configuration, used by the finite-window adjoint
    /// gradient kernel.  Returns `None` for non-finite-window operators.
    fn finite_window_adjoint_config(
        &self,
    ) -> Option<super::finite_window::PstdFiniteWindowBornConfig> {
        None
    }
}

/// First Born approximation. Forward operator is the discrete single-scatter
/// Helmholtz sensitivity used by the analytical gradient in
/// [`super::gradient`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SingleScatterBornOperator;

impl HelmholtzForwardOperator for SingleScatterBornOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::forward::predict_born_rows(
            slowness_s_per_m,
            array,
            frequency_hz,
            config,
            transmissions,
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        false
    }

    fn model_id(&self) -> &'static str {
        "single_scatter_born"
    }
}

/// Dense convergent Born-series solver (Osnabrugge–Leedumrongwatthanakun–
/// Vellekoop 2016) with the free-space dense Green operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseConvergentBornOperator {
    pub iterations: usize,
    pub relative_tolerance: f64,
}

impl HelmholtzForwardOperator for DenseConvergentBornOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::forward::predict_cbs_rows(
            slowness_s_per_m,
            array,
            frequency_hz,
            config,
            transmissions,
            self.cbs_config(),
            GreenOperatorKind::DenseFreeSpace,
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        true
    }

    fn cbs_descriptor(
        &self,
        _config: &Config,
        _frequency_hz: f64,
    ) -> KwaversResult<Option<(CbsConfig, GreenOperatorKind)>> {
        Ok(Some((self.cbs_config(), GreenOperatorKind::DenseFreeSpace)))
    }

    fn validate(&self) -> KwaversResult<()> {
        validate_cbs_parameters(self.iterations, self.relative_tolerance)
    }

    fn model_id(&self) -> &'static str {
        "dense_convergent_born"
    }
}

impl DenseConvergentBornOperator {
    fn cbs_config(&self) -> CbsConfig {
        CbsConfig {
            max_iterations: self.iterations,
            relative_tolerance: self.relative_tolerance,
        }
    }
}

fn validate_cbs_parameters(iterations: usize, relative_tolerance: f64) -> KwaversResult<()> {
    if iterations == 0 {
        return Err(KwaversError::InvalidInput(
            "CBS iterations must be nonzero".to_owned(),
        ));
    }
    if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS relative tolerance must be positive and finite, got {relative_tolerance}"
        )));
    }
    Ok(())
}

/// Spectral (periodic FFT) convergent Born-series solver with selectable
/// absorbing-boundary treatment for FFT wraparound suppression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralConvergentBornOperator {
    pub iterations: usize,
    pub relative_tolerance: f64,
    pub absorbing_boundary: AbsorbingBoundary,
}

impl HelmholtzForwardOperator for SpectralConvergentBornOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::forward::predict_cbs_rows(
            slowness_s_per_m,
            array,
            frequency_hz,
            config,
            transmissions,
            self.cbs_config(),
            GreenOperatorKind::SpectralPeriodic {
                absorbing_boundary: self.absorbing_boundary,
            },
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        true
    }

    fn cbs_descriptor(
        &self,
        _config: &Config,
        _frequency_hz: f64,
    ) -> KwaversResult<Option<(CbsConfig, GreenOperatorKind)>> {
        Ok(Some((
            self.cbs_config(),
            GreenOperatorKind::SpectralPeriodic {
                absorbing_boundary: self.absorbing_boundary,
            },
        )))
    }

    fn validate(&self) -> KwaversResult<()> {
        validate_cbs_parameters(self.iterations, self.relative_tolerance)?;
        self.absorbing_boundary.validate()
    }

    fn validate_for_grid(&self, grid: GridSpec) -> KwaversResult<()> {
        self.absorbing_boundary.validate_for_grid(grid)
    }

    fn model_id(&self) -> &'static str {
        "spectral_convergent_born"
    }
}

impl SpectralConvergentBornOperator {
    fn cbs_config(&self) -> CbsConfig {
        CbsConfig {
            max_iterations: self.iterations,
            relative_tolerance: self.relative_tolerance,
        }
    }
}

/// Spectral convergent Born-series solver whose periodic Green operator uses
/// the homogeneous PSTD leapfrog/k-space modal symbol instead of the continuous
/// Helmholtz symbol.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdSpectralConvergentBornOperator {
    pub iterations: usize,
    pub relative_tolerance: f64,
    pub time_step_s: f64,
    pub temporal_transfer: Option<PstdTemporalTransferConfig>,
    pub absorbing_boundary: AbsorbingBoundary,
}

impl HelmholtzForwardOperator for PstdSpectralConvergentBornOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::forward::predict_cbs_rows(
            slowness_s_per_m,
            array,
            frequency_hz,
            config,
            transmissions,
            self.cbs_config(),
            self.green_operator(
                config.reference_sound_speed_m_s,
                frequency_hz,
                config.spacing_m,
            )?,
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        true
    }

    fn cbs_descriptor(
        &self,
        config: &Config,
        frequency_hz: f64,
    ) -> KwaversResult<Option<(CbsConfig, GreenOperatorKind)>> {
        Ok(Some((
            self.cbs_config(),
            self.green_operator(
                config.reference_sound_speed_m_s,
                frequency_hz,
                config.spacing_m,
            )?,
        )))
    }

    fn validate(&self) -> KwaversResult<()> {
        validate_cbs_parameters(self.iterations, self.relative_tolerance)?;
        if !self.time_step_s.is_finite() || self.time_step_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PSTD spectral Born time_step_s must be positive and finite, got {}",
                self.time_step_s
            )));
        }
        if let Some(temporal_transfer) = self.temporal_transfer {
            temporal_transfer.validate()?;
        }
        self.absorbing_boundary.validate()
    }

    fn validate_for_grid(&self, grid: GridSpec) -> KwaversResult<()> {
        self.absorbing_boundary.validate_for_grid(grid)
    }

    fn model_id(&self) -> &'static str {
        "pstd_spectral_convergent_born"
    }
}

impl PstdSpectralConvergentBornOperator {
    fn cbs_config(&self) -> CbsConfig {
        CbsConfig {
            max_iterations: self.iterations,
            relative_tolerance: self.relative_tolerance,
        }
    }

    fn green_operator(
        self,
        reference_sound_speed_m_s: f64,
        frequency_hz: f64,
        spacing_m: f64,
    ) -> KwaversResult<GreenOperatorKind> {
        let temporal_transfer = self
            .temporal_transfer
            .map(|transfer| {
                transfer.bin_config(
                    frequency_hz,
                    self.time_step_s,
                    spacing_m,
                    reference_sound_speed_m_s,
                )
            })
            .transpose()?;
        Ok(GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s: self.time_step_s,
            reference_sound_speed_m_s,
            temporal_transfer,
            absorbing_boundary: self.absorbing_boundary,
        })
    }
}

/// Finite-window first-order PSTD Born operator with discrete adjoint gradient.
///
/// Unlike the stationary CBS operators, this operator drives the inversion
/// gradient via time-reversal adjoint of the same finite-window PSTD
/// acquisition used to generate the observations.  This closes the
/// operator-equivalence gap between the forward model and the data when
/// the acquisition uses finite drive cycles with a trailing frequency-bin
/// window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdFiniteWindowBornOperator {
    /// PSTD leapfrog time step [s] — must match the dataset generator.
    pub time_step_s: f64,
    /// Scalar pressure-source amplitude [Pa] — must match the dataset generator.
    pub source_amplitude_pa: f64,
    /// Number of drive cycles — must match the dataset generator.
    pub cycles_per_frequency: usize,
    /// Number of trailing bin cycles — must match the dataset generator.
    pub frequency_bin_cycles: usize,
}

impl HelmholtzForwardOperator for PstdFiniteWindowBornOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::finite_window::simulate_pstd_finite_window_born_observation(
            &kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::slowness_to_sound_speed(slowness_s_per_m)?,
            array,
            frequency_hz,
            self.finite_window_config(config),
            transmissions,
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        false
    }

    fn uses_finite_window_adjoint(&self) -> bool {
        true
    }

    fn finite_window_adjoint_config(
        &self,
    ) -> Option<super::finite_window::PstdFiniteWindowBornConfig> {
        Some(super::finite_window::PstdFiniteWindowBornConfig {
            // reference_sound_speed and spacing are filled in by the gradient
            // accumulator from the inversion Config at call time.
            reference_sound_speed_m_s: 0.0,
            spacing_m: 0.0,
            time_step_s: self.time_step_s,
            source_amplitude_pa: self.source_amplitude_pa,
            cycles_per_frequency: self.cycles_per_frequency,
            frequency_bin_cycles: self.frequency_bin_cycles,
        })
    }

    fn validate(&self) -> KwaversResult<()> {
        if !self.time_step_s.is_finite() || self.time_step_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PstdFiniteWindowBornOperator time_step_s must be positive and finite, got {}",
                self.time_step_s
            )));
        }
        if !self.source_amplitude_pa.is_finite() || self.source_amplitude_pa <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PstdFiniteWindowBornOperator source_amplitude_pa must be positive and finite, got {}",
                self.source_amplitude_pa
            )));
        }
        if self.cycles_per_frequency == 0 {
            return Err(KwaversError::InvalidInput(
                "PstdFiniteWindowBornOperator cycles_per_frequency must be nonzero".to_owned(),
            ));
        }
        if self.frequency_bin_cycles == 0 {
            return Err(KwaversError::InvalidInput(
                "PstdFiniteWindowBornOperator frequency_bin_cycles must be nonzero".to_owned(),
            ));
        }
        Ok(())
    }

    fn model_id(&self) -> &'static str {
        "pstd_finite_window_born"
    }
}

impl PstdFiniteWindowBornOperator {
    /// Build the [`PstdFiniteWindowBornConfig`] by merging operator fields
    /// with the inversion `Config`.
    fn finite_window_config(
        &self,
        config: &Config,
    ) -> super::finite_window::PstdFiniteWindowBornConfig {
        super::finite_window::PstdFiniteWindowBornConfig {
            reference_sound_speed_m_s: config.reference_sound_speed_m_s,
            spacing_m: config.spacing_m,
            time_step_s: self.time_step_s,
            source_amplitude_pa: self.source_amplitude_pa,
            cycles_per_frequency: self.cycles_per_frequency,
            frequency_bin_cycles: self.frequency_bin_cycles,
        }
    }
}

/// Finite-window second-order PSTD Born operator.
///
/// Extends the first-order model `p0 + ps1` with the second-order Born-series
/// correction `ps2` whose source is `-chi * accel(ps1)`.  The forward
/// prediction includes `p0 + ps1 + ps2`, while the adjoint gradient reuses
/// the first-order adjoint applied to the second-order residual — a valid
/// descent direction, not the exact second-order adjoint (which requires
/// additional backward-pass storage for `ps1` acceleration history).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PstdFiniteWindowBornSecondOrderOperator {
    /// PSTD leapfrog time step [s] — must match the dataset generator.
    pub time_step_s: f64,
    /// Scalar pressure-source amplitude [Pa] — must match the dataset generator.
    pub source_amplitude_pa: f64,
    /// Number of drive cycles — must match the dataset generator.
    pub cycles_per_frequency: usize,
    /// Number of trailing bin cycles — must match the dataset generator.
    pub frequency_bin_cycles: usize,
}

impl HelmholtzForwardOperator for PstdFiniteWindowBornSecondOrderOperator {
    fn predict_receiver_rows(
        &self,
        slowness_s_per_m: &Array3<f64>,
        array: &MultiRowRingArray,
        frequency_hz: f64,
        config: &Config,
        transmissions: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        super::finite_window::simulate_pstd_finite_window_born_second_order_observation(
            &kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::slowness_to_sound_speed(slowness_s_per_m)?,
            array,
            frequency_hz,
            self.finite_window_config(config),
            transmissions,
        )
    }

    fn uses_volume_field_adjoint(&self) -> bool {
        false
    }

    fn uses_finite_window_adjoint(&self) -> bool {
        true
    }

    fn finite_window_adjoint_config(
        &self,
    ) -> Option<super::finite_window::PstdFiniteWindowBornConfig> {
        Some(super::finite_window::PstdFiniteWindowBornConfig {
            reference_sound_speed_m_s: 0.0,
            spacing_m: 0.0,
            time_step_s: self.time_step_s,
            source_amplitude_pa: self.source_amplitude_pa,
            cycles_per_frequency: self.cycles_per_frequency,
            frequency_bin_cycles: self.frequency_bin_cycles,
        })
    }

    fn validate(&self) -> KwaversResult<()> {
        if !self.time_step_s.is_finite() || self.time_step_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PstdFiniteWindowBornSecondOrderOperator time_step_s must be positive and finite, got {}",
                self.time_step_s
            )));
        }
        if !self.source_amplitude_pa.is_finite() || self.source_amplitude_pa <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PstdFiniteWindowBornSecondOrderOperator source_amplitude_pa must be positive and finite, got {}",
                self.source_amplitude_pa
            )));
        }
        if self.cycles_per_frequency == 0 {
            return Err(KwaversError::InvalidInput(
                "PstdFiniteWindowBornSecondOrderOperator cycles_per_frequency must be nonzero"
                    .to_owned(),
            ));
        }
        if self.frequency_bin_cycles == 0 {
            return Err(KwaversError::InvalidInput(
                "PstdFiniteWindowBornSecondOrderOperator frequency_bin_cycles must be nonzero"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn model_id(&self) -> &'static str {
        "pstd_finite_window_born_second_order"
    }
}

impl PstdFiniteWindowBornSecondOrderOperator {
    fn finite_window_config(
        &self,
        config: &Config,
    ) -> super::finite_window::PstdFiniteWindowBornConfig {
        super::finite_window::PstdFiniteWindowBornConfig {
            reference_sound_speed_m_s: config.reference_sound_speed_m_s,
            spacing_m: config.spacing_m,
            time_step_s: self.time_step_s,
            source_amplitude_pa: self.source_amplitude_pa,
            cycles_per_frequency: self.cycles_per_frequency,
            frequency_bin_cycles: self.frequency_bin_cycles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_are_distinct() {
        let born = SingleScatterBornOperator;
        let dense = DenseConvergentBornOperator {
            iterations: 16,
            relative_tolerance: 1.0e-8,
        };
        let spectral = SpectralConvergentBornOperator {
            iterations: 16,
            relative_tolerance: 1.0e-8,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        };
        let pstd_spectral = PstdSpectralConvergentBornOperator {
            iterations: 16,
            relative_tolerance: 1.0e-8,
            time_step_s: 1.0e-7,
            temporal_transfer: None,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        };
        let pstd_fw = PstdFiniteWindowBornOperator {
            time_step_s: 1.0e-7,
            source_amplitude_pa: 1.0,
            cycles_per_frequency: 4,
            frequency_bin_cycles: 1,
        };

        assert_eq!(born.model_id(), "single_scatter_born");
        assert_eq!(dense.model_id(), "dense_convergent_born");
        assert_eq!(spectral.model_id(), "spectral_convergent_born");
        assert_eq!(pstd_spectral.model_id(), "pstd_spectral_convergent_born");
        assert_eq!(pstd_fw.model_id(), "pstd_finite_window_born");
        let pstd_fw2 = PstdFiniteWindowBornSecondOrderOperator {
            time_step_s: 1.0e-7,
            source_amplitude_pa: 1.0,
            cycles_per_frequency: 4,
            frequency_bin_cycles: 1,
        };
        assert_eq!(pstd_fw2.model_id(), "pstd_finite_window_born_second_order");
    }

    #[test]
    fn cbs_descriptor_present_only_for_cbs_impls() {
        let config = Config::default();

        assert!(SingleScatterBornOperator
            .cbs_descriptor(&config, 200_000.0)
            .unwrap()
            .is_none());
        assert!(DenseConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
        }
        .cbs_descriptor(&config, 200_000.0)
        .unwrap()
        .is_some());
        assert!(SpectralConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }
        .cbs_descriptor(&config, 200_000.0)
        .unwrap()
        .is_some());
        assert!(PstdSpectralConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
            time_step_s: 1.0e-7,
            temporal_transfer: None,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }
        .cbs_descriptor(&config, 200_000.0)
        .unwrap()
        .is_some());
    }

    #[test]
    fn uses_volume_field_adjoint_matches_paths() {
        assert!(!SingleScatterBornOperator.uses_volume_field_adjoint());
        assert!(DenseConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
        }
        .uses_volume_field_adjoint());
        assert!(SpectralConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }
        .uses_volume_field_adjoint());
        assert!(PstdSpectralConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
            time_step_s: 1.0e-7,
            temporal_transfer: None,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }
        .uses_volume_field_adjoint());
    }
}
