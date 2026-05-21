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

use ndarray::{Array2, Array3};
use num_complex::Complex64;

use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;

use super::cbs::{AbsorbingBoundary, CbsConfig, GreenOperatorKind, GridSpec};
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
    fn cbs_descriptor(&self) -> Option<(CbsConfig, GreenOperatorKind)> {
        None
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

    fn cbs_descriptor(&self) -> Option<(CbsConfig, GreenOperatorKind)> {
        Some((self.cbs_config(), GreenOperatorKind::DenseFreeSpace))
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

    fn cbs_descriptor(&self) -> Option<(CbsConfig, GreenOperatorKind)> {
        Some((
            self.cbs_config(),
            GreenOperatorKind::SpectralPeriodic {
                absorbing_boundary: self.absorbing_boundary,
            },
        ))
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

        assert_eq!(born.model_id(), "single_scatter_born");
        assert_eq!(dense.model_id(), "dense_convergent_born");
        assert_eq!(spectral.model_id(), "spectral_convergent_born");
    }

    #[test]
    fn cbs_descriptor_present_only_for_cbs_impls() {
        assert!(SingleScatterBornOperator.cbs_descriptor().is_none());
        assert!(DenseConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
        }
        .cbs_descriptor()
        .is_some());
        assert!(SpectralConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-6,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }
        .cbs_descriptor()
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
    }
}
