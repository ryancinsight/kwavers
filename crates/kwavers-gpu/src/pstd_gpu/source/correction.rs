//! K-space-correction gate shared by the GPU PSTD source schedules.
//!
//! Both the pressure and the x-velocity schedule route their source mode
//! through this one gate so an unsupported mode is rejected identically on
//! both paths.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_source::SourceMode;

/// Select the matching shader source term for one source mode.
///
/// Both schedules route through this one gate so an unsupported mode is
/// rejected identically on the pressure and velocity paths.
pub(crate) fn source_mode_uses_kspace_correction(
    mode: SourceMode,
    source_kind: &str,
) -> KwaversResult<bool> {
    match mode {
        SourceMode::Additive => Ok(true),
        SourceMode::AdditiveNoCorrection => Ok(false),
        SourceMode::Dirichlet => Err(KwaversError::InvalidInput(format!(
            "GPU PSTD does not support Dirichlet {source_kind} sources"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::source_mode_uses_kspace_correction;
    use kwavers_source::SourceMode;

    #[test]
    fn source_mode_selects_the_physical_correction_and_rejects_dirichlet() {
        assert!(
            source_mode_uses_kspace_correction(SourceMode::Additive, "pressure")
                .expect("additive pressure source is supported")
        );
        assert!(
            !source_mode_uses_kspace_correction(SourceMode::AdditiveNoCorrection, "velocity")
                .expect("uncorrected velocity source is supported")
        );

        let error = source_mode_uses_kspace_correction(SourceMode::Dirichlet, "pressure")
            .expect_err("Dirichlet pressure source must be rejected");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD does not support Dirichlet pressure sources"
        );
    }
}
