use crate::analysis::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use crate::core::error::{KwaversError, KwaversResult, NumericalError};

use super::MUSICProcessor;

impl LocalizationProcessor for MUSICProcessor {
    fn localize(
        &self,
        _time_delays: &[f64],
        _sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        Err(KwaversError::Numerical(NumericalError::NotImplemented {
            feature: "MUSIC via time delays (use MUSICProcessor::run with snapshots instead)"
                .to_string(),
        }))
    }

    fn name(&self) -> &str {
        "MUSIC"
    }
}
