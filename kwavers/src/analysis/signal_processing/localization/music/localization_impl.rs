use crate::analysis::signal_processing::localization::tdoa::{
    TDOAConfig, TDOAProcessor, TimeDelayMethod,
};
use crate::analysis::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use crate::core::error::{KwaversError, KwaversResult};

use super::MUSICProcessor;

impl LocalizationProcessor for MUSICProcessor {
    /// Localize from time-delay measurements through the shared TDOA contract.
    ///
    /// MUSIC itself requires complex array snapshots and covariance
    /// eigenspaces, which are available through [`MUSICProcessor::run`]. The
    /// trait boundary supplies only arrival/time-delay data, so the physically
    /// defined operation is the hyperbolic TDOA least-squares solve:
    ///
    /// ```text
    /// r_i(x) = ||x - s_i||/c - t_i,     minimize Σ_i r_i(x)^2.
    /// ```
    ///
    /// This preserves the trait API without fabricating covariance snapshots or
    /// returning a placeholder error for valid time-delay localization input.
    fn localize(
        &self,
        time_delays: &[f64],
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        if sensor_positions.len() != time_delays.len() {
            return Err(KwaversError::InvalidInput(format!(
                "MUSIC time-delay adapter requires one arrival time per sensor; got {} delays and {} sensors",
                time_delays.len(),
                sensor_positions.len()
            )));
        }

        let mut localization_config = self.config.config.clone();
        localization_config.sensor_positions = sensor_positions.to_vec();

        let tdoa_config = TDOAConfig::new(localization_config, TimeDelayMethod::CrossCorrelation)
            .with_refinement_iterations(32)
            .with_convergence_tolerance(1e-12);
        TDOAProcessor::new(&tdoa_config)?.localize(time_delays, sensor_positions)
    }

    fn name(&self) -> &str {
        "MUSIC"
    }
}
