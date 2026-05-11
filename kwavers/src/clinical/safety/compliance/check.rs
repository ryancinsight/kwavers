use super::{ComplianceCheck, ComplianceStatus};

impl ComplianceCheck {
    #[must_use] 
    pub fn new(
        name: String,
        measured: f64,
        limit: f64,
        unit: String,
        warning_threshold: f64,
    ) -> Self {
        let status = if measured > limit {
            ComplianceStatus::NonCompliant
        } else if measured > warning_threshold {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Compliant
        };

        Self {
            name,
            measured,
            limit,
            unit,
            status,
            warning_threshold,
        }
    }

    #[must_use] 
    pub fn percent_of_limit(&self) -> f64 {
        (self.measured / self.limit) * 100.0
    }

    #[must_use] 
    pub fn margin_to_limit(&self) -> f64 {
        100.0 - self.percent_of_limit()
    }
}
