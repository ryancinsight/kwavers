//! TherapyAction type for safety-driven therapy control.

/// Action to take based on safety status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyAction {
    /// Continue therapy normally
    Continue,
    /// Warning: approaching limit, recommend monitoring
    Warning,
    /// Reduce acoustic power to 50% of current
    ReducePower,
    /// Immediately stop therapy (limit exceeded)
    Stop,
}

impl TherapyAction {
    /// Get human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            TherapyAction::Continue => "Therapy safe to continue",
            TherapyAction::Warning => "Approaching safety limit - monitoring recommended",
            TherapyAction::ReducePower => "Safety margin exceeded - reducing power",
            TherapyAction::Stop => "Safety limit exceeded - stopping therapy",
        }
    }

    /// Get priority (higher = more urgent).
    pub fn priority(&self) -> u8 {
        match self {
            TherapyAction::Continue => 0,
            TherapyAction::Warning => 1,
            TherapyAction::ReducePower => 2,
            TherapyAction::Stop => 3,
        }
    }
}
