//! The validation primitives: a [`Check`] is one measured quantity against a limit with the
//! direction of "good" and a signed margin; a [`PhysicsReport`] aggregates a set of checks and gates
//! on `all_pass`.

/// One validated quantity: a measured `value` against a `limit`, with the direction of "good".
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// What is being checked.
    pub name: &'static str,
    /// Measured value.
    pub value: f64,
    /// The limit it is compared against.
    pub limit: f64,
    /// Unit string (for reporting).
    pub unit: &'static str,
    /// Whether the check passes.
    pub pass: bool,
    /// Signed headroom in `unit` (positive = margin to spare, negative = over by that much).
    pub margin: f64,
}

impl Check {
    /// A check that passes when `value ≤ limit` (an upper bound, e.g. temperature rise, IR drop).
    #[must_use]
    pub fn upper(name: &'static str, value: f64, limit: f64, unit: &'static str) -> Self {
        Check {
            name,
            value,
            limit,
            unit,
            pass: value <= limit,
            margin: limit - value,
        }
    }

    /// A check that passes when `value ≥ limit` (a lower bound, e.g. clearance, track width).
    #[must_use]
    pub fn lower(name: &'static str, value: f64, limit: f64, unit: &'static str) -> Self {
        Check {
            name,
            value,
            limit,
            unit,
            pass: value >= limit,
            margin: value - limit,
        }
    }
}

/// A whole-design physics report.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsReport {
    /// The individual checks.
    pub checks: Vec<Check>,
    /// True iff every check passes.
    pub all_pass: bool,
}

impl PhysicsReport {
    /// Build a report from a set of checks.
    #[must_use]
    pub fn new(checks: Vec<Check>) -> Self {
        let all_pass = checks.iter().all(|c| c.pass);
        PhysicsReport { checks, all_pass }
    }
}
