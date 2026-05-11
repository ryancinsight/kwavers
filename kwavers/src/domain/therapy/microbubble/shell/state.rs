use std::fmt;

/// Shell mechanical state according to Marmottant model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellState {
    /// Shell is buckled/compressed (R < R_buckling)
    Buckled,
    /// Shell is in elastic regime (R_buckling ≤ R ≤ R_rupture)
    Elastic,
    /// Shell has ruptured (R > R_rupture)
    Ruptured,
}

impl fmt::Display for ShellState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buckled => write!(f, "Buckled"),
            Self::Elastic => write!(f, "Elastic"),
            Self::Ruptured => write!(f, "Ruptured"),
        }
    }
}
