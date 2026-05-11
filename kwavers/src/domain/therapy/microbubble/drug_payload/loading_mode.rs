//! Drug loading mode enumeration.

use std::fmt;

/// Drug loading configuration
///
/// Describes how drug is loaded into/onto the microbubble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrugLoadingMode {
    /// Drug on shell surface (easy release)
    SurfaceAttached,
    /// Drug in shell lipid bilayer (moderate release)
    ShellEmbedded,
    /// Drug in bubble core (slow release, burst on rupture)
    CoreEncapsulated,
}

impl fmt::Display for DrugLoadingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SurfaceAttached => write!(f, "Surface"),
            Self::ShellEmbedded => write!(f, "Shell"),
            Self::CoreEncapsulated => write!(f, "Core"),
        }
    }
}
