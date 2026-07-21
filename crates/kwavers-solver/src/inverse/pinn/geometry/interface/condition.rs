//! Interface-condition specification.

use std::sync::Arc;

// `dyn Fn` is confined to the cold user-extension boundary where the callback
// type is not known to the library at compile time.
type InterfaceResidualFn =
    Arc<dyn Fn(&[f64], &[f64], &[[f64; 2]; 2], &[[f64; 2]; 2], &[f64]) -> f64 + Send + Sync>;

/// Condition imposed between adjacent PINN material regions.
#[non_exhaustive]
#[derive(Clone)]
pub enum PinnGeometryInterfaceCondition {
    /// Continuity of displacement and traction.
    ElasticContinuity,
    /// Welded contact, equivalent to elastic continuity.
    WeldedContact,
    /// Tangential slip with continuous normal stress.
    SlidingContact,
    /// Zero traction against vacuum or air.
    FreeBoundary,
    /// Coupled fluid-solid condition.
    AcousticElastic {
        /// Fluid density in kilograms per cubic metre.
        fluid_density: f64,
    },
    /// User-defined cold-boundary residual.
    Custom {
        /// Thread-safe residual function.
        residual_fn: InterfaceResidualFn,
    },
}

impl std::fmt::Debug for PinnGeometryInterfaceCondition {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElasticContinuity => formatter.write_str("ElasticContinuity"),
            Self::WeldedContact => formatter.write_str("WeldedContact"),
            Self::SlidingContact => formatter.write_str("SlidingContact"),
            Self::FreeBoundary => formatter.write_str("FreeBoundary"),
            Self::AcousticElastic { fluid_density } => {
                write!(formatter, "AcousticElastic(ρ={fluid_density})")
            }
            Self::Custom { .. } => formatter.write_str("Custom"),
        }
    }
}
