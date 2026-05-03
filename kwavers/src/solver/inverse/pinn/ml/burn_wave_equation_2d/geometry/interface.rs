//! Interface conditions between regions in multi-region PINN domains.

/// Interface conditions between regions in multi-region domains.
pub enum InterfaceCondition {
    /// Continuity of solution and normal derivative (u and ∂u/∂n continuous).
    Continuity,
    /// Continuity of solution only (u continuous, ∂u/∂n discontinuous).
    SolutionContinuity,
    /// Acoustic interface: continuity of pressure and normal velocity.
    AcousticInterface {
        /// Region 1 wave speed.
        c1: f64,
        /// Region 2 wave speed.
        c2: f64,
    },
    /// Custom interface condition with user-defined function.
    Custom {
        /// Boundary condition function.
        condition: Box<dyn Fn(f64, f64, (f64, f64), (f64, f64)) -> f64 + Send + Sync>,
    },
}

impl std::fmt::Debug for InterfaceCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterfaceCondition::Continuity => write!(f, "Continuity"),
            InterfaceCondition::SolutionContinuity => write!(f, "SolutionContinuity"),
            InterfaceCondition::AcousticInterface { c1, c2 } => {
                write!(f, "AcousticInterface(c1={}, c2={})", c1, c2)
            }
            InterfaceCondition::Custom { .. } => write!(f, "Custom{{condition: <function>}}"),
        }
    }
}
