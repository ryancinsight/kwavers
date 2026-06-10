//! Interface conditions between regions in multi-region PINN domains.

/// User-defined interface condition: `f(x, y, (u₁, ∂u₁/∂n), (u₂, ∂u₂/∂n)) -> residual`.
type InterfaceConditionFn = Box<dyn Fn(f64, f64, (f64, f64), (f64, f64)) -> f64 + Send + Sync>;

/// Interface conditions between regions in multi-region domains.
pub enum BurnWave2dInterfaceCondition {
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
        condition: InterfaceConditionFn,
    },
}

impl std::fmt::Debug for BurnWave2dInterfaceCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BurnWave2dInterfaceCondition::Continuity => write!(f, "Continuity"),
            BurnWave2dInterfaceCondition::SolutionContinuity => write!(f, "SolutionContinuity"),
            BurnWave2dInterfaceCondition::AcousticInterface { c1, c2 } => {
                write!(f, "AcousticInterface(c1={}, c2={})", c1, c2)
            }
            BurnWave2dInterfaceCondition::Custom { .. } => {
                write!(f, "Custom{{condition: <function>}}")
            }
        }
    }
}
