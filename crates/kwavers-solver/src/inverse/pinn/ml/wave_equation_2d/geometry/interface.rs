//! Interface conditions between regions in multi-region PINN domains.

/// User-defined interface condition: `f(x, y, (u₁, ∂u₁/∂n), (u₂, ∂u₂/∂n)) -> residual`.
type InterfaceConditionFn = Box<dyn Fn(f64, f64, (f64, f64), (f64, f64)) -> f64 + Send + Sync>;

/// Interface conditions between regions in multi-region domains.
pub enum WaveInterfaceCondition2D {
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

impl std::fmt::Debug for WaveInterfaceCondition2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaveInterfaceCondition2D::Continuity => write!(f, "Continuity"),
            WaveInterfaceCondition2D::SolutionContinuity => write!(f, "SolutionContinuity"),
            WaveInterfaceCondition2D::AcousticInterface { c1, c2 } => {
                write!(f, "AcousticInterface(c1={}, c2={})", c1, c2)
            }
            WaveInterfaceCondition2D::Custom { .. } => {
                write!(f, "Custom{{condition: <function>}}")
            }
        }
    }
}
