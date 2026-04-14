/// Typed recovery action for new recovery call paths.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Caller should fall back to a CPU implementation.
    CpuFallback,
    /// Caller should reduce its timestep.
    ReduceTimestep { factor: f64 },
    /// Caller should switch to a named solver implementation.
    SwitchSolver { target: &'static str },
    /// Recovery completed without an additional caller-side action.
    NoOp,
}
