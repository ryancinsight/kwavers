use crate::solver::inverse::pinn::ml::physics::BoundaryPosition;

/// Electromagnetic problem type
#[derive(Debug, Clone, PartialEq)]
pub enum EMProblemType {
    /// Electrostatics (time-independent E field)
    Electrostatic,
    /// Magnetostatics (time-independent B field)
    Magnetostatic,
    /// Quasi-static electromagnetics (low frequency approximation)
    QuasiStatic,
    /// Full wave propagation (time-dependent Maxwell's equations)
    WavePropagation,
}

/// Electromagnetic boundary condition specification
#[derive(Debug, Clone)]
pub enum ElectromagneticBoundarySpec {
    /// Perfect electric conductor (PEC): E_tangential = 0
    PerfectElectricConductor {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Perfect magnetic conductor (PMC): H_tangential = 0
    PerfectMagneticConductor {
        /// Boundary position
        position: BoundaryPosition,
    },
    /// Impedance boundary: Z E_tangential = -η H_normal
    ImpedanceBoundary {
        /// Boundary position
        position: BoundaryPosition,
        /// Surface impedance (Ω)
        impedance: f64,
    },
    /// Port boundary for waveguide analysis
    Port {
        /// Boundary position
        position: BoundaryPosition,
        /// Port impedance (Ω)
        port_impedance: f64,
        /// Incident mode specification
        mode: usize,
    },
}
