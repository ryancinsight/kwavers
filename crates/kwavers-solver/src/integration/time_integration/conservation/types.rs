//! Conservation quantity types and data structures.

/// Conservation quantities for monitoring
#[derive(Debug, Clone)]
pub struct ConservedQuantities {
    /// Total mass
    pub mass: f64,
    /// Total momentum (x, y, z components)
    pub momentum: (f64, f64, f64),
    /// Total energy
    pub energy: f64,
    /// Total angular momentum
    pub angular_momentum: (f64, f64, f64),
}

/// History of conserved quantities
#[derive(Debug, Clone)]
pub struct ConservationHistory {
    /// Time points
    pub times: Vec<f64>,
    /// Conserved quantities at each time
    pub quantities: Vec<ConservedQuantities>,
}

impl Default for ConservationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationHistory {
    /// Create new empty history
    #[must_use]
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            quantities: Vec::new(),
        }
    }

    /// Add a new entry
    pub fn push(&mut self, time: f64, quantities: ConservedQuantities) {
        self.times.push(time);
        self.quantities.push(quantities);
    }
}

/// Conservation error at a time step
#[derive(Debug, Clone)]
pub struct ConservationError {
    /// Time at which error was measured
    pub time: f64,
    /// Relative mass error
    pub mass_error: f64,
    /// Relative momentum error
    pub momentum_error: f64,
    /// Relative energy error
    pub energy_error: f64,
    /// Relative angular momentum error
    pub angular_momentum_error: f64,
}

impl ConservationError {
    /// Get the maximum error across all conserved quantities
    #[must_use]
    pub fn max_error(&self) -> f64 {
        self.mass_error
            .max(self.momentum_error)
            .max(self.energy_error)
            .max(self.angular_momentum_error)
    }
}
