use super::ConservationTolerances;

impl Default for ConservationTolerances {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            check_interval: 100,
        }
    }
}

impl ConservationTolerances {
    pub fn strict() -> Self {
        Self {
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            check_interval: 10,
        }
    }

    pub fn relaxed() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-4,
            check_interval: 1000,
        }
    }
}
