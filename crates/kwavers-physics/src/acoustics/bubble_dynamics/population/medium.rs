/// Medium properties the population operators need to tie radii to
/// gas content through the equilibrium map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationMedium {
    /// Ambient pressure `p0` [Pa].
    pub ambient_pressure: f64,
    /// Surface tension `σ` [N/m].
    pub surface_tension: f64,
    /// Temperature `T` [K].
    pub temperature: f64,
}

impl PopulationMedium {
    /// Gas content [molecules] of an equilibrium bubble of the
    /// given radius in this medium.
    #[must_use]
    pub fn gas_at_radius(&self, radius: f64) -> f64 {
        crate::acoustics::mechanics::cavitation::core::bader_influx::equilibrium_gas_content(
            radius,
            self.ambient_pressure,
            self.surface_tension,
            self.temperature,
        )
    }

    /// Equilibrium radius [m] of a bubble holding `gas_molecules`
    /// in this medium.
    #[must_use]
    pub fn radius_at_gas(&self, gas_molecules: f64) -> f64 {
        crate::acoustics::mechanics::cavitation::core::bader_influx::residual_radius_equilibrium(
            gas_molecules,
            self.ambient_pressure,
            self.surface_tension,
            self.temperature,
        )
    }
}
