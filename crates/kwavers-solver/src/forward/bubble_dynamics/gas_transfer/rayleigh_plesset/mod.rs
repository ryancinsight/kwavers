//! Rayleigh–Plesset coupled to the transient conservative diffusion update.

use super::{GasTransfer, GasTransferError, TransientGasDiffusion};
use aequitas::systems::si::quantities::{
    AmountOfSubstance, Length, MolarConcentration, Pressure, Time, Velocity,
};
use eunomia::RealField;
use kwavers_physics::acoustics::bubble_dynamics::{
    gas_transfer::{GasDiffusionWall, SphericalGasDiffusion},
    rayleigh_plesset::{PolytropicGas, PolytropicParameters},
};
use thiserror::Error;

/// Initialization or step failure from a diffusive bubble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DiffusiveBubbleError {
    /// Invalid physical data, pressure sample, time interval, or trial gas state.
    #[error("invalid or unrepresentable diffusive bubble {0}")]
    InvalidState(&'static str),
    /// The trial did not converge. The caller can retry with a smaller step.
    #[error("diffusive Rayleigh–Plesset step did not converge")]
    NoConvergence,
    /// The transient liquid diffusion solve failed.
    #[error(transparent)]
    Diffusion(#[from] GasTransferError),
}

/// Accepted spherical wall state and noncondensable gas inventory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffusiveBubbleState<T = f64> {
    /// Instantaneous radius.
    pub radius: Length<T>,
    /// Signed radial velocity, positive toward the liquid.
    pub wall_velocity: Velocity<T>,
    /// Noncondensable gas inventory in moles.
    pub gas_amount: AmountOfSubstance<T>,
}

#[derive(Clone, Copy)]
struct Position<T> {
    radius: T,
    velocity: T,
    // Store the change separately to preserve transfers below one ULP of n₀.
    gas_change: T,
}

/// An isolated spherical bubble with single-species, reversible Henry transfer.
///
/// The gas density sets the polytropic noncondensable pressure used by both
/// Rayleigh–Plesset and the liquid boundary. Keep stepping through acoustic
/// off periods to retain the evolving liquid concentration and gas inventory.
pub struct DiffusiveBubble<T = f64> {
    gas: PolytropicGas<T>,
    diffusion: TransientGasDiffusion<T>,
    henry: T,
    reference_radius: T,
    reference_amount: T,
    velocity_scale: T,
    position: Position<T>,
}

impl<T: RealField> DiffusiveBubble<T> {
    /// Start with a mechanically equilibrated bubble in initially uniform liquid.
    ///
    /// `saturation_concentration` is the dissolved concentration in equilibrium
    /// with a flat interface at the ambient noncondensable partial pressure.
    /// `problem.bulk_concentration` and `reservoir_extent` set the liquid's
    /// initial and far-field concentration and initial wall-to-reservoir distance.
    ///
    /// # Errors
    /// Rejects invalid equilibrium, liquid, or Henry parameters or a reference
    /// state unrepresentable in the selected precision.
    pub fn new(
        mechanics: PolytropicParameters<T>,
        problem: SphericalGasDiffusion<T>,
        saturation_concentration: MolarConcentration<T>,
        reservoir_extent: Length<T>,
        cells: usize,
    ) -> Result<Self, DiffusiveBubbleError> {
        let gas = PolytropicGas::new(problem.reference_radius, mechanics)
            .ok_or(DiffusiveBubbleError::InvalidState("reference state"))?;
        let saturation = saturation_concentration.into_base();
        let henry = saturation / gas.saturation_pressure().into_base();
        let velocity_scale =
            (mechanics.ambient_pressure.into_base() / mechanics.liquid_density.into_base()).sqrt();
        if !henry.is_finite()
            || henry < T::ZERO
            || (henry == T::ZERO && saturation != T::ZERO)
            || !velocity_scale.is_finite()
            || velocity_scale <= T::ZERO
        {
            return Err(DiffusiveBubbleError::InvalidState("reference scales"));
        }
        let pressure = gas
            .gas_pressure(problem.reference_radius, T::ONE)
            .ok_or(DiffusiveBubbleError::InvalidState("reference pressure"))?;
        let reference_radius = problem.reference_radius.into_base();
        Ok(Self {
            gas,
            diffusion: TransientGasDiffusion::new(
                problem,
                reservoir_extent,
                cells,
                MolarConcentration::from_base(henry * pressure.into_base()),
            )?,
            henry,
            reference_radius,
            reference_amount: gas.reference_amount().into_base(),
            velocity_scale,
            position: Position {
                radius: reference_radius,
                velocity: T::ZERO,
                gas_change: T::ZERO,
            },
        })
    }

    /// Most recent accepted wall and gas state.
    #[must_use]
    pub fn state(&self) -> DiffusiveBubbleState<T> {
        DiffusiveBubbleState {
            radius: Length::from_base(self.position.radius),
            wall_velocity: Velocity::from_base(self.position.velocity),
            gas_amount: AmountOfSubstance::from_base(
                self.reference_amount + self.position.gas_change,
            ),
        }
    }

    /// Mechanical equilibrium radius at ambient pressure for the accepted inventory.
    ///
    /// The current radius and liquid concentration are unaltered. The result
    /// uses the same density-based gas closure as the wall acceleration.
    ///
    /// # Errors
    /// Returns an error if the equilibrium radius is not representable.
    pub fn equilibrium_radius(&self) -> Result<Length<T>, DiffusiveBubbleError> {
        self.gas
            .equilibrium_radius(self.state().gas_amount)
            .ok_or(DiffusiveBubbleError::InvalidState("equilibrium radius"))
    }

    /// Signed change from the initial gas amount, retaining small transfers.
    #[must_use]
    pub fn gas_amount_change(&self) -> AmountOfSubstance<T> {
        AmountOfSubstance::from_base(self.position.gas_change)
    }

    /// Current liquid inventory in excess of the initial bulk concentration.
    #[must_use]
    pub fn liquid_excess(&self) -> AmountOfSubstance<T> {
        self.diffusion.liquid_excess()
    }

    /// Advance one interval using the instantaneous absolute far-field pressure
    /// at its midpoint. Supply ambient pressure during an acoustic off interval.
    ///
    /// Implicit midpoint advances the wall. The previously accepted liquid
    /// field supplies each diffusion trial. Commit the endpoint and exactly
    /// its accepted flux together once the scaled endpoint residual is at
    /// floating-point accuracy.
    ///
    /// # Errors
    /// Rejects invalid inputs, inadmissible trial states, or a step failing to
    /// converge. A failed step preserves the bubble state and the liquid history.
    pub fn advance(
        &mut self,
        interval: Time<T>,
        far_field_midpoint: Pressure<T>,
    ) -> Result<GasTransfer<T>, DiffusiveBubbleError> {
        let dt = interval.into_base();
        if !dt.is_finite() || dt <= T::ZERO || !far_field_midpoint.into_base().is_finite() {
            return Err(DiffusiveBubbleError::InvalidState("step inputs"));
        }
        let old = self.position;
        let mut endpoint = old;
        // Allow rounding in the pressure closure, norms and midpoint sums.
        let tolerance = T::from_count(16) * T::EPSILON;
        for _ in 0..24 {
            // Converge the inexpensive wall equations at the trial gas amount
            // before revisiting the radial field.
            endpoint = self.mechanical_endpoint(dt, far_field_midpoint, tolerance, endpoint)?;
            let radius = Length::from_base(endpoint.radius);
            let pressure = self
                .gas
                .gas_pressure(radius, T::ONE + endpoint.gas_change / self.reference_amount)
                .ok_or(DiffusiveBubbleError::InvalidState("endpoint gas"))?;
            let trial = self.diffusion.trial(
                interval,
                GasDiffusionWall {
                    radius,
                    concentration: MolarConcentration::from_base(self.henry * pressure.into_base()),
                },
            )?;
            let gas_change = old.gas_change + trial.exchange().into_bubble.into_base();
            let amount = self.reference_amount + gas_change;
            if !amount.is_finite() || amount <= T::ZERO {
                return Err(DiffusiveBubbleError::InvalidState("trial gas amount"));
            }
            if (gas_change - endpoint.gas_change).abs() / self.reference_amount <= tolerance {
                self.position = Position {
                    gas_change,
                    ..endpoint
                };
                return Ok(trial.commit());
            }
            endpoint.gas_change = gas_change;
        }
        Err(DiffusiveBubbleError::NoConvergence)
    }

    fn mechanical_endpoint(
        &self,
        dt: T,
        far_field_midpoint: Pressure<T>,
        tolerance: T,
        mut endpoint: Position<T>,
    ) -> Result<Position<T>, DiffusiveBubbleError> {
        let old = self.position;
        let half = T::from_f64(0.5);
        let n_mid = (old.gas_change * half + endpoint.gas_change * half) / self.reference_amount;
        for _ in 0..24 {
            let r_mid = Length::from_base(old.radius * half + endpoint.radius * half);
            let v_mid = Velocity::from_base(old.velocity * half + endpoint.velocity * half);
            let gas_mid = self
                .gas
                .gas_pressure(r_mid, T::ONE + n_mid)
                .ok_or(DiffusiveBubbleError::InvalidState("midpoint gas"))?;
            let acceleration = self
                .gas
                .acceleration(r_mid, v_mid, gas_mid, far_field_midpoint);
            let radius = old.radius + dt * v_mid.into_base();
            let velocity = old.velocity + dt * acceleration.into_base();
            if !radius.is_finite() || radius <= T::ZERO || !velocity.is_finite() {
                return Err(DiffusiveBubbleError::InvalidState("trial wall"));
            }
            let converged = (radius - endpoint.radius).abs() / self.reference_radius <= tolerance
                && (velocity - endpoint.velocity).abs() / self.velocity_scale <= tolerance;
            endpoint.radius = radius;
            endpoint.velocity = velocity;
            if converged {
                return Ok(endpoint);
            }
        }
        Err(DiffusiveBubbleError::NoConvergence)
    }
}
