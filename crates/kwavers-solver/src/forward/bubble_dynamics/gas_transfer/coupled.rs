//! Radial bubble and gas heat transport coupled to conservative liquid diffusion.

use super::{
    heat::{GasHeat, HeatGeometry},
    BubbleForcing, DiffusiveBubbleModels, GasTransfer, GasTransferError, RadialMechanics,
    TransientGasDiffusion,
};
use aequitas::systems::si::quantities::{
    AmountOfSubstance, Length, MolarConcentration, Pressure, PressureRate, Time, Velocity,
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
    #[error("diffusive bubble step did not converge")]
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
/// The material-shell liquid transport assumes incompressible flow even
/// when Keller–Miksis supplies the wall trajectory. Heat transfer assumes
/// a fixed wall temperature and uniform pressure inside the gas.
///
/// Wall mechanics and the Henry boundary use the same noncondensable
/// pressure. Rayleigh–Plesset with a fixed polytrope is the default. Keep stepping through acoustic
/// off periods to retain the evolving liquid concentration and gas inventory.
pub struct DiffusiveBubble<T = f64> {
    gas: PolytropicGas<T>,
    mechanics: RadialMechanics<T>,
    heat: Option<GasHeat<T>>,
    exponent: T,
    reference_pressure: T,
    accepted_pressure: Pressure<T>,
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
        Self::new_with_models(
            mechanics,
            problem,
            saturation_concentration,
            reservoir_extent,
            cells,
            DiffusiveBubbleModels {
                mechanics: RadialMechanics::RayleighPlesset,
                heat_transfer: None,
            },
        )
    }

    /// Start a spherical bubble with explicit radial mechanics and heat transport choices.
    ///
    /// The liquid uses the same Henry boundary, reservoir and material-volume
    /// diffusion as [Self::new]. With heat transport, the polytropic parameter
    /// must be one, which selects the relaxed isothermal equilibrium closure.
    /// Use [Self::advance_with_forcing] with Keller–Miksis mechanics.
    ///
    /// # Errors
    /// Invalid or unrepresentable physical parameters, grid or closure choices
    /// are rejected before a bubble is initialized.
    pub fn new_with_models(
        mechanics: PolytropicParameters<T>,
        problem: SphericalGasDiffusion<T>,
        saturation_concentration: MolarConcentration<T>,
        reservoir_extent: Length<T>,
        cells: usize,
        models: DiffusiveBubbleModels<T>,
    ) -> Result<Self, DiffusiveBubbleError> {
        if let RadialMechanics::KellerMiksis { liquid_sound_speed } = models.mechanics {
            let c = liquid_sound_speed.into_base();
            if !c.is_finite() || c <= T::ZERO {
                return Err(DiffusiveBubbleError::InvalidState("liquid sound speed"));
            }
        }
        if models.heat_transfer.is_some() && mechanics.exponent != T::ONE {
            return Err(DiffusiveBubbleError::InvalidState("relaxed gas exponent"));
        }
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
        let heat = models
            .heat_transfer
            .map(|config| GasHeat::new(config, reference_radius))
            .transpose()?;
        Ok(Self {
            gas,
            mechanics: models.mechanics,
            heat,
            exponent: mechanics.exponent,
            reference_pressure: pressure.into_base(),
            accepted_pressure: pressure,
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

    /// Accepted noncondensable partial pressure at the gas-liquid interface.
    #[must_use]
    pub fn gas_pressure(&self) -> Pressure<T> {
        self.accepted_pressure
    }

    /// Mechanical equilibrium radius at ambient pressure for the accepted inventory.
    ///
    /// The current radius and liquid concentration are unaltered. The result
    /// uses the selected fixed polytrope, or the isothermal gas law after
    /// relaxation to the fixed wall temperature if gas heat transport is enabled.
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
    /// converge. Keller–Miksis requires [Self::advance_with_forcing].
    /// A failed step preserves all accepted state.
    pub fn advance(
        &mut self,
        interval: Time<T>,
        far_field_midpoint: Pressure<T>,
    ) -> Result<GasTransfer<T>, DiffusiveBubbleError> {
        if matches!(self.mechanics, RadialMechanics::KellerMiksis { .. }) {
            return Err(DiffusiveBubbleError::InvalidState(
                "Keller–Miksis requires pressure rate",
            ));
        }
        self.advance_with_forcing(interval, BubbleForcing::steady(far_field_midpoint))
    }

    /// Advance one interval using absolute far-field pressure and its rate at
    /// the midpoint. Use ambient pressure with zero rate during acoustic off time.
    ///
    /// The wall, gas temperature and liquid field are advanced by midpoint
    /// coupling. The liquid retains its existing backward Euler start.
    /// All accepted state changes follow the conservative liquid flux.
    ///
    /// # Errors
    /// Invalid input, an inadmissible trial or failed convergence leaves the
    /// accepted wall state, gas temperature and liquid concentration intact.
    pub fn advance_with_forcing(
        &mut self,
        interval: Time<T>,
        forcing: BubbleForcing<T>,
    ) -> Result<GasTransfer<T>, DiffusiveBubbleError> {
        let dt = interval.into_base();
        if !dt.is_finite()
            || dt <= T::ZERO
            || !forcing.pressure.into_base().is_finite()
            || !forcing.pressure_rate.into_base().is_finite()
        {
            return Err(DiffusiveBubbleError::InvalidState("step inputs"));
        }
        let old = self.position;
        if let Some(heat) = &mut self.heat {
            heat.begin(old.radius, old.gas_change / self.reference_amount);
        }
        let mut endpoint = old;
        // Allow rounding in the pressure closure, norms and midpoint sums.
        let tolerance = T::from_count(16) * T::EPSILON;
        for _ in 0..24 {
            let (wall, pressure) = self.mechanical_endpoint(dt, forcing, tolerance, endpoint)?;
            endpoint = wall;
            let trial = self.diffusion.trial(
                interval,
                GasDiffusionWall {
                    radius: Length::from_base(endpoint.radius),
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
                self.accepted_pressure = pressure;
                if let Some(heat) = &mut self.heat {
                    heat.commit();
                }
                return Ok(trial.commit());
            }
            endpoint.gas_change = gas_change;
        }
        Err(DiffusiveBubbleError::NoConvergence)
    }

    fn mechanical_endpoint(
        &mut self,
        dt: T,
        forcing: BubbleForcing<T>,
        tolerance: T,
        mut endpoint: Position<T>,
    ) -> Result<(Position<T>, Pressure<T>), DiffusiveBubbleError> {
        let old = self.position;
        let half = T::from_f64(0.5);
        let n_mid = (old.gas_change * half + endpoint.gas_change * half) / self.reference_amount;
        let n_end = endpoint.gas_change / self.reference_amount;
        for _ in 0..24 {
            let radius_mid = old.radius * half + endpoint.radius * half;
            let velocity_mid = old.velocity * half + endpoint.velocity * half;
            let r_mid = Length::from_base(radius_mid);
            let v_mid = Velocity::from_base(velocity_mid);
            let (p_mid, p_rate, thermal_endpoint) = if let Some(heat) = &mut self.heat {
                let pressure = heat.trial(
                    HeatGeometry {
                        dt,
                        radius: radius_mid,
                        velocity: velocity_mid,
                        endpoint_radius: endpoint.radius,
                        endpoint_gas_fraction: n_end,
                    },
                    tolerance,
                )?;
                (
                    Pressure::from_base(
                        self.reference_pressure
                            * (T::ONE + half * (pressure.old + pressure.endpoint)),
                    ),
                    self.reference_pressure * (pressure.endpoint - pressure.old) / dt,
                    Some(pressure),
                )
            } else {
                let pressure = self
                    .gas
                    .gas_pressure(r_mid, T::ONE + n_mid)
                    .ok_or(DiffusiveBubbleError::InvalidState("midpoint gas"))?;
                let rate = match self.mechanics {
                    RadialMechanics::RayleighPlesset => T::ZERO,
                    RadialMechanics::KellerMiksis { .. } => {
                        let relative_rate = (endpoint.gas_change - old.gas_change)
                            / (dt * self.reference_amount * (T::ONE + n_mid));
                        self.exponent
                            * pressure.into_base()
                            * (relative_rate - T::from_count(3) * velocity_mid / radius_mid)
                    }
                };
                (pressure, rate, None)
            };
            let acceleration = match self.mechanics {
                RadialMechanics::RayleighPlesset => {
                    self.gas.acceleration(r_mid, v_mid, p_mid, forcing.pressure)
                }
                RadialMechanics::KellerMiksis { liquid_sound_speed } => {
                    if velocity_mid.abs() >= liquid_sound_speed.into_base() {
                        return Err(DiffusiveBubbleError::InvalidState("wall Mach number"));
                    }
                    self.gas.compressible_acceleration(
                        r_mid,
                        v_mid,
                        (p_mid, PressureRate::from_base(p_rate)),
                        (forcing.pressure, forcing.pressure_rate),
                        liquid_sound_speed,
                    )
                }
            };
            let radius = old.radius + dt * velocity_mid;
            let velocity = old.velocity + dt * acceleration.into_base();
            if !radius.is_finite() || radius <= T::ZERO || !velocity.is_finite() {
                return Err(DiffusiveBubbleError::InvalidState("trial wall"));
            }
            if let RadialMechanics::KellerMiksis { liquid_sound_speed } = self.mechanics {
                if velocity.abs() >= liquid_sound_speed.into_base() {
                    return Err(DiffusiveBubbleError::InvalidState("wall Mach number"));
                }
            }
            let converged = (radius - endpoint.radius).abs() / self.reference_radius <= tolerance
                && (velocity - endpoint.velocity).abs() / self.velocity_scale <= tolerance;
            endpoint.radius = radius;
            endpoint.velocity = velocity;
            if converged {
                let pressure = match thermal_endpoint {
                    Some(pressure) if pressure.settled => {
                        Pressure::from_base(self.reference_pressure * (T::ONE + pressure.endpoint))
                    }
                    Some(_) => continue,
                    None => self
                        .gas
                        .gas_pressure(Length::from_base(radius), T::ONE + n_end)
                        .ok_or(DiffusiveBubbleError::InvalidState("endpoint gas"))?,
                };
                if !pressure.into_base().is_finite() || pressure.into_base() <= T::ZERO {
                    return Err(DiffusiveBubbleError::InvalidState("endpoint gas"));
                }
                return Ok((endpoint, pressure));
            }
        }
        Err(DiffusiveBubbleError::NoConvergence)
    }
}
