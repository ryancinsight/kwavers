use super::super::bubble_state::{radial_viscous_stress, spherical_capillary_pressure};
use super::wall_acceleration;
use aequitas::systems::si::quantities::{
    Acceleration, AmountOfSubstance, DynamicViscosity, Length, MassDensity, Pressure,
    SurfaceTension, ThermodynamicTemperature, Velocity,
};
use eunomia::FloatElement;
use kwavers_core::constants::{numerical::FOUR_PI, GAS_CONSTANT};

/// Physical constants for a single noncondensable gas and constant vapor pressure.
#[derive(Debug, Clone, Copy)]
pub struct PolytropicParameters<T = f64> {
    /// Absolute far-field pressure at rest.
    pub ambient_pressure: Pressure<T>,
    /// Spatially and temporally constant vapor partial pressure.
    pub vapor_pressure: Pressure<T>,
    /// Newtonian liquid density.
    pub liquid_density: MassDensity<T>,
    /// Newtonian liquid dynamic viscosity.
    pub liquid_viscosity: DynamicViscosity<T>,
    /// Surface tension at the spherical interface.
    pub surface_tension: SurfaceTension<T>,
    /// Initial gas temperature for the ideal-gas reference state.
    pub reference_temperature: ThermodynamicTemperature<T>,
    /// Polytropic exponent of the noncondensable gas, at least one.
    pub exponent: T,
}

/// Density-based gas-pressure closure for Rayleigh–Plesset with changing gas amount.
///
/// For radius R and moles n, p_g/p_g0 = [(n/n0)(R0/R)^3]^κ.
/// The initial ideal-gas reference uses p_g0 = P0 - p_v + 2σ/R0.
/// See Smith & Wang (2022), JFM 951 A12, doi:10.1017/jfm.2022.849.
#[derive(Debug, Clone, Copy)]
pub struct PolytropicGas<T = f64> {
    params: PolytropicParameters<T>,
    radius: T,
    amount: T,
    pressure: T,
    laplace: T,
}

impl<T: FloatElement> PolytropicGas<T> {
    /// Form the equilibrium reference. Returns `None` for invalid or
    /// unrepresentable inputs. Ambient pressure must exceed vapor pressure,
    /// and the polytropic exponent must be at least one.
    #[must_use]
    pub fn new(reference_radius: Length<T>, params: PolytropicParameters<T>) -> Option<Self> {
        let radius = reference_radius.into_base();
        let static_pressure =
            params.ambient_pressure.into_base() - params.vapor_pressure.into_base();
        let density = params.liquid_density.into_base();
        let temperature = params.reference_temperature.into_base();
        let viscosity = params.liquid_viscosity.into_base();
        let tension = params.surface_tension.into_base();
        let vapor = params.vapor_pressure.into_base();
        if [static_pressure, density, temperature, radius]
            .into_iter()
            .any(|x| !x.is_finite() || x <= T::ZERO)
            || !params.exponent.is_finite()
            || params.exponent < T::ONE
            || [viscosity, tension, vapor]
                .into_iter()
                .any(|x| !x.is_finite() || x < T::ZERO)
        {
            return None;
        }
        let laplace = spherical_capillary_pressure(tension, radius);
        let pressure = static_pressure + laplace;
        let volume = (T::from_f64(FOUR_PI) / T::from_count(3)) * radius * radius * radius;
        let amount = pressure * volume / (T::from_f64(GAS_CONSTANT) * temperature);
        if !pressure.is_finite() || !amount.is_finite() || amount <= T::ZERO {
            return None;
        }
        Some(Self {
            params,
            radius,
            amount,
            pressure,
            laplace,
        })
    }

    /// Initial noncondensable gas amount.
    #[must_use]
    pub fn reference_amount(&self) -> AmountOfSubstance<T> {
        AmountOfSubstance::from_base(self.amount)
    }

    /// Flat-interface noncondensable pressure at the ambient far-field pressure.
    #[must_use]
    pub fn saturation_pressure(&self) -> Pressure<T> {
        Pressure::from_base(
            self.params.ambient_pressure.into_base() - self.params.vapor_pressure.into_base(),
        )
    }

    /// Mechanical equilibrium radius at ambient pressure for this gas closure.
    ///
    /// Let x = R/R₀, m = n/n₀, and let a and b be the ambient dry-pressure
    /// and capillary-pressure fractions of the initial dry pressure. Force
    /// balance gives m = x^(3−1/κ) (a x + b)^(1/κ). Both factors increase
    /// on x > 0 for κ ≥ 1, so the positive root is unique. Its bracket comes
    /// from a + b/x lying between 1 and 1/x.
    ///
    /// The isothermal closure (κ = 1) gives the equilibrium after
    /// temperature has relaxed to the reference temperature. Other supported
    /// exponents give the zero-velocity equilibrium of the specified
    /// polytrope. This calculation does not change the mechanical state.
    #[must_use]
    pub fn equilibrium_radius(&self, amount: AmountOfSubstance<T>) -> Option<Length<T>> {
        let amount_ratio = amount.into_base() / self.amount;
        if !amount_ratio.is_finite() || amount_ratio <= T::ZERO {
            return None;
        }
        if amount_ratio == T::ONE {
            return Some(Length::from_base(self.radius));
        }
        let inverse_exponent = T::ONE / self.params.exponent;
        let power = T::from_count(3) - inverse_exponent;
        let ambient_fraction = self.saturation_pressure().into_base() / self.pressure;
        let capillary_fraction = self.laplace / self.pressure;
        let cube_root = amount_ratio.cbrt();
        let normalized_radius = if self.laplace == T::ZERO {
            cube_root
        } else {
            let capillary_limit = amount_ratio.powf(T::ONE / power);
            let (mut low, mut high) = if amount_ratio < T::ONE {
                (capillary_limit, cube_root)
            } else {
                (cube_root, capillary_limit)
            };
            let half = T::from_f64(0.5);
            let amount_at = |radius: T| {
                radius.powf(power)
                    * (ambient_fraction * radius + capillary_fraction).powf(inverse_exponent)
            };
            loop {
                let midpoint = low * half + high * half;
                if midpoint <= low || midpoint >= high {
                    break if (amount_at(low) - amount_ratio).abs()
                        <= (amount_at(high) - amount_ratio).abs()
                    {
                        low
                    } else {
                        high
                    };
                }
                if amount_at(midpoint) < amount_ratio {
                    low = midpoint;
                } else {
                    high = midpoint;
                }
            }
        };
        let radius = self.radius * normalized_radius;
        (radius.is_finite() && radius > T::ZERO).then(|| Length::from_base(radius))
    }

    /// Noncondensable partial pressure at the given radius and gas amount.
    /// `amount_ratio` is the current amount divided by the reference amount.
    /// Returns `None` if the gas state is invalid or unrepresentable.
    #[must_use]
    pub fn gas_pressure(&self, radius: Length<T>, amount_ratio: T) -> Option<Pressure<T>> {
        let radius = radius.into_base();
        if !radius.is_finite()
            || radius <= T::ZERO
            || !amount_ratio.is_finite()
            || amount_ratio <= T::ZERO
        {
            return None;
        }
        let inverse_radius = self.radius / radius;
        let pressure = self.pressure
            * (amount_ratio * inverse_radius * inverse_radius * inverse_radius)
                .powf(self.params.exponent);
        (pressure.is_finite() && pressure > T::ZERO).then(|| Pressure::from_base(pressure))
    }

    /// Rayleigh–Plesset acceleration at a positive radius, given the
    /// noncondensable partial pressure and the instantaneous absolute far-field pressure.
    #[must_use]
    pub fn acceleration(
        &self,
        radius: Length<T>,
        velocity: Velocity<T>,
        gas_pressure: Pressure<T>,
        far_field_pressure: Pressure<T>,
    ) -> Acceleration<T> {
        let r = radius.into_base();
        let v = velocity.into_base();
        let laplace = spherical_capillary_pressure(self.params.surface_tension.into_base(), r);
        let viscous = radial_viscous_stress(self.params.liquid_viscosity.into_base(), v, r);
        // Subtract the static balance first so the reference state is exactly
        // at rest, including in lower precision.
        let net = (gas_pressure.into_base() - self.pressure)
            - (far_field_pressure.into_base() - self.params.ambient_pressure.into_base())
            + (self.laplace - laplace)
            - viscous;
        Acceleration::from_base(wall_acceleration(
            r,
            v,
            self.params.liquid_density.into_base(),
            net,
        ))
    }
}
