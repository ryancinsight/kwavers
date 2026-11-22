use aequitas::systems::si::quantities::{AreaPerTime, Pressure, PressureRate, Velocity};
use eunomia::FloatElement;

/// Radial mechanics for an isolated diffusive bubble.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RadialMechanics<T = f64> {
    /// Incompressible-liquid Rayleigh–Plesset mechanics.
    #[default]
    RayleighPlesset,
    /// First-order liquid compressibility and acoustic radiation.
    KellerMiksis {
        /// Positive small-amplitude sound speed of the liquid.
        liquid_sound_speed: Velocity<T>,
    },
}

/// Uniform-pressure ideal gas with radial heat conduction and a fixed wall temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasHeatTransfer<T = f64> {
    /// Gas ratio of specific heats, strictly greater than one.
    pub heat_capacity_ratio: T,
    /// Positive gas thermal diffusivity at the initial gas pressure and temperature.
    pub reference_diffusivity: AreaPerTime<T>,
    /// Even interval count, at least four, for the gas radial grid.
    pub radial_intervals: usize,
}

/// Independent opt-in radial mechanics and gas heat transport choices.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiffusiveBubbleModels<T = f64> {
    /// Liquid radial mechanics.
    pub mechanics: RadialMechanics<T>,
    /// If present, solve for gas temperature instead of using a fixed polytrope.
    pub heat_transfer: Option<GasHeatTransfer<T>>,
}

/// Far-field forcing values at the time-step midpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BubbleForcing<T = f64> {
    /// Instantaneous absolute far-field pressure.
    pub pressure: Pressure<T>,
    /// Time derivative of absolute far-field pressure at the same instant.
    pub pressure_rate: PressureRate<T>,
}

impl<T: FloatElement> BubbleForcing<T> {
    /// Constant far-field pressure, including an acoustic off interval.
    #[must_use]
    pub fn steady(pressure: Pressure<T>) -> Self {
        Self {
            pressure,
            pressure_rate: PressureRate::from_base(T::ZERO),
        }
    }
}
