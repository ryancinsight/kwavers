//! Clamped circular-plate vibration shared by CMUT membranes and PMUT plates.
//!
//! The fundamental of a clamped circular plate of radius `a`, thickness `h`,
//! Young's modulus `E`, Poisson ratio `ν`, density `ρ` is
//!
//! ```text
//! D   = E h³ / (12 (1−ν²))                          (flexural rigidity)
//! f₀  = (λ²/2π) (h/a²) √(E / (12 ρ (1−ν²)))         (in vacuo, λ²=10.2158)
//! ```
//!
//! Immersion adds radiation mass; Lamb's approximation gives the downshift
//! `f_imm = f₀ / √(1 + Γ ρ_f a /(ρ h))`, `Γ = 0.6689` (fundamental mode).
//!
//! # References
//! - Leissa, A. W. (1969). *Vibration of Plates*, NASA SP-160.
//! - Lamb, H. (1920). "On the vibrations of an elastic plate in contact with water."
//! - Kwak, M. K. (1991). "Vibration of circular plates in contact with water."

use core::f64::consts::TAU;

use aequitas::systems::si::quantities::{
    Dimensionless, FlexuralRigidity, Frequency, Length, Mass, MassDensity, Pressure,
    ReciprocalLength, SpringStiffness,
};
use aequitas::systems::si::units::{Hertz, Joule, Kilogram, Meter, NewtonPerMeter, Pascal};

/// Fundamental clamped-plate eigenvalue λ² (Leissa 1969).
const CLAMPED_PLATE_LAMBDA_SQ: f64 = 10.2158;
/// Lamb fluid-loading added-mass coefficient for the fundamental clamped mode.
const LAMB_GAMMA: f64 = 0.6689;
/// Modal-mass fraction of the clamped-plate fundamental (m_eff = frac · ρ h A).
pub const MODAL_MASS_FRACTION: f64 = 0.1833;

/// Flexural rigidity `D = E h³ / (12 (1−ν²))` \[N·m].
#[must_use]
pub fn flexural_rigidity(
    youngs: Pressure,
    thickness: Length,
    poisson: Dimensionless,
) -> FlexuralRigidity {
    FlexuralRigidity::from_unit::<Joule>(
        youngs.in_unit::<Pascal>() * thickness.in_unit::<Meter>().powi(3)
            / (12.0 * (1.0 - poisson.into_base() * poisson.into_base())),
    )
}

/// In-vacuo fundamental resonance of a clamped circular plate \`Hz`.
#[must_use]
pub fn vacuum_resonance(
    youngs: Pressure,
    thickness: Length,
    poisson: Dimensionless,
    density: MassDensity,
    radius: Length,
) -> Frequency {
    Frequency::from_unit::<Hertz>(
        (CLAMPED_PLATE_LAMBDA_SQ / TAU)
            * (thickness.in_unit::<Meter>() / radius.in_unit::<Meter>().powi(2))
            * (youngs.in_unit::<Pascal>()
                / (12.0
                    * density.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
                    * (1.0 - poisson.into_base() * poisson.into_base())))
            .sqrt(),
    )
}

/// Fluid-loaded (immersion) resonance via Lamb added-mass downshift \`Hz`.
#[must_use]
pub fn immersion_resonance(
    vacuum_freq: Frequency,
    density_plate: MassDensity,
    thickness: Length,
    density_fluid: MassDensity,
    radius: Length,
) -> Frequency {
    let beta = LAMB_GAMMA
        * density_fluid.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
        * radius.in_unit::<Meter>()
        / (density_plate.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
            * thickness.in_unit::<Meter>());
    Frequency::from_unit::<Hertz>(vacuum_freq.in_unit::<Hertz>() / (1.0 + beta).sqrt())
}

/// Effective modal mass \[kg] = fraction · ρ h (π a²).
#[must_use]
pub fn modal_mass(density: MassDensity, thickness: Length, radius: Length) -> Mass {
    Mass::from_unit::<Kilogram>(
        MODAL_MASS_FRACTION
            * density.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
            * thickness.in_unit::<Meter>()
            * core::f64::consts::PI
            * radius.in_unit::<Meter>()
            * radius.in_unit::<Meter>(),
    )
}

/// Effective modal stiffness \[N/m] = (2π f)² m_eff, self-consistent with `f`.
#[must_use]
pub fn modal_stiffness(resonance: Frequency, modal_mass: Mass) -> SpringStiffness {
    let w = TAU * resonance.in_unit::<Hertz>();
    SpringStiffness::from_unit::<NewtonPerMeter>(w * w * modal_mass.in_unit::<Kilogram>())
}

/// Fluid-loading ratio `β = Γ ρ_f a / (ρ_s t)` — added fluid mass relative to the
/// structural areal mass. Larger `β` (lighter structure) ⇒ stronger fluid coupling
/// ⇒ broader bandwidth.
#[must_use]
pub fn fluid_loading_beta(
    density_fluid: MassDensity,
    density_struct: MassDensity,
    thickness: Length,
    radius: Length,
) -> Dimensionless {
    Dimensionless::from_base(
        LAMB_GAMMA
            * density_fluid.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
            * radius.in_unit::<Meter>()
            / (density_struct.in_unit::<aequitas::systems::si::units::KilogramPerCubicMeter>()
                * thickness.in_unit::<Meter>()),
    )
}

/// Practical −6 dB fractional-bandwidth ceiling for a fluid-coupled MUT (~170%).
pub const FBW_MAX: f64 = 1.7;

/// −6 dB fractional bandwidth from the fluid-loading ratio:
/// `FBW = FBW_max · β/(β+1)` — monotone in `β`, saturating below [`FBW_MAX`].
/// Heavy fluid loading (light membrane) approaches the ceiling; a stiff/heavy
/// plate stays well below it.
#[must_use]
pub fn fractional_bandwidth_from_loading(beta: Dimensionless) -> Dimensionless {
    Dimensionless::from_base(FBW_MAX * beta.into_base() / (beta.into_base() + 1.0))
}

/// Forward-radiation efficiency on a finite-stiffness (flexible) backing:
/// `η = k_sub / (k_sub + k_elem) ∈ (0, 1]`. A rigid backing (`k_sub → ∞`) gives
/// `η = 1`; a compliant flexible substrate lets the element recoil into the
/// backing instead of the fluid, reducing forward output. Applies to any MUT on a
/// flexible carrier.
#[must_use]
pub fn flexible_output_factor(
    substrate_stiffness: SpringStiffness,
    element_stiffness: SpringStiffness,
) -> Dimensionless {
    let substrate_stiffness = substrate_stiffness.in_unit::<NewtonPerMeter>();
    let element_stiffness = element_stiffness.in_unit::<NewtonPerMeter>();
    if substrate_stiffness <= 0.0 {
        return Dimensionless::from_base(0.0);
    }
    Dimensionless::from_base(substrate_stiffness / (substrate_stiffness + element_stiffness))
}

/// Plate sag across an element wrapped to radius of curvature `1/curvature`:
/// `δ ≈ ½ κ a²` \`m` (small-deflection geometry).
#[must_use]
pub fn curvature_sag(curvature: ReciprocalLength, radius: Length) -> Length {
    Length::from_unit::<Meter>(
        0.5 * curvature.in_unit::<aequitas::systems::si::units::PerMeter>()
            * radius.in_unit::<Meter>()
            * radius.in_unit::<Meter>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::KilogramPerCubicMeter;

    fn pressure(value: f64) -> Pressure<f64> {
        Pressure::from_unit::<Pascal>(value)
    }

    fn meters(value: f64) -> Length<f64> {
        Length::from_unit::<Meter>(value)
    }

    fn density(value: f64) -> MassDensity<f64> {
        MassDensity::from_unit::<KilogramPerCubicMeter>(value)
    }

    fn dimensionless(value: f64) -> Dimensionless<f64> {
        Dimensionless::from_base(value)
    }

    // Silicon membrane: E=169 GPa, ν=0.22, ρ=2330, a=20 µm, h=1 µm.
    #[test]
    fn resonance_scales_h_over_a_squared() {
        let f1 = vacuum_resonance(
            pressure(169e9),
            meters(1e-6),
            dimensionless(0.22),
            density(2330.0),
            meters(20e-6),
        )
        .in_unit::<Hertz>();
        // double thickness → double f
        let f2 = vacuum_resonance(
            pressure(169e9),
            meters(2e-6),
            dimensionless(0.22),
            density(2330.0),
            meters(20e-6),
        )
        .in_unit::<Hertz>();
        assert!((f2 / f1 - 2.0).abs() < 1e-9);
        // double radius → quarter f (1/a²)
        let f3 = vacuum_resonance(
            pressure(169e9),
            meters(1e-6),
            dimensionless(0.22),
            density(2330.0),
            meters(40e-6),
        )
        .in_unit::<Hertz>();
        assert!((f3 / f1 - 0.25).abs() < 1e-9);
        // sanity: a 1 µm / 20 µm Si membrane resonates in the tens of MHz
        assert!(f1 > 5e6 && f1 < 60e6, "f0 = {f1}");
    }

    #[test]
    fn immersion_lowers_resonance() {
        let f_vac = vacuum_resonance(
            pressure(169e9),
            meters(1e-6),
            dimensionless(0.22),
            density(2330.0),
            meters(20e-6),
        );
        let f_imm = immersion_resonance(
            f_vac,
            density(2330.0),
            meters(1e-6),
            density(1060.0),
            meters(20e-6),
        )
        .in_unit::<Hertz>(); // blood
        let f_vac = f_vac.in_unit::<Hertz>();
        assert!(
            f_imm < f_vac,
            "immersion {f_imm} should be below vacuum {f_vac}"
        );
    }

    #[test]
    fn modal_stiffness_self_consistent_with_resonance() {
        let f = vacuum_resonance(
            pressure(169e9),
            meters(1e-6),
            dimensionless(0.22),
            density(2330.0),
            meters(20e-6),
        );
        let m = modal_mass(density(2330.0), meters(1e-6), meters(20e-6));
        let k = modal_stiffness(f, m);
        // recompute f from √(k/m)/2π → identity
        let f_back = (k.in_unit::<NewtonPerMeter>() / m.in_unit::<Kilogram>()).sqrt() / TAU;
        let f = f.in_unit::<Hertz>();
        assert!((f_back - f).abs() / f < 1e-9);
    }
}
