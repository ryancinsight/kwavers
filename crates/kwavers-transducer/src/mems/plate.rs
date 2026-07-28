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

use aequitas::systems::si::quantities::{
    Energy, Frequency, Length, Mass, MassDensity, Pressure, ReciprocalLength, SpringStiffness,
};
use core::f64::consts::TAU;

/// Fundamental clamped-plate eigenvalue λ² (Leissa 1969).
const CLAMPED_PLATE_LAMBDA_SQ: f64 = 10.2158;
/// Lamb fluid-loading added-mass coefficient for the fundamental clamped mode.
const LAMB_GAMMA: f64 = 0.6689;
/// Modal-mass fraction of the clamped-plate fundamental (m_eff = frac · ρ h A).
pub const MODAL_MASS_FRACTION: f64 = 0.1833;

/// Flexural rigidity `D = E h³ / (12 (1−ν²))` \[N·m].
#[must_use]
pub fn flexural_rigidity(youngs: Pressure, thickness: Length, poisson: f64) -> Energy {
    Energy::from_base(
        youngs.into_base() * thickness.into_base().powi(3) / (12.0 * (1.0 - poisson * poisson)),
    )
}

/// In-vacuo fundamental resonance of a clamped circular plate \`Hz`.
#[must_use]
pub fn vacuum_resonance(
    youngs: Pressure,
    thickness: Length,
    poisson: f64,
    density: MassDensity,
    radius: Length,
) -> Frequency {
    Frequency::from_base(
        (CLAMPED_PLATE_LAMBDA_SQ / TAU)
            * (thickness.into_base() / radius.into_base().powi(2))
            * (youngs.into_base() / (12.0 * density.into_base() * (1.0 - poisson * poisson)))
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
    let beta = LAMB_GAMMA * density_fluid.into_base() * radius.into_base()
        / (density_plate.into_base() * thickness.into_base());
    Frequency::from_base(vacuum_freq.into_base() / (1.0 + beta).sqrt())
}

/// Effective modal mass \[kg] = fraction · ρ h (π a²).
#[must_use]
pub fn modal_mass(density: MassDensity, thickness: Length, radius: Length) -> Mass {
    Mass::from_base(
        MODAL_MASS_FRACTION
            * density.into_base()
            * thickness.into_base()
            * core::f64::consts::PI
            * radius.into_base()
            * radius.into_base(),
    )
}

/// Effective modal stiffness \[N/m] = (2π f)² m_eff, self-consistent with `f`.
#[must_use]
pub fn modal_stiffness(resonance: Frequency, modal_mass: Mass) -> SpringStiffness {
    let w = TAU * resonance.into_base();
    SpringStiffness::from_base(w * w * modal_mass.into_base())
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
) -> f64 {
    LAMB_GAMMA * density_fluid.into_base() * radius.into_base()
        / (density_struct.into_base() * thickness.into_base())
}

/// Practical −6 dB fractional-bandwidth ceiling for a fluid-coupled MUT (~170%).
pub const FBW_MAX: f64 = 1.7;

/// −6 dB fractional bandwidth from the fluid-loading ratio:
/// `FBW = FBW_max · β/(β+1)` — monotone in `β`, saturating below [`FBW_MAX`].
/// Heavy fluid loading (light membrane) approaches the ceiling; a stiff/heavy
/// plate stays well below it.
#[must_use]
pub fn fractional_bandwidth_from_loading(beta: f64) -> f64 {
    FBW_MAX * beta / (beta + 1.0)
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
) -> f64 {
    let substrate_stiffness = substrate_stiffness.into_base();
    let element_stiffness = element_stiffness.into_base();
    if substrate_stiffness <= 0.0 {
        return 0.0;
    }
    substrate_stiffness / (substrate_stiffness + element_stiffness)
}

/// Plate sag across an element wrapped to radius of curvature `1/curvature`:
/// `δ ≈ ½ κ a²` \`m` (small-deflection geometry).
#[must_use]
pub fn curvature_sag(curvature: ReciprocalLength, radius: Length) -> Length {
    Length::from_base(0.5 * curvature.into_base() * radius.into_base() * radius.into_base())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::{
        quantities::{Length, MassDensity, Pressure},
        units::Meter,
    };

    fn length(value: f64) -> Length {
        Length::from_unit::<Meter>(value)
    }

    fn pressure(value: f64) -> Pressure {
        Pressure::from_base(value)
    }

    fn density(value: f64) -> MassDensity {
        MassDensity::from_base(value)
    }

    // Silicon membrane: E=169 GPa, ν=0.22, ρ=2330, a=20 µm, h=1 µm.
    #[test]
    fn resonance_scales_h_over_a_squared() {
        let f1 = vacuum_resonance(
            pressure(169e9),
            length(1e-6),
            0.22,
            density(2330.0),
            length(20e-6),
        );
        // double thickness → double f
        let f2 = vacuum_resonance(
            pressure(169e9),
            length(2e-6),
            0.22,
            density(2330.0),
            length(20e-6),
        );
        assert!((f2.into_base() / f1.into_base() - 2.0).abs() < 1e-9);
        // double radius → quarter f (1/a²)
        let f3 = vacuum_resonance(
            pressure(169e9),
            length(1e-6),
            0.22,
            density(2330.0),
            length(40e-6),
        );
        assert!((f3.into_base() / f1.into_base() - 0.25).abs() < 1e-9);
        // sanity: a 1 µm / 20 µm Si membrane resonates in the tens of MHz
        assert!(
            f1.into_base() > 5e6 && f1.into_base() < 60e6,
            "f0 = {:?}",
            f1
        );
    }

    #[test]
    fn immersion_lowers_resonance() {
        let f_vac = vacuum_resonance(
            pressure(169e9),
            length(1e-6),
            0.22,
            density(2330.0),
            length(20e-6),
        );
        let f_imm = immersion_resonance(
            f_vac,
            density(2330.0),
            length(1e-6),
            density(1060.0),
            length(20e-6),
        ); // blood
        assert!(
            f_imm < f_vac,
            "immersion {:?} should be below vacuum {:?}",
            f_imm,
            f_vac
        );
    }

    #[test]
    fn modal_stiffness_self_consistent_with_resonance() {
        let f = vacuum_resonance(
            pressure(169e9),
            length(1e-6),
            0.22,
            density(2330.0),
            length(20e-6),
        );
        let m = modal_mass(density(2330.0), length(1e-6), length(20e-6));
        let k = modal_stiffness(f, m);
        // recompute f from √(k/m)/2π → identity
        let f_back = (k.into_base() / m.into_base()).sqrt() / TAU;
        assert!((f_back - f.into_base()).abs() / f.into_base() < 1e-9);
    }
}
