//! CMUT — capacitive micromachined ultrasonic transducer cell.
//!
//! A thin membrane suspended over a vacuum gap, actuated electrostatically. The
//! light membrane couples strongly to fluid, giving CMUTs very wide bandwidth and
//! easy CMOS integration; the trade-offs are a DC bias near collapse and a high
//! drive voltage.
//!
//! Models (closed-form, lumped): parallel-plate capacitance and **collapse
//! (pull-in) voltage** `V_c = √(8 k g₀³/(27 ε₀ A))`, bias-dependent
//! electromechanical coupling `k² = (V_dc/V_c)²`, dielectric self-heating, and a
//! radiation-damping-limited fractional bandwidth.
//!
//! # References
//! - Oralkan, Ö., et al. (2002). "Capacitive micromachined ultrasonic transducers."
//!   *IEEE TUFFC*, 49(11).
//! - Khuri-Yakub, B. T., & Oralkan, Ö. (2011). "CMUTs for medical imaging." *J.
//!   Micromech. Microeng.*, 21(5).

use aequitas::systems::si::quantities::{
    Capacitance, DampingCoefficient, ElectricPotential, Frequency, Length, MassDensity, Power,
    Pressure, ReciprocalLength, SpringStiffness, Velocity,
};
use core::f64::consts::{PI, TAU};
use kwavers_core::constants::fundamental::VACUUM_PERMITTIVITY;

use super::plate;

/// A single CMUT cell (Si membrane over a vacuum gap).
#[derive(Debug, Clone, Copy)]
pub struct CmutCell {
    /// Membrane radius `a` \`m`.
    pub radius: Length,
    /// Membrane thickness `h` \`m`.
    pub thickness: Length,
    /// Vacuum gap `g₀` \`m`.
    pub gap: Length,
    /// Membrane Young's modulus `E` \`Pa`.
    pub youngs: Pressure,
    /// Membrane density `ρ` \[kg·m⁻³].
    pub density: MassDensity,
    /// Membrane Poisson ratio `ν`.
    pub poisson: f64,
    /// Loss tangent of the gap/insulator stack (≈1e-3 for Si CMUT).
    pub loss_tangent: f64,
}

impl CmutCell {
    /// Silicon-membrane CMUT preset; `None` for any non-positive geometry.
    #[must_use]
    pub fn silicon(radius: Length, thickness: Length, gap: Length) -> Option<Self> {
        if radius.into_base() > 0.0 && thickness.into_base() > 0.0 && gap.into_base() > 0.0 {
            Some(Self {
                radius,
                thickness,
                gap,
                youngs: Pressure::from_base(169.0e9),
                density: MassDensity::from_base(2330.0),
                poisson: 0.22,
                loss_tangent: 1.0e-3,
            })
        } else {
            None
        }
    }

    /// Membrane area `A = π a²` \`m²`.
    #[must_use]
    pub fn area(&self) -> aequitas::systems::si::quantities::Area {
        aequitas::systems::si::quantities::Area::from_base(
            PI * self.radius.into_base() * self.radius.into_base(),
        )
    }

    /// Small-signal capacitance `C₀ = ε₀ A / g₀` \[F].
    #[must_use]
    pub fn capacitance(&self) -> Capacitance {
        Capacitance::from_base(VACUUM_PERMITTIVITY * self.area().into_base() / self.gap.into_base())
    }

    /// In-vacuo membrane resonance \`Hz`.
    #[must_use]
    pub fn vacuum_resonance(&self) -> Frequency {
        plate::vacuum_resonance(
            self.youngs,
            self.thickness,
            self.poisson,
            self.density,
            self.radius,
        )
    }

    /// Immersion (fluid-loaded) resonance \`Hz`.
    #[must_use]
    pub fn immersion_resonance(&self, density_fluid: MassDensity) -> Frequency {
        plate::immersion_resonance(
            self.vacuum_resonance(),
            self.density,
            self.thickness,
            density_fluid,
            self.radius,
        )
    }

    /// Effective modal stiffness `k` \[N/m] (self-consistent with `f_vac`).
    #[must_use]
    pub fn modal_stiffness(&self) -> SpringStiffness {
        let m = plate::modal_mass(self.density, self.thickness, self.radius);
        plate::modal_stiffness(self.vacuum_resonance(), m)
    }

    /// Electrostatic **collapse (pull-in) voltage** `V_c = √(8 k g₀³/(27 ε₀ A))` \`V`.
    #[must_use]
    pub fn collapse_voltage(&self) -> ElectricPotential {
        let k = self.modal_stiffness();
        ElectricPotential::from_base(
            (8.0 * k.into_base() * self.gap.into_base().powi(3)
                / (27.0 * VACUUM_PERMITTIVITY * self.area().into_base()))
            .sqrt(),
        )
    }

    /// Bias-dependent electromechanical coupling `k² = (V_dc/V_c)²`, capped at
    /// 0.85 (coupling rises toward collapse but stays bounded < 1).
    #[must_use]
    pub fn coupling_k2(&self, bias_voltage: ElectricPotential) -> f64 {
        let vc = self.collapse_voltage();
        if vc.into_base() <= 0.0 {
            return 0.0;
        }
        ((bias_voltage.into_base() / vc.into_base()).powi(2)).min(0.85)
    }

    /// Static **DC pull-down** as a fraction `u = x/g₀` of the gap, the stable
    /// electrostatic equilibrium of the lumped parallel-plate membrane under bias.
    ///
    /// # Nonlinear electrostatics
    ///
    /// Force balance `k x = ε₀ A V²/(2(g₀−x)²)` non-dimensionalises (with
    /// `V_c² = 8 k g₀³/(27 ε₀ A)`) to
    ///
    /// ```text
    /// u (1−u)² = (4/27)(V/V_c)²,   u ∈ [0, 1/3].
    /// ```
    ///
    /// `g(u)=u(1−u)²` is strictly increasing on `[0, 1/3]` (so the root is
    /// unique, found by bisection) and reaches its maximum `4/27` at the
    /// **pull-in** point `u = 1/3` exactly when `V = V_c`. Returns `None` for
    /// `V ≥ V_c` (collapse — no stable equilibrium) or invalid geometry.
    #[must_use]
    pub fn bias_pulldown_fraction(&self, bias_voltage: ElectricPotential) -> Option<f64> {
        let vc = self.collapse_voltage();
        if vc.into_base() <= 0.0
            || bias_voltage.into_base() < 0.0
            || bias_voltage.into_base() >= vc.into_base()
        {
            return None;
        }
        let target = 4.0 / 27.0 * (bias_voltage.into_base() / vc.into_base()).powi(2);
        // Bisection on the strictly-increasing branch u ∈ [0, 1/3].
        let (mut lo, mut hi) = (0.0_f64, 1.0 / 3.0);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if mid * (1.0 - mid) * (1.0 - mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// Biased operating gap `g₀(1 − u)` \`m` under DC bias (the membrane is
    /// pulled down by the electrostatic force). `None` once collapsed (`V ≥ V_c`).
    #[must_use]
    pub fn biased_gap(&self, bias_voltage: ElectricPotential) -> Option<Length> {
        self.bias_pulldown_fraction(bias_voltage)
            .map(|u| Length::from_base(self.gap.into_base() * (1.0 - u)))
    }

    /// Biased small-signal capacitance `C(V) = ε₀ A /(g₀(1−u)) = C₀/(1−u)` \[F].
    /// Rises above `C₀` as the bias pulls the membrane in. `None` once collapsed.
    #[must_use]
    pub fn biased_capacitance(&self, bias_voltage: ElectricPotential) -> Option<Capacitance> {
        self.bias_pulldown_fraction(bias_voltage)
            .map(|u| Capacitance::from_base(self.capacitance().into_base() / (1.0 - u)))
    }

    /// **Spring-softened** resonance under DC bias \`Hz`: the electrostatic force
    /// reduces the effective stiffness to `k_eff = k(1 − 2u/(1−u))`, so the
    /// immersion resonance drops as `f(V) = f_imm·√(k_eff/k)` and **vanishes at
    /// pull-in** (`u = 1/3`, `V = V_c`) — the classic CMUT collapse instability.
    ///
    /// The softened-stiffness factor comes from `dF_elec/dx = 2k u/(1−u)` at the
    /// biased equilibrium. `None` once collapsed.
    #[must_use]
    pub fn bias_softened_resonance(
        &self,
        bias_voltage: ElectricPotential,
        density_fluid: MassDensity,
    ) -> Option<Frequency> {
        let u = self.bias_pulldown_fraction(bias_voltage)?;
        let stiffness_ratio = (2.0 * u / (1.0 - u)).mul_add(-1.0, 1.0).max(0.0);
        Some(Frequency::from_base(
            self.immersion_resonance(density_fluid).into_base() * stiffness_ratio.sqrt(),
        ))
    }

    /// Dielectric self-heating power `P = π f C V_ac² tanδ` \`W` at drive `V_ac`/`freq`.
    #[must_use]
    pub fn self_heating_power(
        &self,
        drive_voltage_ac: ElectricPotential,
        freq: Frequency,
    ) -> Power {
        Power::from_base(
            PI * freq.into_base()
                * self.capacitance().into_base()
                * drive_voltage_ac.into_base()
                * drive_voltage_ac.into_base()
                * self.loss_tangent,
        )
    }

    /// Radiation-damping quality factor `Q = ω₀ m_eff / R_rad` in a fluid
    /// (small-piston radiation resistance, `ka<1`).
    #[must_use]
    pub fn radiation_q(&self, density_fluid: MassDensity, sound_speed_fluid: Velocity) -> f64 {
        let f0 = self.immersion_resonance(density_fluid);
        let w0 = TAU * f0.into_base();
        let m = plate::modal_mass(self.density, self.thickness, self.radius);
        let ka = w0 * self.radius.into_base() / sound_speed_fluid.into_base();
        // R_rad ≈ ρ_f c_f A (ka)²/2 (baffled small piston)
        let r_rad = density_fluid.into_base()
            * sound_speed_fluid.into_base()
            * self.area().into_base()
            * ka
            * ka
            / 2.0;
        if r_rad <= 0.0 {
            return f64::INFINITY;
        }
        w0 * m.into_base() / r_rad
    }

    /// Fluid-loading ratio `β = Γ ρ_f a/(ρ_m h)` (membrane areal mass).
    #[must_use]
    pub fn fluid_loading_beta(&self, density_fluid: MassDensity) -> f64 {
        plate::fluid_loading_beta(density_fluid, self.density, self.thickness, self.radius)
    }

    /// −6 dB fractional bandwidth from fluid loading. CMUTs are fluid-coupling
    /// dominated (light membrane) → very wide bandwidth.
    #[must_use]
    pub fn fractional_bandwidth(&self, density_fluid: MassDensity) -> f64 {
        plate::fractional_bandwidth_from_loading(self.fluid_loading_beta(density_fluid))
    }

    /// Peak transmit surface velocity \[m·s⁻¹]. The membrane is an electrostatic
    /// actuator that **cannot swing more than the gap** before collapse, so the
    /// peak displacement is `g₀·swing_fraction` and `v = ω·g₀·swing_fraction`.
    /// `swing_fraction ≈ 1/3` for conventional pre-collapse bias (stable range),
    /// up to ~1 in collapse-snapback drive. This is a *hard* ceiling — driving
    /// harder cannot exceed it.
    #[must_use]
    pub fn max_surface_velocity(
        &self,
        density_fluid: MassDensity,
        swing_fraction: f64,
    ) -> Velocity {
        let f = self.immersion_resonance(density_fluid);
        Velocity::from_base(TAU * f.into_base() * self.gap.into_base() * swing_fraction)
    }

    /// Gap-limited peak output pressure into the fluid (plane-wave radiation),
    /// `p = ρ c · v_peak` \`Pa`. The defining transmit limitation of CMUTs for
    /// **therapy**: output is capped by the (sub-micron) gap, not the drive.
    #[must_use]
    pub fn max_output_pressure(
        &self,
        density_fluid: MassDensity,
        sound_speed_fluid: Velocity,
        swing_fraction: f64,
    ) -> Pressure {
        Pressure::from_base(
            density_fluid.into_base()
                * sound_speed_fluid.into_base()
                * self
                    .max_surface_velocity(density_fluid, swing_fraction)
                    .into_base(),
        )
    }

    /// Squeeze-film viscous damping coefficient of the gap gas \[N·s·m⁻¹]:
    /// `c = 3π μ a⁴ / (2 g₀³)` (incompressible / low-squeeze-number limit). Applies
    /// to **vented or non-evacuated** gaps; a sealed-vacuum CMUT in immersion is
    /// radiation-damped instead (`fractional_bandwidth`). The strong `a⁴/g₀³`
    /// scaling makes squeeze-film dominant for wide, narrow-gap cells.
    #[must_use]
    pub fn squeeze_film_damping(
        &self,
        gas_viscosity: aequitas::systems::si::quantities::DynamicViscosity,
    ) -> DampingCoefficient {
        DampingCoefficient::from_base(
            3.0 * PI * gas_viscosity.into_base() * self.radius.into_base().powi(4)
                / (2.0 * self.gap.into_base().powi(3)),
        )
    }

    /// Squeeze number `σ = 12 μ ω a² / (p_a g₀²)` at angular frequency `ω = 2πf`.
    /// `σ ≪ 1` → incompressible (viscous damping); `σ ≫ 1` → trapped gas acts as a
    /// spring (stiffening, little damping).
    #[must_use]
    pub fn squeeze_number(
        &self,
        gas_viscosity: aequitas::systems::si::quantities::DynamicViscosity,
        ambient_pressure: Pressure,
        freq: Frequency,
    ) -> f64 {
        if ambient_pressure.into_base() <= 0.0 {
            return f64::INFINITY;
        }
        12.0 * gas_viscosity.into_base()
            * TAU
            * freq.into_base()
            * self.radius.into_base()
            * self.radius.into_base()
            / (ambient_pressure.into_base() * self.gap.into_base() * self.gap.into_base())
    }

    /// Quality factor set by squeeze-film damping alone, `Q = ω₀ m_eff / c`
    /// (in-vacuo resonance). Lower `Q` ⇒ broader, more damped response.
    #[must_use]
    pub fn squeeze_film_quality_factor(
        &self,
        gas_viscosity: aequitas::systems::si::quantities::DynamicViscosity,
    ) -> f64 {
        let c = self.squeeze_film_damping(gas_viscosity);
        if c.into_base() <= 0.0 {
            return f64::INFINITY;
        }
        let w0 = TAU * self.vacuum_resonance().into_base();
        let m = plate::modal_mass(self.density, self.thickness, self.radius);
        w0 * m.into_base() / c.into_base()
    }

    /// Output derating when the cell is **flexed** to curvature `κ`. Wrapping the
    /// die perturbs the sub-micron gap by the sag `δ = ½κa²`; non-uniform gap and
    /// membrane tension spread the collapse voltage and detune cells, so output
    /// falls as `1/(1 + δ/g₀)`. Tighter gaps (higher sensitivity) lose the most —
    /// the reason CMUTs are hard to make flexible without sacrificing output.
    #[must_use]
    pub fn flex_gap_derating(&self, curvature: ReciprocalLength) -> f64 {
        let sag = plate::curvature_sag(curvature, self.radius);
        1.0 / (1.0 + sag.into_base() / self.gap.into_base())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::quantities::{DynamicViscosity, Pressure};
    use aequitas::systems::si::units::{
        Hertz, KilogramPerCubicMeter, Meter, MeterPerSecond, PascalSecond,
    };

    fn length(value: f64) -> Length {
        Length::from_unit::<Meter>(value)
    }

    fn density(value: f64) -> MassDensity {
        MassDensity::from_unit::<KilogramPerCubicMeter>(value)
    }

    fn velocity(value: f64) -> Velocity {
        Velocity::from_unit::<MeterPerSecond>(value)
    }

    fn voltage(value: f64) -> ElectricPotential {
        ElectricPotential::from_base(value)
    }

    fn frequency(value: f64) -> Frequency {
        Frequency::from_unit::<Hertz>(value)
    }

    fn viscosity(value: f64) -> DynamicViscosity {
        DynamicViscosity::from_unit::<PascalSecond>(value)
    }

    fn pressure(value: f64) -> Pressure {
        Pressure::from_base(value)
    }

    fn cell(radius: f64, thickness: f64, gap: f64) -> CmutCell {
        CmutCell::silicon(length(radius), length(thickness), length(gap)).unwrap()
    }

    // IVUS-scale CMUT: a=14 µm, h=0.4 µm, gap=0.12 µm (≈40 MHz in blood).
    fn ivus_cmut() -> CmutCell {
        cell(14e-6, 0.4e-6, 0.12e-6)
    }

    #[test]
    fn capacitance_formula() {
        let c = ivus_cmut();
        let expected = VACUUM_PERMITTIVITY * c.area().into_base() / c.gap.into_base();
        assert!((c.capacitance().into_base() - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn collapse_voltage_scales_gap_to_the_three_halves() {
        let a = cell(14e-6, 0.4e-6, 0.10e-6);
        let b = cell(14e-6, 0.4e-6, 0.20e-6); // 2× gap
        let ratio = b.collapse_voltage().into_base() / a.collapse_voltage().into_base();
        // V_c ∝ g^{3/2}  (k and A unchanged) → ratio = 2^1.5
        assert!(
            (ratio - 2.0_f64.powf(1.5)).abs() / ratio < 1e-6,
            "ratio {ratio}"
        );
        assert!(a.collapse_voltage().into_base() > 0.0);
    }

    #[test]
    fn bias_pulldown_satisfies_force_balance_and_pullin_limit() {
        let c = cell(60e-6, 2.0e-6, 0.2e-6);
        let vc = c.collapse_voltage();
        let k = c.modal_stiffness();
        let eps_a = VACUUM_PERMITTIVITY * c.area().into_base();

        // No bias ⇒ no pull-down.
        assert!(c.bias_pulldown_fraction(voltage(0.0)).unwrap() < 1e-9);

        // Monotone increase toward pull-in.
        let u_low = c
            .bias_pulldown_fraction(voltage(0.5 * vc.into_base()))
            .unwrap();
        let u_high = c
            .bias_pulldown_fraction(voltage(0.9 * vc.into_base()))
            .unwrap();
        assert!(u_low < u_high && u_high < 1.0 / 3.0);

        // Approaches the pull-in displacement g₀/3 as V → V_c⁻. The approach is
        // √-slow (g(u) is flat at its maximum), so even at 0.999 V_c the gap to
        // 1/3 is ~0.017; require it strictly below 1/3 and within 0.03, and that
        // a closer bias gets closer still.
        let u_999 = c
            .bias_pulldown_fraction(voltage(0.999 * vc.into_base()))
            .unwrap();
        let u_9999 = c
            .bias_pulldown_fraction(voltage(0.9999 * vc.into_base()))
            .unwrap();
        assert!(
            u_999 < 1.0 / 3.0 && (1.0 / 3.0 - u_999) < 0.03,
            "near-collapse u={u_999}"
        );
        assert!(
            u_9999 > u_999 && u_9999 < 1.0 / 3.0,
            "tighter bias → closer to pull-in"
        );

        // Differential check: the equilibrium x satisfies kx = ε₀AV²/(2(g₀−x)²).
        for frac in [0.3, 0.6, 0.85] {
            let v = voltage(frac * vc.into_base());
            let x = c.bias_pulldown_fraction(v).unwrap() * c.gap.into_base();
            let spring = k.into_base() * x;
            let electro =
                eps_a * v.into_base() * v.into_base() / (2.0 * (c.gap.into_base() - x).powi(2));
            assert!(
                (spring - electro).abs() <= 1e-6 * spring.max(electro),
                "force balance at V={v:?}: {spring} vs {electro}"
            );
        }

        // Collapsed (V ≥ V_c) ⇒ no stable equilibrium.
        assert!(c.bias_pulldown_fraction(vc).is_none());
        assert!(
            c.bias_pulldown_fraction(voltage(1.5 * vc.into_base()))
                .is_none()
        );
    }

    #[test]
    fn biased_capacitance_rises_and_resonance_softens_to_collapse() {
        let c = cell(60e-6, 2.0e-6, 0.2e-6);
        let vc = c.collapse_voltage();
        let rho = density(1000.0); // water

        // Capacitance C₀/(1−u) exceeds C₀ and grows with bias.
        let c0 = c.capacitance();
        let cap_low = c.biased_capacitance(voltage(0.5 * vc.into_base())).unwrap();
        let cap_high = c.biased_capacitance(voltage(0.9 * vc.into_base())).unwrap();
        assert!(c0 < cap_low && cap_low < cap_high);

        // Resonance softens monotonically with bias (k_eff → 0 at pull-in). The
        // factor is (1−(V/V_c)²)^{1/4}-like near collapse, so it decreases
        // steadily but reaches 0 only at exactly V_c.
        let f0 = c.immersion_resonance(rho);
        let f_low = c
            .bias_softened_resonance(voltage(0.5 * vc.into_base()), rho)
            .unwrap();
        let f_high = c
            .bias_softened_resonance(voltage(0.9 * vc.into_base()), rho)
            .unwrap();
        assert!(
            f_high < f_low && f_low < f0,
            "spring softening: {f_high:?} < {f_low:?} < {f0:?}"
        );
        // At 0.999 V_c the resonance is already strongly reduced (≈0.28 f0)…
        let f_999 = c
            .bias_softened_resonance(voltage(0.999 * vc.into_base()), rho)
            .unwrap();
        assert!(
            f_999.into_base() < 0.35 * f0.into_base(),
            "resonance strongly softened near pull-in: {f_999:?}"
        );
        // …and keeps falling as the bias closes on V_c.
        let f_9999 = c
            .bias_softened_resonance(voltage(0.9999 * vc.into_base()), rho)
            .unwrap();
        assert!(
            f_9999 < f_999,
            "resonance continues to collapse: {f_9999:?} < {f_999:?}"
        );
    }

    #[test]
    fn coupling_rises_with_bias_and_is_bounded() {
        let c = ivus_cmut();
        let vc = c.collapse_voltage();
        assert!(
            c.coupling_k2(voltage(0.5 * vc.into_base()))
                < c.coupling_k2(voltage(0.9 * vc.into_base()))
        );
        assert!(c.coupling_k2(voltage(2.0 * vc.into_base())) <= 0.85); // capped
    }

    #[test]
    fn cmut_is_wide_band_in_blood() {
        let c = ivus_cmut();
        let fbw = c.fractional_bandwidth(density(1060.0)); // blood
        // CMUTs are fluid-coupling dominated → broad fractional bandwidth (>60%)
        assert!(fbw > 0.6, "CMUT FBW {fbw} should be wide");
    }

    #[test]
    fn output_pressure_is_gap_limited_ceiling() {
        // Therapy-scale CMUT (~3 MHz): larger membrane, sub-micron gap.
        let c = cell(60e-6, 2.0e-6, 0.2e-6);
        let (rho, cs) = (density(1000.0), velocity(1500.0)); // water
        let p13 = c.max_output_pressure(rho, cs, 1.0 / 3.0);
        // a bigger gap → proportionally higher ceiling (output ∝ gap)
        let c_big = cell(60e-6, 2.0e-6, 0.4e-6);
        let p_big = c_big.max_output_pressure(rho, cs, 1.0 / 3.0);
        assert!(
            (p_big.into_base() / p13.into_base() - 2.0).abs() < 1e-6,
            "output ∝ gap"
        );
        assert!(p13.into_base() > 0.0);
    }

    #[test]
    fn squeeze_film_damping_scaling_and_q() {
        const AIR_VISC: f64 = 1.8e-5; // Pa·s
        let c = cell(20e-6, 0.5e-6, 0.2e-6);
        let c2a = cell(40e-6, 0.5e-6, 0.2e-6); // 2× radius
        let c2g = cell(20e-6, 0.5e-6, 0.4e-6); // 2× gap
        // c ∝ a⁴ → ×16 ; c ∝ 1/g³ → ÷8
        assert!(
            (c2a.squeeze_film_damping(viscosity(AIR_VISC)).into_base()
                / c.squeeze_film_damping(viscosity(AIR_VISC)).into_base()
                - 16.0)
                .abs()
                < 1e-6
        );
        assert!(
            (c2g.squeeze_film_damping(viscosity(AIR_VISC)).into_base()
                / c.squeeze_film_damping(viscosity(AIR_VISC)).into_base()
                - 0.125)
                .abs()
                < 1e-9
        );
        // more viscous gas → more damping → lower Q
        assert!(
            c.squeeze_film_quality_factor(viscosity(2.0 * AIR_VISC))
                < c.squeeze_film_quality_factor(viscosity(AIR_VISC))
        );
        // squeeze number rises with frequency
        assert!(
            c.squeeze_number(viscosity(AIR_VISC), pressure(101325.0), frequency(2e6))
                > c.squeeze_number(viscosity(AIR_VISC), pressure(101325.0), frequency(1e6))
        );
        assert!(c.squeeze_film_quality_factor(viscosity(AIR_VISC)) > 0.0);
    }

    #[test]
    fn flexing_a_cmut_reduces_output() {
        // sub-micron gap CMUT on a tight catheter wrap (κ = 1/1 mm)
        let c = cell(60e-6, 2.0e-6, 0.2e-6);
        let flat = c
            .flex_gap_derating(aequitas::systems::si::quantities::ReciprocalLength::from_base(0.0));
        let wrapped = c.flex_gap_derating(
            aequitas::systems::si::quantities::ReciprocalLength::from_base(1.0 / 1.0e-3),
        ); // 1 mm radius of curvature
        assert!((flat - 1.0).abs() < 1e-12, "flat = no derating");
        assert!(
            wrapped < flat,
            "flexing must reduce output: {wrapped} < {flat}"
        );
        // a tighter gap is hurt more by the same curvature (the user's concern)
        let tight = cell(60e-6, 2.0e-6, 0.1e-6);
        assert!(
            tight.flex_gap_derating(
                aequitas::systems::si::quantities::ReciprocalLength::from_base(1.0 / 1.0e-3)
            ) < wrapped
        );
    }
}
