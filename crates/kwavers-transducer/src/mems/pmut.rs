//! PMUT — piezoelectric micromachined ultrasonic transducer cell.
//!
//! A piezoelectric thin film on a passive plate (unimorph) that bends when driven.
//! PMUTs operate at low voltage with high transmit sensitivity (piezoelectric
//! drive), but have narrower bandwidth than CMUTs and — with PZT — higher
//! dielectric loss (self-heating).
//!
//! Models (closed-form, lumped): composite clamped-plate resonance, film
//! capacitance, effective electromechanical coupling from `e₃₁,f`, dielectric
//! self-heating, transmit sensitivity, and radiation-limited fractional bandwidth.
//!
//! # References
//! - Muralt, P., et al. (2009). "Piezoelectric micromachined ultrasonic
//!   transducers based on PZT thin films." *IEEE TUFFC*, 52(12).
//! - Jung, J., et al. (2017). "Review of PMUTs." *J. Micromech. Microeng.*, 27(11).

use aequitas::systems::si::quantities::{
    Capacitance, ElectricPotential, Frequency, Length, MassDensity, Power, Pressure,
    PressurePerElectricPotential, SurfaceChargeDensity, Velocity,
};
use core::f64::consts::{PI, TAU};
use kwavers_core::constants::fundamental::VACUUM_PERMITTIVITY;

use super::plate;

/// Piezoelectric thin-film material for the active layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiezoFilm {
    /// Aluminium nitride — low loss, low coupling, CMOS-friendly.
    Aln,
    /// Lead zirconate titanate — high coupling, higher loss.
    Pzt,
}

impl PiezoFilm {
    /// Transverse piezoelectric stress coefficient `e₃₁,f` \[C·m⁻²].
    #[must_use]
    pub fn e31f(self) -> SurfaceChargeDensity {
        match self {
            PiezoFilm::Aln => SurfaceChargeDensity::from_base(-1.05),
            PiezoFilm::Pzt => SurfaceChargeDensity::from_base(-10.0),
        }
    }
    /// Relative permittivity `ε_r`.
    #[must_use]
    pub fn rel_permittivity(self) -> f64 {
        match self {
            PiezoFilm::Aln => 10.5,
            PiezoFilm::Pzt => 1300.0,
        }
    }
    /// Film Young's modulus \`Pa`.
    #[must_use]
    pub fn youngs(self) -> Pressure {
        match self {
            PiezoFilm::Aln => Pressure::from_base(320.0e9),
            PiezoFilm::Pzt => Pressure::from_base(80.0e9),
        }
    }
    /// Dielectric loss tangent `tan δ`.
    #[must_use]
    pub fn loss_tangent(self) -> f64 {
        match self {
            PiezoFilm::Aln => 0.003,
            PiezoFilm::Pzt => 0.02,
        }
    }
}

/// A single PMUT cell: piezo film `t_p` on a passive (Si) plate `t_s`, radius `a`.
#[derive(Debug, Clone, Copy)]
pub struct PmutCell {
    /// Plate radius `a` \`m`.
    pub radius: Length,
    /// Piezo film thickness `t_p` \`m`.
    pub piezo_thickness: Length,
    /// Passive (Si) layer thickness `t_s` \`m`.
    pub passive_thickness: Length,
    /// Piezo film material.
    pub film: PiezoFilm,
}

impl PmutCell {
    /// Construct a PMUT cell; `None` for non-positive geometry.
    #[must_use]
    pub fn new(
        radius: Length,
        piezo_thickness: Length,
        passive_thickness: Length,
        film: PiezoFilm,
    ) -> Option<Self> {
        if radius.into_base() > 0.0
            && piezo_thickness.into_base() > 0.0
            && passive_thickness.into_base() > 0.0
        {
            Some(Self {
                radius,
                piezo_thickness,
                passive_thickness,
                film,
            })
        } else {
            None
        }
    }

    /// Total plate thickness `t_p + t_s` \`m`.
    #[must_use]
    pub fn total_thickness(&self) -> Length {
        Length::from_base(self.piezo_thickness.into_base() + self.passive_thickness.into_base())
    }

    /// Membrane area `A = π a²` \`m²`.
    #[must_use]
    pub fn area(&self) -> aequitas::systems::si::quantities::Area {
        aequitas::systems::si::quantities::Area::from_base(
            PI * self.radius.into_base() * self.radius.into_base(),
        )
    }

    /// Thickness-weighted composite Young's modulus (film + Si passive).
    #[must_use]
    pub fn effective_youngs(&self) -> Pressure {
        let si_youngs = Pressure::from_base(169.0e9);
        Pressure::from_base(
            (self.film.youngs().into_base() * self.piezo_thickness.into_base()
                + si_youngs.into_base() * self.passive_thickness.into_base())
                / self.total_thickness().into_base(),
        )
    }

    /// Thickness-weighted composite density (film + Si passive).
    #[must_use]
    pub fn effective_density(&self) -> MassDensity {
        let si_density = MassDensity::from_base(2330.0);
        let rho_film = match self.film {
            PiezoFilm::Aln => MassDensity::from_base(3260.0),
            PiezoFilm::Pzt => MassDensity::from_base(7600.0),
        };
        MassDensity::from_base(
            (rho_film.into_base() * self.piezo_thickness.into_base()
                + si_density.into_base() * self.passive_thickness.into_base())
                / self.total_thickness().into_base(),
        )
    }

    /// In-vacuo composite-plate resonance \`Hz`.
    #[must_use]
    pub fn vacuum_resonance(&self) -> Frequency {
        plate::vacuum_resonance(
            self.effective_youngs(),
            self.total_thickness(),
            0.25,
            self.effective_density(),
            self.radius,
        )
    }

    /// Immersion (fluid-loaded) resonance \`Hz`.
    #[must_use]
    pub fn immersion_resonance(&self, density_fluid: MassDensity) -> Frequency {
        plate::immersion_resonance(
            self.vacuum_resonance(),
            self.effective_density(),
            self.total_thickness(),
            density_fluid,
            self.radius,
        )
    }

    /// Film capacitance `C₀ = ε₀ ε_r A / t_p` \[F].
    #[must_use]
    pub fn capacitance(&self) -> Capacitance {
        Capacitance::from_base(
            VACUUM_PERMITTIVITY * self.film.rel_permittivity() * self.area().into_base()
                / self.piezo_thickness.into_base(),
        )
    }

    /// Effective electromechanical coupling `k_eff²` from the material coupling
    /// `e₃₁,f²/(ε₀ ε_r · Y)` with a flexural geometric factor (~0.5), bounded < 1.
    #[must_use]
    pub fn coupling_k2(&self) -> f64 {
        const GEOMETRIC_FACTOR: f64 = 0.5;
        let e = self.film.e31f().into_base();
        let k_mat2 = e * e
            / (VACUUM_PERMITTIVITY * self.film.rel_permittivity() * self.film.youngs().into_base());
        (GEOMETRIC_FACTOR * k_mat2).min(0.95)
    }

    /// Dielectric self-heating power `P = π f C V_ac² tan δ` \`W`.
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
                * self.film.loss_tangent(),
        )
    }

    /// Relative transmit sensitivity (output pressure per drive volt), `∝ e₃₁,f / t_p`.
    #[must_use]
    pub fn transmit_sensitivity(&self) -> PressurePerElectricPotential {
        PressurePerElectricPotential::from_base(
            self.film.e31f().into_base().abs() / self.piezo_thickness.into_base(),
        )
    }

    /// Radiation quality factor (same small-piston model as the CMUT).
    #[must_use]
    pub fn radiation_q(&self, density_fluid: MassDensity, sound_speed_fluid: Velocity) -> f64 {
        let f0 = self.immersion_resonance(density_fluid);
        let w0 = TAU * f0.into_base();
        let m = plate::modal_mass(
            self.effective_density(),
            self.total_thickness(),
            self.radius,
        );
        let ka = w0 * self.radius.into_base() / sound_speed_fluid.into_base();
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

    /// Fluid-loading ratio `β = Γ ρ_f a/(ρ_eff t_total)` (composite areal mass).
    #[must_use]
    pub fn fluid_loading_beta(&self, density_fluid: MassDensity) -> f64 {
        plate::fluid_loading_beta(
            density_fluid,
            self.effective_density(),
            self.total_thickness(),
            self.radius,
        )
    }

    /// −6 dB fractional bandwidth from fluid loading. PMUT plates are heavier and
    /// stiffer than CMUT membranes → narrower bandwidth.
    #[must_use]
    pub fn fractional_bandwidth(&self, density_fluid: MassDensity) -> f64 {
        plate::fractional_bandwidth_from_loading(self.fluid_loading_beta(density_fluid))
    }

    /// Piezo-driven peak plate deflection per applied volt \[m·V⁻¹]:
    /// `w/V ≈ η·|e₃₁,f|·a²/(Y_eff·t_total²)` (flexural unimorph, η≈0.3 geometric).
    /// Unlike a CMUT, this is **not gap-limited** — deflection scales with the
    /// drive up to the piezo breakdown field, so PMUTs reach far higher output.
    #[must_use]
    pub fn deflection_per_volt(
        &self,
    ) -> aequitas::systems::si::quantities::LengthPerElectricPotential {
        const ETA: f64 = 0.3;
        aequitas::systems::si::quantities::LengthPerElectricPotential::from_base(
            ETA * self.film.e31f().into_base().abs()
                * self.radius.into_base()
                * self.radius.into_base()
                / (self.effective_youngs().into_base()
                    * self.total_thickness().into_base().powi(2)),
        )
    }

    /// Peak output pressure into the fluid for an AC drive `V` (plane-wave
    /// radiation), `p = ρ c · ω · (w/V)·V` \`Pa`. Scales with drive — the
    /// transmit advantage of PMUTs for therapy.
    #[must_use]
    pub fn max_output_pressure(
        &self,
        drive_voltage: ElectricPotential,
        density_fluid: MassDensity,
        sound_speed_fluid: Velocity,
    ) -> Pressure {
        let f = self.immersion_resonance(density_fluid);
        let w = self.deflection_per_volt().into_base() * drive_voltage.into_base();
        Pressure::from_base(
            density_fluid.into_base() * sound_speed_fluid.into_base() * TAU * f.into_base() * w,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(value: f64) -> Length {
        Length::from_base(value)
    }

    fn density(value: f64) -> MassDensity {
        MassDensity::from_base(value)
    }

    fn velocity(value: f64) -> Velocity {
        Velocity::from_base(value)
    }

    fn voltage(value: f64) -> ElectricPotential {
        ElectricPotential::from_base(value)
    }

    fn frequency(value: f64) -> Frequency {
        Frequency::from_base(value)
    }

    fn ivus_pmut(film: PiezoFilm) -> PmutCell {
        // IVUS-scale PMUT: a=20 µm, 1 µm piezo on 2 µm Si
        PmutCell::new(length(20e-6), length(1e-6), length(2e-6), film).unwrap()
    }

    #[test]
    fn pzt_couples_more_than_aln() {
        assert!(ivus_pmut(PiezoFilm::Pzt).coupling_k2() > ivus_pmut(PiezoFilm::Aln).coupling_k2());
        assert!(ivus_pmut(PiezoFilm::Aln).coupling_k2() > 0.0);
    }

    #[test]
    fn pzt_self_heats_more_than_aln() {
        let (v, f) = (voltage(5.0), frequency(40e6));
        let pzt = ivus_pmut(PiezoFilm::Pzt).self_heating_power(v, f);
        let aln = ivus_pmut(PiezoFilm::Aln).self_heating_power(v, f);
        assert!(
            pzt.into_base() > aln.into_base(),
            "PZT heating {pzt:?} should exceed AlN {aln:?}"
        );
    }

    #[test]
    fn capacitance_and_resonance_are_physical() {
        let p = ivus_pmut(PiezoFilm::Aln);
        assert!(p.capacitance().into_base() > 0.0);
        assert!(p.immersion_resonance(density(1060.0)) < p.vacuum_resonance());
        // IVUS-band resonance (tens of MHz)
        assert!(p.immersion_resonance(density(1060.0)).into_base() > 5e6);
    }

    #[test]
    fn output_pressure_scales_with_drive_not_gap() {
        // therapy-scale PZT PMUT (~3 MHz): larger plate
        let p = PmutCell::new(length(60e-6), length(2e-6), length(4e-6), PiezoFilm::Pzt).unwrap();
        let (rho, c) = (density(1000.0), velocity(1500.0)); // water
        let p10 = p.max_output_pressure(voltage(10.0), rho, c);
        let p20 = p.max_output_pressure(voltage(20.0), rho, c);
        // doubling the drive doubles output (not gap-limited)
        assert!(
            (p20.into_base() / p10.into_base() - 2.0).abs() < 1e-9,
            "PMUT output ∝ drive"
        );
        // PZT drives harder than AlN (higher e31f)
        let aln = PmutCell::new(length(60e-6), length(2e-6), length(4e-6), PiezoFilm::Aln).unwrap();
        assert!(
            p.max_output_pressure(voltage(10.0), rho, c).into_base()
                > aln.max_output_pressure(voltage(10.0), rho, c).into_base()
        );
    }
}
