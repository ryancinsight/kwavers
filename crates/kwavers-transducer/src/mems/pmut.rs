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

#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use aequitas::systems::si::quantities::{
    Area, Capacitance, Dimensionless, ElectricPotential, Frequency, Length, MassDensity, Power,
    Pressure, SurfaceChargeDensity, Velocity, VolumeChargeDensity,
};
use aequitas::systems::si::units::{
    CoulombPerCubicMeter, CoulombPerSquareMeter, Farad, Hertz, KilogramPerCubicMeter, Meter,
    MeterPerSecond, Pascal, SquareMeter, Volt, Watt,
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
            PiezoFilm::Aln => SurfaceChargeDensity::from_unit::<CoulombPerSquareMeter>(-1.05),
            PiezoFilm::Pzt => SurfaceChargeDensity::from_unit::<CoulombPerSquareMeter>(-10.0),
        }
    }
    /// Relative permittivity `ε_r`.
    #[must_use]
    pub fn rel_permittivity(self) -> Dimensionless {
        match self {
            PiezoFilm::Aln => Dimensionless::from_base(10.5),
            PiezoFilm::Pzt => Dimensionless::from_base(1300.0),
        }
    }
    /// Film Young's modulus \`Pa`.
    #[must_use]
    pub fn youngs(self) -> Pressure {
        match self {
            PiezoFilm::Aln => Pressure::from_unit::<Pascal>(320.0e9),
            PiezoFilm::Pzt => Pressure::from_unit::<Pascal>(80.0e9),
        }
    }
    /// Dielectric loss tangent `tan δ`.
    #[must_use]
    pub fn loss_tangent(self) -> Dimensionless {
        match self {
            PiezoFilm::Aln => Dimensionless::from_base(0.003),
            PiezoFilm::Pzt => Dimensionless::from_base(0.02),
        }
    }
}

/// A single PMUT cell: piezo film `t_p` on a passive (Si) plate `t_s`, radius `a`.
#[derive(Debug, Clone, Copy)]
pub struct PmutCell {
    /// Plate radius `a` \`m`.
    pub radius: Length<f64>,
    /// Piezo film thickness `t_p` \`m`.
    pub piezo_thickness: Length<f64>,
    /// Passive (Si) layer thickness `t_s` \`m`.
    pub passive_thickness: Length<f64>,
    /// Piezo film material.
    pub film: PiezoFilm,
}

impl PmutCell {
    #[inline]
    fn raw_dims(&self) -> (f64, f64, f64) {
        (
            self.radius.in_unit::<Meter>(),
            self.piezo_thickness.in_unit::<Meter>(),
            self.passive_thickness.in_unit::<Meter>(),
        )
    }

    /// Construct a PMUT cell; `None` for non-positive geometry.
    #[must_use]
    pub fn new(
        radius: Length<f64>,
        piezo_thickness: Length<f64>,
        passive_thickness: Length<f64>,
        film: PiezoFilm,
    ) -> Option<Self> {
        let (radius_m, piezo_thickness_m, passive_thickness_m) = (
            radius.in_unit::<Meter>(),
            piezo_thickness.in_unit::<Meter>(),
            passive_thickness.in_unit::<Meter>(),
        );

        if radius_m > 0.0 && piezo_thickness_m > 0.0 && passive_thickness_m > 0.0 {
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
        let (_r, tp, ts) = self.raw_dims();
        Length::from_unit::<Meter>(tp + ts)
    }

    /// Membrane area `A = π a²` \`m²`.
    #[must_use]
    pub fn area(&self) -> Area {
        let (r, _tp, _ts) = self.raw_dims();
        Area::from_unit::<SquareMeter>(PI * r * r)
    }

    /// Thickness-weighted composite Young's modulus (film + Si passive).
    #[must_use]
    pub fn effective_youngs(&self) -> Pressure {
        let (_r, tp, ts) = self.raw_dims();
        const SI_YOUNGS: f64 = 169.0e9;
        Pressure::from_unit::<Pascal>(
            (self.film.youngs().in_unit::<Pascal>() * tp + SI_YOUNGS * ts) / (tp + ts),
        )
    }

    /// Thickness-weighted composite density (film + Si passive).
    #[must_use]
    pub fn effective_density(&self) -> MassDensity {
        let (_r, tp, ts) = self.raw_dims();
        const SI_DENSITY: f64 = 2330.0;
        let rho_film = match self.film {
            PiezoFilm::Aln => 3260.0,
            PiezoFilm::Pzt => 7600.0,
        };
        MassDensity::from_unit::<KilogramPerCubicMeter>(
            (rho_film * tp + SI_DENSITY * ts) / (tp + ts),
        )
    }

    /// In-vacuo composite-plate resonance \`Hz`.
    #[must_use]
    pub fn vacuum_resonance(&self) -> Frequency {
        let (r, tp, ts) = self.raw_dims();
        plate::vacuum_resonance(
            self.effective_youngs(),
            Length::from_base(tp + ts),
            Dimensionless::from_base(0.25),
            self.effective_density(),
            Length::from_base(r),
        )
    }

    /// Immersion (fluid-loaded) resonance \`Hz`.
    #[must_use]
    pub fn immersion_resonance(&self, density_fluid: MassDensity) -> Frequency {
        let (r, tp, ts) = self.raw_dims();
        plate::immersion_resonance(
            self.vacuum_resonance(),
            self.effective_density(),
            Length::from_base(tp + ts),
            density_fluid,
            Length::from_base(r),
        )
    }

    /// Film capacitance `C₀ = ε₀ ε_r A / t_p` \[F].
    #[must_use]
    pub fn capacitance(&self) -> Capacitance {
        let (r, tp, _ts) = self.raw_dims();
        Capacitance::from_unit::<Farad>(
            VACUUM_PERMITTIVITY * self.film.rel_permittivity().into_base() * PI * r * r / tp,
        )
    }

    /// Effective electromechanical coupling `k_eff²` from the material coupling
    /// `e₃₁,f²/(ε₀ ε_r · Y)` with a flexural geometric factor (~0.5), bounded < 1.
    #[must_use]
    pub fn coupling_k2(&self) -> Dimensionless {
        let (_r, _tp, _ts) = self.raw_dims();
        const GEOMETRIC_FACTOR: f64 = 0.5;
        let e = self.film.e31f().in_unit::<CoulombPerSquareMeter>();
        let k_mat2 = e * e
            / (VACUUM_PERMITTIVITY
                * self.film.rel_permittivity().into_base()
                * self.film.youngs().in_unit::<Pascal>());
        Dimensionless::from_base((GEOMETRIC_FACTOR * k_mat2).min(0.95))
    }

    /// Dielectric self-heating power `P = π f C V_ac² tan δ` \`W`.
    #[must_use]
    pub fn self_heating_power(
        &self,
        drive_voltage_ac: ElectricPotential,
        freq: Frequency,
    ) -> Power {
        let (_r, _tp, _ts) = self.raw_dims();
        Power::from_unit::<Watt>(
            PI * freq.in_unit::<Hertz>()
                * self.capacitance().in_unit::<Farad>()
                * drive_voltage_ac.in_unit::<Volt>()
                * drive_voltage_ac.in_unit::<Volt>()
                * self.film.loss_tangent().into_base(),
        )
    }

    /// Piezoelectric volume-charge-density coefficient `|e₃₁,f|/t_p`.
    #[must_use]
    pub fn charge_density_gradient(&self) -> VolumeChargeDensity {
        let (_r, tp, _ts) = self.raw_dims();
        VolumeChargeDensity::from_unit::<CoulombPerCubicMeter>(
            self.film.e31f().in_unit::<CoulombPerSquareMeter>().abs() / tp,
        )
    }

    /// Radiation quality factor (same small-piston model as the CMUT).
    #[must_use]
    pub fn radiation_q(
        &self,
        density_fluid: MassDensity,
        sound_speed_fluid: Velocity,
    ) -> Dimensionless {
        let (r, tp, ts) = self.raw_dims();
        let f0 = self.immersion_resonance(density_fluid).in_unit::<Hertz>();
        let w0 = TAU * f0;
        let m = plate::modal_mass(
            self.effective_density(),
            Length::from_base(tp + ts),
            Length::from_base(r),
        )
        .in_unit::<aequitas::systems::si::units::Kilogram>();
        let ka = w0 * r / sound_speed_fluid.in_unit::<MeterPerSecond>();
        let r_rad = density_fluid.in_unit::<KilogramPerCubicMeter>()
            * sound_speed_fluid.in_unit::<MeterPerSecond>()
            * PI
            * r
            * r
            * ka
            * ka
            / 2.0;
        if r_rad <= 0.0 {
            return Dimensionless::from_base(f64::INFINITY);
        }
        Dimensionless::from_base(w0 * m / r_rad)
    }

    /// Fluid-loading ratio `β = Γ ρ_f a/(ρ_eff t_total)` (composite areal mass).
    #[must_use]
    pub fn fluid_loading_beta(&self, density_fluid: MassDensity) -> Dimensionless {
        let (r, tp, ts) = self.raw_dims();
        plate::fluid_loading_beta(
            density_fluid,
            self.effective_density(),
            Length::from_base(tp + ts),
            Length::from_base(r),
        )
    }

    /// −6 dB fractional bandwidth from fluid loading. PMUT plates are heavier and
    /// stiffer than CMUT membranes → narrower bandwidth.
    #[must_use]
    pub fn fractional_bandwidth(&self, density_fluid: MassDensity) -> Dimensionless {
        let (_r, _tp, _ts) = self.raw_dims();
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
        let (r, tp, ts) = self.raw_dims();
        const ETA: f64 = 0.3;
        aequitas::systems::si::quantities::LengthPerElectricPotential::from_unit::<
            aequitas::systems::si::units::MeterPerVolt,
        >(
            ETA * self.film.e31f().in_unit::<CoulombPerSquareMeter>().abs() * r * r
                / (self.effective_youngs().in_unit::<Pascal>() * (tp + ts).powi(2)),
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
        let (_r, _tp, _ts) = self.raw_dims();
        let f = self.immersion_resonance(density_fluid).in_unit::<Hertz>();
        let w = self
            .deflection_per_volt()
            .in_unit::<aequitas::systems::si::units::MeterPerVolt>()
            * drive_voltage.in_unit::<Volt>();
        Pressure::from_unit::<Pascal>(
            density_fluid.in_unit::<KilogramPerCubicMeter>()
                * sound_speed_fluid.in_unit::<MeterPerSecond>()
                * TAU
                * f
                * w,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meters(value: f64) -> Length<f64> {
        Length::from_unit::<Meter>(value)
    }

    fn ivus_pmut(film: PiezoFilm) -> PmutCell {
        // IVUS-scale PMUT: a=20 µm, 1 µm piezo on 2 µm Si
        PmutCell::new(meters(20e-6), meters(1e-6), meters(2e-6), film).unwrap()
    }

    #[test]
    fn pzt_couples_more_than_aln() {
        assert!(
            ivus_pmut(PiezoFilm::Pzt).coupling_k2().into_base()
                > ivus_pmut(PiezoFilm::Aln).coupling_k2().into_base()
        );
        assert!(ivus_pmut(PiezoFilm::Aln).coupling_k2().into_base() > 0.0);
    }

    #[test]
    fn pzt_self_heats_more_than_aln() {
        let v = ElectricPotential::from_unit::<Volt>(5.0);
        let f = Frequency::from_unit::<Hertz>(40e6);
        let pzt = ivus_pmut(PiezoFilm::Pzt).self_heating_power(v, f);
        let aln = ivus_pmut(PiezoFilm::Aln).self_heating_power(v, f);
        assert!(
            pzt.in_unit::<Watt>() > aln.in_unit::<Watt>(),
            "PZT heating {} should exceed AlN {}",
            pzt.in_unit::<Watt>(),
            aln.in_unit::<Watt>()
        );
    }

    #[test]
    fn capacitance_and_resonance_are_physical() {
        let p = ivus_pmut(PiezoFilm::Aln);
        assert!(p.capacitance().in_unit::<Farad>() > 0.0);
        let blood = MassDensity::from_unit::<KilogramPerCubicMeter>(1060.0);
        let immersion = p.immersion_resonance(blood).in_unit::<Hertz>();
        let vacuum = p.vacuum_resonance().in_unit::<Hertz>();
        assert!(immersion < vacuum);
        // IVUS-band resonance (tens of MHz)
        assert!(immersion > 5e6);
    }

    #[test]
    fn output_pressure_scales_with_drive_not_gap() {
        // therapy-scale PZT PMUT (~3 MHz): larger plate
        let p = PmutCell::new(meters(60e-6), meters(2e-6), meters(4e-6), PiezoFilm::Pzt).unwrap();
        let (rho, c) = (
            MassDensity::from_unit::<KilogramPerCubicMeter>(1000.0),
            Velocity::from_unit::<MeterPerSecond>(1500.0),
        ); // water
        let p10 = p
            .max_output_pressure(ElectricPotential::from_unit::<Volt>(10.0), rho, c)
            .in_unit::<Pascal>();
        let p20 = p
            .max_output_pressure(ElectricPotential::from_unit::<Volt>(20.0), rho, c)
            .in_unit::<Pascal>();
        // doubling the drive doubles output (not gap-limited)
        assert!((p20 / p10 - 2.0).abs() < 1e-9, "PMUT output ∝ drive");
        // PZT drives harder than AlN (higher e31f)
        let aln = PmutCell::new(meters(60e-6), meters(2e-6), meters(4e-6), PiezoFilm::Aln).unwrap();
        assert!(
            p.max_output_pressure(ElectricPotential::from_unit::<Volt>(10.0), rho, c)
                .in_unit::<Pascal>()
                > aln
                    .max_output_pressure(ElectricPotential::from_unit::<Volt>(10.0), rho, c)
                    .in_unit::<Pascal>()
        );
    }
}
