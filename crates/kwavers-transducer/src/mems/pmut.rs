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
    pub fn e31f(self) -> f64 {
        match self {
            PiezoFilm::Aln => -1.05,
            PiezoFilm::Pzt => -10.0,
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
    /// Film Young's modulus \[Pa].
    #[must_use]
    pub fn youngs(self) -> f64 {
        match self {
            PiezoFilm::Aln => 320.0e9,
            PiezoFilm::Pzt => 80.0e9,
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
    /// Plate radius `a` \[m].
    pub radius: f64,
    /// Piezo film thickness `t_p` \[m].
    pub piezo_thickness: f64,
    /// Passive (Si) layer thickness `t_s` \[m].
    pub passive_thickness: f64,
    /// Piezo film material.
    pub film: PiezoFilm,
}

impl PmutCell {
    /// Construct a PMUT cell; `None` for non-positive geometry.
    #[must_use]
    pub fn new(
        radius: f64,
        piezo_thickness: f64,
        passive_thickness: f64,
        film: PiezoFilm,
    ) -> Option<Self> {
        if radius > 0.0 && piezo_thickness > 0.0 && passive_thickness > 0.0 {
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

    /// Total plate thickness `t_p + t_s` \[m].
    #[must_use]
    pub fn total_thickness(&self) -> f64 {
        self.piezo_thickness + self.passive_thickness
    }

    /// Membrane area `A = π a²` \[m²].
    #[must_use]
    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    /// Thickness-weighted composite Young's modulus (film + Si passive).
    #[must_use]
    pub fn effective_youngs(&self) -> f64 {
        const SI_YOUNGS: f64 = 169.0e9;
        (self.film.youngs() * self.piezo_thickness + SI_YOUNGS * self.passive_thickness)
            / self.total_thickness()
    }

    /// Thickness-weighted composite density (film + Si passive).
    #[must_use]
    pub fn effective_density(&self) -> f64 {
        const SI_DENSITY: f64 = 2330.0;
        let rho_film = match self.film {
            PiezoFilm::Aln => 3260.0,
            PiezoFilm::Pzt => 7600.0,
        };
        (rho_film * self.piezo_thickness + SI_DENSITY * self.passive_thickness)
            / self.total_thickness()
    }

    /// In-vacuo composite-plate resonance \[Hz].
    #[must_use]
    pub fn vacuum_resonance(&self) -> f64 {
        plate::vacuum_resonance(
            self.effective_youngs(),
            self.total_thickness(),
            0.25,
            self.effective_density(),
            self.radius,
        )
    }

    /// Immersion (fluid-loaded) resonance \[Hz].
    #[must_use]
    pub fn immersion_resonance(&self, density_fluid: f64) -> f64 {
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
    pub fn capacitance(&self) -> f64 {
        VACUUM_PERMITTIVITY * self.film.rel_permittivity() * self.area() / self.piezo_thickness
    }

    /// Effective electromechanical coupling `k_eff²` from the material coupling
    /// `e₃₁,f²/(ε₀ ε_r · Y)` with a flexural geometric factor (~0.5), bounded < 1.
    #[must_use]
    pub fn coupling_k2(&self) -> f64 {
        const GEOMETRIC_FACTOR: f64 = 0.5;
        let e = self.film.e31f();
        let k_mat2 =
            e * e / (VACUUM_PERMITTIVITY * self.film.rel_permittivity() * self.film.youngs());
        (GEOMETRIC_FACTOR * k_mat2).min(0.95)
    }

    /// Dielectric self-heating power `P = π f C V_ac² tan δ` \[W].
    #[must_use]
    pub fn self_heating_power(&self, drive_voltage_ac: f64, freq: f64) -> f64 {
        PI * freq
            * self.capacitance()
            * drive_voltage_ac
            * drive_voltage_ac
            * self.film.loss_tangent()
    }

    /// Relative transmit sensitivity (output pressure per drive volt), `∝ e₃₁,f / t_p`.
    #[must_use]
    pub fn transmit_sensitivity(&self) -> f64 {
        self.film.e31f().abs() / self.piezo_thickness
    }

    /// Radiation quality factor (same small-piston model as the CMUT).
    #[must_use]
    pub fn radiation_q(&self, density_fluid: f64, sound_speed_fluid: f64) -> f64 {
        let f0 = self.immersion_resonance(density_fluid);
        let w0 = TAU * f0;
        let m = plate::modal_mass(
            self.effective_density(),
            self.total_thickness(),
            self.radius,
        );
        let ka = w0 * self.radius / sound_speed_fluid;
        let r_rad = density_fluid * sound_speed_fluid * self.area() * ka * ka / 2.0;
        if r_rad <= 0.0 {
            return f64::INFINITY;
        }
        w0 * m / r_rad
    }

    /// Fluid-loading ratio `β = Γ ρ_f a/(ρ_eff t_total)` (composite areal mass).
    #[must_use]
    pub fn fluid_loading_beta(&self, density_fluid: f64) -> f64 {
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
    pub fn fractional_bandwidth(&self, density_fluid: f64) -> f64 {
        plate::fractional_bandwidth_from_loading(self.fluid_loading_beta(density_fluid))
    }

    /// Piezo-driven peak plate deflection per applied volt \[m·V⁻¹]:
    /// `w/V ≈ η·|e₃₁,f|·a²/(Y_eff·t_total²)` (flexural unimorph, η≈0.3 geometric).
    /// Unlike a CMUT, this is **not gap-limited** — deflection scales with the
    /// drive up to the piezo breakdown field, so PMUTs reach far higher output.
    #[must_use]
    pub fn deflection_per_volt(&self) -> f64 {
        const ETA: f64 = 0.3;
        ETA * self.film.e31f().abs() * self.radius * self.radius
            / (self.effective_youngs() * self.total_thickness().powi(2))
    }

    /// Peak output pressure into the fluid for an AC drive `V` (plane-wave
    /// radiation), `p = ρ c · ω · (w/V)·V` \[Pa]. Scales with drive — the
    /// transmit advantage of PMUTs for therapy.
    #[must_use]
    pub fn max_output_pressure(
        &self,
        drive_voltage: f64,
        density_fluid: f64,
        sound_speed_fluid: f64,
    ) -> f64 {
        let f = self.immersion_resonance(density_fluid);
        let w = self.deflection_per_volt() * drive_voltage;
        density_fluid * sound_speed_fluid * TAU * f * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ivus_pmut(film: PiezoFilm) -> PmutCell {
        // IVUS-scale PMUT: a=20 µm, 1 µm piezo on 2 µm Si
        PmutCell::new(20e-6, 1e-6, 2e-6, film).unwrap()
    }

    #[test]
    fn pzt_couples_more_than_aln() {
        assert!(ivus_pmut(PiezoFilm::Pzt).coupling_k2() > ivus_pmut(PiezoFilm::Aln).coupling_k2());
        assert!(ivus_pmut(PiezoFilm::Aln).coupling_k2() > 0.0);
    }

    #[test]
    fn pzt_self_heats_more_than_aln() {
        let (v, f) = (5.0, 40e6);
        let pzt = ivus_pmut(PiezoFilm::Pzt).self_heating_power(v, f);
        let aln = ivus_pmut(PiezoFilm::Aln).self_heating_power(v, f);
        assert!(pzt > aln, "PZT heating {pzt} should exceed AlN {aln}");
    }

    #[test]
    fn capacitance_and_resonance_are_physical() {
        let p = ivus_pmut(PiezoFilm::Aln);
        assert!(p.capacitance() > 0.0);
        assert!(p.immersion_resonance(1060.0) < p.vacuum_resonance());
        // IVUS-band resonance (tens of MHz)
        assert!(p.immersion_resonance(1060.0) > 5e6);
    }

    #[test]
    fn output_pressure_scales_with_drive_not_gap() {
        // therapy-scale PZT PMUT (~3 MHz): larger plate
        let p = PmutCell::new(60e-6, 2e-6, 4e-6, PiezoFilm::Pzt).unwrap();
        let (rho, c) = (1000.0, 1500.0); // water
        let p10 = p.max_output_pressure(10.0, rho, c);
        let p20 = p.max_output_pressure(20.0, rho, c);
        // doubling the drive doubles output (not gap-limited)
        assert!((p20 / p10 - 2.0).abs() < 1e-9, "PMUT output ∝ drive");
        // PZT drives harder than AlN (higher e31f)
        let aln = PmutCell::new(60e-6, 2e-6, 4e-6, PiezoFilm::Aln).unwrap();
        assert!(p.max_output_pressure(10.0, rho, c) > aln.max_output_pressure(10.0, rho, c));
    }
}
