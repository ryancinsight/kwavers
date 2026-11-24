//! Population evolution under an imaging pulse train
//! (Build 14 / B2, KW-PHYS-011).
//!
//! A triplet-pulse acquisition insonifies the cloud it images:
//! Kuji 2024 (§2.6) acquires four imaging pulses (phases
//! 120°/0°/180°/240°) at 200 µs intervals, 1 ms after the heating
//! burst; Iwasaki 2018 (§2.4) three pulses at 5 kHz PRF, 0.6 ms
//! after the burst. This module evolves a bubble pulse-to-pulse
//! over such a train:
//!
//! * **During each pulse:** rectified diffusion at the
//!   Fyrillas–Szeri / Louisnard–Gomez cycle-averaged rate — the
//!   law itself is **not** duplicated here; the caller supplies
//!   the rate from
//!   [`crate::acoustics::mechanics::cavitation::core::louisnard_gomez_growth_rate`]
//!   evaluated on the bubble's Keller–Miksis history for that
//!   pulse, and this module applies `ΔR = rate × on-time`.
//! * **Between pulses:** Epstein–Plesset dissolution (full form:
//!   Laplace term + transient term, the formation harness's
//!   `b2_gap.ep_shrink_factor`), integrated over the inter-pulse
//!   gap at the bubble's advancing age.
//! * **Fragmentation gate:** the project's Stage-7 kernel gate,
//!   `Rmax/R0 > 5` ([`IMAGING_FRAGMENTATION_GATE_RMAX_R0`]),
//!   recorded per pulse. When it fires the caller may apply the
//!   population fragmentation operators
//!   ([`super::fragmentation`]); this module does not fragment on
//!   its own (in the validated regime below it never fires).
//! * **Removal:** a bubble shrunk below 0.1 µm is dissolved
//!   (population loss), the B2 removal rule.
//!
//! # Validation — B14 B2 (validated NEGATIVE result)
//!
//! Validated against Build 14 stage B2 (`B14_B2_RESULTS.md`,
//! Kuji medium: 25 °C, D = 2.0e-9 m²/s, C_sat = 0.023 kg/m³,
//! ρ_g0 = 1.18 kg/m³, σ = 0.072 N/m, saturation ratio 0.175 —
//! Kuji §2.5's degassed condition; bubble age 1 ms at imaging):
//!
//! * **Population loss is 0 % at every imaging PNP
//!   (0.12 / 0.26 / 0.72 / 1.23 MPa) and every radius** over the
//!   4-pulse / 600 µs acquisition.
//! * The dominant change is the PNP-**independent**
//!   Epstein–Plesset shrink over the three 200 µs gaps:
//!   **−1.74 % (R0 = 1.5 µm), −1.05 % (2 µm), −0.53 % (3 µm),
//!   −0.24 % (5 µm)** — reproduced by the tests below.
//! * Rectified diffusion during the pulses contributes
//!   ≤ 0.0006 µm (total acoustic on-time ≈ 8 µs at 5 cycles/pulse)
//!   — four orders below the gap dissolution.
//! * The fragmentation gate fires only for R0 ≈ 1 µm bubbles
//!   (outside the standard clouds) at 0.72 / 1.23 MPa; for
//!   R0 ≥ 1.5 µm the maximum Rmax/R0 over the B2 grid is 4.55.
//!
//! Conclusion carried by this module: within one imaging
//! acquisition the cloud is, to ≤ 2 % in radius and exactly in
//! count, the cloud the trigger made — PNP-dependent evolution
//! during imaging does **not** create Kuji's 0.26 MPa contrast
//! peak (neither its rising nor its falling side).
//!
//! # Regime
//!
//! **Imaging pressures only** (PNP ≈ 0.1–1.3 MPa, µs-scale total
//! on-time, sub-ms spans). This module must not be applied to
//! trigger/burst pressures (10–20 MPa, thousands of cycles):
//! there the per-pulse linearised rectified term and the
//! no-fragmentation bookkeeping are both invalid, and the
//! formation-side laws (Bader influx, Louisnard–Gomez per-cycle
//! influx, the population operators) are the correct components.
//! The Louisnard–Gomez law's own validated regime
//! (p = PA/P0 ∈ [1, 5], f = 20–100 kHz) covers the 0.12 and
//! 0.26 MPa imaging points; 0.72 / 1.23 MPa are extrapolations of
//! that law — flagged in B2, and immaterial there because the
//! on-time integral is negligible either way.

/// Fragmentation gate of the project's Stage-7 kernel, as a
/// per-pulse `Rmax/R0` threshold. Distinct from the Chomas
/// max/min diameter indicator
/// ([`super::fragmentation::CHOMAS_FRAGMENTATION_DIAMETER_RATIO`]):
/// this gate compares the expansion ratio against equilibrium.
pub const IMAGING_FRAGMENTATION_GATE_RMAX_R0: f64 = 5.0;

/// Dissolution removal radius [m] (B2 rule): below this a bubble
/// is counted as dissolved (population loss).
pub const DISSOLUTION_REMOVAL_RADIUS_M: f64 = 0.1e-6;

/// Medium + saturation state for Epstein–Plesset dissolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpsteinPlessetMedium {
    /// Gas diffusivity in the liquid `D` [m²/s].
    pub diffusivity_m2_s: f64,
    /// Saturation mass concentration `C_sat` [kg/m³].
    pub saturation_concentration_kg_m3: f64,
    /// Gas density at ambient conditions `ρ_g0` [kg/m³].
    pub gas_density_kg_m3: f64,
    /// Surface tension `σ` [N/m].
    pub surface_tension_n_m: f64,
    /// Ambient pressure `p0` [Pa].
    pub ambient_pressure_pa: f64,
    /// Far-field saturation ratio `C∞ / C_sat` (dimensionless).
    pub saturation_ratio: f64,
}

impl EpsteinPlessetMedium {
    /// Kuji 2024 imaging medium (B2): 25 °C, D = 2.0e-9 m²/s,
    /// C_sat = 0.023 kg/m³, ρ_g0 = 1.18 kg/m³, σ = 0.072 N/m,
    /// saturation ratio 0.175 (paper §2.5: tank water degassed to
    /// 15–20 % dissolved oxygen; tissue degassed 3 h at 0.08 MPa).
    #[must_use]
    pub const fn kuji_degassed() -> Self {
        Self {
            diffusivity_m2_s: 2.0e-9,
            saturation_concentration_kg_m3: 0.023,
            gas_density_kg_m3: 1.18,
            surface_tension_n_m: 0.072,
            ambient_pressure_pa: 101_325.0,
            saturation_ratio: 0.175,
        }
    }
}

/// One Epstein–Plesset dissolution step over `[age, age + dt]`,
/// holding R constant over the step and integrating the transient
/// term analytically (the formation harness's full form):
///
/// `dR² = −2D (C_s − C∞) R / (ρ_g0 (R + 2a/3)) · (dt + 2R(√(age+dt) − √age)/√(πD))`,
/// with `a = 2σ/p0` and `C_s = C_sat (1 + a/R)`.
///
/// Returns the new radius [m]; 0 when the bubble dissolves within
/// the step.
#[must_use]
pub fn epstein_plesset_step(
    radius_m: f64,
    age_s: f64,
    dt_s: f64,
    medium: &EpsteinPlessetMedium,
) -> f64 {
    if radius_m <= 0.0 || dt_s <= 0.0 {
        return radius_m.max(0.0);
    }
    let a = 2.0 * medium.surface_tension_n_m / medium.ambient_pressure_pa;
    let c_s = medium.saturation_concentration_kg_m3 * (1.0 + a / radius_m);
    let c_inf = medium.saturation_ratio * medium.saturation_concentration_kg_m3;
    let t_int = dt_s
        + 2.0 * radius_m * ((age_s + dt_s).sqrt() - age_s.sqrt())
            / (std::f64::consts::PI * medium.diffusivity_m2_s).sqrt();
    let d_r2 = -2.0 * medium.diffusivity_m2_s * (c_s - c_inf) * radius_m
        / (medium.gas_density_kg_m3 * (radius_m + 2.0 * a / 3.0))
        * t_int;
    let r2 = radius_m * radius_m + d_r2;
    if r2 <= 0.0 {
        0.0
    } else {
        r2.sqrt()
    }
}

/// Imaging-train protocol (pulse count, spacing, ages).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagingTrain {
    /// Number of imaging pulses in one acquisition.
    pub n_pulses: usize,
    /// Inter-pulse gap [s] (both papers: 200 µs, 5 kHz PRF).
    pub gap_s: f64,
    /// Bubble age at the first pulse [s] (burst → imaging delay).
    pub initial_age_s: f64,
    /// Acoustic on-time per pulse [s].
    pub pulse_on_time_s: f64,
}

impl ImagingTrain {
    /// Kuji 2024 acquisition (§2.6): 4 pulses, 200 µs gaps,
    /// 1 ms after the burst, 5 cycles at 2.45 MHz per pulse
    /// (pulse length is not stated in the paper; 5 cycles is the
    /// B2 primary, 12 cycles its sensitivity — the verdict is
    /// insensitive to the choice).
    #[must_use]
    pub const fn kuji_2024() -> Self {
        Self {
            n_pulses: 4,
            gap_s: 200e-6,
            initial_age_s: 1e-3,
            pulse_on_time_s: 5.0 / 2.45e6,
        }
    }

    /// Iwasaki 2018 acquisition (§2.4): 3 pulses, 200 µs gaps,
    /// 0.6 ms after the burst, 5 cycles at 1.74 MHz per pulse.
    #[must_use]
    pub const fn iwasaki_2018() -> Self {
        Self {
            n_pulses: 3,
            gap_s: 200e-6,
            initial_age_s: 0.6e-3,
            pulse_on_time_s: 5.0 / 1.74e6,
        }
    }
}

/// The effect of one imaging pulse on one bubble, as computed by
/// the caller from the bubble's dynamics solve for that pulse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagingPulseEffect {
    /// Rectified-diffusion radius rate during the pulse [m/s],
    /// from the Louisnard–Gomez law (see the module docs — the
    /// law is reused, not duplicated).
    pub rectified_rate_m_s: f64,
    /// Expansion ratio `Rmax/R0` reached during the pulse, for
    /// the fragmentation gate.
    pub rmax_over_r0: f64,
}

/// Outcome of evolving one bubble over an imaging train.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagingEvolution {
    /// Radius after the last pulse [m] (0 if dissolved).
    pub final_radius_m: f64,
    /// Signed fractional change `(final − initial) / initial`.
    pub fractional_change: f64,
    /// Whether the bubble survives the acquisition (radius at or
    /// above [`DISSOLUTION_REMOVAL_RADIUS_M`]).
    pub survives: bool,
    /// Number of pulses on which the fragmentation gate fired.
    pub fragmentation_gate_pulses: usize,
}

/// Evolve one bubble over an imaging train: per pulse, apply the
/// rectified-diffusion increment `rate × on-time` and record the
/// fragmentation gate; between pulses, apply one Epstein–Plesset
/// gap step at the advancing age. `pulses` must hold exactly
/// `train.n_pulses` entries (one per pulse, in protocol order);
/// a shorter slice is padded with inert pulses (no rectified
/// term, gate not fired).
#[must_use]
pub fn evolve_under_imaging_train(
    r0_m: f64,
    pulses: &[ImagingPulseEffect],
    medium: &EpsteinPlessetMedium,
    train: &ImagingTrain,
) -> ImagingEvolution {
    let mut radius = r0_m;
    let mut age = train.initial_age_s;
    let mut gate_pulses = 0_usize;
    for i in 0..train.n_pulses {
        let effect = pulses.get(i);
        if let Some(e) = effect {
            if e.rmax_over_r0 > IMAGING_FRAGMENTATION_GATE_RMAX_R0 {
                gate_pulses += 1;
            }
            radius += e.rectified_rate_m_s * train.pulse_on_time_s;
            if radius <= 0.0 {
                radius = 0.0;
                break;
            }
        }
        if i + 1 < train.n_pulses {
            radius = epstein_plesset_step(radius, age, train.gap_s, medium);
            age += train.gap_s;
            if radius <= 0.0 {
                break;
            }
        }
    }
    ImagingEvolution {
        final_radius_m: radius,
        fractional_change: if r0_m > 0.0 {
            (radius - r0_m) / r0_m
        } else {
            0.0
        },
        survives: radius >= DISSOLUTION_REMOVAL_RADIUS_M,
        fragmentation_gate_pulses: gate_pulses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inert_pulses(n: usize) -> Vec<ImagingPulseEffect> {
        vec![
            ImagingPulseEffect {
                rectified_rate_m_s: 0.0,
                rmax_over_r0: 1.0,
            };
            n
        ]
    }

    /// B2 headline validation: PNP-independent Epstein–Plesset
    /// shrink over the Kuji acquisition reproduces −1.74 / −1.05 /
    /// −0.53 / −0.24 % for R0 = 1.5 / 2 / 3 / 5 µm, with zero
    /// population loss. (B2's totals include the ≤ 0.0006 µm
    /// rectified contribution; tolerance 0.02 percentage points
    /// covers it.)
    #[test]
    fn b2_epstein_plesset_shrink_reproduced() {
        let medium = EpsteinPlessetMedium::kuji_degassed();
        let train = ImagingTrain::kuji_2024();
        let expected_pct = [(1.5e-6, -1.74), (2.0e-6, -1.05), (3.0e-6, -0.53), (5.0e-6, -0.24)];
        for (r0, pct) in expected_pct {
            let out = evolve_under_imaging_train(r0, &inert_pulses(4), &medium, &train);
            let got_pct = out.fractional_change * 100.0;
            assert!(
                (got_pct - pct).abs() < 0.02,
                "R0 {r0}: shrink {got_pct:.3}% vs B2 {pct}%"
            );
            assert!(out.survives, "R0 {r0}: B2 population loss is 0%");
            assert_eq!(out.fragmentation_gate_pulses, 0);
        }
    }

    /// B2 with the actual Fyrillas–Szeri rates at 0.26 MPa
    /// (C* = 0.2): the rectified term moves the result by
    /// ≤ 0.01 percentage points — the evolution is PNP-blind.
    #[test]
    fn b2_rectified_term_is_negligible_during_pulses() {
        let medium = EpsteinPlessetMedium::kuji_degassed();
        let train = ImagingTrain::kuji_2024();
        // B2 table, PNP 0.26 MPa: dR/dt [µm/s] and Rmax/R0.
        let cases = [
            (1.5e-6, 1.6e-6, 2.78),
            (2.0e-6, -10.3e-6, 1.93),
            (3.0e-6, -3.0e-6, 1.80),
            (5.0e-6, -7.1e-6, 1.19),
        ];
        for (r0, rate, rmax) in cases {
            let pulses = vec![
                ImagingPulseEffect {
                    rectified_rate_m_s: rate,
                    rmax_over_r0: rmax,
                };
                4
            ];
            let with = evolve_under_imaging_train(r0, &pulses, &medium, &train);
            let without = evolve_under_imaging_train(r0, &inert_pulses(4), &medium, &train);
            assert!(
                (with.fractional_change - without.fractional_change).abs() < 1e-4,
                "R0 {r0}: rectified term moved the shrink by {}",
                with.fractional_change - without.fractional_change
            );
            assert!(with.survives);
            assert_eq!(with.fragmentation_gate_pulses, 0);
        }
    }

    /// B2 fragmentation gate: max Rmax/R0 for R0 ≥ 1.5 µm is 4.55
    /// (1.5 µm at 1.23 MPa) — never fires; the R0 = 1 µm
    /// hypothetical (5.17 at 0.72 MPa) fires on every pulse.
    #[test]
    fn b2_fragmentation_gate_threshold() {
        let medium = EpsteinPlessetMedium::kuji_degassed();
        let train = ImagingTrain::kuji_2024();
        let below = vec![
            ImagingPulseEffect {
                rectified_rate_m_s: 0.0,
                rmax_over_r0: 4.55,
            };
            4
        ];
        let out = evolve_under_imaging_train(1.5e-6, &below, &medium, &train);
        assert_eq!(out.fragmentation_gate_pulses, 0);
        let above = vec![
            ImagingPulseEffect {
                rectified_rate_m_s: 0.0,
                rmax_over_r0: 5.17,
            };
            4
        ];
        let out = evolve_under_imaging_train(1.0e-6, &above, &medium, &train);
        assert_eq!(out.fragmentation_gate_pulses, 4);
    }

    #[test]
    fn single_ep_step_matches_b2_gap_arithmetic() {
        // One 200 µs gap from age 1 ms at R0 = 1.5 µm: B2's
        // three-gap total is −0.0261 µm; the first gap carries the
        // largest share (transient term decays with age).
        let medium = EpsteinPlessetMedium::kuji_degassed();
        let r1 = epstein_plesset_step(1.5e-6, 1e-3, 200e-6, &medium);
        assert!(r1 < 1.5e-6 && r1 > 1.49e-6, "one-gap radius {r1}");
        // Saturated medium (C∞ = C_sat): Laplace term alone still
        // dissolves a small bubble, but far more slowly.
        let mut saturated = medium;
        saturated.saturation_ratio = 1.0;
        let r_sat = epstein_plesset_step(1.5e-6, 1e-3, 200e-6, &saturated);
        assert!(r_sat > r1, "saturated {r_sat} vs degassed {r1}");
    }
}
