//! Expansion-phase gas influx for shocked histotripsy pulses
//! (Bader & Bollen / Bader et al. law family), the Bader & Holland
//! maximum-radius fit, and the residual-radius (gas content ↔ radius)
//! maps.
//!
//! # Laws
//!
//! **Per-pulse influx** (Bader et al. 2021, *Phys. Med. Biol.* 66,
//! 165004, Eq. (7); operationalising Bader et al. 2019a, *Phys. Med.
//! Biol.* 64, 115012, Eqs. (4)–(8), and Bader & Bollen 2018, *J.
//! Acoust. Soc. Am.* 143, EL481–EL486):
//!
//! ```text
//! Δn = 8 · C∞ · √(π·D·t_M) · R_M²        [molecules per pulse]
//! ```
//!
//! with `R_M` the maximum bubble radius in the pulse, `t_M` the time
//! to maximum radius (≈ the tensile-phase duration, `1/(2f)` per
//! Bader & Holland 2016, *Phys. Med. Biol.* 61, 2947–2966), and `C∞`
//! the far-field dissolved-gas **number** concentration. Gas gained
//! during expansion is retained through the inertial collapse (Church
//! 1989; Bader et al. 2021 §2D).
//!
//! **Units trap:** Bader's Eq. (8) uses `k_B`, so `n` in this family
//! is a **molecule count**, not moles, despite the papers' surrounding
//! text saying "moles". Every function in this module works in
//! molecules; divide by Avogadro's number at a molar interface.
//!
//! **Signed (C_s-corrected) form:** Bader et al. 2019a Eqs. (4)–(6)
//! put the bubble's own surface concentration `C_s` (Henry's law at
//! the gas pressure at `R_M`) in the diffusion gradient. With the
//! zero-floor removed, the gradient can reverse:
//! [`bader_signed_expansion_influx`] is the generalisation the
//! triplet-pulse prototype (Build 6) used to expose the collapse-phase
//! outflux, together with [`compression_phase_outflux`]. In the
//! histotripsy regime the outflux term is ~1e-12 of the influx (Bader
//! 2019a §2) — which is why the published law drops it — but that
//! justification does not carry to long sub-threshold tone bursts,
//! where the system sits at the choked-gradient boundary.
//!
//! **Maximum radius:** [`bader_holland_dmax`] is the Bader & Holland
//! (2016) Eq. (10) fit, shock-scattering branch:
//! `D_MAX [µm] = 18.18 · |p−|^0.56 · f^(−1)` with `|p−|` in MPa and
//! `f` in MHz (fit range p− 15–25 MPa, f 0.75–1 MHz, 5–20 cycles,
//! r² = 0.999). The fit is an upper limit evaluated at a 20 nm
//! nucleus. Below the 14.7 MPa continual-growth threshold (their
//! §3.1, Fig. 8) the bubble does not grow cycle-over-cycle, though it
//! still re-expands to ~the same `R_M` and collapses every cycle.
//!
//! # Residual-radius maps — read before choosing
//!
//! Two maps from gas content `n` to equilibrium (residual) radius are
//! implemented, and they are **not interchangeable**:
//!
//! - [`residual_radius_equilibrium`] (implicit): the mechanical
//!   equilibrium relation `R³·(p0 + 2σ/R) = 3·k_B·T·n/(4π)`. This is
//!   the physically correct map and the formation model's primary.
//! - [`residual_radius_as_printed`]: Bader & Bollen (2018) Eq. (4) =
//!   Bader et al. (2021) Eq. (8) = Bader et al. (2019a) Eq. (8), which
//!   evaluates the Laplace pressure at the **pre-exposure nucleus
//!   radius** `R_nuc` — Bader et al. (2021) §2D states this explicitly
//!   ("the subscript 'nuc' refers to the properties of the bubble
//!   nucleus prior to histotripsy exposure"). The as-printed form is
//!   not a mechanical equilibrium: at n = 1e9 molecules from a 10 nm
//!   nucleus it implies an internal pressure of ~11.3 MPa against a
//!   restoring pressure of ~0.39 MPa (29× over-pressured), so such a
//!   bubble must expand ~3.8× in radius to equilibrate.
//!
//! Measurement arbitrates in favour of the implicit map: the Bader
//! 2021 residuals (an independent measurement via the dissolution
//! model, cross-validated by dissolution times 161.6 vs 167.0 ms)
//! give implicit 2.72 µm vs measured 2.9 µm agarose (−6%), as-printed
//! 0.69 µm (−76%). The 2018 diameters that favour the as-printed form
//! are Eq. (4) outputs, not independent measurements, so that check is
//! circular. The as-printed form is retained because it is the
//! published law and because the Eq. (7) influx was calibrated
//! against it — but do not silently swap the two (see
//! `formation/RNUC_RESOLUTION.md` in the prototype record).
//!
//! # Validity regime
//!
//! Single spherical bubble; ~1 MHz; p− 14.5–18.3 MPa at 3–5 cycles
//! (Bader & Bollen 2018) and 25–40 MPa at 2 cycles (Bader et al.
//! 2021); 90% gas saturation; 20 nm nucleus; no bubble–bubble
//! interaction; no vapour. Eq. (7) is a **single-pulse** law for
//! short, shocked, above-threshold pulses: applying it per cycle to
//! ~44,000 sub-threshold tone-burst cycles overstates the influx by
//! ~300–500× (prototype Build 6 verdict) — the cycle-averaged law for
//! that regime is
//! [`louisnard_gomez_growth_rate`](super::louisnard_gomez_growth_rate).
//! [`bader_expansion_influx_checked`] refuses inputs outside the
//! published regime.

use kwavers_core::constants::fundamental::{AVOGADRO, BOLTZMANN};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use std::f64::consts::PI;

/// Lower peak-negative-pressure bound of the published influx regime
/// [Pa] (Bader & Bollen 2018: 14.5 MPa).
pub const BADER_INFLUX_MIN_PRESSURE_PA: f64 = 14.5e6;
/// Upper peak-negative-pressure bound of the published influx regime
/// [Pa] (Bader et al. 2021: 40 MPa).
pub const BADER_INFLUX_MAX_PRESSURE_PA: f64 = 40.0e6;
/// Maximum pulse length of the published influx regime [cycles]
/// (validated to 5 cycles).
pub const BADER_INFLUX_MAX_CYCLES: u32 = 5;
/// Continual-growth threshold [Pa] (Bader & Holland 2016 §3.1:
/// numerical 14.7 MPa; analytic Eq. (9) gives 12.3 MPa). Below this
/// peak negative pressure a bubble does not grow cycle-over-cycle.
pub const CONTINUAL_GROWTH_THRESHOLD_PA: f64 = 14.7e6;

/// Which Bader & Holland (2016) Eq. (10) fit bounds a `D_MAX` call
/// exceeds. The fit stays evaluable outside its range — the flags
/// exist so the extrapolation is never silent (the formation model's
/// Iwasaki burst, 5.39 MPa saturated, sets three of the four).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BaderHollandFlags {
    /// p− below the 15 MPa fit range.
    pub below_fit_pressure: bool,
    /// p− above the 25 MPa fit range.
    pub above_fit_pressure: bool,
    /// Frequency outside 0.75–1.0 MHz.
    pub outside_fit_frequency: bool,
    /// p− below the 14.7 MPa continual-growth threshold.
    pub below_continual_growth_threshold: bool,
}

impl BaderHollandFlags {
    /// True when the call is inside the fit's validated range and
    /// above the continual-growth threshold.
    #[must_use]
    pub fn in_fit_range(&self) -> bool {
        !self.below_fit_pressure
            && !self.above_fit_pressure
            && !self.outside_fit_frequency
            && !self.below_continual_growth_threshold
    }
}

/// Maximum bubble diameter [m] for a shock-scattering histotripsy
/// pulse (Bader & Holland 2016, Eq. (10), shock-scattering branch),
/// with the fit-range flags for the call.
///
/// `peak_negative_pressure` is the peak negative pressure magnitude
/// [Pa]; `frequency` is in Hz (converted to MHz internally, as the
/// fit is stated).
#[must_use]
pub fn bader_holland_dmax(peak_negative_pressure: f64, frequency: f64) -> (f64, BaderHollandFlags) {
    let p_mpa = peak_negative_pressure.abs() / 1.0e6;
    let f_mhz = frequency / 1.0e6;
    let flags = BaderHollandFlags {
        below_fit_pressure: p_mpa < 15.0,
        above_fit_pressure: p_mpa > 25.0,
        outside_fit_frequency: !(0.75..=1.0).contains(&f_mhz),
        below_continual_growth_threshold: peak_negative_pressure.abs()
            < CONTINUAL_GROWTH_THRESHOLD_PA,
    };
    let d_max_um = 18.18 * p_mpa.powf(0.56) * f_mhz.powf(-1.0);
    (d_max_um * 1e-6, flags)
}

/// Time to maximum radius [s], approximated as the tensile-phase
/// duration `1/(2f)` (Bader & Holland 2016 Eq. (7) relation; Bader et
/// al. 2021 §2D).
#[must_use]
pub fn tensile_phase_duration(frequency: f64) -> f64 {
    if frequency <= 0.0 {
        return 0.0;
    }
    0.5 / frequency
}

/// Per-pulse expansion-phase gas influx [molecules] (Bader et al.
/// 2021, Eq. (7)): `Δn = 8·C∞·√(π·D·t_M)·R_M²`.
///
/// `max_radius` is `R_M` [m], `time_to_max` is `t_M` [s],
/// `diffusivity` is `D` [m²/s], and `far_field_concentration` is `C∞`
/// in **molecules/m³** (multiply a molar concentration by Avogadro's
/// number). Degenerate inputs return 0.0. This function does not
/// enforce the published regime — see
/// [`bader_expansion_influx_checked`].
#[must_use]
pub fn bader_expansion_influx(
    max_radius: f64,
    time_to_max: f64,
    diffusivity: f64,
    far_field_concentration: f64,
) -> f64 {
    if max_radius <= 0.0 || time_to_max <= 0.0 || diffusivity <= 0.0 {
        return 0.0;
    }
    8.0 * far_field_concentration * (PI * diffusivity * time_to_max).sqrt() * max_radius.powi(2)
}

/// Checked form of [`bader_expansion_influx`]: refuses peak negative
/// pressures outside 14.5–40 MPa and pulses longer than 5 cycles —
/// the published regime of the law.
///
/// # Errors
/// - [`KwaversError::Physics`] with `PhysicsError::InvalidParameter`
///   when `peak_negative_pressure` is outside
///   [`BADER_INFLUX_MIN_PRESSURE_PA`]..=[`BADER_INFLUX_MAX_PRESSURE_PA`]
///   or `cycles` exceeds [`BADER_INFLUX_MAX_CYCLES`]. Per-cycle
///   application to a long tone burst is refused for the same reason:
///   Eq. (7) is a single-pulse law (see the module docs).
pub fn bader_expansion_influx_checked(
    max_radius: f64,
    time_to_max: f64,
    diffusivity: f64,
    far_field_concentration: f64,
    peak_negative_pressure: f64,
    cycles: u32,
) -> KwaversResult<f64> {
    let p = peak_negative_pressure.abs();
    if !(BADER_INFLUX_MIN_PRESSURE_PA..=BADER_INFLUX_MAX_PRESSURE_PA).contains(&p) {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "peak_negative_pressure".to_owned(),
            value: peak_negative_pressure,
            reason: format!(
                "Bader Eq. (7) influx validated at {BADER_INFLUX_MIN_PRESSURE_PA}..={BADER_INFLUX_MAX_PRESSURE_PA} Pa peak negative pressure; got {peak_negative_pressure}"
            ),
        }));
    }
    if cycles > BADER_INFLUX_MAX_CYCLES {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: "cycles".to_owned(),
            value: f64::from(cycles),
            reason: format!(
                "Bader Eq. (7) is a single-pulse law validated to {BADER_INFLUX_MAX_CYCLES} cycles; per-cycle application to long bursts overstates the influx ~300-500x (use the Louisnard-Gomez cycle-averaged law there)"
            ),
        }));
    }
    Ok(bader_expansion_influx(
        max_radius,
        time_to_max,
        diffusivity,
        far_field_concentration,
    ))
}

/// Surface concentration [molecules/m³] at the bubble wall from
/// Henry's law at the bubble's own gas pressure at `R_M`:
/// `C_s = C_0·P_g/P_0` with `P_g = n·k_B·T/((4π/3)·R_M³)` (Bader et
/// al. 2019a, Eqs. (4)–(6)).
///
/// `saturation_concentration` is `C_0`, the dissolved-gas number
/// concentration at saturation [molecules/m³].
#[must_use]
pub fn henry_surface_concentration(
    gas_content_molecules: f64,
    max_radius: f64,
    temperature: f64,
    saturation_concentration: f64,
    ambient_pressure: f64,
) -> f64 {
    if max_radius <= 0.0 || temperature <= 0.0 || ambient_pressure <= 0.0 {
        return 0.0;
    }
    let p_gas =
        gas_content_molecules * BOLTZMANN * temperature / ((4.0 / 3.0) * PI * max_radius.powi(3));
    saturation_concentration * p_gas / ambient_pressure
}

/// Signed expansion-phase influx [molecules] with the surface
/// concentration in the gradient (Bader et al. 2019a, Eqs. (4)–(6),
/// generalised form):
/// `Δn = 8·√(π·D·t_M)·R_M²·(C∞ − C_s)`.
///
/// Reduces to [`bader_expansion_influx`] at `C_s = 0`; a negative
/// gradient (bubble gas pressure above the far-field equilibrium)
/// drives gas **out** during expansion. The published law floors this
/// at zero — the floor is a caller decision, not applied here.
#[must_use]
pub fn bader_signed_expansion_influx(
    max_radius: f64,
    time_to_max: f64,
    diffusivity: f64,
    far_field_concentration: f64,
    surface_concentration: f64,
) -> f64 {
    if max_radius <= 0.0 || time_to_max <= 0.0 || diffusivity <= 0.0 {
        return 0.0;
    }
    8.0 * (PI * diffusivity * time_to_max).sqrt()
        * max_radius.powi(2)
        * (far_field_concentration - surface_concentration)
}

/// Net gas transfer [molecules] over a compression-phase radius
/// segment (prototype Build 6 generalisation of the Bader 2019a
/// Eq. (4) thin-boundary-layer flux):
///
/// ```text
/// Δn = ∫ 4π·R(t)²·D·(C∞ − C_s(R(t))) / δ(t) dt,
/// δ(t) = √(π·D·t),   C_s(R) = C_0·P_g(R)/P_0,
/// P_g(R) = P_gM·(R_M/R)^(3γ),  P_gM = n·k_B·T/((4π/3)·R_M³)
/// ```
///
/// Signed: positive = net influx during compression, negative = net
/// outflux. The outflux is driven by `C_s ~ R^(−3γ)` and dominated by
/// the small-`R` state. `time` is measured from the **cycle start**
/// (the boundary layer grows from cycle start in the primary
/// formulation), and the samples should cover the compression phase
/// (`R` falling from `R_M`). This is a generalisation anchored to the
/// published flux form, not itself a published law — it was
/// investigated in the prototype because the published influx-only
/// law cannot close the formation gas budget; its verdict there was
/// that the outflux is real but closes only ~1.33× of the gap.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn compression_phase_outflux(
    radius: &[f64],
    time: &[f64],
    diffusivity: f64,
    far_field_concentration: f64,
    saturation_concentration: f64,
    gas_content_molecules: f64,
    max_radius: f64,
    temperature: f64,
    polytropic_exponent: f64,
    ambient_pressure: f64,
) -> f64 {
    if radius.len() < 2
        || radius.len() != time.len()
        || diffusivity <= 0.0
        || max_radius <= 0.0
        || temperature <= 0.0
        || ambient_pressure <= 0.0
    {
        return 0.0;
    }
    let p_gm =
        gas_content_molecules * BOLTZMANN * temperature / ((4.0 / 3.0) * PI * max_radius.powi(3));
    let integrand: Vec<f64> = radius
        .iter()
        .zip(time.iter())
        .map(|(&r, &t)| {
            if r <= 0.0 || t <= 0.0 {
                return 0.0;
            }
            let p_g = p_gm * (max_radius / r).powf(3.0 * polytropic_exponent);
            let c_s = saturation_concentration * p_g / ambient_pressure;
            let delta = (PI * diffusivity * t).sqrt();
            4.0 * PI * r * r * diffusivity * (far_field_concentration - c_s) / delta
        })
        .collect();
    let mut integral = 0.0;
    for i in 1..integrand.len() {
        integral += 0.5 * (integrand[i - 1] + integrand[i]) * (time[i] - time[i - 1]);
    }
    integral
}

/// Gas content [molecules] of a bubble at mechanical equilibrium:
/// `n = (p0 + 2σ/R)·(4π/3)·R³/(k_B·T)`. The forward direction of the
/// implicit residual-radius map.
#[must_use]
pub fn equilibrium_gas_content(
    radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if radius <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    (ambient_pressure + 2.0 * surface_tension / radius) * (4.0 / 3.0) * PI * radius.powi(3)
        / (BOLTZMANN * temperature)
}

/// `dn/dR` [molecules/m] of the equilibrium gas-content map:
/// `(4π·p0·R² + 16π·σ·R/3)/(k_B·T)`. The conversion between a molar /
/// molecular flux and a radius rate in quasi-static integration.
#[must_use]
pub fn equilibrium_gas_content_derivative(
    radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if temperature <= 0.0 {
        return 0.0;
    }
    (4.0 * PI * ambient_pressure * radius * radius + 16.0 * PI * surface_tension * radius / 3.0)
        / (BOLTZMANN * temperature)
}

/// Solve `R³·(P + 2σ/R) = 3·k_B·T·n/(4π)` for `R` [m] by bisection —
/// the equilibrium radius of a bubble holding `gas_molecules` under a
/// total external pressure `external_pressure` [Pa].
///
/// With `external_pressure = p0` this is the implicit residual-radius
/// map; with `external_pressure = p0 + p+` it is the pressure-balance
/// (collapse-floor) radius of the Build 6 outflux analysis — a bubble
/// **can** contract below its `p0`-equilibrium radius when the
/// external pressure exceeds `p0` by ~100×, so the collapse floor is
/// this radius, not the residual radius.
#[must_use]
pub fn equilibrium_radius_for_pressure(
    gas_molecules: f64,
    external_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if gas_molecules <= 0.0 || external_pressure <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    let rhs = 3.0 * BOLTZMANN * temperature * gas_molecules / (4.0 * PI);
    let f = |r: f64| external_pressure * r.powi(3) + 2.0 * surface_tension * r.powi(2) - rhs;
    let (mut lo, mut hi) = (1e-12_f64, 1e-2_f64);
    if f(lo) > 0.0 {
        return lo;
    }
    if f(hi) < 0.0 {
        return hi;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Residual (equilibrium) radius [m] from gas content [molecules] —
/// the implicit mechanical-equilibrium map
/// `R³·(p0 + 2σ/R) = 3·k_B·T·n/(4π)`. The formation model's primary
/// map; see the module docs for the arbitration against the
/// as-printed form.
#[must_use]
pub fn residual_radius_equilibrium(
    gas_molecules: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    equilibrium_radius_for_pressure(
        gas_molecules,
        ambient_pressure,
        surface_tension,
        temperature,
    )
}

/// Residual radius [m] from gas content [molecules] — the as-printed
/// Bader Eq. (8) map, `R = [3·k_B·T·n/(4π·(p0 + 2σ/R_nuc))]^(1/3)`,
/// with the Laplace pressure evaluated at the **pre-exposure nucleus
/// radius** `nucleus_radius` [m].
///
/// Retained as the published law and as a labelled sensitivity; it is
/// **not** a mechanical equilibrium (see the module docs). Do not
/// silently swap it for [`residual_radius_equilibrium`].
#[must_use]
pub fn residual_radius_as_printed(
    gas_molecules: f64,
    nucleus_radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if gas_molecules <= 0.0 || nucleus_radius <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    let rhs = 3.0 * BOLTZMANN * temperature * gas_molecules / (4.0 * PI);
    let p_eff = ambient_pressure + 2.0 * surface_tension / nucleus_radius;
    (rhs / p_eff).powf(1.0 / 3.0)
}

/// Avogadro conversion kept beside the laws that need it: molecules
/// per mole (the papers in this family mix the two; see the module
/// docs' units trap).
pub const MOLECULES_PER_MOLE: f64 = AVOGADRO;

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

    const P0: f64 = ATMOSPHERIC_PRESSURE;
    const SIGMA_BADER: f64 = 0.056;
    const T_BADER: f64 = 310.0;
    const D_BADER: f64 = 2.42e-9;
    const C0_BADER_MOL: f64 = 0.822;

    #[test]
    fn test_nucleus_gas_content_check_value() {
        // Bader et al. (2019a) §2 check: a 20 nm-diameter nucleus at
        // equilibrium holds ~1.11e4 molecules.
        let n = equilibrium_gas_content(10e-9, P0, SIGMA_BADER, T_BADER);
        assert!(
            (n - 1.11e4).abs() / 1.11e4 < 0.02,
            "nucleus gas content {n} vs check value 1.11e4"
        );
    }

    #[test]
    fn test_residual_radius_round_trip() {
        for r in [0.5e-6, 3e-6, 10e-6] {
            let n = equilibrium_gas_content(r, P0, SIGMA_BADER, T_BADER);
            let back = residual_radius_equilibrium(n, P0, SIGMA_BADER, T_BADER);
            assert_relative_eq!(back, r, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_as_printed_vs_implicit_at_1e9_molecules() {
        // RNUC_RESOLUTION diagnostic: at n = 1e9 from a 10 nm nucleus,
        // as-printed gives D ≈ 0.90 µm (29× over-pressured), the
        // implicit equilibrium gives D ≈ 3.4–3.7 µm.
        let n = 1e9;
        let r_printed = residual_radius_as_printed(n, 10e-9, P0, SIGMA_BADER, T_BADER);
        let r_implicit = residual_radius_equilibrium(n, P0, SIGMA_BADER, T_BADER);
        assert!(
            (2.0 * r_printed - 0.90e-6).abs() / 0.90e-6 < 0.06,
            "as-printed diameter {} µm",
            2.0 * r_printed * 1e6
        );
        assert!(
            (2.0 * r_implicit - 3.55e-6).abs() / 3.55e-6 < 0.10,
            "implicit diameter {} µm",
            2.0 * r_implicit * 1e6
        );
        assert!(r_implicit > r_printed);
    }

    #[test]
    fn test_dmax_fit_and_flags() {
        // Centre of the fit range: no flags; D_MAX = 18.18·20^0.56 µm.
        let (d, flags) = bader_holland_dmax(20e6, 1.0e6);
        assert!(flags.in_fit_range());
        assert_relative_eq!(d, 18.18 * 20.0_f64.powf(0.56) * 1e-6, epsilon = 1e-9);
        // Iwasaki burst (saturated 5.39 MPa, 1.25 MHz): three flags.
        let (_, flags) = bader_holland_dmax(5.39e6, 1.25e6);
        assert!(flags.below_fit_pressure);
        assert!(flags.outside_fit_frequency);
        assert!(flags.below_continual_growth_threshold);
        assert!(!flags.above_fit_pressure);
    }

    #[test]
    fn test_bader_2018_benchmark_diameters() {
        // Bader & Bollen (2018) residual diameters via Eq. (7) influx
        // + the as-printed map (the map those diameters were computed
        // with). Paper: 0.80 µm (1 cycle, 18.3 MPa, 90%) and 0.95 µm
        // (3 cycles, 14.5 MPa, 90%). The prototype reproduced 0.70 /
        // 0.92 µm; the ±20% band is the B-lit acceptance used there.
        let c_inf = 0.9 * C0_BADER_MOL * AVOGADRO;
        let t_m = tensile_phase_duration(1.0e6);
        for (p_pa, cycles, paper_d_um) in [(18.3e6, 1_u32, 0.80), (14.5e6, 3, 0.95)] {
            let (d_max, _) = bader_holland_dmax(p_pa, 1.0e6);
            let dn = bader_expansion_influx(d_max / 2.0, t_m, D_BADER, c_inf);
            let n = f64::from(cycles) * dn;
            let d_um = 2.0 * residual_radius_as_printed(n, 10e-9, P0, SIGMA_BADER, T_BADER) * 1e6;
            assert!(
                (d_um - paper_d_um).abs() / paper_d_um < 0.20,
                "Bader 2018 row p = {p_pa} Pa, {cycles} cycles: got {d_um} µm, paper {paper_d_um} µm"
            );
        }
    }

    #[test]
    fn test_checked_influx_regime() {
        let c_inf = 0.9 * C0_BADER_MOL * AVOGADRO;
        // In regime the checked form is the unchecked law, bit for bit.
        let checked =
            bader_expansion_influx_checked(46e-6, 0.5e-6, D_BADER, c_inf, 18.3e6, 3).unwrap();
        let plain = bader_expansion_influx(46e-6, 0.5e-6, D_BADER, c_inf);
        assert_eq!(checked.to_bits(), plain.to_bits());
        // Below-threshold burst pressure: refused, naming the pressure.
        let below = bader_expansion_influx_checked(46e-6, 0.5e-6, D_BADER, c_inf, 5.39e6, 3);
        assert!(matches!(
            below,
            Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                ref parameter,
                value,
                ..
            })) if parameter == "peak_negative_pressure" && value.to_bits() == 5.39e6_f64.to_bits()
        ));
        // Per-cycle application to a long burst: refused, naming the cycles.
        let long_burst =
            bader_expansion_influx_checked(46e-6, 0.5e-6, D_BADER, c_inf, 18.3e6, 44_000);
        assert!(matches!(
            long_burst,
            Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                ref parameter,
                value,
                ..
            })) if parameter == "cycles" && value.to_bits() == 44_000.0_f64.to_bits()
        ));
    }

    #[test]
    fn test_signed_influx_reduces_to_published() {
        let c_inf = 0.9 * C0_BADER_MOL * AVOGADRO;
        let plain = bader_expansion_influx(46e-6, 0.5e-6, D_BADER, c_inf);
        let signed = bader_signed_expansion_influx(46e-6, 0.5e-6, D_BADER, c_inf, 0.0);
        assert_relative_eq!(signed, plain, epsilon = 1e-12);
        // A supersaturated wall concentration reverses the flux.
        let reversed = bader_signed_expansion_influx(46e-6, 0.5e-6, D_BADER, c_inf, 2.0 * c_inf);
        assert!(reversed < 0.0);
        assert_relative_eq!(reversed, -plain, epsilon = 1e-12);
    }

    #[test]
    fn test_pressure_balance_radius_below_residual() {
        // Under a +100 MPa compressive peak the same gas content
        // balances at a much smaller radius than the p0 residual.
        let n = equilibrium_gas_content(3e-6, P0, SIGMA_BADER, T_BADER);
        let r_bal = equilibrium_radius_for_pressure(n, P0 + 100e6, SIGMA_BADER, T_BADER);
        assert!(r_bal < 3e-6);
        assert!(r_bal > 0.0);
    }
}
