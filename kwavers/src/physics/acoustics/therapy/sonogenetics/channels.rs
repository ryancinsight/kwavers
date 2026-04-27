//! Mechanosensitive ion channel gating models for sonogenetics.
//!
//! # Supported channels
//!
//! | Channel    | Organism         | Gating model         | Reference                           |
//! |------------|------------------|----------------------|-------------------------------------|
//! | MscL-G22S  | E. coli (GOF)    | Boltzmann / tension  | Xian 2023; Li 2026                  |
//! | MscL-G22N  | E. coli (GOF)    | Boltzmann / tension  | Li 2026; Sawada 2015                |
//! | MscS       | E. coli          | Boltzmann / tension  | Li 2026; Nomura 2012                |
//! | Piezo1     | Mammalian        | Boltzmann / tension  | Cox 2016; Lewis 2017                |
//! | TRPC6      | Mammalian        | Boltzmann / tension  | Shimojo 2024; Matsushita 2024       |
//! | hsTRPA1    | H. salinarum     | Pressure threshold   | Ibsen 2015; Szablowski 2022         |
//!
//! # Boltzmann two-state gating (tension-activated)
//!
//! Derived from chemical equilibrium of a two-state Markov chain (Hamill & Martinac 2001):
//!
//!   P_open(ΔT) = 1 / (1 + exp(-A_gate · (ΔT − T_half) / (k_B · T_temp)))
//!
//! where:
//! - A_gate   = in-plane gating area [m²]  (mechanochemical coupling coefficient)
//! - ΔT       = membrane tension increment [N/m]
//! - T_half   = tension at half-maximum activation [N/m]
//! - k_B      = Boltzmann constant = 1.380649×10⁻²³ J/K
//! - T_temp   = absolute temperature [K]
//!
//! At ΔT = T_half: argument = 0 → P_open = 0.5. ✓
//!
//! # Pressure-threshold gating (hsTRPA1)
//!
//! Activation follows a sigmoidal radiation-pressure threshold (Ibsen 2015):
//!
//!   P_open(P_rad) = 1 / (1 + exp(-(P_rad − P_half) / s))
//!
//! where s [Pa] is the sigmoid steepness derived from the MI activation threshold.
//!
//! # Ion current
//!
//! Single-channel conductance model (Hille 2001 §2):
//!
//!   I_out = g_single · n_channels · P_open · (V_m − E_rev)
//!   I_inj = -I_out = g_single · n_channels · P_open · (E_rev − V_m)
//!
//! `I_out` is the electrophysiology outward-current convention. `ion_current`
//! returns `I_inj`, the depolarizing injected-current convention consumed by
//! the LIF neuron model. Units depend on whether `n_channels` is a density
//! [m⁻²] or absolute count.
//!
//! # References
//!
//! - Sukharev, S.I. et al. (1997). Mechanosensitive channel MscL in E. coli.
//!   *Biophysical Journal*, 72(1), 193-203.
//! - Hamill, O.P. & Martinac, B. (2001). *Physiological Reviews*, 81(2), 685-740.
//! - Ibsen, S. et al. (2015). Sonogenetics in C. elegans.
//!   *Nature Nanotechnology*, 10(9), 810-815.
//! - Cox, C.D. et al. (2016). Removal of the mechanoprotective influence of the cytoskeleton
//!   reveals PIEZO1 is gated by bilayer tension. *Nature Communications*, 7, 10366.
//! - Hille, B. (2001). *Ion Channels of Excitable Membranes*, 3rd ed. Sinauer.
//! - Suchyna, T.M. et al. (2000). Identification of a peptide toxin for a mechano-sensitive channel.
//!   *Journal of General Physiology*, 115(5), 583-598.
//! - Szablowski, J.O. et al. (2022). Sonogenetics. *Curr. Opin. Neurobiol.*, 73, 102515.
//! - Duque, M. et al. (2023). *Science*, 380(6649), 1084-1090.
//! - Xian, Q. et al. (2023). Modulation of deep neural circuits with sonogenetics.
//!   *PNAS*, 120(23), e2220575120.
//! - Li, X. et al. (2026). Channel-specific differential effects of bacterial
//!   mechanosensitive channels for ultrasound neuromodulation in precision
//!   sonogenetics. *Theranostics*, 16(5), 2447-2465.
//! - Shimojo, D. et al. (2024). TRPC6 is a mechanosensitive channel essential for
//!   ultrasound neuromodulation in the mammalian brain. *PNAS*, 121.
//! - Matsushita, S. et al. (2024). *PNAS*, 121(14), e2314729121.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array3, Zip};

/// Boltzmann constant [J/K].
const K_B: f64 = 1.380_649e-23;

/// Canonical body temperature [K] = 37 °C.
pub const BODY_TEMP_K: f64 = 310.15;

// ─────────────────────────────────────────────────────────────────────────────
// Gating parameter types
// ─────────────────────────────────────────────────────────────────────────────

/// Two-state Boltzmann gating parameters for tension-activated channels.
///
/// # Physical units
///
/// - `gating_area_m2`:              in-plane gating area A_gate [m²]
/// - `half_tension_n_per_m`:        half-activation membrane tension T_half [N/m]
/// - `single_channel_conductance_s`: unitary conductance g_single [S]
/// - `reversal_potential_v`:         reversal (Nernst) potential E_rev [V]
#[derive(Debug, Clone)]
pub struct BoltzmannGatingParams {
    /// In-plane gating area A_gate [m²].
    pub gating_area_m2: f64,
    /// Membrane tension at half-maximum activation T_half [N/m].
    pub half_tension_n_per_m: f64,
    /// Unitary single-channel conductance g_single [S].
    pub single_channel_conductance_s: f64,
    /// Reversal potential E_rev [V].
    pub reversal_potential_v: f64,
}

/// Pressure-threshold gating parameters for hsTRPA1.
///
/// # Physical units
///
/// - `half_pressure_pa`:             radiation pressure at half-maximum activation P_half [Pa]
/// - `steepness_pa`:                 sigmoid steepness s [Pa]
/// - `single_channel_conductance_s`: unitary conductance [S]
/// - `reversal_potential_v`:         reversal potential [V]
#[derive(Debug, Clone)]
pub struct PressureThresholdParams {
    /// Radiation pressure at half-maximum activation P_half [Pa].
    ///
    /// Derived from Ibsen 2015: MI threshold ≈ 0.4 at 1 MHz → P_peak ≈ 400 kPa.
    /// Radiation pressure P_rad = P_peak² / (2ρc²) ≈ 35.6 Pa for water
    /// (ρ = 1000 kg/m³, c = 1500 m/s).
    pub half_pressure_pa: f64,
    /// Sigmoid steepness parameter s [Pa].
    pub steepness_pa: f64,
    /// Unitary conductance [S].
    pub single_channel_conductance_s: f64,
    /// Reversal potential [V].
    pub reversal_potential_v: f64,
}

/// Mechanosensitive channel gating model.
#[derive(Debug, Clone)]
pub enum GatingModel {
    /// Two-state Boltzmann tension-activated gating (MscL-G22S, Piezo1, TRPC6).
    Boltzmann(BoltzmannGatingParams),
    /// Sigmoidal pressure-threshold gating (hsTRPA1).
    PressureThreshold(PressureThresholdParams),
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel identity and canonical parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Mechanosensitive ion channel identity.
///
/// Each variant carries canonical literature-backed gating parameters via
/// [`canonical_params`][MechanoChannel::canonical_params].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanoChannel {
    /// *E. coli* MscL gain-of-function mutant G22S.
    ///
    /// The primary sonogenetics tool for deep-brain neuromodulation.
    /// G22S lowers the tension threshold ~40% relative to wild-type,
    /// enabling activation at physiologically safe acoustic intensities.
    ///
    /// References: Sukharev 1997; Duque 2023; Prieto 2023.
    MscLG22S,

    /// *E. coli* MscL gain-of-function mutant G22N.
    ///
    /// Li 2026 reports weaker ultrasound responses than G22S despite lower
    /// mechanical threshold, so this variant encodes the mechanochemical gate
    /// without treating lower threshold as higher ultrasound efficacy.
    MscLG22N,

    /// *E. coli* small-conductance mechanosensitive channel.
    ///
    /// Li 2026 reports a smaller transcriptomic perturbation footprint for
    /// MscS than MscL variants and diminished high-intensity responses. This
    /// variant represents the channel-opening relation only.
    MscS,

    /// Mammalian stretch-activated channel Piezo1.
    ///
    /// Large gating area (~20 nm²) gives steep activation curve.
    /// Expressed in endothelial cells, red blood cells, and can be
    /// heterologously expressed in neurons (Prieto 2023).
    ///
    /// References: Cox 2016; Lewis 2017.
    Piezo1,

    /// Mammalian mechanosensitive TRP channel C6.
    ///
    /// Endogenously expressed in smooth muscle and some neurons.
    /// Matsushita 2024 demonstrated selective sonogenetic activation via TRPC6.
    ///
    /// References: Matsushita 2024; Suchyna 2000.
    Trpc6,

    /// *Halobacterium salinarum* TRPA1 homologue.
    ///
    /// Activates above an acoustic pressure threshold; the first heterologous
    /// channel used for sonogenetics in *C. elegans* (Ibsen 2015).
    ///
    /// References: Ibsen 2015; Szablowski 2022.
    HsTrpa1,
}

impl MechanoChannel {
    /// Canonical literature-derived gating parameters for this channel.
    ///
    /// # Theorem: two-state tension gating is half-active at `T_half`
    ///
    /// Let `P(T) = 1/(1 + exp(-A(T - T_half)/(k_B theta)))` with `A > 0`,
    /// `k_B > 0`, and `theta > 0`. At `T = T_half`, the exponent is zero, so
    /// `P = 1/(1 + exp(0)) = 1/2`. The derivative is positive for finite `T`,
    /// so the open probability is strictly increasing with membrane tension.
    ///
    /// # MscL-G22S (Sukharev 1997; Duque 2023)
    ///
    /// Wild-type MscL: A_gate = 6.5 nm², T_half = 7.8 mN/m (Sukharev 1997 Table I).
    /// G22S GOF mutation shifts T_half by approximately −40% to 4.7 mN/m (Duque 2023
    /// supplementary electrophysiology), enabling activation at ISPTA ≈ 1–5 W/cm².
    ///
    /// # MscL-G22N (Sawada 2015; Li 2026)
    ///
    /// G22N is a stronger gain-of-function MscL mutant with spontaneous opening
    /// and reduced mechanical threshold. Li 2026 reports weaker ultrasound
    /// response than G22S despite lower mechanical threshold; network response
    /// must therefore remain separate from the gating threshold.
    ///
    /// # MscS (Nomura 2012; Li 2026)
    ///
    /// MscS is a smaller-conductance bacterial mechanosensitive channel. Li 2026
    /// reports lower off-target transcriptomic alteration than MscL variants and
    /// reduced response at high ultrasound intensity; this implementation
    /// represents the channel-opening physics only.
    ///
    /// # Piezo1 (Cox 2016)
    ///
    /// Patch-clamp in HEK293T lipid bilayer bleb: A_gate = 20.0 nm²,
    /// T_half = 2.5 mN/m (Cox 2016 Supplementary Table 3).
    ///
    /// # TRPC6 (Suchyna 2000; Matsushita 2024)
    ///
    /// Fitted from Matsushita 2024 activation at mean tension ≈ 2.4 mN/m;
    /// A_gate = 4.5 nm² from Suchyna 2000 analogous mechanosensitive TRP.
    ///
    /// # hsTRPA1 (Ibsen 2015)
    ///
    /// Threshold MI ≈ 0.4 at 1 MHz → P_peak ≈ 0.4 × √(f[Hz] × ρc) ≈ 400 kPa.
    /// Radiation pressure P_rad = P_peak² / (2ρc²) = (400e3)²/(2 × 1000 × 1500²) = 35.6 Pa.
    /// Steepness s = 10 Pa (sigmoidal fit, Ibsen 2015 Fig. 3).
    #[must_use]
    pub fn canonical_params(&self) -> GatingModel {
        match self {
            MechanoChannel::MscLG22S => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 6.5e-18,              // Sukharev 1997 Table I
                half_tension_n_per_m: 4.7e-3,         // G22S GOF shift; Duque 2023
                single_channel_conductance_s: 3.0e-9, // ~3 nS; Sukharev 1997
                reversal_potential_v: 0.0,            // Non-selective cation; Sukharev 1997
            }),
            MechanoChannel::MscLG22N => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 6.5e-18,              // MscL pore-family area; Sukharev 1997
                half_tension_n_per_m: 2.35e-3,        // lower-threshold GOF seed; Sawada 2015
                single_channel_conductance_s: 3.0e-9, // large-conductance MscL pore
                reversal_potential_v: 0.0,            // Non-selective cation
            }),
            MechanoChannel::MscS => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 1.2e-18,              // smaller mechanosensitive gate area
                half_tension_n_per_m: 5.5e-3,         // bacterial MscS tension-gating scale
                single_channel_conductance_s: 1.0e-9, // small-conductance channel scale
                reversal_potential_v: 0.0,            // Non-selective bacterial MS channel
            }),
            MechanoChannel::Piezo1 => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 20.0e-18,               // Cox 2016 Suppl. Table 3
                half_tension_n_per_m: 2.5e-3,           // Cox 2016 Suppl. Table 3
                single_channel_conductance_s: 35.0e-12, // ~35 pS; Cox 2016
                reversal_potential_v: 0.0,              // Non-selective cation; Cox 2016
            }),
            MechanoChannel::Trpc6 => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 4.5e-18,                // Suchyna 2000 analogue
                half_tension_n_per_m: 5.0e-3,           // Matsushita 2024 fitted
                single_channel_conductance_s: 28.0e-12, // ~28 pS; Suchyna 2000
                reversal_potential_v: 5.0e-3,           // +5 mV; Suchyna 2000
            }),
            MechanoChannel::HsTrpa1 => {
                GatingModel::PressureThreshold(PressureThresholdParams {
                    half_pressure_pa: 35.6,                 // P_peak = 400 kPa; P_rad = 35.6 Pa
                    steepness_pa: 10.0,                     // Ibsen 2015 Fig. 3 fit
                    single_channel_conductance_s: 60.0e-12, // ~60 pS; Ibsen 2015
                    reversal_potential_v: 0.0,              // Non-selective cation; Ibsen 2015
                })
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gating computations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-voxel open probability using the Boltzmann two-state model.
///
/// # Formula
///
/// P_open = 1 / (1 + exp(-A_gate · (ΔT − T_half) / (k_B · T_temp)))
///
/// # Arguments
///
/// - `membrane_tension` — ΔT_membrane(x) [N/m]
/// - `params`           — Boltzmann gating parameters
/// - `temperature_k`    — absolute temperature [K]; must be > 0
///
/// # Returns
///
/// Per-voxel open probability ∈ (0, 1).
///
/// # Errors
///
/// Returns `Err` if `temperature_k ≤ 0`.
pub fn boltzmann_p_open(
    membrane_tension: &Array3<f64>,
    params: &BoltzmannGatingParams,
    temperature_k: f64,
) -> KwaversResult<Array3<f64>> {
    if temperature_k <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "temperature_k".to_string(),
            value: temperature_k,
            reason: "absolute temperature must be strictly positive".to_string(),
        }));
    }
    let kbt = K_B * temperature_k;
    let a = params.gating_area_m2;
    let t_half = params.half_tension_n_per_m;
    let mut out = Array3::<f64>::zeros(membrane_tension.dim());
    Zip::from(&mut out)
        .and(membrane_tension)
        .par_for_each(|p, &dt| {
            let exponent = -a * (dt - t_half) / kbt;
            *p = 1.0 / (1.0 + exponent.exp());
        });
    Ok(out)
}

/// Compute per-voxel open probability using the sigmoidal pressure-threshold model.
///
/// # Formula
///
/// P_open = 1 / (1 + exp(-(P_rad − P_half) / s))
///
/// # Arguments
///
/// - `radiation_pressure` — P_rad(x) = I(x)/c(x) [Pa]
/// - `params`             — pressure-threshold parameters; `steepness_pa` must be > 0
///
/// # Errors
///
/// Returns `Err` if `steepness_pa ≤ 0`.
pub fn pressure_threshold_p_open(
    radiation_pressure: &Array3<f64>,
    params: &PressureThresholdParams,
) -> KwaversResult<Array3<f64>> {
    if params.steepness_pa <= 0.0 {
        return Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: "steepness_pa".to_string(),
            value: params.steepness_pa,
            reason: "sigmoid steepness must be strictly positive".to_string(),
        }));
    }
    let p_half = params.half_pressure_pa;
    let s = params.steepness_pa;
    let mut out = Array3::<f64>::zeros(radiation_pressure.dim());
    Zip::from(&mut out)
        .and(radiation_pressure)
        .par_for_each(|p, &p_rad| {
            *p = 1.0 / (1.0 + (-(p_rad - p_half) / s).exp());
        });
    Ok(out)
}

/// Dispatch to the appropriate gating model.
///
/// # Arguments
///
/// - `model`              — gating model variant
/// - `membrane_tension`   — ΔT_membrane(x) [N/m] (consumed by Boltzmann)
/// - `radiation_pressure` — P_rad(x) = I/c [Pa] (consumed by PressureThreshold)
/// - `temperature_k`      — absolute temperature [K] (consumed by Boltzmann)
///
/// # Errors
///
/// Propagates errors from the underlying model functions.
pub fn compute_p_open(
    model: &GatingModel,
    membrane_tension: &Array3<f64>,
    radiation_pressure: &Array3<f64>,
    temperature_k: f64,
) -> KwaversResult<Array3<f64>> {
    match model {
        GatingModel::Boltzmann(params) => boltzmann_p_open(membrane_tension, params, temperature_k),
        GatingModel::PressureThreshold(params) => {
            pressure_threshold_p_open(radiation_pressure, params)
        }
    }
}

/// Compute per-voxel ion current from channel open probability.
///
/// # Formula
///
/// I_inj = g_single · n_channels · P_open · (E_rev − V_m)
///
/// # Theorem: reversal-potential null current
///
/// For finite `g_single`, `n_channels`, and `P_open`, if `V_m = E_rev`, then
/// `I_inj = g n P (E_rev - V_m) = 0`. Nonselective cation channels with
/// `E_rev > V_m` produce positive depolarizing injected current under the LIF
/// sign convention.
///
/// # Arguments
///
/// - `p_open`     — per-voxel open probability ∈ [0, 1]
/// - `g_single`   — single-channel conductance [S]
/// - `n_channels` — channel density [m⁻²] or absolute count
/// - `v_membrane` — membrane potential V_m [V]
/// - `e_rev`      — reversal potential E_rev [V]
///
/// # Returns
///
/// Per-voxel injected ion current (units: A × n_channels_unit).
#[must_use]
pub fn ion_current(
    p_open: &Array3<f64>,
    g_single: f64,
    n_channels: f64,
    v_membrane: f64,
    e_rev: f64,
) -> Array3<f64> {
    let scale = g_single * n_channels * (e_rev - v_membrane);
    p_open.mapv(|p| p * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    // ─────────────────────────────────────────────────────────────────────────
    // Boltzmann model correctness
    // ─────────────────────────────────────────────────────────────────────────

    /// At ΔT = T_half, P_open must equal exactly 0.5.
    #[test]
    fn test_boltzmann_half_activation() {
        let params = MechanoChannel::MscLG22S.canonical_params();
        let GatingModel::Boltzmann(ref bp) = params else {
            panic!("MscLG22S must use Boltzmann model");
        };
        let tension = Array3::from_elem((2, 2, 2), bp.half_tension_n_per_m);
        let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
        for &v in p_open.iter() {
            assert_relative_eq!(v, 0.5, max_relative = 1e-12);
        }
    }

    /// Well below T_half (ΔT = 0): P_open must be small (channel predominantly closed).
    ///
    /// Analytical value at ΔT = 0 for MscL-G22S:
    ///   exponent = −A·(0 − T_half)/(k_B·T) = +A·T_half/(k_B·T)
    ///            = +6.5e-18 × 4.7e-3 / (1.380649e-23 × 310.15)
    ///            ≈ +7.14
    ///   P_open = 1 / (1 + e^7.14) ≈ 7.96e-4  (< 0.1% open probability)
    ///
    /// This is physically correct: MscL has a small spontaneous open probability
    /// at zero membrane tension (Sukharev 1997).  The threshold < 0.01 (1%) is
    /// conservative relative to the analytical bound.
    #[test]
    fn test_boltzmann_deep_closed() {
        let params = MechanoChannel::MscLG22S.canonical_params();
        let GatingModel::Boltzmann(ref bp) = params else {
            panic!("expected Boltzmann");
        };
        let tension = Array3::zeros((2, 2, 2));
        let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
        for &v in p_open.iter() {
            assert!(
                v < 0.01,
                "P_open at ΔT=0 should be < 1% (channel predominantly closed), got {v:.3e}"
            );
        }
    }

    /// Well above T_half (ΔT = 10·T_half): P_open must be close to 1.
    #[test]
    fn test_boltzmann_deep_open() {
        let params = MechanoChannel::MscLG22S.canonical_params();
        let GatingModel::Boltzmann(ref bp) = params else {
            panic!("expected Boltzmann");
        };
        let tension = Array3::from_elem((2, 2, 2), 10.0 * bp.half_tension_n_per_m);
        let p_open = boltzmann_p_open(&tension, bp, BODY_TEMP_K).unwrap();
        for &v in p_open.iter() {
            assert!(
                v > 1.0 - 1e-10,
                "P_open at 10×T_half should approach 1, got {v:.6}"
            );
        }
    }

    /// Boltzmann is monotonically increasing with tension.
    #[test]
    fn test_boltzmann_monotone() {
        let params = MechanoChannel::Piezo1.canonical_params();
        let GatingModel::Boltzmann(ref bp) = params else {
            panic!("expected Boltzmann");
        };
        let tensions: Vec<f64> = (0..10)
            .map(|i| i as f64 * bp.half_tension_n_per_m / 4.0)
            .collect();
        let mut prev = 0.0_f64;
        for &t in &tensions {
            let arr = Array3::from_elem((1, 1, 1), t);
            let p = boltzmann_p_open(&arr, bp, BODY_TEMP_K).unwrap()[[0, 0, 0]];
            assert!(
                p > prev,
                "P_open must increase with tension: {prev} → {p} at t={t}"
            );
            prev = p;
        }
    }

    /// Temperature must be positive.
    #[test]
    fn test_boltzmann_zero_temperature_is_error() {
        let params = MechanoChannel::MscLG22S.canonical_params();
        let GatingModel::Boltzmann(ref bp) = params else {
            panic!("expected Boltzmann");
        };
        let tension = Array3::zeros((2, 2, 2));
        assert!(boltzmann_p_open(&tension, bp, 0.0).is_err());
        assert!(boltzmann_p_open(&tension, bp, -1.0).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pressure-threshold model correctness
    // ─────────────────────────────────────────────────────────────────────────

    /// At P_rad = P_half, P_open = 0.5.
    #[test]
    fn test_pressure_threshold_half_activation() {
        let params = MechanoChannel::HsTrpa1.canonical_params();
        let GatingModel::PressureThreshold(ref pp) = params else {
            panic!("hsTRPA1 must use PressureThreshold model");
        };
        let p_rad = Array3::from_elem((2, 2, 2), pp.half_pressure_pa);
        let p_open = pressure_threshold_p_open(&p_rad, pp).unwrap();
        for &v in p_open.iter() {
            assert_relative_eq!(v, 0.5, max_relative = 1e-12);
        }
    }

    /// Zero steepness is rejected.
    #[test]
    fn test_pressure_threshold_zero_steepness_is_error() {
        let mut pp = PressureThresholdParams {
            half_pressure_pa: 35.6,
            steepness_pa: 0.0,
            single_channel_conductance_s: 60.0e-12,
            reversal_potential_v: 0.0,
        };
        let p_rad = Array3::zeros((2, 2, 2));
        assert!(pressure_threshold_p_open(&p_rad, &pp).is_err());
        pp.steepness_pa = -1.0;
        assert!(pressure_threshold_p_open(&p_rad, &pp).is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Ion current
    // ─────────────────────────────────────────────────────────────────────────

    /// I_ion = g · n · P_open · (E_rev − V_m).
    ///
    /// At P_open = 0.5, g = 3e-9 S, n = 1000, V_m = −60 mV, E_rev = 0 mV:
    ///   I_ion = 3e-9 × 1000 × 0.5 × (0 − (−0.06)) = +90e-9 A
    #[test]
    fn test_ion_current_analytical() {
        let p_open = Array3::from_elem((2, 2, 2), 0.5_f64);
        let current = ion_current(&p_open, 3.0e-9, 1000.0, -60.0e-3, 0.0);
        let expected = 3.0e-9 * 1000.0 * 0.5 * (0.0 - (-60.0e-3));
        for &v in current.iter() {
            assert_relative_eq!(v, expected, max_relative = 1e-12);
        }
    }

    /// At P_open = 0 (closed): current is zero regardless of driving force.
    #[test]
    fn test_ion_current_zero_when_closed() {
        let p_open = Array3::zeros((2, 2, 2));
        let current = ion_current(&p_open, 3.0e-9, 1000.0, -60.0e-3, 0.0);
        for &v in current.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// At V_m = E_rev: current is zero regardless of P_open.
    #[test]
    fn test_ion_current_zero_at_reversal() {
        let p_open = Array3::from_elem((2, 2, 2), 0.8_f64);
        let e_rev = -10.0e-3_f64;
        let current = ion_current(&p_open, 35.0e-12, 5000.0, e_rev, e_rev);
        for &v in current.iter() {
            assert_eq!(v, 0.0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Canonical parameter sanity
    // ─────────────────────────────────────────────────────────────────────────

    /// All canonical parameters must produce strictly positive gating areas,
    /// half-tensions/pressures, and conductances.
    #[test]
    fn test_canonical_params_are_positive() {
        for ch in &[
            MechanoChannel::MscLG22S,
            MechanoChannel::MscLG22N,
            MechanoChannel::MscS,
            MechanoChannel::Piezo1,
            MechanoChannel::Trpc6,
            MechanoChannel::HsTrpa1,
        ] {
            match ch.canonical_params() {
                GatingModel::Boltzmann(p) => {
                    assert!(p.gating_area_m2 > 0.0, "{ch:?}: gating_area must be > 0");
                    assert!(
                        p.half_tension_n_per_m > 0.0,
                        "{ch:?}: half_tension must be > 0"
                    );
                    assert!(
                        p.single_channel_conductance_s > 0.0,
                        "{ch:?}: conductance must be > 0"
                    );
                }
                GatingModel::PressureThreshold(p) => {
                    assert!(
                        p.half_pressure_pa > 0.0,
                        "{ch:?}: half_pressure must be > 0"
                    );
                    assert!(p.steepness_pa > 0.0, "{ch:?}: steepness must be > 0");
                    assert!(
                        p.single_channel_conductance_s > 0.0,
                        "{ch:?}: conductance must be > 0"
                    );
                }
            }
        }
    }

    /// 2026 bacterial-channel modernization: G22N encodes a lower mechanical
    /// threshold than G22S, while MscS remains lower-conductance than MscL.
    #[test]
    fn test_bacterial_channel_variant_ordering() {
        let GatingModel::Boltzmann(g22s) = MechanoChannel::MscLG22S.canonical_params() else {
            panic!("MscLG22S must use Boltzmann model");
        };
        let GatingModel::Boltzmann(g22n) = MechanoChannel::MscLG22N.canonical_params() else {
            panic!("MscLG22N must use Boltzmann model");
        };
        let GatingModel::Boltzmann(mscs) = MechanoChannel::MscS.canonical_params() else {
            panic!("MscS must use Boltzmann model");
        };

        assert!(
            g22n.half_tension_n_per_m < g22s.half_tension_n_per_m,
            "G22N must encode lower mechanical threshold than G22S"
        );
        assert!(
            mscs.single_channel_conductance_s < g22s.single_channel_conductance_s,
            "MscS must remain lower-conductance than MscL variants"
        );
    }
}
