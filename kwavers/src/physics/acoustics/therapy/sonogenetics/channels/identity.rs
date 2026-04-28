//! Channel identities and canonical literature-backed parameters.

use super::params::{BoltzmannGatingParams, GatingModel, PressureThresholdParams};

/// Mechanosensitive ion channel identity.
///
/// Each variant carries canonical literature-backed gating parameters via
/// [`canonical_params`][MechanoChannel::canonical_params].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanoChannel {
    /// *E. coli* MscL gain-of-function mutant G22S.
    ///
    /// G22S lowers the tension threshold relative to wild-type, enabling
    /// activation at physiologically safe acoustic intensities.
    MscLG22S,

    /// *E. coli* MscL gain-of-function mutant G22N.
    ///
    /// Li 2026 reports weaker ultrasound responses than G22S despite lower
    /// mechanical threshold; network response remains separate from the gate.
    MscLG22N,

    /// *E. coli* small-conductance mechanosensitive channel.
    MscS,

    /// Mammalian stretch-activated channel Piezo1.
    Piezo1,

    /// Mammalian mechanosensitive TRP channel C6.
    Trpc6,

    /// *Halobacterium salinarum* TRPA1 homologue.
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
    /// Wild-type MscL: A_gate = 6.5 nm^2, T_half = 7.8 mN/m. G22S shifts
    /// T_half by approximately -40% to 4.7 mN/m, enabling activation at
    /// sonogenetic intensities.
    ///
    /// # MscL-G22N (Sawada 2015; Li 2026)
    ///
    /// G22N is a stronger gain-of-function MscL mutant with reduced mechanical
    /// threshold. Li 2026 reports weaker ultrasound response than G22S despite
    /// lower mechanical threshold, so efficacy is not encoded as threshold.
    ///
    /// # MscS (Nomura 2012; Li 2026)
    ///
    /// MscS is a smaller-conductance bacterial mechanosensitive channel. This
    /// parameter set represents channel-opening physics only.
    ///
    /// # Piezo1 (Cox 2016)
    ///
    /// Patch-clamp in HEK293T lipid bilayer bleb: A_gate = 20.0 nm^2,
    /// T_half = 2.5 mN/m.
    ///
    /// # TRPC6 (Suchyna 2000; Matsushita 2024)
    ///
    /// Activation is fitted around mean tension about 2.4 mN/m, with A_gate =
    /// 4.5 nm^2 from mechanosensitive TRP-channel scale.
    ///
    /// # hsTRPA1 (Ibsen 2015)
    ///
    /// Threshold MI about 0.4 at 1 MHz gives P_peak about 400 kPa. Radiation
    /// pressure P_rad = P_peak^2 / (2 rho c^2) = 35.6 Pa for water.
    #[must_use]
    pub fn canonical_params(&self) -> GatingModel {
        match self {
            MechanoChannel::MscLG22S => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 6.5e-18,
                half_tension_n_per_m: 4.7e-3,
                single_channel_conductance_s: 3.0e-9,
                reversal_potential_v: 0.0,
            }),
            MechanoChannel::MscLG22N => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 6.5e-18,
                half_tension_n_per_m: 2.35e-3,
                single_channel_conductance_s: 3.0e-9,
                reversal_potential_v: 0.0,
            }),
            MechanoChannel::MscS => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 1.2e-18,
                half_tension_n_per_m: 5.5e-3,
                single_channel_conductance_s: 1.0e-9,
                reversal_potential_v: 0.0,
            }),
            MechanoChannel::Piezo1 => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 20.0e-18,
                half_tension_n_per_m: 2.5e-3,
                single_channel_conductance_s: 35.0e-12,
                reversal_potential_v: 0.0,
            }),
            MechanoChannel::Trpc6 => GatingModel::Boltzmann(BoltzmannGatingParams {
                gating_area_m2: 4.5e-18,
                half_tension_n_per_m: 5.0e-3,
                single_channel_conductance_s: 28.0e-12,
                reversal_potential_v: 5.0e-3,
            }),
            MechanoChannel::HsTrpa1 => GatingModel::PressureThreshold(PressureThresholdParams {
                half_pressure_pa: 35.6,
                steepness_pa: 10.0,
                single_channel_conductance_s: 60.0e-12,
                reversal_potential_v: 0.0,
            }),
        }
    }
}
