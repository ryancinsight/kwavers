//! T/R switch model for combined transmit/receive ultrasound front-ends.
//!
//! T/R switches protect the sensitive receive amplifier from the high-voltage
//! transmit pulses while passing the low-voltage echoes to the RX chain.
//! Modern ultrasound pulsers (e.g., MAX14815) integrate T/R switches on-chip,
//! eliminating external clamping diodes and reducing board area.
//!
//! # Evidence tier
//!
//! Closed-form circuit physics (clamp voltage, RC time constant, diode
//! conduction loss), verified against typical values from MAX14815 datasheet.

/// T/R switch configuration.
#[derive(Debug, Clone, Copy)]
pub struct TrSwitchConfig {
    /// Clamp voltage — RX input is protected above this level (V).
    pub clamp_v: f64,
    /// On-resistance of the T/R switch in receive mode (Ω).
    pub r_on_ohm: f64,
    /// Off-capacitance in transmit mode (pF).
    pub c_off_pf: f64,
    /// Maximum transmit voltage before breakdown (V).
    pub v_max_v: f64,
    /// Whether the T/R switch is integrated on-chip with the pulser.
    pub integrated: bool,
}

/// Standard T/R switch profiles.
#[must_use]
pub fn tr_switch_profiles() -> Vec<(&'static str, TrSwitchConfig)> {
    vec![
        (
            "MAX14815 (integrated)",
            TrSwitchConfig {
                clamp_v: 1.5,
                r_on_ohm: 12.0,
                c_off_pf: 2.5,
                v_max_v: 200.0,
                integrated: true,
            },
        ),
        (
            "External diode clamp",
            TrSwitchConfig {
                clamp_v: 0.7,
                r_on_ohm: 5.0,
                c_off_pf: 1.5,
                v_max_v: 200.0,
                integrated: false,
            },
        ),
        (
            "HV7355 + external limiter",
            TrSwitchConfig {
                clamp_v: 1.8,
                r_on_ohm: 15.0,
                c_off_pf: 3.0,
                v_max_v: 150.0,
                integrated: false,
            },
        ),
    ]
}

/// Clamp power dissipation during receive mode (W) — the energy dissipated
/// when the transmit pulse leaks through and the clamp conducts. For a
/// well-designed T/R switch, this is negligible.
#[must_use]
pub fn tr_clamp_dissipation_w(
    tr: &TrSwitchConfig,
    tx_pulse_v: f64,
    prf_hz: f64,
    pulse_width_s: f64,
) -> f64 {
    if tx_pulse_v <= tr.clamp_v {
        return 0.0;
    }
    let overdrive = tx_pulse_v - tr.clamp_v;
    let i_clamp = overdrive / tr.r_on_ohm;
    let p_pulse = i_clamp * overdrive;
    p_pulse * pulse_width_s * prf_hz
}

/// RX noise figure penalty from T/R switch on-resistance: `NF = 10·log₁₀(1 + R_on/50)`.
/// Thermal noise from the switch series resistance adds to the RX noise figure.
#[must_use]
pub fn tr_noise_figure_db(tr: &TrSwitchConfig) -> f64 {
    10.0 * (1.0 + tr.r_on_ohm / 50.0).log10()
}

/// Board area saved by integrating the T/R switch (per-channel mm²).
/// External diode clamps typically consume ~5 mm² per channel (two diodes + bypass).
#[must_use]
pub fn tr_area_saving_mm2(tr: &TrSwitchConfig) -> f64 {
    if tr.integrated {
        5.0 // saved vs external diode clamp per channel
    } else {
        0.0
    }
}

/// Whether the T/R switch is adequate for a given transmit voltage.
#[must_use]
pub fn tr_adequate(tr: &TrSwitchConfig, tx_v: f64) -> bool {
    tr.v_max_v >= tx_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_tr_area_saving() {
        let int = tr_switch_profiles();
        let max15 = int.iter().find(|(n, _)| n.contains("MAX14815")).unwrap();
        assert!(tr_area_saving_mm2(&max15.1) > 0.0);
        let ext = int.iter().find(|(n, _)| n.contains("External")).unwrap();
        assert_eq!(tr_area_saving_mm2(&ext.1), 0.0);
    }

    #[test]
    fn tr_noise_figure_is_small() {
        for (name, cfg) in tr_switch_profiles() {
            let nf = tr_noise_figure_db(&cfg);
            assert!(
                nf < 3.0,
                "T/R switch {name} NF={nf:.1} dB — should be <3 dB for 50Ω RX chain"
            );
        }
    }

    #[test]
    fn clamp_dissipation_scales_with_overdrive() {
        let ext = tr_switch_profiles()[1].1;
        let p0 = tr_clamp_dissipation_w(&ext, 0.3, 1000.0, 1e-6); // below clamp (0.3V &lt; 0.7V)
        assert_eq!(p0, 0.0);
        let p1 = tr_clamp_dissipation_w(&ext, 50.0, 1000.0, 1e-6);
        assert!(p1 > 0.0, "overdrive should dissipate power");
    }

    #[test]
    fn tr_adequacy_checks_voltage_rating() {
        let max15 = tr_switch_profiles()[0].1;
        assert!(tr_adequate(&max15, 150.0));
        assert!(!tr_adequate(&max15, 250.0));
    }
}
