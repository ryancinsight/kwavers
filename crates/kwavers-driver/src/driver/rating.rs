//! Thermal-duty and package-power-rating limits: the duty headroom before the board hits its
//! thermal ceiling, the chip-passive power rating lookup and its board-wide overload check, and the
//! hot-Rds_on efficiency derating that couples the thermal field back into the electrical model.

use super::pulser::{pulser_dissipation, PulserOp};
use super::reactive::driver_efficiency;

/// Maximum safe burst **duty cycle** before the board hits its thermal limit. The steady-state
/// temperature rise is ~linear in dissipation, and pulser dissipation is ~linear in duty, so from a
/// solved operating point (`current_duty`, `current_rise_k`) the headroom scales: `D_max = D ·
/// (ΔT_limit / ΔT)`, capped at 1.0 (CW). The binding driver-design constraint linking the loss model
/// to the thermal envelope.
#[must_use]
pub fn max_safe_duty(current_duty: f64, current_rise_k: f64, limit_rise_k: f64) -> f64 {
    if current_rise_k <= 0.0 || current_duty <= 0.0 {
        return 1.0;
    }
    (current_duty * limit_rise_k / current_rise_k).min(1.0)
}

/// Nominal continuous power rating (W) of a chip passive, inferred from the imperial size code in its
/// footprint name (`…0402…`, `…1206…`). Standard derated-to-70 °C values. `None` if no size is found.
#[must_use]
pub fn chip_power_rating_w(footprint_name: &str) -> Option<f64> {
    // Imperial size code → rating (W). Ordered longest-first so "1206" isn't matched as "0201"-style.
    const TABLE: &[(&str, f64)] = &[
        ("0201", 0.05),
        ("0402", 0.063),
        ("0603", 0.10),
        ("0805", 0.125),
        ("1206", 0.25),
        ("1210", 0.50),
        ("2010", 0.75),
        ("2512", 1.0),
    ];
    TABLE
        .iter()
        .find(|(code, _)| footprint_name.contains(code))
        .map(|&(_, w)| w)
}

/// A component dissipating beyond its package's power rating: `(refdes, dissipation W, rating W)`.
pub type PowerOverload = (String, f64, f64);

/// Component power-rating findings.
#[derive(Debug, Clone, Default)]
pub struct PowerRatingReport {
    /// Components whose modelled dissipation exceeds their package rating (overheating risk).
    pub overloaded: Vec<PowerOverload>,
    /// Whether every dissipating part is within its package rating.
    pub pass: bool,
}

/// Check that no chip passive dissipates beyond its package power rating. `dissipation_w(fp)` gives the
/// per-component heat (e.g. the pulser loss model's series-resistor share); a part with a known chip
/// rating and dissipation above it is flagged — the loss model catching an under-sized damping resistor
/// before it cooks. Parts without a recognised chip size (ICs, connectors) are skipped.
#[must_use]
pub fn power_rating_check(
    comps: &[crate::place::component::Component],
    lib: &[crate::place::footprint::FootprintDef],
    dissipation_w: impl Fn(&crate::place::footprint::FootprintDef) -> f64,
) -> PowerRatingReport {
    let mut overloaded = Vec::new();
    for c in comps {
        let fp = &lib[c.fp];
        let Some(rating) = chip_power_rating_w(&fp.name) else {
            continue;
        };
        let p = dissipation_w(fp);
        if p > rating {
            overloaded.push((c.refdes.clone(), p, rating));
        }
    }
    overloaded.sort_by(|a, b| a.0.cmp(&b.0));
    PowerRatingReport {
        pass: overloaded.is_empty(),
        overloaded,
    }
}

/// Thermally-derated efficiency accounting for hot Rds_on.
///
/// The FET on-resistance increases with junction temperature:
/// `R_on(T_j) = R_on_25 · (1 + α · (T_j − 298))`,  `α ≈ 0.6 %/K` for Si LDMOSFETs
/// (typical for the HV7355 class of power switches; exact value from the I_D vs V_DS
/// curves at elevated temperature on the datasheet).
///
/// This raises the `dynamic_device` loss component (R_on fraction of `C·V²·f`) and
/// lowers efficiency at high temperature — the key coupling between the thermal field
/// and the driver electrical model. At 80 °C junction rise (T_j = 378 K), a typical
/// `α = 0.6 %/K` gives `R_on(T_j) ≈ 1.48 × R_on(25 °C)`, reducing efficiency by a
/// significant margin relative to the cold-start model.
///
/// Returns the derated efficiency (0–1).
///
/// - `op` — operating point at 25 °C (cold Rds_on)
/// - `t_j_k` — junction temperature (K); use `crate::physics::thermal::junction_temperature_k`
/// - `alpha_per_k` — temperature coefficient of Rds_on (K⁻¹); typical Si LDMOS ≈ 6.0e-3
/// - `p_acoustic_w` — acoustic output power delivered (W)
#[must_use]
pub fn thermally_derated_efficiency(
    op: &PulserOp,
    t_j_k: f64,
    alpha_per_k: f64,
    p_acoustic_w: f64,
) -> f64 {
    let t_ref_k = 298.0; // 25 °C
    let r_on_hot = op.r_on_ohm * (1.0 + alpha_per_k * (t_j_k - t_ref_k).max(0.0));
    let op_hot = PulserOp {
        r_on_ohm: r_on_hot,
        ..*op
    };
    let d = pulser_dissipation(&op_hot);
    driver_efficiency(p_acoustic_w, d.device_total)
}
