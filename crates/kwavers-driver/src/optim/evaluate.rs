//! [`evaluate_design_point`] — the orchestrator that drives every physics module once for a single
//! operating point and assembles the [`DesignReport`]. Adds no new physics; it wires the kernels.

use crate::driver::{
    driver_efficiency, pulser_dissipation, switching_node_ringing_v, thermally_derated_efficiency,
    PulserOp,
};
use crate::physics::acoustic::{
    acoustic_intensity_w_per_m2, array_factor, focal_pressure_gain, nonlinear_shock_parameter,
};
use crate::physics::emi::{capacitive_drive_current_a, loop_inductance_nh, radiated_emi_dbuv_m};
use crate::physics::pdn::{anti_resonance_hz, pdn_impedance_at_freq, target_impedance_ohm};
use crate::physics::thermal::junction_temperature_k;

use super::context::{ArrayGeometry, EmiContext, PdnConfig, ThermalContext};
use super::report::DesignReport;

/// Evaluate the full physics co-design point.
///
/// This function combines all physics modules into one structured report so the designer
/// can evaluate a proposed IC + layout strategy without building the individual calls.
/// The report surfaces the binding constraints (thermal headroom, PDN margin, EMI) alongside
/// the acoustic output figures.
///
/// # Arguments
/// - `op` — electrical operating point (frequency, duty, C_load, V_pp, R_on, …)
/// - `array` — transducer geometry
/// - `thermal` — board thermal context (ambient temperature, board rise from the thermal solver)
/// - `pdn` — PDN capacitor bank and ripple budget
/// - `emi` — regulatory EMI test context
/// - `p_acoustic_w` — acoustic output power delivered to the medium (W)
/// - `b_over_a` — nonlinear parameter of the medium (water ≈ 5.2, tissue ≈ 6.0)
/// - `z0_rayl` — characteristic acoustic impedance of the medium (Rayl = Pa·s/m)
#[must_use]
#[allow(clippy::too_many_arguments)] // physics co-design kernel: each argument is an irreducible physical input
pub fn evaluate_design_point(
    op: &PulserOp,
    array: &ArrayGeometry,
    thermal: &ThermalContext,
    pdn: &PdnConfig,
    emi: &EmiContext,
    p_acoustic_w: f64,
    b_over_a: f64,
    z0_rayl: f64,
) -> DesignReport {
    // ── Electrical ──────────────────────────────────────────────────────────────────────────
    let diss_cold = pulser_dissipation(op);
    let p_device_cold_w = diss_cold.device_total;
    let efficiency_cold = driver_efficiency(p_acoustic_w, p_device_cold_w);

    // Switching node ringing: use capacitive drive current as peak current estimate.
    let i_peak = capacitive_drive_current_a(op.c_load_f, op.v_pp, 1.0 / (2.0 * op.drive_hz));
    let loop_nh = loop_inductance_nh(pdn.loop_area_mm2);
    let switching_ring_v = switching_node_ringing_v(i_peak, loop_nh, op.c_load_f);

    // ── Thermal ─────────────────────────────────────────────────────────────────────────────
    let t_j_k = junction_temperature_k(
        thermal.t_ambient_k,
        thermal.board_rise_k,
        thermal.theta_jc_k_per_w,
        p_device_cold_w,
    );
    let thermal_pass = t_j_k <= thermal.t_j_max_k;
    let efficiency_derated =
        thermally_derated_efficiency(op, t_j_k, thermal.alpha_rds_per_k, p_acoustic_w);
    // Compute derated dissipation from derated efficiency.
    let p_device_derated_w = if efficiency_derated > 0.0 && efficiency_derated < 1.0 {
        p_acoustic_w * (1.0 - efficiency_derated) / efficiency_derated
    } else {
        p_device_cold_w
    };

    // ── Acoustic ────────────────────────────────────────────────────────────────────────────
    let fp_gain = focal_pressure_gain(array.n_elements);
    // Intensity model: 1 Pa source at a single element; coherent gain N^2 for intensity.
    let acoustic_intensity_w_m2_per_pa2 = acoustic_intensity_w_per_m2(1.0, z0_rayl);
    let sigma = nonlinear_shock_parameter(
        // Representative peak pressure estimate from acoustic output power and aperture.
        // p_acoustic ≈ sqrt(2 · I · Z0); I ≈ p_acoustic_w / (π/4 · aperture²).
        {
            let aperture_m = array.pitch_m * (array.n_elements as f64 - 1.0);
            let area = std::f64::consts::PI / 4.0 * aperture_m * aperture_m;
            let i = if area > 0.0 { p_acoustic_w / area } else { 0.0 };
            (2.0 * i * z0_rayl).sqrt()
        },
        array.freq_hz,
        array.focal_m,
        1050.0, // tissue density kg/m³
        array.speed_m_s,
        b_over_a,
    );
    let af_steer = array_factor(
        array.n_elements,
        array.pitch_m,
        array.lambda_m,
        array.steer_deg,
        array.steer_deg,
    );
    let af_broadside = array_factor(
        array.n_elements,
        array.pitch_m,
        array.lambda_m,
        array.steer_deg,
        0.0,
    );

    // ── PDN ─────────────────────────────────────────────────────────────────────────────────
    let pdn_z = pdn_impedance_at_freq(&pdn.caps, op.drive_hz);
    let pdn_z_target = target_impedance_ohm(pdn.v_ripple, pdn.i_transient_a);
    let pdn_pass = pdn_z <= pdn_z_target;

    // Anti-resonance between the bulk cap (approximated as largest-value cap + highest ESL)
    // and the local cap (smallest ESL cap).
    let ar_hz = if pdn.caps.len() >= 2 {
        let l_bulk = pdn.caps.iter().map(|c| c.2).fold(0.0_f64, f64::max);
        let c_local = pdn
            .caps
            .iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.0)
            .unwrap_or(0.0);
        anti_resonance_hz(l_bulk, c_local)
    } else {
        f64::INFINITY
    };

    // ── EMI ─────────────────────────────────────────────────────────────────────────────────
    let emi_dbuvpm =
        radiated_emi_dbuv_m(op.drive_hz, pdn.loop_area_mm2, i_peak, emi.test_distance_m);
    let emi_pass = emi_dbuvpm <= emi.limit_dbuv_m;

    DesignReport {
        p_device_cold_w,
        efficiency_cold,
        switching_ring_v,
        t_j_k,
        thermal_pass,
        efficiency_derated,
        p_device_derated_w,
        focal_pressure_gain: fp_gain,
        acoustic_intensity_w_m2_per_pa2,
        nonlinear_sigma: sigma,
        array_factor_at_steer: af_steer,
        array_factor_broadside: af_broadside,
        pdn_z_at_freq_ohm: pdn_z,
        pdn_z_target_ohm: pdn_z_target,
        pdn_pass,
        anti_resonance_hz: ar_hz,
        radiated_emi_dbuv_m: emi_dbuvpm,
        emi_pass,
    }
}
