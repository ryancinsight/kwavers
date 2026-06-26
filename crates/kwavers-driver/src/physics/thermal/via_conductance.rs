//! Thermal-via conductance to the back/inner plane — a per-via barrel `R = L/(k·A)` in parallel.

/// Thermal conductance to the back/inner plane (W/K) provided by `n` thermal vias of the given
/// drill and plating under a power device. Each via is a copper barrel `R = L/(k·A)` in parallel;
/// `k_cu ≈ 400 W/m·K`. More vias ⇒ lower junction-to-board thermal resistance.
#[must_use]
pub fn thermal_via_conductance(
    n: usize,
    drill_m: f64,
    plating_m: f64,
    board_thickness_m: f64,
) -> f64 {
    if n == 0 || board_thickness_m <= 0.0 {
        return 0.0;
    }
    let k_cu = 400.0;
    // Annular copper barrel cross-section ≈ π·d·plating.
    let area = std::f64::consts::PI * drill_m * plating_m;
    let g_one = k_cu * area / board_thickness_m; // W/K per via
    g_one * n as f64
}
