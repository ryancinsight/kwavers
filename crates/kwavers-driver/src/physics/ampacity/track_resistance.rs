use super::ipc2221::copper_thickness_m;

/// DC resistance (Ω) of a copper track of length/width (metres) and weight (oz). ρ_Cu = 1.68e-8 Ω·m.
#[must_use]
pub fn track_resistance(len_m: f64, width_m: f64, copper_oz: f64) -> f64 {
    let rho = 1.68e-8;
    let t = copper_thickness_m(copper_oz);
    if width_m <= 0.0 || t <= 0.0 {
        return 0.0;
    }
    rho * len_m / (width_m * t)
}
