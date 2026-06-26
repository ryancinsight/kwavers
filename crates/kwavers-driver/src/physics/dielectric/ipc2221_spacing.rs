//! IPC-2221B Table 6-1 B1 minimum conductor spacing (mm) for `voltage_v` between conductors,
//! external uncoated, below 3050 m.
//!
//! Piecewise table from the standard; above 500 V the spacing grows at `0.005 mm/V`. The 150 V
//! floor is **0.60 mm** — that exact value is the SSOT that
//! [`crate::rules::CreepageRule::holohv`] matches, so the routing creepage rule, the DRU
//! HV-creepage rule, and the `.kicad_dru` emission can never silently drift apart.

/// IPC-2221B Table 6-1 B1 minimum conductor spacing (mm) for `voltage_v` between conductors,
/// external uncoated, below 3050 m. Piecewise from the standard's B1 table; above 500 V it grows
/// at 0.005 mm/V.
#[must_use]
pub fn ipc2221_min_spacing_mm(voltage_v: f64) -> f64 {
    let v = voltage_v.abs();
    if v <= 15.0 {
        0.13
    } else if v <= 30.0 {
        0.25
    } else if v <= 50.0 {
        0.40
    } else if v <= 100.0 {
        0.50
    } else if v <= 150.0 {
        0.60 // IPC-2221B Table 6-1 B1 (external uncoated)
    } else if v <= 300.0 {
        1.25
    } else if v <= 500.0 {
        2.50
    } else {
        2.50 + (v - 500.0) * 0.005
    }
}
