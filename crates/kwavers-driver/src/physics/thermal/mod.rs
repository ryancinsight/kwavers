//! Steady-state thermal field of the board, by 2D heat conduction — Phase 3b slice.
//!
//! # Phase 1a migration roadmap
//!
//! Public functions accept `f64` watts / kelvin / m·K / celsius. The signatures stay flat-`f64`
//! for Phase 1a — replacing them with `Watt` / `Kelvin` / `Celsius` (from `crate::units`) requires
//! touching every caller (especially `crate::driver`, which threads power figures through
//! `pulser_dissipation`). Phase 2 will migrate the function parameters and return values at the
//! time the `physics/thermal` vertical slice gets its full lift; Phase 1a only ships the
//! underlying newtypes so future imports can opt into compile-time unit safety on a per-call-site
//! basis.
//!
//! The HV power stage is the measured hot spot in the source paper, with passive heatsinks on the
//! board. This solves the steady-state heat equation in the board plane,
//!
//! ```text
//! -∇²T = f,    f = q / k_eff     (q: volumetric power W/m³, k_eff: effective conductivity)
//! ```
//!
//! with Dirichlet `T = 0` (ambient rise) on the board edge — the perimeter heatsink boundary — by
//! Gauss–Seidel relaxation. The temperature-rise field feeds the adversarial loop: power devices
//! are pushed apart and toward the cooled edge to hold peak temperature down.

use crate::geom::{GridSpec, Point};

/// A solved temperature-rise field (K above ambient) over the grid's in-plane columns.
#[derive(Debug, Clone)]
pub struct ThermalField {
    /// The grid the field is sampled on.
    pub spec: GridSpec,
    /// Temperature rise per in-plane column `(ix, iy)`.
    pub temp: Vec<f64>,
}

impl ThermalField {
    /// Peak temperature rise anywhere on the board (K).
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.temp.iter().copied().fold(0.0, f64::max)
    }

    /// Columns whose temperature exceeds `frac` of the peak — the thermal hotspots.
    #[must_use]
    pub fn hotspots(&self, frac: f64) -> Vec<Point> {
        let thresh = self.peak() * frac;
        let mut pts = Vec::new();
        for iy in 0..self.spec.ny {
            for ix in 0..self.spec.nx {
                if self.temp[iy * self.spec.nx + ix] >= thresh && thresh > 0.0 {
                    pts.push(self.spec.point_of(ix, iy));
                }
            }
        }
        pts
    }
}

pub mod electrothermal;
pub mod ir_drop;
pub mod joule_source;
pub mod transient;
pub mod via_conductance;

pub use electrothermal::{power_source, solve_board, solve_electrothermal, solve_poisson};
pub use ir_drop::{ir_drop, IrDrop};
pub use joule_source::joule_source;
pub use transient::{
    junction_temperature_k, temperature_derated_resistance, thermal_time_constant_s, transient_rise_k,
};
pub use via_conductance::thermal_via_conductance;

#[cfg(test)]
mod tests;
