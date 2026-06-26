//! Electro-thermal solve family: Poisson solver, source-field builders, and the convenience
//! `solve_board` + `solve_electrothermal` entry points that combine device dissipation with
//! [`super::joule_source()`] track heating into a single [`super::ThermalField`].

use crate::geom::GridSpec;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

use super::ThermalField;

/// Solve `-∇²T + βT = f` on the grid with Dirichlet `T = 0` on the boundary, by Gauss–Seidel.
/// Here β = 2 * h_conv / (k_eff * thickness) models surface convection (top & bottom faces).
///
/// `f` is the per-column source term `q/k_eff` (units K/m²). `iters` Gauss–Seidel sweeps.
#[must_use]
pub fn solve_poisson(
    spec: GridSpec,
    f: &[f64],
    k_eff: f64,
    thickness: f64,
    h_conv: f64,
    iters: usize,
) -> ThermalField {
    let (nx, ny) = (spec.nx, spec.ny);
    let h = spec.pitch.to_mm() * 1.0e-3; // metres
    let h2 = h * h;
    let mut t = vec![0.0f64; nx * ny];
    let beta = if k_eff > 0.0 && thickness > 0.0 {
        (2.0 * h_conv) / (k_eff * thickness)
    } else {
        0.0
    };
    let denom = 4.0 + beta * h2;
    if nx >= 3 && ny >= 3 {
        for _ in 0..iters {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let i = iy * nx + ix;
                    t[i] = (t[i - 1] + t[i + 1] + t[i - nx] + t[i + nx] + h2 * f[i]) / denom;
                }
            }
        }
    }
    ThermalField { spec, temp: t }
}

/// Build the source field `f = q/k_eff` from placed components: each component's dissipation
/// (from `watts`) is spread uniformly over its courtyard cells.
///
/// `k_eff` is the effective in-plane conductivity (W/m·K) of the copper+laminate stack; `thickness`
/// the effective conducting thickness (m).
#[must_use]
pub fn power_source(
    spec: GridSpec,
    comps: &[Component],
    lib: &[FootprintDef],
    watts: impl Fn(&FootprintDef) -> f64,
    k_eff: f64,
    thickness: f64,
) -> Vec<f64> {
    let mut f = vec![0.0f64; spec.nx * spec.ny];
    let cell_area = (spec.pitch.to_mm() * 1.0e-3).powi(2);
    let vol = cell_area * thickness; // m³ per cell
    for c in comps {
        let p = watts(&lib[c.fp]);
        if p <= 0.0 {
            continue;
        }
        let rect = c.courtyard(lib);
        let cells = spec.cells_in_rect(
            c.placement.pos,
            crate::geom::Nm((rect.max.x - rect.min.x).0 / 2),
            crate::geom::Nm((rect.max.y - rect.min.y).0 / 2),
        );
        if cells.is_empty() {
            continue;
        }
        let p_per_cell = p / cells.len() as f64;
        for (ix, iy) in cells {
            let q = p_per_cell / vol; // W/m³
            f[iy * spec.nx + ix] += q / k_eff; // K/m²
        }
    }
    f
}

/// Convenience: solve the board's thermal field from placed components.
#[allow(clippy::too_many_arguments)] // a thermal kernel: each input is a distinct physical quantity.
#[must_use]
pub fn solve_board(
    spec: GridSpec,
    comps: &[Component],
    lib: &[FootprintDef],
    watts: impl Fn(&FootprintDef) -> f64,
    k_eff: f64,
    thickness: f64,
    h_conv: f64,
    iters: usize,
) -> ThermalField {
    let f = power_source(spec, comps, lib, watts, k_eff, thickness);
    solve_poisson(spec, &f, k_eff, thickness, h_conv, iters)
}

/// Electro-thermal solve: device dissipation plus track Joule heating in one temperature field.
#[allow(clippy::too_many_arguments)] // an electro-thermal kernel: each input is physical and required
#[must_use]
pub fn solve_electrothermal(
    spec: GridSpec,
    comps: &[Component],
    lib: &[FootprintDef],
    board: &crate::board::Board,
    watts: impl Fn(&FootprintDef) -> f64,
    current_a: impl Fn(crate::board::NetId) -> f64,
    copper_oz: f64,
    k_eff: f64,
    thickness: f64,
    h_conv: f64,
    iters: usize,
) -> ThermalField {
    let mut f = power_source(spec, comps, lib, watts, k_eff, thickness);
    let j = super::joule_source::joule_source(spec, board, current_a, copper_oz, k_eff, thickness);
    for (a, b) in f.iter_mut().zip(j.iter()) {
        *a += b;
    }
    solve_poisson(spec, &f, k_eff, thickness, h_conv, iters)
}
