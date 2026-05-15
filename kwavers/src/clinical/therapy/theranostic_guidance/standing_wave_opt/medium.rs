use ndarray::Array2;

use super::config::StandingWaveOptConfig;

/// Build the (nx, ny) sound-speed field with the reflective slab.
pub(super) fn build_sound_speed(config: &StandingWaveOptConfig) -> Array2<f64> {
    let mut c = Array2::from_elem((config.nx, config.ny), config.c_ref_m_s);
    for xi in config.layer_x_start..config.layer_x_end.min(config.nx) {
        for yi in 0..config.ny {
            c[[xi, yi]] = config.c_layer_m_s;
        }
    }
    c
}

/// PML absorbing boundary: quadratic sponge, shape (nx, ny).
///
/// Cells within `pml_cells` of any edge are exponentially attenuated.
/// `mask[[xi, yi]] = exp(−0.20 × depth²)` where `depth ∈ [0, 1]`.
pub(super) fn pml_damping(config: &StandingWaveOptConfig) -> Array2<f64> {
    let nx = config.nx;
    let ny = config.ny;
    let p = config.pml_cells;
    let mut mask = Array2::ones((nx, ny));
    for xi in 0..nx {
        for yi in 0..ny {
            let dist = xi.min(yi).min(nx - 1 - xi).min(ny - 1 - yi);
            if dist < p {
                let depth = (p - dist) as f64 / p as f64;
                mask[[xi, yi]] = (-0.20 * depth * depth).exp();
            }
        }
    }
    mask
}
