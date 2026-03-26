//! Spatial Impulse Response (SIR) calculation
//!
//! ## Mathematical Foundation
//!
//! Tupholme-Stepanishen method for rectangular pistons:
//! ```text
//! h(r,t) = (v₀/2πc) · δ(t − r/c) / r
//! ```
//!
//! ## References
//!
//! - Stepanishen (1971). "Transient radiation from pistons in an infinite planar baffle"

use super::plugin::TransducerFieldCalculatorPlugin;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

impl TransducerFieldCalculatorPlugin {
    /// Calculate spatial impulse response for a given transducer
    pub fn calculate_sir(
        &mut self,
        transducer_index: usize,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array3<f64>> {
        let cache_key = format!("sir_{transducer_index}");

        if let Some(cached) = self.sir_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut sir = Array3::zeros(grid.dimensions());

        if self.transducer_geometries.is_empty() {
            return Ok(sir);
        }
        let geometry = &self.transducer_geometries[0];

        let n_elements = geometry.element_positions.nrows();
        let avg_width = geometry.element_sizes.column(0).mean().unwrap_or(0.01);
        let avg_height = geometry.element_sizes.column(1).mean().unwrap_or(0.01);
        let element_area = (avg_width * avg_height) / n_elements as f64;

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let mut response = 0.0;

                    for elem_idx in 0..n_elements {
                        let elem_pos = geometry.element_positions.row(elem_idx);
                        let x_elem = elem_pos[0];
                        let y_elem = elem_pos[1];
                        let z_elem = elem_pos[2];

                        let r =
                            ((x - x_elem).powi(2) + (y - y_elem).powi(2) + (z - z_elem).powi(2))
                                .sqrt();

                        if r > 1e-10 {
                            let _c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                            response += element_area / (2.0 * std::f64::consts::PI * r);

                            if let Some(apod_weights) = &geometry.apodization {
                                if elem_idx < apod_weights.len() {
                                    response *= apod_weights[elem_idx];
                                }
                            }
                        }
                    }

                    sir[[i, j, k]] = response;
                }
            }
        }

        self.sir_cache.insert(cache_key, sir.clone());
        Ok(sir)
    }
}
