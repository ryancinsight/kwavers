//! `ElasticProperties` and `ElasticArrayAccess` impls for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::grid::Grid;
use crate::medium::absorption::TISSUE_PROPERTIES;
use crate::medium::elastic::{ElasticArrayAccess, ElasticProperties};
use ndarray::Array3;

impl ElasticProperties for HeterogeneousTissueMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).lame_lambda
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).lame_mu
    }
}

impl ElasticArrayAccess for HeterogeneousTissueMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                arr[[i, j, k]] = props.lame_lambda;
            }
        }
        arr
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                arr[[i, j, k]] = props.lame_mu;
            }
        }
        arr
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        // Mathematical specification: c_s = sqrt(μ / ρ)
        // Compute shear wave speed from tissue properties at each grid point
        let mut arr = Array3::zeros(self.tissue_map.dim());
        for ((i, j, k), tissue_type) in self.tissue_map.indexed_iter() {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                // Get density from tissue properties
                let density = props.density;
                let mu = props.lame_mu;

                // Compute shear wave speed: c_s = sqrt(μ / ρ)
                arr[[i, j, k]] = if density > 0.0 {
                    (mu / density).sqrt()
                } else {
                    0.0
                };
            }
        }
        arr
    }
}
