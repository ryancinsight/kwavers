use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

#[derive(Debug)]
pub struct DiscontinuityDetector {
    threshold: f64,
}

impl DiscontinuityDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl super::traits::DiscontinuityDetection for DiscontinuityDetector {
    fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>> {
        let mut mask = Array3::from_elem(field.dim(), false);
        if grid.nx < 3 || grid.ny < 3 || grid.nz < 3 {
            return Ok(mask);
        }

        let inv_dx = 1.0 / grid.dx;
        let inv_dy = 1.0 / grid.dy;
        let inv_dz = 1.0 / grid.dz;

        let th2 = self.threshold * self.threshold;
        for k in 1..grid.nz - 1 {
            for j in 1..grid.ny - 1 {
                for i in 1..grid.nx - 1 {
                    let dfdx = 0.5 * (field[[i + 1, j, k]] - field[[i - 1, j, k]]) * inv_dx;
                    let dfdy = 0.5 * (field[[i, j + 1, k]] - field[[i, j - 1, k]]) * inv_dy;
                    let dfdz = 0.5 * (field[[i, j, k + 1]] - field[[i, j, k - 1]]) * inv_dz;
                    let grad2 = dfdx * dfdx + dfdy * dfdy + dfdz * dfdz;
                    mask[[i, j, k]] = grad2 > th2;
                }
            }
        }

        Ok(mask)
    }

    fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}
