use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;

#[derive(Debug)]
pub struct DiscontinuityDetector {
    threshold: f64,
}

impl DiscontinuityDetector {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Detect discontinuities into caller-owned storage.
    ///
    /// # Errors
    /// Returns an error when field, grid, and output dimensions diverge.
    pub fn detect_into(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
        output: &mut Array3<bool>,
    ) -> KwaversResult<()> {
        if field.dim() != (grid.nx, grid.ny, grid.nz) || output.dim() != field.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "DiscontinuityDetector dimension mismatch: field={:?}, output={:?}, grid=({}, {}, {})",
                field.dim(),
                output.dim(),
                grid.nx,
                grid.ny,
                grid.nz
            )));
        }

        output.fill(false);
        if grid.nx == 1 && grid.ny == 1 && grid.nz == 1 {
            return Ok(());
        }

        let th2 = self.threshold * self.threshold;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let dfdx =
                        axis_gradient(field, [grid.nx, grid.ny, grid.nz], [i, j, k], 0, grid.dx);
                    let dfdy =
                        axis_gradient(field, [grid.nx, grid.ny, grid.nz], [i, j, k], 1, grid.dy);
                    let dfdz =
                        axis_gradient(field, [grid.nx, grid.ny, grid.nz], [i, j, k], 2, grid.dz);
                    let grad2 = dfdx * dfdx + dfdy * dfdy + dfdz * dfdz;
                    output[[i, j, k]] = grad2 > th2;
                }
            }
        }
        Ok(())
    }
}

impl super::traits::DiscontinuityDetection for DiscontinuityDetector {
    fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>> {
        let mut mask = Array3::from_elem(field.dim(), false);
        self.detect_into(field, grid, &mut mask)?;
        Ok(mask)
    }

    fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

fn axis_gradient(
    field: &Array3<f64>,
    dims: [usize; 3],
    index: [usize; 3],
    axis: usize,
    spacing: f64,
) -> f64 {
    let n = dims[axis];
    if n == 1 {
        return 0.0;
    }

    let mut left = index;
    let mut right = index;
    if n == 2 {
        left[axis] = 0;
        right[axis] = 1;
        return (field[[right[0], right[1], right[2]]] - field[[left[0], left[1], left[2]]])
            / spacing;
    }

    left[axis] = if index[axis] == 0 {
        n - 1
    } else {
        index[axis] - 1
    };
    right[axis] = (index[axis] + 1) % n;
    0.5 * (field[[right[0], right[1], right[2]]] - field[[left[0], left[1], left[2]]]) / spacing
}

#[cfg(test)]
mod tests {
    use super::DiscontinuityDetector;
    use crate::forward::pstd::dg::traits::DiscontinuityDetection;
    use kwavers_grid::Grid;
    use leto::Array3;

    #[test]
    fn detector_marks_embedded_1d_jump() {
        let grid = Grid::new(4, 1, 1, 1.0, 1.0, 1.0).unwrap();
        let field = Array3::from_shape_vec((4, 1, 1), vec![0.0, 0.0, 2.0, 2.0]).unwrap();
        let detector = DiscontinuityDetector::new(0.5);

        let mask = detector.detect(&field, &grid).unwrap();

        assert!(mask.iter().any(|&flag| flag));
    }

    #[test]
    fn detector_marks_embedded_2d_jump() {
        let grid = Grid::new(4, 4, 1, 1.0, 1.0, 1.0).unwrap();
        let field = Array3::from_shape_fn(
            (4, 4, 1),
            |(i, j, _)| {
                if i >= 2 || j >= 2 {
                    3.0
                } else {
                    0.0
                }
            },
        );
        let detector = DiscontinuityDetector::new(0.5);

        let mask = detector.detect(&field, &grid).unwrap();

        assert!(mask.iter().any(|&flag| flag));
    }
}
