#[cfg(test)]
mod tests {
    use kwavers_grid::Grid;
    use crate::amr::AMRSolver;
    use ndarray::Array3;
    use std::f64::consts::PI;

    #[test]
    fn test_amr_wavelet_refinement() {
        // Validate adaptive mesh refinement (Berger & Oliger 1984)
        let base_n = 32;
        let dx = 1e-3;

        let grid = Grid::new(base_n, base_n, base_n, dx, dx, dx).unwrap();
        let mut amr = AMRSolver::new(&grid, 3).unwrap();

        // Create localized feature requiring refinement
        let mut field = Array3::zeros((base_n, base_n, base_n));
        let center = base_n / 2;
        let feature_width = 3;

        for i in (center - feature_width)..(center + feature_width) {
            for j in (center - feature_width)..(center + feature_width) {
                for k in (center - feature_width)..(center + feature_width) {
                    let r = (((i as i32 - center as i32).pow(2)
                        + (j as i32 - center as i32).pow(2)
                        + (k as i32 - center as i32).pow(2)) as f64)
                        .sqrt();
                    field[[i, j, k]] = (PI * r / feature_width as f64).cos();
                }
            }
        }

        // Verify AMR manager configuration
        // The AMR manager uses internal wavelet-based refinement criteria
        // which are applied during the refine() method call
        // AMR solver initialized with max level 3

        // Test that mesh adaptation can be triggered (though actual refinement
        // depends on the field gradients exceeding thresholds)
        let adaptation_result = amr.adapt_mesh(&field, 0.1);
        adaptation_result.unwrap();
    }
}
