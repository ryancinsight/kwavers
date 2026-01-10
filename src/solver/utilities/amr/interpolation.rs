//! Interpolation schemes for mesh refinement

use super::octree::Octree;
use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Interpolation scheme for refinement/coarsening
#[derive(Debug, Clone, Copy)]
pub enum InterpolationScheme {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Conservative interpolation
    Conservative,
}

/// Conservative interpolator for AMR
#[derive(Debug)]
pub struct ConservativeInterpolator {
    scheme: InterpolationScheme,
}

impl Default for ConservativeInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservativeInterpolator {
    /// Create a new conservative interpolator
    #[must_use]
    pub fn new() -> Self {
        Self {
            scheme: InterpolationScheme::Conservative,
        }
    }

    /// Interpolate field to refined mesh using octree structure
    ///
    /// Traverses the octree and interpolates field values from coarse cells
    /// to refined cells based on the selected interpolation scheme.
    ///
    /// Algorithm:
    /// 1. Traverse octree depth-first
    /// 2. For each refined leaf node, interpolate from parent
    /// 3. Use scheme-specific interpolation (linear/cubic/conservative)
    ///
    /// References:
    /// - Berger & Colella (1989): "Local adaptive mesh refinement for shock hydrodynamics"
    /// - Berger & Oliger (1984): "Adaptive mesh refinement for hyperbolic PDEs"
    pub fn interpolate_to_refined(
        &self,
        octree: &Octree,
        field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        // Create output field with same dimensions initially
        // Will be refined where octree indicates refinement
        let mut refined_field = field.clone();

        // Traverse octree and interpolate refined regions
        self.interpolate_node(octree.root(), field, &mut refined_field, nx, ny, nz)?;

        Ok(refined_field)
    }

    /// Recursively interpolate field values for octree nodes
    fn interpolate_node(
        &self,
        node: &super::octree::OctreeNode,
        coarse_field: &Array3<f64>,
        refined_field: &mut Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // If leaf node, use current field values
        if node.is_leaf() {
            return Ok(());
        }

        // If internal node with children, interpolate for each child
        if let Some(ref children) = node.children {
            for child in children.iter() {
                // Calculate child region bounds in grid coordinates
                let bounds = &child.bounds;

                // Map spatial bounds to grid indices
                let i_min = ((bounds.min[0] / (bounds.max[0] - bounds.min[0])) * nx as f64).max(0.0)
                    as usize;
                let i_max = ((bounds.max[0] / (bounds.max[0] - bounds.min[0])) * nx as f64)
                    .min(nx as f64) as usize;
                let j_min = ((bounds.min[1] / (bounds.max[1] - bounds.min[1])) * ny as f64).max(0.0)
                    as usize;
                let j_max = ((bounds.max[1] / (bounds.max[1] - bounds.min[1])) * ny as f64)
                    .min(ny as f64) as usize;
                let k_min = ((bounds.min[2] / (bounds.max[2] - bounds.min[2])) * nz as f64).max(0.0)
                    as usize;
                let k_max = ((bounds.max[2] / (bounds.max[2] - bounds.min[2])) * nz as f64)
                    .min(nz as f64) as usize;

                // Extract subregion from coarse field
                if i_max > i_min
                    && j_max > j_min
                    && k_max > k_min
                    && i_max <= nx
                    && j_max <= ny
                    && k_max <= nz
                {
                    // Get coarse subregion
                    let coarse_region =
                        coarse_field.slice(ndarray::s![i_min..i_max, j_min..j_max, k_min..k_max]);

                    // Interpolate to refined mesh
                    let refined_region = match self.scheme {
                        InterpolationScheme::Linear => self.prolongate(&coarse_region.to_owned()),
                        InterpolationScheme::Cubic => self.prolongate(&coarse_region.to_owned()),
                        InterpolationScheme::Conservative => {
                            self.prolongate(&coarse_region.to_owned())
                        }
                    };

                    // Write refined values back to output (if dimensions match)
                    let (ref_nx, ref_ny, ref_nz) = refined_region.dim();
                    let write_i_max = (i_min + ref_nx).min(nx);
                    let write_j_max = (j_min + ref_ny).min(ny);
                    let write_k_max = (k_min + ref_nz).min(nz);

                    for i in i_min..write_i_max {
                        for j in j_min..write_j_max {
                            for k in k_min..write_k_max {
                                let ri = i - i_min;
                                let rj = j - j_min;
                                let rk = k - k_min;
                                if ri < ref_nx && rj < ref_ny && rk < ref_nz {
                                    refined_field[[i, j, k]] = refined_region[[ri, rj, rk]];
                                }
                            }
                        }
                    }
                }

                // Recursively process children
                self.interpolate_node(child, coarse_field, refined_field, nx, ny, nz)?;
            }
        }

        Ok(())
    }

    /// Prolongation: coarse to fine
    #[must_use]
    pub fn prolongate(&self, coarse: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = coarse.dim();
        let mut fine = Array3::zeros((nx * 2, ny * 2, nz * 2));

        match self.scheme {
            InterpolationScheme::Linear => self.linear_prolongation(coarse, &mut fine),
            InterpolationScheme::Cubic => self.cubic_prolongation(coarse, &mut fine),
            InterpolationScheme::Conservative => self.conservative_prolongation(coarse, &mut fine),
        }

        fine
    }

    /// Restriction: fine to coarse
    #[must_use]
    pub fn restrict(&self, fine: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = fine.dim();
        let mut coarse = Array3::zeros((nx / 2, ny / 2, nz / 2));

        match self.scheme {
            InterpolationScheme::Linear => self.linear_restriction(fine, &mut coarse),
            InterpolationScheme::Cubic => self.cubic_restriction(fine, &mut coarse),
            InterpolationScheme::Conservative => self.conservative_restriction(fine, &mut coarse),
        }

        coarse
    }

    /// Linear prolongation (injection with linear interpolation)
    fn linear_prolongation(&self, coarse: &Array3<f64>, fine: &mut Array3<f64>) {
        let (nx, ny, nz) = coarse.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let val = coarse[[i, j, k]];

                    // Direct injection
                    fine[[2 * i, 2 * j, 2 * k]] = val;

                    // Linear interpolation for other points
                    if i < nx - 1 {
                        fine[[2 * i + 1, 2 * j, 2 * k]] = 0.5 * (val + coarse[[i + 1, j, k]]);
                    }
                    if j < ny - 1 {
                        fine[[2 * i, 2 * j + 1, 2 * k]] = 0.5 * (val + coarse[[i, j + 1, k]]);
                    }
                    if k < nz - 1 {
                        fine[[2 * i, 2 * j, 2 * k + 1]] = 0.5 * (val + coarse[[i, j, k + 1]]);
                    }

                    // Trilinear for interior points
                    if i < nx - 1 && j < ny - 1 && k < nz - 1 {
                        fine[[2 * i + 1, 2 * j + 1, 2 * k + 1]] = 0.125
                            * (coarse[[i, j, k]]
                                + coarse[[i + 1, j, k]]
                                + coarse[[i, j + 1, k]]
                                + coarse[[i + 1, j + 1, k]]
                                + coarse[[i, j, k + 1]]
                                + coarse[[i + 1, j, k + 1]]
                                + coarse[[i, j + 1, k + 1]]
                                + coarse[[i + 1, j + 1, k + 1]]);
                    }
                }
            }
        }
    }

    /// Conservative prolongation (preserves integral)
    fn conservative_prolongation(&self, coarse: &Array3<f64>, fine: &mut Array3<f64>) {
        let (nx, ny, nz) = coarse.dim();

        // Each coarse cell is divided into 8 fine cells
        // The value is distributed equally to preserve the integral
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let val = coarse[[i, j, k]];

                    // Distribute value to 8 fine cells
                    for di in 0..2 {
                        for dj in 0..2 {
                            for dk in 0..2 {
                                let fi = 2 * i + di;
                                let fj = 2 * j + dj;
                                let fk = 2 * k + dk;

                                if fi < fine.dim().0 && fj < fine.dim().1 && fk < fine.dim().2 {
                                    fine[[fi, fj, fk]] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Cubic prolongation (higher-order interpolation)
    ///
    /// Currently uses linear interpolation as cubic spline interpolation requires
    /// solving tridiagonal systems for each dimension which adds computational complexity.
    /// Linear interpolation is second-order accurate which is sufficient for AMR
    /// applications where the refinement factor is typically 2:1.
    ///
    /// Full cubic implementation deferred to Sprint 122+ pending performance requirements.
    fn cubic_prolongation(&self, coarse: &Array3<f64>, fine: &mut Array3<f64>) {
        // Use linear prolongation - sufficient for 2:1 refinement ratios
        self.linear_prolongation(coarse, fine);
    }

    /// Linear restriction (averaging)
    fn linear_restriction(&self, fine: &Array3<f64>, coarse: &mut Array3<f64>) {
        let (nx, ny, nz) = coarse.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Average 8 fine cells to get coarse value
                    let mut sum = 0.0;
                    let mut count = 0;

                    for di in 0..2 {
                        for dj in 0..2 {
                            for dk in 0..2 {
                                let fi = 2 * i + di;
                                let fj = 2 * j + dj;
                                let fk = 2 * k + dk;

                                if fi < fine.dim().0 && fj < fine.dim().1 && fk < fine.dim().2 {
                                    sum += fine[[fi, fj, fk]];
                                    count += 1;
                                }
                            }
                        }
                    }

                    if count > 0 {
                        coarse[[i, j, k]] = sum / f64::from(count);
                    }
                }
            }
        }
    }

    /// Conservative restriction (volume-weighted averaging)
    fn conservative_restriction(&self, fine: &Array3<f64>, coarse: &mut Array3<f64>) {
        // Same as linear for uniform grids
        // Would differ for non-uniform grids
        self.linear_restriction(fine, coarse);
    }

    /// Cubic restriction
    ///
    /// Currently uses linear restriction for computational efficiency.
    /// Cubic restriction would require weighted averaging with cubic kernel support
    /// which is rarely needed in practice as simple averaging is conservative.
    fn cubic_restriction(&self, fine: &Array3<f64>, coarse: &mut Array3<f64>) {
        // Linear restriction is sufficient for most AMR applications
        self.linear_restriction(fine, coarse);
    }
}
