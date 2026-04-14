// FDTD dynamic source injection — extracted from solver.rs for SRP compliance.
//
// # Responsibility
//
// This module manages the injection of `Arc<dyn Source>` dynamic sources into the FDTD
// field arrays at each time step. It is responsible for:
//
// 1. **Source registration** (`add_source_arc`): converting a dynamic `Source` to a
//    pre-computed sparse index and determining the injection mode (Boundary vs Additive).
//
// 2. **Per-step injection** (`apply_dynamic_pressure_sources`,
//    `apply_dynamic_pressure_dirichlet`, `apply_dynamic_velocity_sources`): O(nnz) update
//    of the flat field slices using the pre-built sparse index.
//
// # Injection Modes
//
// ## Boundary (Dirichlet enforcement)
// When the source mask is entirely concentrated on a single domain-boundary plane
// (x=0, x=Nₓ−1, y=0, y=Nᵧ−1, z=0, z=N_z−1), injection uses Dirichlet assignment:
// ```text
//   p[flat_idx] = amplitude(t)
// ```
// This is the physical model for a planar transducer producing a prescribed pressure
// waveform at the boundary. The sparse index stores only voxels with |m| > 1e-12.
//
// ## Additive
// Interior or distributed sources use additive injection with L1-normalised mask:
// ```text
//   p[flat_idx] += m * amplitude(t)   where Σ m = 1.0
// ```
// L1 normalisation preserves total injected energy independent of the number of
// active voxels, which is the correct scaling for a distributed acoustic source
// (Treeby & Cox 2010, §II.B).
//
// # Sparse Index
//
// At `add_source_arc` time, `source.create_mask(grid)` is scanned once to produce
// a contiguous `Vec<(flat_index, mask_value)>` for each source. Per-step injection
// iterates only over non-zero voxels (O(nnz)), rather than the full O(nx·ny·nz)
// dense scan. For a phased-array transducer with 128 elements on a 256³ grid:
//   nnz ≈ 1536 cells vs 16.8M total — ~11,000× fewer iterations.
//
// # References
// - Treeby, B.E. & Cox, B.T. (2010). J. Biomed. Opt. 15(2), 021314.
//   doi:10.1117/1.3360308. §II.B (additive pressure source scaling).
// - k-Wave MATLAB kspaceFirstOrder3D.m, lines 780–820 (source injection).

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::source::{Source, SourceField, SourceInjectionMode};
use ndarray::{s, Array3};
use std::sync::Arc;

use crate::domain::field::wave::WaveFields;

/// FDTD Dynamic Source Injector
///
/// Encapsulates dynamic sources, precomputing a sparse indexing mask to preserve O(nnz) updates
/// rather than dense O(nx*ny*nz) updates during every simulation timestep.
#[derive(Debug)]
pub(crate) struct DynamicSourceInjector {
    sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    sparse: Vec<(usize, f64)>,
    offsets: Vec<usize>,
    modes: Vec<SourceInjectionMode>,
}

impl DynamicSourceInjector {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            sparse: Vec::new(),
            offsets: vec![0], // Match prefix-sum layout semantics
            modes: Vec::new(),
        }
    }

    pub fn sources_len(&self) -> usize {
        self.sources.len()
    }

    pub fn add_source_arc(&mut self, source: Arc<dyn Source>, grid: &Grid) -> KwaversResult<()> {
        let mask = source.create_mask(grid);
        let mode = Self::determine_injection_mode(&mask);

        let sparse: Vec<(usize, f64)> = mask
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, v)| v.abs() > 1e-12)
            .collect();

        self.sources.push((source, mask));
        self.sparse.extend(sparse);
        self.offsets.push(self.sparse.len());
        self.modes.push(mode);
        Ok(())
    }

    pub(crate) fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
        let shape = mask.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        let x0_count = mask
            .slice(s![0, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let xn_count = mask
            .slice(s![nx - 1, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let y0_count = mask
            .slice(s![.., 0, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let yn_count = mask
            .slice(s![.., ny - 1, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let z0_count = mask
            .slice(s![.., .., 0])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let zn_count = mask
            .slice(s![.., .., nz - 1])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        let mut mask_sum = 0.0_f64;
        let mut nonzero_count = 0_usize;
        for &val in mask.iter() {
            if val > 0.0 {
                nonzero_count += 1;
                mask_sum += val;
            }
        }

        let is_boundary_plane = nonzero_count > 0
            && (x0_count == nonzero_count
                || xn_count == nonzero_count
                || y0_count == nonzero_count
                || yn_count == nonzero_count
                || z0_count == nonzero_count
                || zn_count == nonzero_count);

        if is_boundary_plane {
            SourceInjectionMode::Boundary
        } else {
            let scale = if mask_sum > 0.0 { 1.0 / mask_sum } else { 1.0 };
            SourceInjectionMode::Additive { scale }
        }
    }

    pub(crate) fn apply_dynamic_pressure_sources(
        &self,
        dt: f64,
        time_step_index: usize,
        fields: &mut WaveFields,
    ) {
        let t = time_step_index as f64 * dt;
        let p_flat = fields.p.as_slice_mut().expect("p C-contiguous");

        for (idx, (source, _mask)) in self.sources.iter().enumerate() {
            if source.source_type() != SourceField::Pressure {
                continue;
            }
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            let mode = self.modes[idx];
            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];

            match mode {
                SourceInjectionMode::Boundary => {
                    for &(fi, _m) in &self.sparse[start..end] {
                        p_flat[fi] = amp;
                    }
                }
                SourceInjectionMode::Additive { .. } => {
                    for &(fi, m) in &self.sparse[start..end] {
                        p_flat[fi] += m * amp;
                    }
                }
            }
        }
    }

    pub(crate) fn apply_dynamic_pressure_dirichlet(
        &self,
        dt: f64,
        time_step_index: usize,
        fields: &mut WaveFields,
    ) {
        let t = time_step_index as f64 * dt;
        let p_flat = fields.p.as_slice_mut().expect("p C-contiguous");

        for (idx, (source, _mask)) in self.sources.iter().enumerate() {
            if source.source_type() != SourceField::Pressure {
                continue;
            }
            if self.modes[idx] != SourceInjectionMode::Boundary {
                continue;
            }
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }
            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];
            for &(fi, _m) in &self.sparse[start..end] {
                p_flat[fi] = amp;
            }
        }
    }

    pub(crate) fn apply_dynamic_velocity_sources(
        &self,
        dt: f64,
        time_step_index: usize,
        fields: &mut WaveFields,
    ) {
        let t = time_step_index as f64 * dt;
        for (idx, (source, _mask)) in self.sources.iter().enumerate() {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];
            match source.source_type() {
                SourceField::Pressure => {}
                SourceField::VelocityX => {
                    let flat = fields.ux.as_slice_mut().expect("ux C-contiguous");
                    for &(fi, m) in &self.sparse[start..end] {
                        flat[fi] += m * amp;
                    }
                }
                SourceField::VelocityY => {
                    let flat = fields.uy.as_slice_mut().expect("uy C-contiguous");
                    for &(fi, m) in &self.sparse[start..end] {
                        flat[fi] += m * amp;
                    }
                }
                SourceField::VelocityZ => {
                    let flat = fields.uz.as_slice_mut().expect("uz C-contiguous");
                    for &(fi, m) in &self.sparse[start..end] {
                        flat[fi] += m * amp;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_boundary_plane_x0_is_boundary_mode() {
        let nx = 8;
        let mut mask = Array3::<f64>::zeros((nx, nx, nx));
        mask.slice_mut(ndarray::s![0, .., ..]).fill(1.0);
        let mode = DynamicSourceInjector::determine_injection_mode(&mask);
        assert!(
            matches!(mode, SourceInjectionMode::Boundary),
            "x=0 plane must be Boundary; got {mode:?}"
        );
    }

    #[test]
    fn test_interior_source_is_additive_mode() {
        let nx = 8;
        let mut mask = Array3::<f64>::zeros((nx, nx, nx));
        mask[[2, 3, 4]] = 1.0;
        mask[[3, 3, 4]] = 1.0;
        let mode = DynamicSourceInjector::determine_injection_mode(&mask);
        assert!(
            matches!(mode, SourceInjectionMode::Additive { .. }),
            "Interior source must be Additive; got {mode:?}"
        );
    }

    #[test]
    fn test_additive_scale_l1_normalised() {
        let nx = 8;
        let mut mask = Array3::<f64>::zeros((nx, nx, nx));
        mask[[2, 2, 2]] = 1.0;
        mask[[3, 3, 3]] = 1.0; // sum = 2.0 → scale = 0.5
        let mode = DynamicSourceInjector::determine_injection_mode(&mask);
        if let SourceInjectionMode::Additive { scale } = mode {
            assert!(
                (scale - 0.5).abs() < 1e-12,
                "L1 scale for sum=2 mask must be 0.5; got {scale}"
            );
        } else {
            panic!("Expected Additive mode; got {mode:?}");
        }
    }

    #[test]
    fn test_empty_mask_safe_default() {
        let nx = 4;
        let mask = Array3::<f64>::zeros((nx, nx, nx));
        let mode = DynamicSourceInjector::determine_injection_mode(&mask);
        assert!(
            matches!(mode, SourceInjectionMode::Additive { scale } if (scale - 1.0).abs() < 1e-12),
            "Empty mask must produce Additive{{scale=1.0}}; got {mode:?}"
        );
    }

    #[test]
    fn test_add_source_arc_builds_sparse_index() {
        use crate::domain::signal::NullSignal;
        use crate::domain::source::PointSource;
        use std::sync::Arc;

        let nx = 8usize;
        let dx = 1e-3;
        let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");

        let mut injector = DynamicSourceInjector::new();

        // Place a point source at cell (1,1,1) → interior → Additive mode
        let center_x = 1.5 * dx; // grid.indices_to_coordinates(1,1,1) ≈ 1.5 mm
        let src = Arc::new(PointSource::new(
            (center_x, center_x, center_x),
            Arc::new(NullSignal),
        ));

        let n_before = injector.sources_len();
        injector.add_source_arc(src, &grid).expect("add source");
        assert_eq!(
            injector.sources_len(),
            n_before + 1,
            "Source must be registered"
        );
        // Sparse index must have exactly 1 entry for a single-cell point source
        let start = injector.offsets[n_before];
        let end = injector.offsets[n_before + 1];
        assert_eq!(
            end - start,
            1,
            "PointSource mask has 1 non-zero cell → 1 sparse entry"
        );
    }
}
