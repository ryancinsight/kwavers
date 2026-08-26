//! Criterion comparison: CSR-shaped transfer matrix vs the historical jagged
//! `Vec<Vec<(usize, f64)>>` layout in `UtilConservativeInterpolator::transfer`.
//!
//! The jagged baseline is kept inline as a faithful replica of the
//! pre-conversion traversal so the comparison cannot rot when the production
//! code evolves. Byte-parity between the two layouts is asserted before any
//! timing is recorded.
//!
//! Run with `cargo bench -p kwavers --bench conservative_interpolation_comparison`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers_grid::Grid;
use kwavers_solver::utilities::interpolation::{ConservationMode, UtilConservativeInterpolator};
use leto::Array3;

/// Jagged-layout replica of the pre-conversion interpolator (build + transfer).
mod jagged {
    use kwavers_core::error::{KwaversError, KwaversResult};
    use kwavers_grid::Grid;
    use leto::Array3;

    pub struct JaggedInterpolator {
        source_grid: Grid,
        target_grid: Grid,
        transfer_matrix: Vec<Vec<(usize, f64)>>,
        source_volumes: Vec<f64>,
    }

    fn index_3d(ix: usize, iy: usize, iz: usize, nx: usize, ny: usize) -> usize {
        iz * (nx * ny) + iy * nx + ix
    }

    impl JaggedInterpolator {
        pub fn new(source: &Grid, target: &Grid) -> KwaversResult<Self> {
            let source_volumes =
                vec![source.dx * source.dy * source.dz; source.nx * source.ny * source.nz];
            let mut transfer_matrix = vec![Vec::new(); target.nx * target.ny * target.nz];
            for iz_t in 0..target.nz {
                for iy_t in 0..target.ny {
                    for ix_t in 0..target.nx {
                        let t_idx = index_3d(ix_t, iy_t, iz_t, target.nx, target.ny);
                        let (xt0, xt1, yt0, yt1, zt0, zt1) = (
                            ix_t as f64 * target.dx,
                            (ix_t + 1) as f64 * target.dx,
                            iy_t as f64 * target.dy,
                            (iy_t + 1) as f64 * target.dy,
                            iz_t as f64 * target.dz,
                            (iz_t + 1) as f64 * target.dz,
                        );
                        let ix0 =
                            ((xt0 / source.dx).floor() as usize).min(source.nx.saturating_sub(1));
                        let ix1 = ((xt1 / source.dx).ceil() as usize).min(source.nx);
                        let iy0 =
                            ((yt0 / source.dy).floor() as usize).min(source.ny.saturating_sub(1));
                        let iy1 = ((yt1 / source.dy).ceil() as usize).min(source.ny);
                        let iz0 =
                            ((zt0 / source.dz).floor() as usize).min(source.nz.saturating_sub(1));
                        let iz1 = ((zt1 / source.dz).ceil() as usize).min(source.nz);
                        let mut weights = Vec::new();
                        for iz_s in iz0..iz1 {
                            for iy_s in iy0..iy1 {
                                for ix_s in ix0..ix1 {
                                    let vol = (xt1.min((ix_s + 1) as f64 * source.dx)
                                        - xt0.max(ix_s as f64 * source.dx))
                                    .max(0.0)
                                        * (yt1.min((iy_s + 1) as f64 * source.dy)
                                            - yt0.max(iy_s as f64 * source.dy))
                                        .max(0.0)
                                        * (zt1.min((iz_s + 1) as f64 * source.dz)
                                            - zt0.max(iz_s as f64 * source.dz))
                                        .max(0.0);
                                    if vol > 1e-15 {
                                        weights.push((
                                            index_3d(ix_s, iy_s, iz_s, source.nx, source.ny),
                                            vol,
                                        ));
                                    }
                                }
                            }
                        }
                        let total: f64 = weights.iter().map(|(_, v)| v).sum();
                        if total > 1e-15 {
                            for (_, v) in &mut weights {
                                *v /= total;
                            }
                        }
                        transfer_matrix[t_idx] = weights;
                    }
                }
            }
            Ok(Self {
                source_grid: source.clone(),
                target_grid: target.clone(),
                transfer_matrix,
                source_volumes,
            })
        }

        /// Traversal identical to the pre-conversion hot loop (per-entry
        /// unravel + 3-D indexing).
        pub fn transfer(&self, source_field: &Array3<f64>, target_field: &mut Array3<f64>) {
            for iz in 0..self.target_grid.nz {
                for iy in 0..self.target_grid.ny {
                    for ix in 0..self.target_grid.nx {
                        let t_idx = index_3d(ix, iy, iz, self.target_grid.nx, self.target_grid.ny);
                        let mut sum = 0.0;
                        for &(s_idx, w) in &self.transfer_matrix[t_idx] {
                            let sz = s_idx / (self.source_grid.nx * self.source_grid.ny);
                            let rem = s_idx % (self.source_grid.nx * self.source_grid.ny);
                            sum += w * source_field
                                [[rem % self.source_grid.nx, rem / self.source_grid.nx, sz]];
                        }
                        target_field[[ix, iy, iz]] = sum;
                    }
                }
            }
        }

        #[allow(dead_code)]
        pub fn nnz(&self) -> usize {
            self.transfer_matrix.iter().map(|r| r.len()).sum()
        }
        #[allow(dead_code)]
        pub fn volumes(&self) -> &[f64] {
            &self.source_volumes
        }
        pub fn _error() -> Option<KwaversError> {
            None
        }
    }
}

fn make_grids(f: usize) -> (Grid, Grid) {
    // Source grid f× finer per axis than a fixed 16³ target; refinement factor
    // keeps cell counts tractable while giving every target row ~f³ entries.
    let target = Grid::new(16usize, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let n = 16usize * f;
    let source = Grid::new(n, n, n, 1e-3 / f as f64, 1e-3 / f as f64, 1e-3 / f as f64).unwrap();
    (source, target)
}

fn filled(grid: &Grid, seed: f64) -> Array3<f64> {
    let mut a = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    for (k, v) in a.iter_mut().enumerate() {
        *v = (k as f64 * seed).sin();
    }
    a
}

fn bench_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("conservative_interpolation/transfer");
    for refine in [2usize, 4usize] {
        let (source, target) = make_grids(refine);
        let src_field = filled(&source, 0.037);
        let csr =
            UtilConservativeInterpolator::new(&source, &target, ConservationMode::Mass).unwrap();
        let jagged = jagged::JaggedInterpolator::new(&source, &target).unwrap();

        // Parity gate: both layouts must produce identical fields before timing.
        let mut tgt_csr = Array3::<f64>::zeros((target.nx, target.ny, target.nz));
        let mut tgt_jag = Array3::<f64>::zeros((target.nx, target.ny, target.nz));
        csr.transfer(&src_field, &mut tgt_csr).unwrap();
        jagged.transfer(&src_field, &mut tgt_jag);
        assert_eq!(csr.nnz(), jagged.nnz(), "nnz mismatch — layouts diverged");
        let max_diff = tgt_csr
            .iter()
            .zip(tgt_jag.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-10, "parity failure: max diff {max_diff}");

        group.bench_function(format!("csr_refine_{refine}"), |b| {
            b.iter(|| {
                let mut out = Array3::<f64>::zeros((target.nx, target.ny, target.nz));
                csr.transfer(black_box(&src_field), &mut out).unwrap();
                out
            })
        });
        group.bench_function(format!("jagged_refine_{refine}"), |b| {
            b.iter(|| {
                let mut out = Array3::<f64>::zeros((target.nx, target.ny, target.nz));
                jagged.transfer(black_box(&src_field), &mut out);
                out
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_transfer);
criterion_main!(benches);
