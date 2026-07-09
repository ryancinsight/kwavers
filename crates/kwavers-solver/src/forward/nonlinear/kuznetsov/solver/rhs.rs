//! Right-hand Side Computation for the Kuznetsov Equation
//!
//! ## Theorem (heterogeneous RHS parallelisation strategy)
//!
//! The `Medium` and `Source` trait objects are `?Sync`; they cannot be shared
//! across Rayon threads without an explicit `Sync` bound.  The heterogeneous
//! path therefore separates computation into two phases:
//!
//! **Phase 1 — serial property sampling** (O(N) trait-object calls):
//! Evaluate ρ(x,y,z), c₀(x,y,z), B/A(x,y,z), and S(t,x,y,z) at every grid
//! point and store into the pre-allocated cache arrays in `KuznetsovWorkspace`.
//!
//! **Phase 2 — parallel RHS combination** (embarrassingly parallel):
//! Combine laplacian, pressure history, and the cached property arrays using
//! `Zip::indexed(...).par_for_each()`.  All inputs are `Array3<f64>` borrows
//! (`Sync` by construction), so Rayon parallelism is race-free.
//!
//! This pattern achieves full multi-core utilisation for the arithmetic-heavy
//! RHS while keeping trait dispatch in a single serial sweep.

use super::wave::KuznetsovWave;
use crate::forward::nonlinear::kuznetsov::config::AcousticEquationMode;
use crate::forward::nonlinear::kuznetsov::diffusion::compute_diffusive_term_workspace;
use crate::forward::nonlinear::kuznetsov::nonlinear::compute_nonlinear_term_workspace;
use kwavers_core::constants::numerical::{B_OVER_A_DIVISOR, NONLINEARITY_COEFFICIENT_OFFSET};
use kwavers_medium::Medium;
use kwavers_source::Source;
use moirai_parallel::ParallelSliceMut;
use ndarray::Zip;

impl KuznetsovWave {
    /// Compute the right-hand side of the Kuznetsov equation.
    ///
    /// For heterogeneous media, nonlinear and diffusive terms are computed using
    /// local material properties at each grid point. For homogeneous media,
    /// properties are computed once for efficiency.
    pub(super) fn compute_rhs(
        &mut self,
        source: &dyn Source,
        medium: &dyn Medium,
        t: f64,
        dt: f64,
    ) {
        let pressure = &self.pressure_current;
        let is_heterogeneous = !medium.is_homogeneous();

        let (uniform_density, uniform_sound_speed, _uniform_nonlinearity, uniform_diffusivity) =
            if !is_heterogeneous {
                let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
                let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
                let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
                (
                    kwavers_medium::density_at(medium, center_x, center_y, center_z, &self.grid),
                    kwavers_medium::sound_speed_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    kwavers_medium::nonlinearity_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    self.config.acoustic_diffusivity,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        // 1. Compute linear term: c₀²∇²p using spectral methods
        self.workspace.spectral_op.compute_laplacian_workspace(
            pressure,
            &mut self.workspace.laplacian,
            &self.grid,
        );

        let rhs = &mut self.workspace.k1;

        let include_nonlinearity = matches!(
            self.config.equation_mode,
            AcousticEquationMode::FullKuznetsov
                | AcousticEquationMode::KZK
                | AcousticEquationMode::Westervelt
        );

        if include_nonlinearity && !is_heterogeneous {
            compute_nonlinear_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                dt,
                uniform_density,
                uniform_sound_speed,
                self.config.nonlinearity_coefficient * self.nonlinearity_scaling,
                &mut self.workspace.nonlinear_term,
            );
        }

        let include_diffusion = matches!(
            self.config.equation_mode,
            AcousticEquationMode::FullKuznetsov | AcousticEquationMode::KZK
        ) && self.config.acoustic_diffusivity > 0.0;

        if include_diffusion && !is_heterogeneous {
            compute_diffusive_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                &self.workspace.pressure_prev3,
                dt,
                uniform_sound_speed,
                uniform_diffusivity,
                &mut self.workspace.diffusive_term,
            );
        }

        if is_heterogeneous {
            // ------------------------------------------------------------------
            // Phase 1: sample medium/source properties serially.
            //
            // Medium and Source trait objects are ?Sync; they cannot be shared
            // across Rayon threads.  Fill the pre-allocated cache arrays with a
            // single serial sweep so Phase 2 only touches plain Array3<f64>.
            // ------------------------------------------------------------------
            let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                        self.workspace.cache_density[[i, j, k]] =
                            kwavers_medium::density_at(medium, x, y, z, &self.grid);
                        self.workspace.cache_sound_speed[[i, j, k]] =
                            kwavers_medium::sound_speed_at(medium, x, y, z, &self.grid);
                        if include_nonlinearity {
                            self.workspace.cache_nonlinearity[[i, j, k]] =
                                kwavers_medium::nonlinearity_at(medium, x, y, z, &self.grid);
                        }
                        self.workspace.cache_source[[i, j, k]] =
                            source.get_source_term(t, x, y, z, &self.grid);
                    }
                }
            }

            // ------------------------------------------------------------------
            // Phase 2: compute RHS in parallel.
            //
            // All captured arrays are Array3<f64> (&-borrows → Sync).  The
            // only mutable write is to rhs via the Zip element reference `r`.
            // Each task writes to a distinct (i,j,k) element; all reads are
            // shared → race-free.
            //
            // We use Zip::indexed(rhs_mut).par_for_each and access the
            // remaining arrays directly by (i,j,k) inside the closure.  This
            // avoids the ndarray Zip 6-producer limit that would apply if we
            // chained every array with `.and()`.
            // ------------------------------------------------------------------
            let cache_density = &self.workspace.cache_density;
            let cache_c0 = &self.workspace.cache_sound_speed;
            let cache_nl = &self.workspace.cache_nonlinearity;
            let cache_src = &self.workspace.cache_source;
            let laplacian = &self.workspace.laplacian;
            let pressure_prev = &self.workspace.pressure_prev;
            let pressure_prev2 = &self.workspace.pressure_prev2;
            let pressure_prev3 = &self.workspace.pressure_prev3;
            let diffusivity = self.config.acoustic_diffusivity;

            Zip::indexed(rhs.view_mut()).par_for_each(|(i, j, k), r| {
                let rho = cache_density[[i, j, k]];
                let c0 = cache_c0[[i, j, k]];
                let p = pressure[[i, j, k]];
                let lap = laplacian[[i, j, k]];
                let p_prev = pressure_prev[[i, j, k]];

                *r = c0 * c0 * lap;

                if include_nonlinearity {
                    let b_over_a = cache_nl[[i, j, k]];
                    let beta = NONLINEARITY_COEFFICIENT_OFFSET + b_over_a / B_OVER_A_DIVISOR;
                    // Explicit-form: +(β/ρc₀²)∂²(p²)/∂t²  [positive; c² not c⁴]
                    let coeff = beta / (rho * c0.powi(2));
                    let p2 = p * p;
                    let p2_prev = p_prev * p_prev;
                    let p_prev2_v = pressure_prev2[[i, j, k]];
                    let p2_prev2 = p_prev2_v * p_prev2_v;
                    let d2p2_dt2 = (2.0f64.mul_add(-p2_prev, p2) + p2_prev2) / (dt * dt);
                    *r += coeff * d2p2_dt2; // positive: derived from rearranging operator form
                }

                if include_diffusion {
                    let p_prev2_v = pressure_prev2[[i, j, k]];
                    let p_prev3_v = pressure_prev3[[i, j, k]];
                    // ∂³p/∂t³ ≈ (p[n] − 3p[n−1] + 3p[n−2] − p[n−3]) / Δt³
                    let d3p_dt3 = (3.0f64.mul_add(p_prev2_v, 3.0f64.mul_add(-p_prev, p))
                        - p_prev3_v)
                        / dt.powi(3);
                    // Explicit-form: +(δ/c₀²)∂³p/∂t³  [positive; c² not c⁴]
                    *r += (diffusivity / c0.powi(2)) * d3p_dt3;
                }

                *r += cache_src[[i, j, k]];
            });
        } else {
            // Homogeneous: uniform properties — fully parallel from the start.
            let c0_squared = uniform_sound_speed * uniform_sound_speed;
            // Slice 6 site 2: linear term (homogeneous) -- 1 mut + 1 immut layout
            assert!(
                rhs.is_standard_layout(),
                "rhs must be C-contiguous (default Array3 layout) for the migration"
            );
            assert!(
                self.workspace.laplacian.is_standard_layout(),
                "laplacian must be C-contiguous (default Array3 layout) for the migration"
            );
            {
                let r_slice = rhs
                    .as_slice_mut()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                let lap_slice = self.workspace.laplacian
                    .as_slice()
                    .expect("laplacian: standard-layout asserted just above; layout matched");
                r_slice.par_mut().enumerate(|idx, r: &mut f64| {
                    let lap = lap_slice[idx];
                    *r = c0_squared * lap;
                });
            }

            // Source term: sample serially (Source is ?Sync), accumulate into cache.
            {
                let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                            self.workspace.cache_source[[i, j, k]] =
                                source.get_source_term(t, x, y, z, &self.grid);
                        }
                    }
                }
            }
            // Slice 6 site 3: source accumulation (homogeneous) -- 1 mut + 1 immut layout
            assert!(
                rhs.is_standard_layout(),
                "rhs must be C-contiguous (default Array3 layout) for the migration"
            );
            assert!(
                self.workspace.cache_source.is_standard_layout(),
                "cache_source must be C-contiguous (default Array3 layout) for the migration"
            );
            {
                let r_slice = rhs
                    .as_slice_mut()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                let src_slice = self.workspace.cache_source
                    .as_slice()
                    .expect("cache_source: standard-layout asserted just above; layout matched");
                r_slice.par_mut().enumerate(|idx, r: &mut f64| {
                    let s = src_slice[idx];
                    *r += s;
                });
            }

            if include_nonlinearity {
                // Slice 6 site 4: nonlinearity add (homogeneous) -- 1 mut + 1 immut
                assert!(
                    rhs.is_standard_layout(),
                    "rhs must be C-contiguous (default Array3 layout) for the migration"
                );
                assert!(
                    self.workspace.nonlinear_term.is_standard_layout(),
                    "nonlinear_term must be C-contiguous (default Array3 layout) for the migration"
                );
                {
                    let r_slice = rhs
                        .as_slice_mut()
                        .expect("rhs: standard-layout asserted just above; layout matched");
                    let nl_slice = self.workspace.nonlinear_term
                        .as_slice()
                        .expect("nonlinear_term: standard-layout asserted just above; layout matched");
                    r_slice.par_mut().enumerate(|idx, r: &mut f64| {
                        let nl = nl_slice[idx];
                        *r += nl;
                    });
                }
            }

            if include_diffusion {
                // Slice 6 site 5: diffusion add (homogeneous) -- 1 mut + 1 immut
                assert!(
                    rhs.is_standard_layout(),
                    "rhs must be C-contiguous (default Array3 layout) for the migration"
                );
                assert!(
                    self.workspace.diffusive_term.is_standard_layout(),
                    "diffusive_term must be C-contiguous (default Array3 layout) for the migration"
                );
                {
                    let r_slice = rhs
                        .as_slice_mut()
                        .expect("rhs: standard-layout asserted just above; layout matched");
                    let diff_slice = self.workspace.diffusive_term
                        .as_slice()
                        .expect("diffusive_term: standard-layout asserted just above; layout matched");
                    r_slice.par_mut().enumerate(|idx, r: &mut f64| {
                        let diff = diff_slice[idx];
                        *r += diff;
                    });
                }
            }
        }
    }
}
