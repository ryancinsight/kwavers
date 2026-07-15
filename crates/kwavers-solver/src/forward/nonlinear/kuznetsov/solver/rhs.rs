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
//! `moirai_parallel` chunked dispatch.  All inputs are `Array3<f64>` borrows
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
use moirai_parallel::{enumerate_mut_with, Adaptive};
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
            // All captured arrays are Array3<f64> (&-borrows → Sync). The only
            // mutable write is to rhs. Each task writes to a distinct (i,j,k)
            // element; all reads are shared → race-free.
            //
            // SLICE 6b: MIGRATED the deferred heterogeneous Zip::indexed site
            // to verbose is_standard_layout() layout precondition + flat-slice
            // reads via `<op>_slice[idx]`. The Zip chain is dropped entirely:
            // with 1 mut + N immut and no view-based boundary checks, the
            // canonical flat-slice pattern from slices 1-6 (r_slice.par_mut
            // ().enumerate(|idx, r: &mut f64| { ... immut_slice[idx] ... }))
            // is the most direct replacement. ndarray's 6-producer Zip-limit
            // that previously drove the heterogeneous-by-(i,j,k) workaround
            // is now bypassed at the source.
            //
            // IDX-TO-(i,j,k) STRIDE-ARITHMETIC DECOMPOSITION (for source
            // symmetry with sibling sites): for shape (nx, ny, nz) C-
            // contiguous, idx = i*(ny*nz) + j*nz + k. The inverse mapping is
            //   i = idx / (ny * nz)
            //   j = (idx / nz) % ny
            //   k = idx % nz
            // The current closure body reads via flat slice[idx] and does
            // not consume (i, j, k); the formula is documented above for
            // future maintainers who may switch to view-based patterns.
            //
            // WHY NOT HELPER: kwavers_safety::with_zip_standard_layout is the
            // canonical SSOT for future Batch #2 work, but was not adopted in
            // slice 6b because: (a) the verbose-form assert pattern is the
            // established Batch #1 SSOT across slices 1-7 (helper adoption
            // in 0 of 8 migrated sites so far); (b) helper ergonomics need
            // validation for sites with N>5 closure-captured immuts before
            // broader adoption. Slice 6b deliberately matches the verbose-
            // form pattern for source-level consistency with slices 1-7 and
            // to defer the helper-validation work to Batch #2.
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

            let r_slice = rhs
                .as_slice_mut()
                .expect("rhs: standard-layout asserted just above; layout matched");
            let p_slice = pressure
                .as_slice()
                .expect("pressure: standard-layout asserted just above; layout matched");
            let cd_slice = cache_density
                .as_slice()
                .expect("cache_density: standard-layout asserted just above; layout matched");
            let c0_slice = cache_c0
                .as_slice()
                .expect("cache_c0: standard-layout asserted just above; layout matched");
            let nl_slice = cache_nl
                .as_slice()
                .expect("cache_nl: standard-layout asserted just above; layout matched");
            let src_slice = cache_src
                .as_slice()
                .expect("cache_src: standard-layout asserted just above; layout matched");
            let lap_slice = laplacian
                .as_slice()
                .expect("laplacian: standard-layout asserted just above; layout matched");
            let pp_slice = pressure_prev
                .as_slice()
                .expect("pressure_prev: standard-layout asserted just above; layout matched");
            let pp2_slice = pressure_prev2
                .as_slice()
                .expect("pressure_prev2: standard-layout asserted just above; layout matched");
            let pp3_slice = pressure_prev3
                .as_slice()
                .expect("pressure_prev3: standard-layout asserted just above; layout matched");

            enumerate_mut_with::<Adaptive, _, _>(r_slice, |idx, r: &mut f64| {
                // idx-to-(i,j,k) stride-arithmetic decomposition is documented
                // at the site-level comment above; the closure body reads via
                // flat slice[idx] (no view-based boundary checks needed).
                let rho = cd_slice[idx];
                let c0 = c0_slice[idx];
                let p = p_slice[idx];
                let lap = lap_slice[idx];
                let p_prev = pp_slice[idx];

                *r = c0 * c0 * lap;

                if include_nonlinearity {
                    let b_over_a = nl_slice[idx];
                    let beta = NONLINEARITY_COEFFICIENT_OFFSET + b_over_a / B_OVER_A_DIVISOR;
                    // Explicit-form: +(β/ρc₀²)∂²(p²)/∂t²  [positive; c² not c⁴]
                    let coeff = beta / (rho * c0.powi(2));
                    let p2 = p * p;
                    let p2_prev = p_prev * p_prev;
                    let p_prev2_v = pp2_slice[idx];
                    let p2_prev2 = p_prev2_v * p_prev2_v;
                    let d2p2_dt2 = (2.0f64.mul_add(-p2_prev, p2) + p2_prev2) / (dt * dt);
                    *r += coeff * d2p2_dt2; // positive: derived from rearranging operator form
                }

                if include_diffusion {
                    let p_prev2_v = pp2_slice[idx];
                    let p_prev3_v = pp3_slice[idx];
                    // ∂³p/∂t³ ≈ (p[n] − 3p[n−1] + 3p[n−2] − p[n−3]) / Δt³
                    let d3p_dt3 = (3.0f64.mul_add(p_prev2_v, 3.0f64.mul_add(-p_prev, p))
                        - p_prev3_v)
                        / dt.powi(3);
                    // Explicit-form: +(δ/c₀²)∂³p/∂t³  [positive; c² not c⁴]
                    *r += (diffusivity / c0.powi(2)) * d3p_dt3;
                }

                *r += src_slice[idx];
            });
        } else {
            // Homogeneous: uniform properties — fully parallel from the start.
            let c0_squared = uniform_sound_speed * uniform_sound_speed;
            {
                let r_slice = rhs
                    .as_slice_mut()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                let lap_slice = self
                    .workspace
                    .laplacian
                    .as_slice()
                    .expect("laplacian: standard-layout asserted just above; layout matched");
                enumerate_mut_with::<Adaptive, _, _>(r_slice, |idx, r: &mut f64| {
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
            {
                let r_slice = rhs
                    .as_slice_mut()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                let src_slice =
                    self.workspace.cache_source.as_slice().expect(
                        "cache_source: standard-layout asserted just above; layout matched",
                    );
                enumerate_mut_with::<Adaptive, _, _>(r_slice, |idx, r: &mut f64| {
                    let s = src_slice[idx];
                    *r += s;
                });
            }

            if include_nonlinearity {
                {
                    let r_slice = rhs
                        .as_slice_mut()
                        .expect("rhs: standard-layout asserted just above; layout matched");
                    let nl_slice = self.workspace.nonlinear_term.as_slice().expect(
                        "nonlinear_term: standard-layout asserted just above; layout matched",
                    );
                    enumerate_mut_with::<Adaptive, _, _>(r_slice, |idx, r: &mut f64| {
                        let nl = nl_slice[idx];
                        *r += nl;
                    });
                }
            }

            if include_diffusion {
                {
                    let r_slice = rhs
                        .as_slice_mut()
                        .expect("rhs: standard-layout asserted just above; layout matched");
                    let diff_slice = self.workspace.diffusive_term.as_slice().expect(
                        "diffusive_term: standard-layout asserted just above; layout matched",
                    );
                    enumerate_mut_with::<Adaptive, _, _>(r_slice, |idx, r: &mut f64| {
                        let diff = diff_slice[idx];
                        *r += diff;
                    });
                }
            }
        }
    }
}
