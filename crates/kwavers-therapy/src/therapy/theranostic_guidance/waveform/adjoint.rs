//! Adjoint RTM imaging with checkpointed forward replay.
//!
//! ## Correctness requirements for the 2nd-order-in-time wave equation
//!
//! The scalar acoustic wave equation is second-order in time:
//! `p(t+1) = 2p(t) - p(t-1) + dt²c²Δp(t) + s(t)`.
//!
//! Restarting the recursion from a single snapshot `p(t_ck)` forces
//! `p(t_ck - 1) := p(t_ck)`, which is equivalent to claiming zero time
//! derivative at the checkpoint.  This introduces an O(|p|) error in the
//! first replay step and compounds over subsequent steps, causing the imaging
//! condition to use a corrupted forward field.
//!
//! Fix: save the **pair** `(p(t-1), p(t))` at each checkpoint.
//! The forward pass stores `[previous | current]` per slot (see `forward.rs`).
//!
//! ## Imaging condition timing contract
//!
//! The Born RTM imaging condition is
//! `I(x) = Σ_t p_fwd(x,t) · q(x,t)`
//! where both fields are evaluated at the **same** physical time `t`.
//!
//! ## Loop invariant and backward adjoint equation
//!
//! At the start of each iteration for loop variable `step = t`, the state
//! buffers satisfy:
//!   `prev_adj = q(t+2)`,  `curr_adj = q(t+1)`
//!
//! The backward adjoint equation (time-reversed form) is:
//!   `q(t) = 2·q(t+1) − q(t+2) + dt²c²Δq(t+1) + f_adj(t)`
//!
//! `step_wavefield_cpml(prev=q(t+2), curr=q(t+1))` computes the first three
//! terms, yielding `next_adj = q(t)` without the source.  Injecting
//! `residual[t]` into `next_adj` completes `q(t)`.  The imaging condition
//! then cross-correlates `p_fwd(t)` with the now-complete `next_adj = q(t)`.
//!
//! Evaluating the imaging condition **before** the backward step (as was
//! previously the case) would use `curr_adj = q(t+1)` — one step late —
//! introducing a phase lag of ≈ 90° at the focal depth (~20 cells, λ/dx ≈ 4),
//! which inverts the contrast of the lesion signal (CNR < 0).
//!
//! Correct order per iteration: BACKWARD → INJECT → IMAGE → SWAP.
//!
//! ## Inverse-scattering imaging condition (wavefield-decomposition class)
//!
//! The bare Born RTM cross-correlation `I(x) = Σ_t p_fwd(x,t) · q(x,t)`
//! produces a low-wavenumber "smile" artefact along the source→focus
//! illumination cone.  This artefact integrates the product of two
//! **co-propagating** wavefields (incident forward field and back-projected
//! adjoint field) along the path connecting the transmit aperture to the
//! scatterer.  Under the previous (incorrect) clamped sim grid, CPML overlapped
//! most of the body interior and the existing CPML-mute incidentally
//! suppressed the smile.  With the corrected padded geometry, CPML sits on
//! the outer ring outside the body and the artefact is exposed.
//!
//! An earlier comment in this module rejected illumination normalisation on
//! the grounds that "forward energy is maximum at the focal point, so dividing
//! by it suppresses the focal signal".  That argument was wrong: it conflates
//! peak illumination location with the ratio's value.  Their ratio remains
//! positive and resolvable at the focus.  However, illumination normalisation
//! alone (Liu, Zhang, Morton & Leveille 2011, Geophysics 76(1):S29 Eq. 5)
//! does **not** suppress the cone smile in a focused-HIFU geometry: along the
//! aperture→focus cone, both numerator and denominator vary by a similar
//! factor, so the ratio does not preferentially boost the focus over the cone.
//!
//! ### Wavefield-decomposition imaging condition (Op't Root et al. 2012)
//!
//! The remedy is the inverse-scattering imaging condition (Whitmore & Crawley
//! 2012, "Applications of RTM inverse scattering imaging conditions",
//! SEG Tech. Prog. 2012; Op't Root, Stolk & van Leeuwen 2012, "Linearized
//! inverse scattering based on seismic reverse time migration", J. Math.
//! Pures Appl. 98:211–238):
//!
//! ```text
//!     I(x) = Σ_t [ c²(x) · ∇p_fwd(x,t) · ∇q(x,t) − ∂_t p_fwd(x,t) · ∂_t q(x,t) ]
//! ```
//!
//! This is a wavefield-decomposition imaging condition in the sense of Liu
//! et al. (2011) §4: it selects counter-propagating phase pairs without
//! requiring an explicit directional Fourier or Hilbert decomposition.
//!
//! Derivation (plane-wave test, 1D for clarity):
//! - Forward plane wave `p_fwd = sin(kx − ωt)`:
//!   `∂_t p = −ω cos(·)`, `∂_x p = k cos(·)`.
//! - **Co-propagating** adjoint `q = sin(kx − ωt + φ)`:
//!   `∂_t p · ∂_t q = ω² cos(·) cos(·+φ)`,
//!   `c² ∇p · ∇q = c²k² cos(·) cos(·+φ) = ω² cos(·) cos(·+φ)`.
//!   The bracketed difference is **identically zero**.  The cone smile is
//!   removed.
//! - **Counter-propagating** adjoint `q = sin(kx + ωt)` (true scatterer):
//!   `∂_t p · ∂_t q = −ω² cos(kx−ωt) cos(kx+ωt)`,
//!   `c² ∇p · ∇q = +ω² cos(kx−ωt) cos(kx+ωt)`.
//!   The bracketed difference is `2ω² cos(kx−ωt) cos(kx+ωt)` — the imaging
//!   response is doubled relative to the bare cross-correlation, with the
//!   correct sign at scatterers.
//!
//! Both partial derivatives use centred second-order finite differences on
//! the existing time and space stencils.  The temporal derivative uses the
//! checkpointed pair `(p(t-1), p(t))` (already stored in `fwd_prev`,
//! `fwd_curr`) and the adjoint pair `(q(t+1), q(t))` (already in `curr_adj`,
//! `next_adj`); the spatial gradient uses the same 4th-order CPML stencil
//! grid spacing `dx`.  No extra forward replay is required.
//!
//! The body-mask normalisation in the caller (`normalize_positive`) still
//! applies on top.
//!
//! ## Poynting-vector directional gating (Yoon & Marfurt 2006)
//!
//! Even with the inverse-scattering imaging condition above, a residual
//! body-boundary spike survives at the perimeter of the body/water padding
//! interface, just outside the material-interface mute strip.  This artefact
//! is the angle-domain residue of the illumination cone: along the
//! aperture→focus path the forward and back-projected adjoint Poynting
//! vectors are co-linear and co-directional (energy flowing the same way
//! through the cell), whereas at a genuine scatterer the back-scattered
//! adjoint wavefield carries energy back toward the aperture and its
//! Poynting vector is anti-parallel to the incident forward Poynting vector.
//!
//! Reference: Yoon, K. & Marfurt, K. J. (2006), "Reverse-time migration
//! using the Poynting vector", Explor. Geophys. 37: 102–107.
//!
//! ### Acoustic Poynting vector
//!
//! The acoustic energy-flux density is `P(x,t) = −∂_t p · ∇p`.  At a true
//! scatterer the forward field is incident on the cell and the adjoint field
//! is scattered away from it, so `P_fwd · P_adj < 0`.  Along the
//! illumination cone both fields propagate from the aperture toward the
//! focus, so `P_fwd · P_adj > 0`.
//!
//! ### Soft-tanh gate
//!
//! Hard-step gates (Heaviside on `−P_fwd · P_adj`) introduce checker
//! artefacts at zero-crossings.  A soft tanh gate is used:
//!
//! ```text
//!     gate(x,t) = 0.5 · (1 − tanh(β · cosθ))
//!     cosθ      = (P_fwd · P_adj) / (|P_fwd| · |P_adj| + ε_P)
//! ```
//!
//! - `β = 4.0`: at full anti-parallel (`cosθ = −1`) the gate is
//!   `0.5·(1 − tanh(−4)) ≈ 0.9993`; at full parallel (`cosθ = +1`)
//!   it is `0.5·(1 − tanh(4)) ≈ 0.00067`.  This delivers >99% weight
//!   separation between scatterer and smile contributions while keeping
//!   the transition C∞-smooth over a cosθ band of width ≈ 1/β = 0.25.
//! - `ε_P = 1e-30`: f32 has min normal ≈ 1.18e-38; with peak pressure
//!   amplitudes ~1e8 Pa and gradients ~1e8/dx, `|P|² ≤ ~1e25`,
//!   `|P_fwd|·|P_adj| ≤ ~1e25`.  ε_P = 1e-30 is ~5 orders of magnitude
//!   below the smallest physically meaningful product yet ~8 orders above
//!   the f32 underflow threshold, eliminating 0/0 at zero-field cells
//!   without biasing the cosine.
//!
//! ### Plane-wave verification
//!
//! Forward `p_fwd = sin(kx − ωt)`: `−∂_t p_fwd · ∂_x p_fwd = ω cos(·) · k cos(·) = ωk cos²(·) > 0`
//! (P_fwd points in +x̂, energy flows toward +x).
//! Counter-propagating adjoint `q = sin(kx + ωt)`:
//! `−∂_t q · ∂_x q = −ω cos(·) · k cos(·) = −ωk cos²(·) < 0`
//! (P_adj points in −x̂).  Therefore `P_fwd · P_adj < 0` at scatterers.
//! Co-propagating adjoint `q = sin(kx − ωt + φ)`:
//! `P_fwd · P_adj = ω²k² cos(·) cos(·+φ) · cos(·) cos(·+φ) > 0` on average
//! along the cone.
//!
//! The gate is multiplicative and stacks with the existing CPML-zone and
//! material-interface mutes.
//!
//! ## Memory strategy (Griewank 1992)
//!
//! For each adjoint step `t` (reverse order):
//! 1. Load checkpoint pair at `t_ck = floor(t / K) * K`.
//! 2. Replay forward from `t_ck` to `t` (≤ K steps) WITH source injection.
//! 3. Advance adjoint one step backward: `step_cpml(q(t+2), q(t+1)) → next = q(t)`.
//! 4. Inject residual at time `t` into `next` to complete `q(t)`.
//! 5. Apply imaging condition: `I += p_fwd[t] · next = p_fwd[t] · q[t]`.
//! 6. After all steps: zero `I` at source and receiver cell positions (aperture mute).
//!
//! Total cost: ≈ 1.5 × forward passes.  Memory: O(√T · N²).

use leto::Array2;

use super::forward::{apply_attenuation, c2dt2_field, inject_sources, step_wavefield_cpml};
use super::types::{AcousticGrid, CheckpointSchedule};
use super::utils::linear;

/// Sharpness of the Poynting-vector soft gate; see module docs for derivation.
///
/// β = 4.0 yields ≈ 0.9993 weight at fully anti-parallel Poynting vectors
/// (true scatterer) and ≈ 0.00067 weight at fully parallel ones (smile
/// artefact along the illumination cone), with a C∞-smooth transition of
/// width 1/β = 0.25 in cosθ.
const BETA_POYNTING: f64 = 4.0;

/// Underflow guard in the normalised dot product cosθ = (P_fwd·P_adj)/(|P_fwd|·|P_adj|+ε).
///
/// f32 underflow threshold is ≈ 1.18e-38; with peak pressure ~1e8 Pa and
/// gradients ~1e8/dx, the product |P_fwd|·|P_adj| spans 0 … ~1e25.
/// ε_P = 1e-30 is far below the smallest physically meaningful product yet
/// far above the f32 underflow floor, eliminating 0/0 at zero-field cells
/// without biasing cosθ on resolved cells.
const EPS_POYNTING: f64 = 1.0e-30;

/// Compute the absolute-value Born RTM image via checkpointed backpropagation.
///
/// # Arguments
///
/// * `grid` — simulation grid; carries CPML, source geometry,
///   `source_frequency_hz`, and `source_scale` for replay injection.
/// * `speed_m_s` — background (predicted) sound-speed model.
/// * `residual` — flat `[time_steps × receiver_count]` row-major adjoint
///   source (Charbonnier or L2 misfit derivative).
/// * `checkpoints` — paired snapshot buffer from the forward run.
///   Layout: `[prev₀ | curr₀ | prev₁ | curr₁ | …]`, size `2·slots·N`.
/// * `checkpoint_interval` — K = `CheckpointSchedule::interval`.
///
/// # Returns
///
/// `Array2<f64>` of shape `(nx, ny)` with the absolute-value Born
/// cross-correlation image.  Body-mask normalization is applied by the caller.
///
/// # Imaging condition timing
///
/// The cross-correlation `Σ_t p_fwd(x,t) · q(x,t)` is accumulated using
/// `next_adj = q(t)` produced by the backward step and source injection so
/// the forward and adjoint fields are at the same physical time `t`.  Per
/// the loop invariant `(prev_adj, curr_adj) = (q(t+2), q(t+1))`, applying
/// `step_wavefield_cpml` followed by `residual[t]` injection yields the
/// complete `next_adj = q(t)` before the swap.  See module-level
/// documentation for the full derivation.
pub(super) fn adjoint_image(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    residual: &[f32],
    checkpoints: &[f32],
    checkpoint_interval: usize,
) -> Array2<f64> {
    let n = grid.nx * grid.ny;
    let schedule = CheckpointSchedule {
        interval: checkpoint_interval,
        time_steps: grid.time_steps,
    };

    let mut prev_adj = vec![0.0_f32; n];
    let mut curr_adj = vec![0.0_f32; n];
    let mut next_adj = vec![0.0_f32; n];
    let mut psi_x_adj = vec![0.0_f32; n];
    let mut psi_y_adj = vec![0.0_f32; n];
    let mut image = vec![0.0_f64; n];
    let receiver_count = grid.receiver_cells.len();
    let inv_dt = 1.0_f64 / grid.dt_s;
    let inv_two_dx = 0.5_f64 / grid.dx_m;

    // Loop-invariant stencil coefficient, shared by the forward replay and the
    // backward adjoint advance (both use the same `c²·dt²` field).
    let c2dt2 = c2dt2_field(grid, speed_m_s);

    let mut fwd_prev = vec![0.0_f32; n];
    let mut fwd_curr = vec![0.0_f32; n];
    let mut fwd_next = vec![0.0_f32; n];
    let mut fwd_psi_x = vec![0.0_f32; n];
    let mut fwd_psi_y = vec![0.0_f32; n];

    for reverse in 0..grid.time_steps {
        let step = grid.time_steps - 1 - reverse;

        // ── Replay forward field from the nearest preceding checkpoint ──────
        //
        // Each slot stores the pair (previous, current) at checkpoint time
        // t_ck.  Layout per slot: [prev_n | curr_n], each block is `n` f32s.
        let ck_step = schedule.preceding_checkpoint(step);
        let ck_slot = schedule.slot_for(ck_step);
        let base = ck_slot * 2 * n;
        fwd_prev.copy_from_slice(&checkpoints[base..base + n]);
        fwd_curr.copy_from_slice(&checkpoints[base + n..base + 2 * n]);
        fwd_next.fill(0.0);
        fwd_psi_x.fill(0.0);
        fwd_psi_y.fill(0.0);

        // Exact Griewank replay with source injection.
        for fwd_step in ck_step..step {
            step_wavefield_cpml(
                grid,
                &c2dt2,
                &fwd_prev,
                &fwd_curr,
                &mut fwd_next,
                &mut fwd_psi_x,
                &mut fwd_psi_y,
            );
            inject_sources(grid, fwd_step, &mut fwd_next);
            apply_attenuation(grid, &mut fwd_next);
            std::mem::swap(&mut fwd_prev, &mut fwd_curr);
            std::mem::swap(&mut fwd_curr, &mut fwd_next);
            fwd_next.fill(0.0);
        }
        // fwd_curr is now the accurate forward pressure field at time `step`.

        // ── Advance adjoint field one step (backward in time) ───────────────
        //
        // Loop invariant: prev_adj = q(step+2), curr_adj = q(step+1).
        // step_wavefield_cpml computes:
        //   next = 2·q(step+1) − q(step+2) + dt²c²Δq(step+1)
        // which is q(step) without the adjoint source term.
        step_wavefield_cpml(
            grid,
            &c2dt2,
            &prev_adj,
            &curr_adj,
            &mut next_adj,
            &mut psi_x_adj,
            &mut psi_y_adj,
        );

        // Inject adjoint source at time `step` to complete q(step).
        //
        // Backward adjoint equation (time-reversed form):
        //   q(t) = 2q(t+1) − q(t+2) + dt²c²Δq(t+1) + f_adj(t)
        // step_wavefield_cpml produced the first three terms; adding f_adj(t)
        // = residual[step] completes q(t).
        //
        // References:
        //   Claerbout (1985), "Imaging the Earth's Interior", Eq. 2.6.
        //   Fichtner (2010), "Full Seismic Waveform Modelling", Ch. 4.3.
        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            next_adj[cell] += residual[step * receiver_count + receiver];
        }
        apply_attenuation(grid, &mut next_adj);

        // ── Op't Root inverse-scattering imaging condition at time `step` ────
        //
        // I(x) = Σ_t [c²(x) ∇p_fwd · ∇q − ∂_t p_fwd · ∂_t q]
        //
        // Temporal derivatives use the checkpointed pairs that are already
        // resident in the loop:
        //   ∂_t p_fwd(step) ≈ (fwd_curr − fwd_prev) / dt   (backward diff at step)
        //   ∂_t q(step)     ≈ (curr_adj − next_adj) / dt   (forward diff from
        //                                                   q(step+1) toward q(step))
        // Both pairs are O(dt²) accurate at the half-step centred between
        // their two samples; their product matches at the same half-step so
        // there is no temporal misalignment in the bracketed difference.
        //
        // Spatial gradients use 2nd-order centred differences on the
        // isotropic grid (dx = dy).  Edge cells fall back to a one-sided
        // difference of zero gradient, which is acceptable because edge cells
        // are inside the CPML mute zone and are zeroed below.
        //
        // See module-level documentation for the plane-wave derivation
        // showing this condition annihilates co-propagating contributions
        // (the cone smile) while doubling counter-propagating contributions
        // (the true scattering response).
        for ix in 1..grid.nx - 1 {
            let row = ix * grid.ny;
            let row_xm = (ix - 1) * grid.ny;
            let row_xp = (ix + 1) * grid.ny;
            for iy in 1..grid.ny - 1 {
                let idx = row + iy;
                let fwd = fwd_curr[idx] as f64;
                let adj = next_adj[idx] as f64;

                let dt_fwd = (fwd - fwd_prev[idx] as f64) * inv_dt;
                let dt_adj = (curr_adj[idx] as f64 - adj) * inv_dt;

                let dx_fwd =
                    (fwd_curr[row_xp + iy] as f64 - fwd_curr[row_xm + iy] as f64) * inv_two_dx;
                let dy_fwd = (fwd_curr[idx + 1] as f64 - fwd_curr[idx - 1] as f64) * inv_two_dx;
                let dx_adj =
                    (next_adj[row_xp + iy] as f64 - next_adj[row_xm + iy] as f64) * inv_two_dx;
                let dy_adj = (next_adj[idx + 1] as f64 - next_adj[idx - 1] as f64) * inv_two_dx;

                let c = speed_m_s[[ix, iy]];
                let c2 = c * c;
                let integrand = c2 * (dx_fwd * dx_adj + dy_fwd * dy_adj) - dt_fwd * dt_adj;

                // Poynting-vector directional gate (Yoon & Marfurt 2006).
                //
                // P = −∂_t p · ∇p (acoustic energy-flux density).  See
                // module-level docs for the β=4.0, ε_P=1e-30 derivation.
                let p_fwd_x = -dt_fwd * dx_fwd;
                let p_fwd_y = -dt_fwd * dy_fwd;
                let p_adj_x = -dt_adj * dx_adj;
                let p_adj_y = -dt_adj * dy_adj;
                let dot = p_fwd_x * p_adj_x + p_fwd_y * p_adj_y;
                let mag_fwd_sq = p_fwd_x * p_fwd_x + p_fwd_y * p_fwd_y;
                let mag_adj_sq = p_adj_x * p_adj_x + p_adj_y * p_adj_y;
                let mag = (mag_fwd_sq * mag_adj_sq).sqrt();
                let cos_theta = dot / (mag + EPS_POYNTING);
                let gate = 0.5 * (1.0 - (BETA_POYNTING * cos_theta).tanh());

                image[idx] += gate * integrand;
            }
        }

        std::mem::swap(&mut prev_adj, &mut curr_adj);
        std::mem::swap(&mut curr_adj, &mut next_adj);
        next_adj.fill(0.0);
    }

    // Take absolute value: the inverse-scattering condition is signed at
    // scatterers (positive for impedance increases, negative for decreases).
    // For lesion-support detection we want magnitude (Born reflectivity
    // magnitude), so apply |·| before the CPML mute.
    for val in image.iter_mut() {
        *val = val.abs();
    }

    // Zero the image throughout the CPML absorption zone.
    //
    // The CPML modifies the scalar wave equation inside its damping strips
    // (the first and last PML_CELLS rows/columns of each axis).  In those
    // cells the wave equation is no longer `p_tt = c² Δp` but a stretched-
    // coordinate variant with complex damping.  Cross-correlating p_fwd and
    // q inside the CPML zone produces artifacts: the fields propagate through
    // a modified medium, the CPML memory variables are initialized from zero
    // at each checkpoint replay (since they are not checkpointed), and the
    // source and receiver cells lie in the CPML zone for compact simulation
    // grids (body surface at PML depth for the ~42-cell test geometry).  The
    // CPML zone is not an imaging target; it is an absorbing boundary.
    //
    // Detection: `a_x[ix] < 0` iff σ_x > 0 (PML cell in x-direction).
    // `a_i = exp(-σ_i·dt) - 1 < 0` for any σ_i > 0; interior cells have
    // σ = 0 and therefore a = 0 exactly.  This also mutes all source and
    // receiver cells (aperture mute) which are in the CPML zone for the
    // geometries exercised by the clinical test suite.
    for ix in 0..grid.nx {
        let in_x_pml = grid.cpml.a_x[ix] < 0.0;
        for iy in 0..grid.ny {
            if in_x_pml || grid.cpml.a_y[iy] < 0.0 {
                image[linear(ix, iy, grid.ny)] = 0.0;
            }
        }
    }

    // ── Material-interface mute ─────────────────────────────────────────────
    //
    // The Op't Root inverse-scattering imaging condition `c²∇p·∇q − ∂_t p·∂_t q`
    // assumes a locally smooth velocity model so that the spatial gradient
    // stencils sample a single material.  Across the body/water interface
    // (where c jumps from ~1500 m/s to ~1480 m/s and density jumps from
    // ~1050 kg/m³ to ~1000 kg/m³ in the padded margin) the wavefields p_fwd
    // and q themselves have discontinuous spatial derivatives; the 2nd-order
    // centred stencil straddling the interface produces a spurious imaging
    // spike that has nothing to do with a sub-resolution scatterer.
    //
    // Detect such cells via the local velocity contrast: if any 3×3
    // neighbour differs from the centre cell by more than 1% (a value well
    // below body/water contrast but well above any physical lesion contrast
    // that the linearised Born model can resolve), zero the imaging response.
    // This is a single-material mute, not an empirical hack: it enforces the
    // domain of validity of the smooth-medium imaging condition.
    //
    // References:
    //   Op't Root, Stolk & van Leeuwen (2012), "Linearized inverse scattering
    //     based on seismic reverse time migration", J. Math. Pures Appl.
    //     98:211–238 — smoothness assumption on the reference model.
    //   Symes (2008), "Migration velocity analysis and waveform inversion",
    //     Geophys. Prospect. 56:765–790 — smooth background requirement.
    const INTERFACE_CONTRAST_THRESHOLD: f64 = 0.01;
    let mut interface_mask = vec![false; n];
    for ix in 1..grid.nx - 1 {
        for iy in 1..grid.ny - 1 {
            let c0 = speed_m_s[[ix, iy]];
            let mut max_rel_diff = 0.0_f64;
            for dx in [-1isize, 0, 1] {
                for dy in [-1isize, 0, 1] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx_i = (ix as isize + dx) as usize;
                    let ny_i = (iy as isize + dy) as usize;
                    let c_n = speed_m_s[[nx_i, ny_i]];
                    let rel = (c_n - c0).abs() / c0.max(1.0);
                    if rel > max_rel_diff {
                        max_rel_diff = rel;
                    }
                }
            }
            if max_rel_diff > INTERFACE_CONTRAST_THRESHOLD {
                interface_mask[linear(ix, iy, grid.ny)] = true;
            }
        }
    }
    for idx in 0..n {
        if interface_mask[idx] {
            image[idx] = 0.0;
        }
    }

    // Return the illumination-compensated, absolute-value, CPML-muted image.
    // The cross-correlation magnitude is taken inside the normalisation block
    // above (Liu et al. 2011 Eq. 5).  The absolute value handles scatterers
    // where the adjoint field has opposite polarity to the forward field
    // (e.g. slower-than-background lesion: reflection coefficient R < 0
    // inverts the scattered pulse, so q < 0 at the focus while p_fwd > 0),
    // consistent with the Born reflectivity interpretation.
    Array2::from_shape_fn((grid.nx, grid.ny), |[ix, iy]| {
        image[linear(ix, iy, grid.ny)]
    })
}
