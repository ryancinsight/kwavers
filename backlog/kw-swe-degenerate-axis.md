<a id="kw-swe-degenerate-axis"></a>

## KW-SWE-DEGENERATE-AXIS — Wall clamp annihilates thin-slab and plane-strain domains [patch] [fix] — todo

priority: correctness; needs: none; scope: `crates/kwavers-solver/src/forward/elastic/swe/integration/integrator/clamp.rs`, `crates/kwavers-solver/src/forward/elastic/swe/integration/tests.rs`

- **Finding.** The KW-SWE-EDGE-GROWTH clamp zeroes all six wall planes unconditionally. On a degenerate axis (`nz = 1`, the 2D FWI grids and the plane-strain kernel test; likewise any axis with fewer than 3 cells) the wall pair covers the whole domain, so velocity is zeroed everywhere each step and the field freezes at its initial value: 4 elastic-FWI inverse lib failures on main (`plane_strain_kernel_matches_full_history_oracle`, `recovers_stiff_inclusion`, `forward_misfit_zero_at_true_model_positive_off_it`, `k_mu_gradient_is_valid_descent_direction`), all red since #1023. Thick-grid behavior is unaffected.
- **Acceptance:** an axis pair is clamped only with interior cells (`n >= 3`); a thin-slab regression probe evolves (not frozen) and stays bounded; the 4 FWI tests plus the SWE suites pass; pre-push hook green.
- **Next step:** guard `zero_wall_planes` per axis, add the slab probe, verify the FWI suite with `clinical-imaging`, push through the hook.
