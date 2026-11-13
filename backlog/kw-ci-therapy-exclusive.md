<a id="kw-ci-therapy-exclusive"></a>

## KW-CI-THERAPY-EXCLUSIVE — Four kwavers-therapy tests still run alone [patch] — todo

priority: verification; needs: none; basis: 4a5c8db8889e807011518f4aaf8013f18729e698; scope: `.config/nextest.toml`, `crates/kwavers-therapy/src/theranostic_guidance/tests/`

- **Finding.** At 4a5c8db88, `.config/nextest.toml:84-88` puts four kwavers-therapy tests (`abdominal_preprocessing_selects_one_connected_treatment_component`, `abdominal_theranostic_inverse_recovers_lesion_support`, `abdominal_preprocessing_keeps_external_skin_between_target_and_aperture`, `brain_focused_bowl_layout_uses_requested_element_count`) in `full-grid-sim` with `threads-required = "num-test-threads"`, and `:138-141` does the same in the `ci` profile. The 2026-07-22 review record said KW-CI-068 deleted this serialization (a containment for Moirai bounded-admission failures); the override is still there. Exclusivity is for an external shared resource, and these tests use none that is recorded.
- **Acceptance:** the four tests run under ordinary scheduling in every profile, measured as kwavers#1066 measured the solver tests (a full `-p kwavers-therapy` run with and without the override, per-test times recorded), with no failure, timeout or test over its bound; or a named external resource is recorded beside the override.
- **Next step:** run that measurement and remove the override if it holds.
