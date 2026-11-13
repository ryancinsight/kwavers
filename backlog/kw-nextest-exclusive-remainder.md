<a id="kw-nextest-exclusive-remainder"></a>

## KW-NEXTEST-EXCLUSIVE-REMAINDER — Root-package binaries and `ci` overrides still hold nextest slots exclusively, and the 25 m bound is underived [patch] [ci] — todo

priority: tightening; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `.config/nextest.toml`

- **Source.** kwavers#1066 removed the kwavers-solver tests from `full-grid-sim` and slot reservation (`.config/nextest.toml:64-70`) and left the rest unmeasured. The four kwavers-therapy tests (`:84-88`) belong to `KW-CI-THERAPY-EXCLUSIVE`.
- **Default profile.** `:102-105` puts ten `kwavers` binaries (`cpml_absorption_quality` to `swe_3d_validation`) in `full-grid-sim` with `threads-required = "num-test-threads"`; no measurement of that serialization is recorded. `:17` keeps `global-timeout = "25m"`; its comment (`:11-14`) says the serial sum of the remaining group users is not re-measured.
- **`ci` profile.** `:138-141` and `:148-151` keep the four solver FWI tests (beside the four therapy tests) and `kwavers-solver` `/kuznetsov/` tests in the group; `:143-146` keep five root binaries. Hosted CI invokes the profile with `-p kwavers` (`ci.yml:949`; `scripts/integration_tests.py:67` (`-p kwavers --tests`) and `:76` (`--profile ci`), run at `ci.yml:439`), where the solver filters select nothing; the alias `test-timeout` (`.cargo/config.toml:26`) runs it over every member's lib tests, where they apply.
- **Acceptance:** for the root binaries (`-p kwavers --tests`) and for the solver FWI and kuznetsov tests (`cargo test-timeout`), a run with and without the group and slot reservation, per-test times recorded (the method of #1066), shows no failure, timeout or test over its bound and the override is removed, or the external resource is named beside it; `global-timeout` is re-derived from the measured serial sum of the remaining group members and its comment states the figure.
- **Next step:** run the root-binary measurement on the default profile, then settle the `ci` overrides and the global bound.
