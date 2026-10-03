<a id="kw-examples-oversized-923"></a>

## KW-EXAMPLES-OVERSIZED-923 — Examples and tests grown or touched by #923 stay over the 500-line target [patch] — todo

priority: tightening; needs: none; scope: the nine examples and five tests listed below under `crates/kwavers/` and `crates/kwavers-physics/`

- **Finding.** PR #923 routed example output through `std::io` and carried files past the 500-line target; line counts at origin/main below (net growth by #923 in parentheses). The ten examples that grew past 500 in the same PR were split by 72bdaf94482 and are excluded here; the counts below were re-read at a23f2f86cb0 and are unchanged.
- **Examples (9):** `liver_theranostic_reconstruction` 1648 (+131), `skull_ct_phase_correction` 1517 (+3), `literature_validation_safe` 1172 (+202), `safe_vectorization_benchmarks` 846 (+65), `brain_theranostic_monitor` 675 (+32), `swe_3d_liver_fibrosis` 673 (+146), `transcranial_ct_mri_reconstruction` 593 (+30), `pstd_fdtd_comparison` 552 (+30), `dg_common/dg_acoustic_common` 530 (+27); all under `crates/kwavers/examples/`.
- **Tests touched by #923, over 500 (5; no net growth, the two the PR is blamed for could not be singled out by line delta):** `crates/kwavers/tests/imaging_literature_validation.rs` 1604, `kwave_reference_parity.rs` 1526, `nl_swe_convergence_tests.rs` 1166, `comparative_solver_tests.rs` 853, and `crates/kwavers-physics/src/analytical/cavitation/passive_dose/tests.rs` 1180.
- **Outcome:** each file splits by concern into leaf modules of at most 500 lines (examples under `examples/<name>/` declared from the example root with `#[path]`, the convention `comprehensive_clinical_workflow` and `seismic_imaging` use; integration tests into one harness module tree), with no behavior or assertion change.
- **Acceptance:** the `oversized_files` baseline falls by the number of files split and is not raised; `cargo build --examples` and the affected Nextest packages pass; git moved-line detection shows the diff is a relocation.
