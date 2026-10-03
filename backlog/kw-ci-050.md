<a id="kw-ci-050"></a>

## KW-CI-050 — Restore hosted format and CUDA prerequisites [patch] — review

priority: verification; needs: none; scope: the finite-window PSTD test formatting; CUDA container prerequisites in the CI workflow

- **Acceptance:** repository rustfmt passes for the corrected test and the CUDA build image provides the OpenSSL development metadata `openssl-sys` requires; no test workload or assertion changes.
- **Re-validation finding (2026-10-03):** the scoped `architecture-validation.yml` no longer exists (folded into `ci.yml` by 1279abc1d61); `ci.yml`'s `cuda-runtime-build` job installs `libssl-dev` before the CUDA compile step.
- **Next step:** confirm the finite-window PSTD test is rustfmt-clean at the default branch, then close.
