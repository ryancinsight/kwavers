<a id="kw-subspace-jacobi-eigen"></a>

## KW-SUBSPACE-JACOBI-EIGEN — MUSIC and ESMV beamformers still use the classical Jacobi Hermitian eigensolver [patch] [perf] — todo

priority: tightening; needs: none; scope: `crates/kwavers-analysis/src/signal_processing/beamforming/adaptive/subspace/{music,esmv}.rs`, then `localization/music/spectrum.rs` and `crates/kwavers-python/src/pam_bindings.rs` if the profile supports it

- **Finding (at origin/main a23f2f86cb0).** `music.rs:90`, `esmv.rs:111` and `esmv.rs:222` call `hermitian_eigen_jacobi`; kwavers#926 replaced it in the adaptive clutter filter with the Householder/QL `SymmetricEigenWorkspace`, where Jacobi measured 207 ms per 120x120 matrix. `localization/music/spectrum.rs:194` and `pam_bindings.rs:83` call it too.
- **Outcome:** a measurement decides. If the eigensolve dominates a MUSIC or ESMV pseudospectrum call at its operating covariance sizes, the callers adopt the Householder/QL Hermitian path (upstream in Leto if the complex Hermitian form is missing, never a local copy) and the classical solver loses those callers. If it does not dominate, the item closes with the profile recorded.
- **Next step:** a criterion bench of the MUSIC pseudospectrum and the ESMV weight computation at covariance sizes 8, 32 and 120, with a sampling profile of the eigensolve share.
- **Acceptance:** the bench and profile are attached to the PR; on adoption, eigenvalues and subspace projectors agree with the Jacobi path within a bound derived from machine epsilon and the matrix norm, and pseudospectrum peak positions are unchanged on the existing MUSIC and ESMV tests.
