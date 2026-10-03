<a id="kwavers-coupling-contract-001"></a>

## KWAVERS-COUPLING-CONTRACT-001 — Medium-aware field-coupling inputs [minor] — todo

priority: feature; needs: none; scope: `crates/kwavers-solver/src/multiphysics/field_coupling/`, direct callers and tests

- **Outcome:** add a typed medium-property provider to `MultiphysicsFieldCoupler` for photoelastic, optical-absorption and frequency-dependent acoustic-absorption coefficients; retain scalar extraction only at the field-update boundary.
- **Context:** the current field-coupler API accepts only collocated field volumes, so its nominal water/tissue coefficients are real defaults rather than hidden input or a silent fallback. The specialized `AcousticOpticalSolver` already accepts a caller-supplied photoelastic coefficient.
- **Acceptance:** the coupler takes the provider, every in-workspace caller and test compiles against it, and value-semantic tests cover each coefficient.
