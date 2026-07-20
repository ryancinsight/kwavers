# ADR 043: Asclepius response ownership

## Context

Kwavers contains repeated implementations of three biological-response laws:

- cumulative equivalent minutes at 43 degrees Celsius in thermal fields,
  safety monitors, HIFU planning, therapy tracking, analytical helpers, and
  Python bindings;
- first-order Arrhenius thermal damage in analytical helpers and ablation
  kinetics;
- independent mechanical and thermal insult composition.

Helios needs the same law families. Asclepius now provides the public,
Aequitas-typed implementations, including zero-allocation temperature streams,
validated single-step increments, a validated CEM43 rate, typed Arrhenius
parameters, and a const-generic independent-insult strategy.

Keeping Kwavers formulas would preserve two sources of truth. Wrapping the old
functions around Asclepius would preserve the obsolete public surface and hide
the ownership boundary.

## Decision

Depend directly on public Asclepius from each crate that evaluates a biological
response. Delete the superseded analytical functions and update every in-tree
caller in the same change.

Convert Celsius storage to Aequitas absolute temperature at the consumer
boundary and express steps, rates, activation energy, gas constant, damage,
and probabilities with their quantity or validating newtype. No response
arithmetic remains in Kwavers.

Kwavers retains:

- voxel grids, treatment histories, lesion masks, safety limits, and therapy
  workflow state;
- tissue-specific Arrhenius parameter presets and ablation thresholds;
- clinical dose thresholds such as 240 CEM43 minutes;
- the solver validation implementation derived independently from the
  Sapareto-Dewey reference.

The independent validation implementation must not call Asclepius. Agreement
between it and migrated consumers is cross-verification; agreement between two
calls to Asclepius would only be self-consistency.

## Failure atomicity

Persistent fields must not be partially updated when Asclepius rejects an
observation. Spatial kernels compute checked response increments into
caller-owned reusable scratch storage, return the first typed failure before
mutating accumulated state, and apply the scratch values only after complete
validation. The valid hot path allocates no per update, performs no dynamic
dispatch, and selects the provider once per monomorphized kernel.

Scalar and history APIs propagate provider failures before committing their
new state. PyO3 bindings map those failures to `ValueError`; Python owns no
domain arithmetic.

## Proof obligations

Asclepius owns the model proofs:

1. Positive time and compensation factors make every CEM43 increment
   non-negative, so cumulative exposure is monotone.
2. At the canonical reference, 42, 43, and 44 degrees Celsius contribute
   0.25, 1, and 2 equivalent minutes per elapsed minute.
3. Positive Arrhenius parameters and absolute temperature make the damage rate
   non-negative; damage is monotone and survival is `exp(-damage)`.
4. Independent insults compose as one minus the product of survival
   probabilities, which remains in the unit interval.

Kwavers verifies these obligations at its boundaries with the independent
published CEM43 oracle, analytical Arrhenius reference cases, invalid-domain
regressions, and Python/Rust value equivalence.

Floating-point comparisons use bounds derived from operation count and machine
epsilon. Reordered parallel reductions are not part of these elementwise laws.

## Rejected alternatives

- Retain forwarding wrappers: rejected because they preserve compatibility
  soup and a second public owner.
- Move grids, tissue catalogs, or clinical thresholds into Asclepius: rejected
  because those are Kwavers treatment and simulation concerns.
- Depend on Asclepius only through the facade crate: rejected because it hides
  the actual dependency and prevents narrow package verification.
- Reuse Asclepius in the solver oracle: rejected because it would remove the
  independent cross-verification path.

## Verification

- residue scans for the deleted formula bodies and public function names;
- focused warning-denied Clippy and Nextest for physics, therapy, Python, and
  their callers;
- doctests, Rustdoc, dependency policy, and the public-breaking SemVer gate;
- hosted CI at the exact pull-request head before merge.
