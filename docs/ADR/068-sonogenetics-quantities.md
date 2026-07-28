# ADR 068: Type sonogenetics physical contracts

- Status: Accepted
- Date: 2026-07-27

## Context

The sonogenetics channel and leaky-integrate-and-fire models exposed membrane
geometry, membrane tension, pressure thresholds, conductance, voltage,
capacitance, current, and time as unit-documented `f64` values. This allowed
metres, farads, siemens, volts, amperes, and seconds to cross public contracts
without dimensional checking. Aequitas also lacked the electrical dimensions
needed by the model.

Dense Leto fields remain scalar storage because a quantity wrapper per voxel
would change the array boundary and allocation model. Open probabilities,
channel counts, and phenomenological thermal-coupling coefficients are
dimensionless or model-specific and remain scalar by contract.

## Decision

Add Aequitas `ElectricCharge`, `ElectricPotential`, `ElectricConductance`, and
`Capacitance` dimensions with coherent Coulomb, Volt, Siemens, and Farad unit
markers. Type sonogenetics public membrane, channel, ion-current, and LIF
contracts with the provider quantities. Extract base scalars only when a typed
quantity enters a dense array, exponential formula, or other numerical storage
boundary.

Rename fields whose unit suffix duplicated the new type contract. Migrate every
in-repository caller and test in one cutover; no compatibility wrapper or
parallel raw-field surface remains.

## Alternatives rejected

1. Keep unit-suffixed scalars and rely on documentation. Rejected because the
   compiler cannot reject cross-unit assignments or invalid electrical algebra.
2. Add Kwavers-local electrical wrappers. Rejected because Aequitas owns SI
   vocabulary and a consumer wrapper would duplicate the provider contract.
3. Wrap every dense voxel value in a quantity. Rejected because it adds storage
   and traversal overhead where Leto arrays are the numerical boundary.

## Verification

- Aequitas dimension-law and layout tests cover charge, potential,
  conductance, and capacitance composition and unit markers.
- Kwavers `kwavers-physics` package check and warning-denied Clippy pass.
- Kwavers Nextest passes 1,556/1,556 tests with one repository-declared skip.
- Kwavers doctests, targeted rustfmt, and Rustdoc pass.
