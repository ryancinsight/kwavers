<a id="kw-power-law-inverse-forms"></a>

## KW-POWER-LAW-INVERSE-FORMS — Inverse, rad/s and constant power-law conversions still spell the factors [patch] — todo

priority: tightening; needs: KW-POWER-LAW-SWEEP; scope: `crates/kwavers-physics/src/acoustics/mechanics/absorption.rs`, `crates/kwavers-core/src/units.rs`, `const` prefactors in therapy, physics and diagnostics, the KZK Hz^y prefactor

- **Finding (basis `0da4122f3`):** `PowerLawAbsorption` owns the forward α(f) law. About 25 sites still convert by hand (`git grep -n "NP_TO_DB\|DB_TO_NP"`): `np_m_to_power_law_db_cm`, `power_law_db_cm_to_np_omega_m` duplicated by `kwavers_core::units::db_per_mhz_cm_to_neper_per_rad_s_m` and its inverse, `const` dB/(cm·MHz) → Np/(m·MHz) items, and the KZK Np/(m·Hz^y) prefactor. Other α₀·f^y spellings take a different prefactor unit or return dB: dB-valued (`absorption/tissue.rs:393`, `properties/acoustic.rs:163`, `analytical/tissue.rs:124`, `analytical/wave/linear.rs:317`, `analytical/skull.rs:34`, `kwavers-driver/src/physics/acoustic/safety.rs:43,123`), Np-prefactor (`skull/attenuation/physics.rs:21`, `wave_propagation/nonlinear/parameters.rs:152`), and unit-agnostic (`kuznetsov/diffusion.rs:134`, `propagation/attenuation.rs:87,106`, `analytical/wave/linear.rs:301`, `absorption/spatially_varying/computation.rs:17,39`).
- **Acceptance:** one owner each for the inverse and the rad/s prefactor beside `PowerLawAbsorption`; constant prefactors expressed through it or through a typed amplitude-decibel unit; the dB-valued and unit-agnostic spellings evaluate one law with a typed prefactor.
- **Next step:** decide whether aequitas gains an amplitude-decibel reciprocal-length unit, which would let the `const` prefactors convert through aequitas.
