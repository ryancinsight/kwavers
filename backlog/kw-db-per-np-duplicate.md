<a id="kw-db-per-np-duplicate"></a>

## KW-DB-PER-NP-DUPLICATE — kwavers-core defines 20/ln 10 twice, one ULP apart [patch] — todo

priority: tightening; needs: none; scope: crates/kwavers-core/src/units.rs, crates/kwavers-core/src/constants/acoustic_parameters.rs

- **Finding.** `units::DB_PER_NP = 20.0 / LN_10` evaluates to `0x1.15f2ced384f28p+3`. `constants::acoustic_parameters::NP_TO_DB` is the literal `8.685_889_638_065_037`, which is `0x1.15f2ced384f29p+3`, the correctly rounded 20/ln 10. Two conversions that should agree differ in the last bit; found by the #963 judge.
- **Next step.** Keep one definition, the correctly rounded literal in `acoustic_parameters`. Have `alpha_db_per_mhz_cm_to_np_per_m` use it and delete `DB_PER_NP`.
- **Acceptance.** `git grep DB_PER_NP` is empty. A unit test asserts the conversion equals `α·f^y·100 / NP_TO_DB` bitwise for a published value.
