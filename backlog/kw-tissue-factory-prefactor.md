<a id="kw-tissue-factory-prefactor"></a>

## KW-TISSUE-FACTORY-PREFACTOR — TissueFactory scales α₀ by 0.18 before PSTD reads it as the 1 MHz prefactor [patch] [fix] — todo

priority: correctness; needs: none; scope: `crates/kwavers-medium/src/heterogeneous/factory/tissue.rs`

- **Finding (basis `0da4122f3`):** `tissue.rs:116-121` stores α₀·(180 kHz / 1 MHz)^1 = 0.09 with `alpha_power` 1.5 and `reference_frequency` 180 kHz. PSTD reads `alpha_coefficient` as α₀ at 1 MHz and converts it with `power_law_db_cm_to_np_omega_m` (`kwavers-solver/src/forward/pstd/physics/absorption/init.rs:207-220`), so tissue absorption there is 5.6× low. `absorption_coefficient` at 1 MHz gives 0.09·5.56^1.5 ≈ 1.18 dB/cm instead of 0.5. The scaling also uses exponent 1 while the field holds 1.5.
- **Acceptance:** the factory stores α₀ = `ACOUSTIC_ABSORPTION_TISSUE`; a test pins `alpha_coefficient` = 0.5 dB/(MHz^y·cm) and the attenuation at 1 MHz.
- **Next step:** delete the scaling and test the factory's acoustic fields.
