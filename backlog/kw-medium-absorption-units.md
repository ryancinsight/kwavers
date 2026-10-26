<a id="kw-medium-absorption-units"></a>

## KW-MEDIUM-ABSORPTION-UNITS — Medium absorption getters return dB/cm under an Np/m contract [patch] [fix] — todo

priority: correctness; needs: KW-POWER-LAW-TYPED; scope: `crates/kwavers-medium/src/{acoustic,core}.rs`, `crates/kwavers-medium/src/{homogeneous,heterogeneous}/`, the consumers below

- **Finding (basis `0da4122f3`):** every medium stores the dB/(MHz^y·cm) prefactor α₀ that `alpha_coefficient` documents and PSTD, KZK and GPU consume. `AcousticProperties::absorption_coefficient`, documented Np/m, returns α₀·(f/f_ref)^y in dB/cm for `HomogeneousMedium` (`homogeneous/implementation/traits_physical.rs:28`) and `HeterogeneousMedium` (`heterogeneous/traits/acoustic/properties.rs:115-119`). `CoreMedium::absorption` (`ReciprocalLength`, Np/m) and `absorption_array` return α₀ unconverted. Only `HeterogeneousTissueMedium::absorption_coefficient` converts. The trait default `AcousticProperties::attenuation` (`acoustic.rs:19-22`) evaluates the dB prefactor at f in Hz with exponent 1.05, about 4.4e3 "Np/m" for water at 1 MHz, where 0.025 is correct.
- **Impact:** every consumer that assumes Np/m is low by 100·ln(10)/20 = 11.513: therapy heating Q = 2αI (`kwavers-simulation/src/therapy/calculator.rs:139-152`), Eckart streaming, time-reversal gain, acoustic diffusivity, radical initiation, Westervelt exp(−αcΔt) (`kwavers-solver/.../westervelt/update.rs:137`), and the property iterators. `validate_absorption_model` now detects it (error ≈ 0.91 for water at 1 MHz).
- **Design constraint:** `absorption_array` is a raw view and its only writer, `residual_gas_coupling.rs:235-255`, writes dB. So the stored field stays α₀ at 1 MHz in dB/(MHz^y·cm), documented as such, and the Np/m getters convert through `PowerLawAbsorption`. The change is a unit-contract correction with a physical consequence, so an ADR comes first.
- **Acceptance:** each Np/m getter matches `PowerLawAbsorption` against a closed form per medium; `validate_absorption_model` passes for `HomogeneousMedium::water`; consumer tests that pinned dB values are re-derived.
- **Next step:** draft the ADR on the absorption unit contract.
