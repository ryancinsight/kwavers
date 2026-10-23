<a id="kw-annular-element-panics"></a>

## KW-ANNULAR-ELEMENT-PANICS — `add_annular_element` panics on invalid input [patch] — todo

priority: correctness; needs: none; scope: crates/kwavers-transducer/src/kwave_array/construction.rs, crates/kwavers-python/src/kwave_array_py.rs

- **Finding.** `KWaveArray::add_annular_element` validates its radii with `assert!` (`construction.rs:328`). An input-dependent violation panics, and in Python it surfaces as `PanicException`. ADR 136 made `add_rect_*` return `KwaversResult` for the same case.
- **Next step.** Return `KwaversError::Config(ConfigError::InvalidValue)` naming the parameter, map it to `PyValueError`, and audit the other `add_*_element` constructors for `assert!` on inputs.
- **Acceptance.** No `add_*_element` panics on input. A `matches!` test covers each rejected parameter.
