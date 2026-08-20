# kwavers-field

Numerical field definitions for [kwavers](https://github.com/ryancinsight/kwavers):
component indices (SSOT), field-type mapping, field operations and statistics, and the
bubble/electromagnetic field states.

A kwavers simulation carries its state as one dense multi-field array. This crate defines
what each component of that array *means* — the index layout every solver, physics module,
and recorder agrees on — so a field index is never re-derived at a call site.

## What it provides

| Module | Responsibility |
|---|---|
| `indices` | Component index constants (`PRESSURE_IDX`, `VX_IDX`, `STRESS_XX_IDX`, …) and `TOTAL_FIELDS` |
| `type` | `UnifiedFieldType` — the named field taxonomy |
| `mapping` | Field-type to component-index mapping |
| `operations` | `FieldOperations`, `FieldStatistics` over field arrays |
| `wave` | Wave-field state definitions |
| `bubble` | `BubbleStateFields` — cavitation state components |
| `electromagnetic` | `EMFields`, `PoyntingVector` |

The `leto` array types the fields are stored in (`Array3`, `Array4`, `ArrayView3`,
`ArrayViewMut3`, …) are re-exported so consumers need one import path.

## Example

The index constants are the single source of truth for the component layout:

```rust
use kwavers_field::indices::{PRESSURE_IDX, TEMPERATURE_IDX, VX_IDX, TOTAL_FIELDS};
use kwavers_field::Array4;

// Allocate the full field stack over a small grid.
let fields = Array4::<f64>::zeros((TOTAL_FIELDS, 8, 8, 8));

assert_eq!(PRESSURE_IDX, 0);
assert!(TEMPERATURE_IDX < TOTAL_FIELDS);
assert_eq!(fields[[VX_IDX, 0, 0, 0]], 0.0);
```

## Documentation

- API reference: <https://docs.rs/kwavers-field>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
