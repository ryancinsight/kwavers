# kwavers-therapy

Clinical therapy and care-delivery layer for
[kwavers](https://github.com/ryancinsight/kwavers): HIFU, histotripsy, and lithotripsy
planning, theranostic image-guided guidance, dose and safety monitoring, regulatory
support, and patient management.

This is an application layer. It orchestrates the physics, solver, simulation, and analysis
crates into therapeutic workflows and contains no numerical kernels. Its diagnostic sibling
is [`kwavers-diagnostics`](https://docs.rs/kwavers-diagnostics).

## What it provides

| Module | Responsibility |
|---|---|
| `therapy` | HIFU/LIFU, histotripsy, and lithotripsy planning; theranostic guidance; `ClinicalTherapyParameters` |
| `safety` | IEC 60601-2-37 compliance: mechanical index, dose control, interlocks, audit logging |
| `regulatory` | FDA 510(k) submission records: device class, predicate devices, risk and evidence |
| `patient_management` | Encounters, consent, medical history, treatment plans, vital signs |

## Safety posture

Safety is modeled as enforced state, not as advice. `ClinicalSafetyLimits` and
`ClinicalSafetyMonitor` evaluate exposure against the standard's limits;
`MechanicalIndexCalculator` reports an explicit `MechanicalIndexSafetyStatus` per tissue
type; `InterlockSystem` and `DoseController` gate delivery; and `SafetyAuditLogger` records
every violation and override as an `AuditEntry` for the compliance report.

## Documentation

- API reference: <https://docs.rs/kwavers-therapy>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
