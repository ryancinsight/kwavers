# kwavers-diagnostics

Clinical diagnostic imaging workflows for
[kwavers](https://github.com/ryancinsight/kwavers): reconstruction pipelines, multi-modal
fusion, Doppler, spectroscopy, functional ultrasound, and decision support.

This is an application layer. It composes the physics, solver, simulation, and analysis
crates into the workflows a diagnostic study actually runs — and contains no numerical
kernels of its own. Its therapeutic sibling is
[`kwavers-therapy`](https://docs.rs/kwavers-therapy).

## What it provides

| Module | Responsibility |
|---|---|
| `reconstruction` | Image reconstruction pipelines over recorded channel data |
| `photoacoustic` | Photoacoustic reconstruction and spectroscopic unmixing |
| `functional_ultrasound` | Functional ultrasound (fUS) with vascular-based neuronavigation |
| `workflows` | `ClinicalWorkflowOrchestrator`, protocols, plane-wave compounding, timing metrics |

Workflow outputs are clinical records rather than raw arrays:
`ClinicalExaminationResult`, `DiagnosticRecommendation`, and `DiagnosticUrgency` carry the
result plus the confidence and urgency a downstream decision needs.

## References

The functional-ultrasound neuronavigation workflow follows Nouhoum et al. (2021),
"A functional ultrasound brain GPS for automatic vascular-based neuronavigation",
*Scientific Reports*, [doi:10.1038/s41598-021-94764-7](https://doi.org/10.1038/s41598-021-94764-7).

## Documentation

- API reference: <https://docs.rs/kwavers-diagnostics>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
