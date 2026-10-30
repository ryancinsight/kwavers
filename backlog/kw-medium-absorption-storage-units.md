<a id="kw-swe-optical-attenuation"></a>
## KW-SWE-OPTICAL-ATTENUATION — Nonlinear SWE solver seeds attenuation from the optical coefficient [defect][minor] — todo
- outcome: `NonlinearSWE` attenuation comes from the acoustic absorption model at its operating frequency, not from optical μa.
- evidence 2026-10-22: `crates/kwavers-solver/src/forward/elastic/nonlinear/solver/core.rs:29` sets `attenuation_np_per_m` from `medium.optical_absorption_coefficient(...)` (Beer–Lambert optical μa). Dimensionally L⁻¹ either way, so aequitas typing cannot catch it; found while typing that getter as `ReciprocalLength`.
- next: confirm intended physics; likely replace with `AcousticProperties::absorption_coefficient` at the solver frequency, or delete the field if unused downstream.
- priority: defect; needs: none; scope: `crates/kwavers-solver/src/forward/elastic/nonlinear`.
- see also: KW-TISSUE-FACTORY-PREFACTOR (done 2026-10-26 on `feat/medium-bubble-properties-typing`; item and index line deleted by the delivering change).
