<a id="kw-medium-absorption-storage-units"></a>
## KW-MEDIUM-ABSORPTION-STORAGE-UNITS — Unify absorption storage units behind the typed scalar boundary [tightening][minor] — todo
- outcome: `CoreMedium::absorption(i,j,k)` (typed Np/m) and `absorption_array()` agree with `absorption_coefficient(f)` in units for every medium; one documented storage convention for the Szabo prefactor.
- evidence 2026-10-22: homogeneous `absorption()` returns `ReciprocalLength::from_base(self.absorption_alpha)` — the raw dB/(cm·MHz^y) prefactor wearing an Np/m type (11.513× off, the defect fixed for `absorption_coefficient` in the bubble/elastic/optical slice); `absorption_array()` fills the same dB-based numeric; heterogeneous/tissue return their `absorption` fields directly. Solver alpha paths mostly consume `alpha_coefficient()` (documented dB prefactor converted at the solver boundary), so storage changes need a consumer audit first.
- next: audit every `absorption()`/`absorption_array()` consumer; pick one convention (α₀ dB-based storage with conversion at typed getters, matching `absorption_coefficient`); add a water-valued regression test beside `water_absorption_is_np_per_m_at_reference`.
- priority: tightening; needs: none; scope: `crates/kwavers-medium/src/{core,homogeneous,heterogeneous}` + solver consumers.

<a id="kw-swe-optical-attenuation"></a>
## KW-SWE-OPTICAL-ATTENUATION — Nonlinear SWE solver seeds attenuation from the optical coefficient [defect][minor] — todo
- outcome: `NonlinearSWE` attenuation comes from the acoustic absorption model at its operating frequency, not from optical μa.
- evidence 2026-10-22: `crates/kwavers-solver/src/forward/elastic/nonlinear/solver/core.rs:29` sets `attenuation_np_per_m` from `medium.optical_absorption_coefficient(...)` (Beer–Lambert optical μa). Dimensionally L⁻¹ either way, so aequitas typing cannot catch it; found while typing that getter as `ReciprocalLength`.
- next: confirm intended physics; likely replace with `AcousticProperties::absorption_coefficient` at the solver frequency, or delete the field if unused downstream.
- priority: defect; needs: none; scope: `crates/kwavers-solver/src/forward/elastic/nonlinear`.
