<a id="kw-phys-001"></a>

## KW-PHYS-001 — Rectified-diffusion rate is not the Eller–Flynn rate (missing oscillation-amplitude factor) [patch] — todo

priority: correctness; needs: none; scope: `rectified_diffusion_rate` in `kwavers-physics` (`acoustics/mechanics/cavitation/core/growth.rs`); milestone: none

- **Outcome:** `rectified_diffusion_rate` returns the genuine Eller–Flynn rate: at Eller's (1969) conditions it gives growth of the order of the measured ~13 µm/min, not 515–527 µm/min — so the function is the theory it claims to implement, and no caller can mistake its output for a validated growth rate.
- **Driver:** Stage 7 of the triplet-pulse prototype (Nov 17, 2026) found the kwavers formula returns 515–527 µm/min at Eller's 1969 conditions against ~13 µm/min measured — a 40× overprediction — while the genuine Eller–Flynn theory UNDERpredicts the same measurement by >20× (Smith & Wang 2022, J. Fluid Mech. 939, A28 §1). The kwavers formula is the Eller–Flynn form with the oscillation amplitude set to unity; the amplitude factor (ε-dependent) is missing. The Stage 6 audit's retired numbers (986/1904 µm) and its shell/depletion bounds are rate-independent and unaffected. The regime guard (`rectified_diffusion_rate_checked`, refuses P_ac/P0 > 3) is already in place; this entry is the rate correction itself.
- **Acceptance:** rate at Eller's conditions within a factor of ~2 of the measured ~13 µm/min (or of the genuine Eller–Flynn value with the amplitude factor restored); the existing `growth` tests still pass, including the checked-variant guard tests; the amplitude dependence is documented on the function, with Smith & Wang (2022) noted as the more accurate in-regime alternative (two-ODE model: radius + gas mass, fading-memory history integral).
