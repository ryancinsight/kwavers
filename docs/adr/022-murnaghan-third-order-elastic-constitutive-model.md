# ADR 022 — Murnaghan Third-Order Elastic Constitutive Model

**Status:** Accepted (constitutive core); small-on-large tensor and time-domain PDE staged
**Change class:** [major] (new physics constitutive model; foundational layer for nonlinear
elastodynamics — follow-on to ADR 014)
**Date:** 2026-06-09

## Context

ADR 014 delivered the *analytical* acousto-elastic relation
`ρc_S²(σ₀) = μ + A·σ₀`, `A = (m+n)/(2(λ+μ))`
(`kwavers_physics::analytical::elastography::{acoustoelastic_sensitivity,
acoustoelastic_shear_speed, estimate_prestress}`), explicitly scoping the *full third-order
PDE* as deferred. Backlog #7 asks for "the time-domain nonlinear forward field behind the
analytical acousto-elastic relation."

Verifying the codebase: the existing nonlinear elastic solver
`kwavers_solver::forward::elastic::nonlinear::NonlinearElasticWaveSolver` implements **hyperelastic
invariant** models (Neo-Hookean, Mooney-Rivlin) — *geometric/finite-strain* nonlinearity — and its
own module header lists, under **"Not yet implemented"**, *"Third-order elastic constants: M and N"*
and *"Acoustoelastic tensor"*. So the **Murnaghan third-order constitutive model** — the material
law whose small-strain pre-stress limit *is* ADR 014's `A` — is genuinely absent. It is also the
prerequisite layer for any time-domain third-order forward solve.

The full deliverable (constitutive law → small-on-large acousto-elastic tangent → time-domain
PDE solver) is large and correctness-sensitive (continuum-mechanics conventions are a classic
source of subtle error). This ADR scopes the **constitutive core** as the first complete,
independently-verifiable increment, with the tensor and PDE as staged follow-ons.

## Decision

Implement a zero-dependency Murnaghan constitutive model in
`kwavers_physics::analytical::murnaghan`, using the **power-sum invariant** strain-energy form of
Chapter 11 §11.9.1 over the symmetric Green–Lagrange strain `E` (3×3, `[[f64;3];3]`):

```text
W(E) = (λ/2)(tr E)² + μ tr(E²) + (l/3)(tr E)³ + m (tr E) tr(E²) + n tr(E³)
```

**Convention choice (load-bearing).** Murnaghan constants are not convention-free: the
principal-invariant (Hughes–Kelly) form `… (l+2m)/3 I₁³ − 2m I₁ I₂ + n I₃` and this power-sum form
yield *different* `(l,m,n)` for the same material (e.g. `n_principal = 3 n_powersum`). The codebase
already commits to the **power-sum** convention — Chapter 11 §11.9.1 writes `W` in `tr E², tr E³`,
and `elastography::acoustoelastic_sensitivity` consumes the `(m,n)` of that form via
`A = (m+n)/(2(λ+μ))`. To keep `(l,m,n)` the SSOT-same across the constitutive model and the
existing AE relation (and the future small-on-large link), this model uses the power-sum form.

- **Reduction check.** The second-order part is exactly St-Venant–Kirchhoff
  `W₂ = (λ/2)(tr E)² + μ tr(E²)`, so `l=m=n=0` ⇒ StVK and the small-strain limit ⇒ Hooke `(λ,μ)`.
- **Second Piola–Kirchhoff stress** `S = ∂W/∂E`, from `∂(tr E)/∂E = I`, `∂ tr(E²)/∂E = 2E`,
  `∂ tr(E³)/∂E = 3E²`:

  ```text
  S = [λ tr E + l (tr E)² + m tr(E²)] I + (2μ + 2m tr E) E + 3n E²
  ```

- **Reference tangent** `ℂ₀ = ∂²W/∂E²|_{E=0}` equals the isotropic linear stiffness
  `ℂ₀ : H = λ tr(H) I + 2μ H`, recovering Lamé `(λ, μ)` exactly.

Public API: `MurnaghanConstants { lambda, mu, l, m, n }`, `strain_energy(&self, &E) -> f64`,
`second_pk_stress(&self, &E) -> [[f64;3];3]`, the reference tangent
`apply_reference_tangent(&self, &H)`, and the **finite-strain material tangent**
`material_tangent(&self, &E, &H) -> [[f64;3];3]` returning `ℂ(E):H` where
`ℂ(E) = ∂²W/∂E² = ∂S/∂E`:

```text
ℂ(E):H = [λ trH + 2l I₁ trH + 2m (E:H)] I + 2m trH·E + (2μ + 2m I₁) H + 3n (H·E + E·H)
```

plus small symmetric-3×3 helpers (trace, double-dot, matmul). The tangent reduces to the
reference stiffness at `E=0` and is major-symmetric; it is the prerequisite for the acousto-elastic
acoustic tensor and for implicit time integration.

## Verification (value-semantic, closed-form)

- **StVK recovery** — with `l=m=n=0`, `S(E) == λ(tr E)I + 2μ E` for arbitrary symmetric `E`
  (exact equality).
- **Linear limit** — as `E→0`, `‖S(E) − Hooke(E)‖/‖Hooke(E)‖ → 0` (third-order terms are `O(E²)`).
- **Uniaxial closed form** — for `E = diag(e,0,0)`: `S_xx == (λ+2μ)e + (l+3m+3n)e²` and
  `S_yy = S_zz == λe + (l+m)e²`; exact.
- **Energy–stress consistency** — `S == ∂W/∂E` checked by central finite differences of
  `strain_energy` against `second_pk_stress` (every component, bounded ε).
- **Symmetry** — `S` symmetric for symmetric `E`.
- **Reference tangent** — `apply_reference_tangent` equals `λ tr(H)I + 2μH`; recovers `(λ,μ)`.
- **Finite-strain tangent** — `material_tangent(E,·)` equals `∂S/∂E` by central finite difference of
  `second_pk_stress` (component-wise); reduces to the reference tangent at `E=0`; is major-symmetric
  (`H₁:ℂ:H₂ == H₂:ℂ:H₁`) and gives symmetric output for symmetric `H`.

## Alternatives considered

- **Extend the solver's `HyperelasticModel` (Neo-Hookean/Mooney-Rivlin) with a Murnaghan
  variant.** Rejected for placement: that type lives in the *solver* crate, while its own
  architectural note states constitutive laws belong in the *physics* layer. The Murnaghan law is
  pure material physics (no discretization), so it lives in `kwavers_physics`, co-located with the
  AE relation it generalizes. The solver will consume it when the PDE follow-on lands.
- **Landau–Lifshitz `(A,B,C)` constants.** Equivalent to Murnaghan `(l,m,n)` by a linear map; the
  codebase already speaks Murnaghan `(m,n)` (ADR 014), so Murnaghan is the SSOT-consistent choice.
- **`nalgebra`/`ndarray` 3×3.** Unnecessary dependency / allocation for fixed 3×3; plain
  `[[f64;3];3]` is zero-cost and zero-dep.

## Consequences

- Closes the "third-order elastic constants" gap the nonlinear solver flagged; provides the
  material SSOT for nonlinear elastodynamics.
- The model's small-strain pre-stress limit reproduces ADR 014's `A` — verifying that link
  (the **small-on-large acousto-elastic tangent**, `cᵢⱼₖₗ + cᵢⱼₖₗ^{AE}(σ₀)`) is the **next staged
  item**: it requires the push-forward to the pre-stressed configuration (Thurston & Brugger 1964).
- The **time-domain third-order PDE solver** (`∂²u/∂t² = ∇·S(∇u)` with the Murnaghan `S`) is the
  final staged item, consuming this constitutive core inside the existing elastic stepper
  framework.

## Status / staging

Constitutive core **Accepted** and implemented, including the **finite-strain material tangent**
`ℂ(E) = ∂²W/∂E²` (`material_tangent`). Staged follow-ons (own [major] items, each ADR-referenced
here):

1. **Small-on-large acousto-elastic acoustic tensor + differential test vs ADR 014's `A`.** Builds
   on `material_tangent`: form the incremental (instantaneous) modulus
   `A⁰_{ijkl} = ℂ(E₀) + initial-stress geometric terms`, then the Christoffel tensor
   `Γ_{ik}(N) = A⁰_{jikl} N_j N_l` whose eigenvalues are `ρc²`. **Note (correctness):** reproducing
   the codebase's specific `A = (m+n)/(2(λ+μ))` requires matching the exact Hughes–Kelly (1953)
   configuration assumptions and including the initial-stress geometric terms (Thurston & Brugger
   1964) — the material tangent alone (energy curvature) is insufficient. This step is deferred
   precisely to get those terms right rather than ship a configuration-mismatched coefficient.
2. **Time-domain Murnaghan forward PDE solve** (`∂²u/∂t² = ∇·S(∇u)`) + harmonic-generation
   verification, consuming this constitutive core inside the existing elastic stepper framework.
