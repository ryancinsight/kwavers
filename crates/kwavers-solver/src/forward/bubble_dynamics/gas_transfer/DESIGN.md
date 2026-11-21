# Transient transport at an isolated spherical interface

The solver advances one dissolved species in an incompressible liquid given
the bubble radius and liquid-side interfacial concentration at successive
times. It returns signed moles transferred into the bubble and into the
outer reservoir. Callers supply their own mechanical and thermodynamic
boundary model. The liquid concentration field persists during every
interval, including when the acoustic drive is off.

## Governing equation and conservation

Let `a = ((R/R_ref)^3 + 3x)^(4/3)` in the material coordinate
`x = (r^3 - R^3)/(3 R_ref^3)`, and use `c = C - C_bulk`.
The equation is `c_t = (D/R_ref^2) (a c_x)_x`. The inner boundary
condition uses the supplied wall concentration. The outer material surface
is held at bulk concentration and the liquid starts uniformly at bulk.

Cell averages carry inventory `4π R_ref^3 Δx c`. Fluxes shared by
adjacent cells cancel in the sum of the update equations. The liquid
inventory change plus the two reported outward exchanges therefore sums to
zero in exact arithmetic. The inner flux uses the wall concentration and
the first two cell averages to reproduce the wall gradient of any quadratic
in `x`. Internal face gradients use neighboring averages on a
smoothly graded grid. The first backward Euler step damps high modes after
an initial wall discontinuity. Implicit midpoint thereafter approximates
the time integral of flux at second order for smooth histories.

## Interfaces and scope

The physics crate owns the spherical material coordinate and diffusion
metric. The solver crate owns the mesh, the implicit update, and the
retained liquid state. The same code uses the caller's floating-point
scalar throughout. For `N` radial cells, each step takes O(N) time and
the solver stores O(N) values. The step commits the new wall and liquid
state after the solve and flux checks succeed.

The equilibrium-radius conversion in the fixtures linearizes
`n = (4π/3) (P_0 R_eq^3 + 2σ R_eq^2)/(R_g T)` at the initial radius.
This yields a signed rate for checking flux. It does not evolve the
prescribed radius from the gas inventory. Coupling it to bubble mechanics,
fitting surfactant or streaming physics, and validating a finite
experimental reservoir require separate evidence. The finite outer
distance and time and space resolution belong to the problem setup.

A leading thin-layer history kernel preserves part of the transport at
lower cost. This increment instead retains the full spherical metric so
the solution also passes the fixed-sphere curvature and transient limits.
An independent wall-following, nodal difference scheme with BDF2 tests the
material-volume implementation using different spatial and time operators.

## References

- Smith & Wang (2022), JFM 939 A28, equations (2.4)–(2.7).
  https://doi.org/10.1017/jfm.2022.218
- Peñas-López et al. (2016), JFM 800, 180–212.
  https://doi.org/10.1017/jfm.2016.401
