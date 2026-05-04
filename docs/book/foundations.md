# Chapter 1 — Wave Physics Fundamentals

> **Prerequisite mathematics:** vector calculus (gradient, divergence, curl);
> basic thermodynamics (ideal-gas and isentropic relations); complex-number
> notation for harmonic signals.

---

## 1.1 Scope and physical picture

An acoustic wave is a mechanical disturbance that propagates through a
compressible medium by alternating compression and rarefaction of the medium's
molecules.  Unlike electromagnetic waves, acoustic waves require a material
medium: they cannot propagate in vacuum.

In clinical ultrasound the quantities of primary interest are

| Quantity | Symbol | SI unit |
|----------|--------|---------|
| Acoustic pressure | $p(\mathbf{x},t)$ | Pa |
| Particle velocity | $\mathbf{u}(\mathbf{x},t)$ | m s⁻¹ |
| Density perturbation | $\rho'(\mathbf{x},t)$ | kg m⁻³ |
| Acoustic intensity | $I(\mathbf{x},t)$ | W m⁻² |

The *small-signal* or *linear* regime is defined by the conditions

$$
\frac{|p|}{p_0} \ll 1, \qquad
\frac{|\rho'|}{\rho_0} \ll 1, \qquad
|\mathbf{u}| \ll c_0,
$$

where $p_0 \approx 10^5\,\text{Pa}$ is the ambient (atmospheric) pressure,
$\rho_0$ the equilibrium density, and $c_0$ the small-signal speed of sound.
Diagnostic ultrasound typically satisfies these conditions (peak pressures of
$0.1$–$1\,\text{MPa} \ll p_0$ is the caveat: diagnostics are in the nonlinear
regime because $p \gg p_0$, but the *relative* nonlinear distortion per
wavelength is small enough to treat propagation as approximately linear).
High-intensity therapy (HIFU, lithotripsy) explicitly violates the linear
assumption; those cases are treated in Chapters 3 and 6.

This chapter derives the linear acoustic wave equation from first principles,
establishes its solutions, and characterises the physical parameters needed to
describe wave propagation in tissue.

---

## 1.2 Conservation laws for a compressible fluid

### 1.2.1 Mass conservation (continuity equation)

Consider a fixed control volume $V$ with boundary $\partial V$.  The total mass
in $V$ changes only through flux across $\partial V$:

$$
\frac{\partial}{\partial t} \int_V \rho \, dV
= -\oint_{\partial V} \rho \mathbf{u} \cdot \hat{n} \, dA.
$$

By the divergence theorem and the arbitrariness of $V$,

$$
\boxed{
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0.
}
\tag{1.1}
$$

### 1.2.2 Momentum conservation (Euler equation)

For an inviscid fluid the momentum balance on a fluid parcel is Newton's second
law: mass times acceleration equals the pressure force.  In Eulerian form,

$$
\boxed{
\rho \left(
  \frac{\partial \mathbf{u}}{\partial t}
  + (\mathbf{u} \cdot \nabla)\mathbf{u}
\right) = -\nabla p.
}
\tag{1.2}
$$

Viscous stresses, absent here, are treated in §1.8 when we discuss absorption.

### 1.2.3 Equation of state

A fourth unknown (two field unknowns $p, \rho$ beyond $\mathbf{u}$) requires a
closure.  For isentropic processes (reversible, adiabatic — a good model for
ultrasound at MHz frequencies where heat diffusion is negligible),

$$
p = p(\rho), \qquad
c_0^2 = \left.\frac{\partial p}{\partial \rho}\right|_{\rho_0}.
\tag{1.3}
$$

A Taylor expansion about the equilibrium state $(\rho_0, p_0)$ gives the
*exact* isentropic pressure for an ideal gas and an approximate one for
liquids:

$$
p - p_0
= c_0^2 (\rho - \rho_0)
  + \frac{c_0^2}{2\rho_0} \frac{B}{A} (\rho - \rho_0)^2
  + \mathcal{O}\!\left((\rho-\rho_0)^3\right),
\tag{1.4}
$$

where $B/A$ is the **acoustic nonlinearity parameter** of the medium (see §1.9).

---

## 1.3 Linearised acoustic equations

**Linearisation assumption.** Write $\rho = \rho_0 + \rho'$, $p = p_0 + p'$,
and $\mathbf{u} = \mathbf{u}'$ where primed quantities are first-order small.
Neglect second-order products $(\rho' \nabla p', \rho' \partial_t \mathbf{u}',
\ldots)$.

Substituting into (1.1)–(1.3) and dropping the primes:

$$
\boxed{
\frac{\partial \rho}{\partial t} + \rho_0 \nabla \cdot \mathbf{u} = 0,
}
\tag{1.5}
$$

$$
\boxed{
\rho_0 \frac{\partial \mathbf{u}}{\partial t} = -\nabla p,
}
\tag{1.6}
$$

$$
\boxed{
p = c_0^2 \rho.
}
\tag{1.7}
$$

Equations (1.5)–(1.7) are the **first-order linear acoustic equations**.  They
are the exact governing equations implemented in kwavers' PSTD solver
(`kwavers::solver::forward::pstd`).

---

## 1.4 The acoustic wave equation

**Theorem 1.1 (Acoustic Wave Equation).** *Let $p$, $\mathbf{u}$, $\rho$
satisfy the linearised equations (1.5)–(1.7) in a homogeneous medium with
constant $\rho_0$ and $c_0$.  Then $p$ satisfies*

$$
\boxed{
\frac{\partial^2 p}{\partial t^2} - c_0^2 \nabla^2 p = 0.
}
\tag{1.8}
$$

*Furthermore, every component of $\mathbf{u}$ and $\rho$ satisfies the same
equation.*

**Proof.**
Take $\partial/\partial t$ of the continuity equation (1.5):

$$
\frac{\partial^2 \rho}{\partial t^2}
+ \rho_0 \nabla \cdot \frac{\partial \mathbf{u}}{\partial t} = 0.
\tag{i}
$$

Take $\nabla \cdot$ of the momentum equation (1.6):

$$
\rho_0 \nabla \cdot \frac{\partial \mathbf{u}}{\partial t}
= -\nabla^2 p.
\tag{ii}
$$

Eliminate $\nabla \cdot (\partial_t \mathbf{u})$ between (i) and (ii):

$$
\frac{\partial^2 \rho}{\partial t^2} = \nabla^2 p.
\tag{iii}
$$

Use the constitutive relation $p = c_0^2 \rho$, hence
$\partial^2 \rho / \partial t^2 = c_0^{-2} \partial^2 p / \partial t^2$:

$$
\frac{1}{c_0^2}\frac{\partial^2 p}{\partial t^2} = \nabla^2 p,
$$

which is equation (1.8). $\square$

**Corollary 1.1.** For a spatially varying but time-independent medium with
$c_0 = c_0(\mathbf{x})$ and $\rho_0 = \rho_0(\mathbf{x})$, eliminating
$\mathbf{u}$ from (1.5)–(1.7) produces the *variable-coefficient* wave
equation:

$$
\frac{1}{\rho_0 c_0^2}\frac{\partial^2 p}{\partial t^2}
- \nabla \cdot \left(\frac{1}{\rho_0} \nabla p\right) = 0.
\tag{1.9}
$$

This form is used in the kwavers FDTD solver for heterogeneous media (see
`kwavers::solver::forward::fdtd`).

---

## 1.5 Plane-wave solutions

A **plane wave** propagating in direction $\hat{\mathbf{k}}$ is the fundamental
building block of all acoustic wave fields via Fourier superposition.

**Theorem 1.2 (d'Alembert solution in one dimension).** *The general solution
of*

$$
\frac{\partial^2 p}{\partial t^2} - c_0^2 \frac{\partial^2 p}{\partial x^2} = 0
$$

*is*

$$
p(x, t) = f(x - c_0 t) + g(x + c_0 t),
$$

*where $f$ and $g$ are arbitrary twice-differentiable functions determined by
initial conditions $p(x,0)$ and $\dot{p}(x,0)$.*

**Proof.**  Change variables $\xi = x - c_0 t$, $\eta = x + c_0 t$.  The chain
rule gives

$$
\frac{\partial^2 p}{\partial t^2} = c_0^2 \left(
  \frac{\partial^2 p}{\partial \xi^2}
  - 2\frac{\partial^2 p}{\partial \xi \partial \eta}
  + \frac{\partial^2 p}{\partial \eta^2}
\right),
\qquad
c_0^2 \frac{\partial^2 p}{\partial x^2} = c_0^2 \left(
  \frac{\partial^2 p}{\partial \xi^2}
  + 2\frac{\partial^2 p}{\partial \xi \partial \eta}
  + \frac{\partial^2 p}{\partial \eta^2}
\right).
$$

Subtracting: the wave equation becomes
$4 c_0^2 \,\partial^2 p / \partial \xi \partial \eta = 0$, with general
solution $p = f(\xi) + g(\eta)$.  $\square$

**Remark.**  In three dimensions the plane wave

$$
p(\mathbf{x}, t)
= A \, e^{i(\mathbf{k} \cdot \mathbf{x} - \omega t)}
+ B \, e^{-i(\mathbf{k} \cdot \mathbf{x} - \omega t)}
\tag{1.10}
$$

(complex notation, physical field = real part) is a solution of (1.8) provided

$$
\boxed{\omega = c_0 |\mathbf{k}|,}
\tag{1.11}
$$

which is the **dispersion relation** for a non-dispersive medium.  Here
$\omega$ is angular frequency [rad s⁻¹] and $|\mathbf{k}|$ is the wavenumber
magnitude [rad m⁻¹].

The corresponding particle velocity follows from (1.6):

$$
\mathbf{u}(\mathbf{x},t)
= \frac{\hat{\mathbf{k}}}{\rho_0 c_0} p(\mathbf{x},t).
\tag{1.12}
$$

---

## 1.6 Spherical wave solutions and the free-space Green's function

A point source at the origin generates a **spherical wave**:

$$
\boxed{
p(r, t) = \frac{A}{r} \, f\!\left(t - \frac{r}{c_0}\right),
\qquad r = |\mathbf{x}|.
}
\tag{1.13}
$$

The $1/r$ amplitude decay is geometric spreading — not absorption.

**Theorem 1.3 (Free-space Green's function).** *The outgoing Green's function
of the wave equation (1.8) satisfying*
$(\partial_{tt} - c_0^2 \nabla^2) G = \delta(\mathbf{x})\, \delta(t)$
*is*

$$
G(\mathbf{x}, t)
= \frac{1}{4\pi c_0^2 r} \delta\!\left(t - \frac{r}{c_0}\right).
\tag{1.14}
$$

**Proof.**  In spherical coordinates the wave equation for a radially symmetric
solution $G(r,t)$ is $\partial_{tt}G = c_0^2 (\partial_{rr} + 2r^{-1}\partial_r) G$.
The substitution $G = v(r,t)/r$ converts this to the 1D wave equation
$\partial_{tt} v = c_0^2 \partial_{rr} v$, whose outgoing solution is
$v = \delta(t - r/c_0)/(4\pi c_0^2)$ by direct integration of the source term.
Dividing by $r$ gives (1.14).  $\square$

**Physical consequence (Huygens principle).** Any pressure field radiated by
an extended source $S(\mathbf{x}', t)$ can be written as the superposition

$$
p(\mathbf{x}, t)
= \int G(\mathbf{x} - \mathbf{x}', t) \star_t S(\mathbf{x}', t) \, d^3x',
$$

where $\star_t$ denotes convolution in time.  This is the foundation of the
angular-spectrum and k-space propagation methods.

---

## 1.7 Acoustic impedance and boundary conditions

**Definition 1.1 (Specific acoustic impedance).**  For a harmonic plane wave,

$$
Z_0 = \frac{p}{\mathbf{u} \cdot \hat{\mathbf{k}}} = \rho_0 c_0
\quad [\text{Pa s m}^{-1} = \text{kg m}^{-2} \text{s}^{-1}].
\tag{1.15}
$$

The impedance $Z_0$ is a real scalar for a lossless medium.  Representative
values:

| Medium | $\rho_0$ (kg m⁻³) | $c_0$ (m s⁻¹) | $Z_0$ (MRayl) |
|--------|-------------------|----------------|----------------|
| Air (20 °C) | 1.21 | 343 | 0.000415 |
| Water (20 °C) | 998 | 1 481 | 1.478 |
| Soft tissue | 1 060 | 1 540 | 1.632 |
| Bone (cortical) | 1 912 | 3 500 | 6.7 |
| Fat | 928 | 1 440 | 1.337 |
| Blood | 1 060 | 1 575 | 1.670 |

**Theorem 1.4 (Normal-incidence reflection and transmission).** *At a planar
interface between two media with impedances $Z_1$ and $Z_2$, the pressure
reflection and transmission coefficients for a normally incident plane wave are*

$$
\mathcal{R} = \frac{Z_2 - Z_1}{Z_2 + Z_1},
\qquad
\mathcal{T} = \frac{2 Z_2}{Z_2 + Z_1}.
\tag{1.16}
$$

**Proof.**  Boundary conditions require continuity of pressure and normal
particle velocity at the interface ($x = 0$):

$$
p_i + p_r = p_t, \qquad
\frac{p_i - p_r}{Z_1} = \frac{p_t}{Z_2},
$$

where $p_i$, $p_r$, $p_t$ are the amplitudes of the incident, reflected, and
transmitted waves.  Solving the $2 \times 2$ linear system gives (1.16).
$\square$

**Corollary 1.2 (Intensity reflection and transmission).**

$$
R_I = \mathcal{R}^2
= \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2,
\qquad
T_I = 1 - R_I = \frac{4 Z_1 Z_2}{(Z_1 + Z_2)^2}.
\tag{1.17}
$$

**Clinical example.**  At a soft-tissue–bone interface:
$\mathcal{R} = (6.7 - 1.63)/(6.7 + 1.63) = 0.608$, so $R_I = 37\,\%$ of the
incident intensity is reflected.  This strong reflection is the basis of
bone-interface echo signatures in B-mode imaging.

---

## 1.8 Acoustic energy and intensity

**Definition 1.2 (Acoustic energy density).**  For a linear lossless medium,

$$
\mathcal{E}
= \frac{p^2}{2\rho_0 c_0^2}
+ \frac{\rho_0 |\mathbf{u}|^2}{2}
\qquad [\text{J m}^{-3}],
\tag{1.18}
$$

where the two terms represent potential (compressive) and kinetic energy,
respectively.

**Definition 1.3 (Instantaneous acoustic intensity).**

$$
\mathbf{I}(\mathbf{x},t) = p(\mathbf{x},t)\, \mathbf{u}(\mathbf{x},t)
\qquad [\text{W m}^{-2}].
\tag{1.19}
$$

**Theorem 1.5 (Energy conservation — acoustic Poynting theorem).**
*In a lossless medium,*

$$
\frac{\partial \mathcal{E}}{\partial t} + \nabla \cdot \mathbf{I} = 0.
\tag{1.20}
$$

**Proof.**  Multiply (1.5) by $p/(\rho_0 c_0^2)$ and (1.6) by $\mathbf{u}$,
then add.  Each term matches the time-derivative of the energy density (1.18),
and the cross-term produces $\nabla \cdot (p\mathbf{u})$.  $\square$

**Theorem 1.6 (Plane-wave time-averaged intensity).**  *For a single harmonic
plane wave $p(t) = A \cos(\omega t)$ and $u(t) = (A/Z_0)\cos(\omega t)$,*

$$
\langle I \rangle
= \langle p u \rangle
= \frac{A^2}{2 Z_0}
= \frac{A^2}{2\rho_0 c_0}.
\tag{1.21}
$$

**Proof.**  $\langle \cos^2(\omega t) \rangle = 1/2$; substitute.  $\square$

**Definition 1.4 (RMS pressure).**  $p_\mathrm{rms} = A/\sqrt{2}$ for a single
harmonic.  Thus $\langle I \rangle = p_\mathrm{rms}^2 / Z_0$.

**Clinical intensity scales:**

| Application | Typical $I_{\text{SPTA}}$ |
|-------------|--------------------------|
| Diagnostic B-mode | 0.001–0.1 W cm⁻² |
| Physiotherapy (thermal) | 0.1–3 W cm⁻² |
| Lithotripsy (peak) | 10³–10⁴ W cm⁻² |
| HIFU ablation | 100–10 000 W cm⁻² |
| FDA limit (diagnostic) | 0.72 W cm⁻² ($I_{\text{SPTA}}$) |

---

## 1.9 Acoustic absorption

### 1.9.1 Physical mechanisms

Real biological tissue absorbs acoustic energy through at least three mechanisms:

1. **Viscous dissipation.**  Velocity gradients drive irreversible momentum
   transfer.  The equation of motion gains a term
   $(\eta + \eta_B/3)\nabla(\nabla\cdot\mathbf{u}) - \eta\nabla\times\nabla\times\mathbf{u}$,
   where $\eta$ is shear viscosity and $\eta_B$ bulk viscosity.

2. **Thermal conduction.**  The process is not perfectly isentropic;
   temperature gradients near compressions drive heat conduction, dissipating
   energy.

3. **Molecular relaxation.**  Internal degrees of freedom (rotation, vibration,
   chemical equilibria in tissue) lag behind pressure changes and dissipate
   energy near characteristic frequencies.

### 1.9.2 The power-law absorption model

Empirically, biological tissues follow a **power-law** attenuation over the
ultrasound frequency range 0.5–10 MHz (Duck 1990; Szabo 2004):

$$
\boxed{
\alpha(f) = \alpha_0 f^y \qquad [\text{Np m}^{-1}],
}
\tag{1.22}
$$

where $f$ is frequency in MHz, $\alpha_0$ is the absorption coefficient at
1 MHz, and $y \in [1, 2]$ is the power-law exponent.

**Representative tissue values (from Duck 1990):**

| Tissue | $\alpha_0$ (dB cm⁻¹ MHz⁻$y$) | $y$ |
|--------|-------------------------------|-----|
| Blood | 0.18 | 1.21 |
| Fat | 0.48 | 1.0 |
| Soft tissue (average) | 0.52 | 1.0 |
| Muscle (along fibres) | 0.57 | 1.0 |
| Liver | 0.50 | 1.05 |
| Kidney | 1.0 | 1.0 |
| Bone (cancellous) | 9.94 | 1.0 |

Unit conversion: $1\,\text{dB cm}^{-1} = 0.1151\,\text{Np cm}^{-1}
= 11.51\,\text{Np m}^{-1}$.

### 1.9.3 Fractional Laplacian formulation

Treeby and Cox (2010) showed that the power-law model (1.22) can be incorporated
*exactly* into the time-domain acoustic equations by augmenting the density
update with a fractional-Laplacian operator.

**Theorem 1.7 (Treeby–Cox fractional absorption operators).**  *Define the
causal power-law absorption operators*

$$
\mathcal{L}_1 = -2\alpha_0 \omega_0^{-y} (-\nabla^2)^{(y+1)/2},
\qquad
\mathcal{L}_2 = 2\alpha_0 \omega_0^{-y} c_0 (-\nabla^2)^{y/2},
\tag{1.23}
$$

*where $\omega_0 = 2\pi \times 1\,\text{MHz}$ is the reference frequency.
Then the first-order system*

$$
\frac{\partial \mathbf{u}}{\partial t}
= -\frac{1}{\rho_0} \nabla p,
\qquad
\frac{\partial \rho}{\partial t}
= -\rho_0 \nabla \cdot \mathbf{u}
  + \mathcal{L}_1 p + \mathcal{L}_2 \rho,
\tag{1.24}
$$

*produces attenuation $\alpha \propto \omega^y$ and phase velocity
$c_\mathrm{ph}(\omega) = c_0 \left[1 + \alpha_0 \omega_0^{-y} c_0 \omega^{y-1}
\cot(y\pi/2)\right]^{-1}$ consistent with the Kramers–Kronig relations.*

**Proof sketch.**  Fourier-transform the system (1.24) in space.  The
fractional Laplacian $(-\nabla^2)^s$ becomes multiplication by $|\mathbf{k}|^{2s}$
in Fourier space.  Substituting the plane-wave ansatz
$p, \rho \sim e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$ and eliminating
density yields a complex dispersion relation.  The imaginary part of the complex
wavenumber gives $\alpha(\omega)$, which matches (1.22).  The real part gives
the phase velocity, which is shown to satisfy the Kramers–Kronig relations by
the causal construction of $\mathcal{L}_1$.
Full proof: Treeby & Cox, *J. Acoust. Soc. Am.* 127(5), 2010, §III.  $\square$

**Implementation reference.**  This is exactly what kwavers implements for the
CPU PSTD solver (see
`kwavers::solver::forward::pstd::physics::absorption`).  The fractional
Laplacian $(-\nabla^2)^s$ is computed as $\mathrm{IFFT}[|\mathbf{k}|^{2s}
\cdot \mathrm{FFT}[\cdot]]$, using Apollo's 3D FFT plan.

---

## 1.10 The nonlinearity parameter $B/A$

As the amplitude of a pressure wave increases, the constitutive relation (1.7)
is no longer adequate.  The leading nonlinear correction is captured by the
equation of state (1.4), in which the $B/A$ term drives second-harmonic
generation and shock formation.

**Definition 1.5 (B/A parameter).** For an isentropic fluid,

$$
\frac{B}{A}
= 2\rho_0 c_0 \left.\frac{\partial c}{\partial p}\right|_{\rho_0}
= \rho_0 \left.\frac{\partial^2 p}{\partial \rho^2}\right|_{\rho_0}
  \Big/ c_0^2.
\tag{1.25}
$$

It measures the fractional change in compressibility with pressure.  The
combined nonlinearity coefficient used in the Westervelt and Kuznetsov
equations is

$$
\beta = 1 + \frac{B}{2A}.
\tag{1.26}
$$

**Representative $B/A$ values (Beyer 1997; Hamilton & Blackstock 1998):**

| Medium | $B/A$ | $\beta$ |
|--------|-------|---------|
| Water (20 °C) | 5.0 | 3.5 |
| Water (37 °C) | 5.4 | 3.7 |
| Blood | 6.1 | 4.05 |
| Soft tissue (average) | 7.4 | 4.7 |
| Fat | 9.5 | 5.75 |
| Amniotic fluid | 6.2 | 4.1 |
| Air (0 °C) | 0.4 | 1.2 |

Higher $B/A$ means faster nonlinear distortion: harmonics grow more rapidly and
shocks form at shorter propagation distances.

**Theorem 1.8 (Harmonic generation from quadratic nonlinearity).**  *A
monochromatic plane wave $p(x,0) = p_0 \cos(kx)$ propagating in a lossless
nonlinear medium generates harmonics at integer multiples of the fundamental
frequency.  To leading order, the second-harmonic amplitude grows as*

$$
p_2(x) \approx \frac{\beta k p_0^2}{4 \rho_0 c_0^3} \, x.
\tag{1.27}
$$

**Proof sketch.**  Substitute $p = p_0\cos(kx - \omega t) + p_2$ into the
Westervelt equation (see Chapter 3, eq. 3.1) and collect terms at frequency
$2\omega$.  The driving term on the right-hand side is
$(\beta/\rho_0 c_0^4) \partial^2(p_0^2 \cos^2)/\partial t^2
= (\beta p_0^2 \omega^2 / \rho_0 c_0^4) \cos(2\omega t - 2kx)$,
which resonantly drives $p_2$ linearly in $x$.  Integrating gives (1.27).
Full derivation: Hamilton & Blackstock (1998), §1.4.  $\square$

**Clinical implication.**  Second-harmonic imaging (tissue harmonic imaging)
exploits the $p_0^2$ dependence in (1.27): grating-lobe artefacts and
reverberation clutter, which scale linearly with $p_0$, are suppressed relative
to the tissue harmonic signal.

---

## 1.11 Speed of sound in tissue

The small-signal speed of sound in soft tissue depends on temperature,
hydration, and lipid content:

$$
c_0(T) \approx c_{37} + \alpha_c (T - 37),
\qquad T\,[\text{°C}],
\tag{1.28}
$$

where $c_{37} \approx 1\,540\,\text{m s}^{-1}$ for most soft tissues and
$\alpha_c \approx 1$–$2\,\text{m s}^{-1}\,\text{°C}^{-1}$.

In water the temperature dependence is more pronounced:

$$
c_\text{water}(T)
= 1\,402.7 + 4.83T - 0.048T^2 + 1.47 \times 10^{-4} T^3
\quad [\text{m s}^{-1}].
\tag{1.29}
$$

**Implementation reference.**  kwavers stores $c_0$ as a 3D scalar field in
`kwavers::domain::medium::material_fields::GenericMaterialFields`.  The
temperature-dependent correction (1.28) can be applied by updating the field
after each thermal step in a coupled simulation (see Chapter 6).

---

## 1.12 Summary of governing constants in kwavers

The physical constants relevant to this chapter are defined as single-source-of-truth
constants in `kwavers::core::constants`:

```rust
// From kwavers::core::constants::fundamental
pub const SOUND_SPEED_WATER_20C: f64 = 1_481.0;   // m/s
pub const DENSITY_WATER_20C: f64    = 998.0;       // kg/m³
pub const WATER_IMPEDANCE: f64      = 1_477.938;   // kg/(m²·s) = Pa·s/m

// From kwavers::core::constants::acoustic_parameters
pub const WATER_NONLINEARITY_B_A: f64       = 5.0;
pub const WATER_ABSORPTION_ALPHA_0: f64     = 0.002; // Np/(m·MHz^y)
pub const WATER_ABSORPTION_POWER: f64       = 2.0;   // y (viscothermal)

// Tissue defaults (Duck 1990)
pub const SOFT_TISSUE_SOUND_SPEED: f64      = 1_540.0;    // m/s
pub const SOFT_TISSUE_DENSITY: f64          = 1_060.0;    // kg/m³
pub const SOFT_TISSUE_ABSORPTION_ALPHA: f64 = 0.52;       // dB/(cm·MHz^y)
pub const SOFT_TISSUE_ABSORPTION_POWER: f64 = 1.0;        // y
pub const SOFT_TISSUE_NONLINEARITY_B_A: f64 = 7.4;
```

---

## 1.13 Worked example — standing-wave field in a 1D water column

**Setup.** A 50 mm water column ($c_0 = 1\,481\,\text{m s}^{-1}$,
$\rho_0 = 998\,\text{kg m}^{-3}$) with rigid boundaries at $x = 0$ and
$x = L$ is excited by an initial pressure distribution
$p(x, 0) = p_0 \sin(kx)$ with $k = \pi/L$ (fundamental mode) and
$p_0 = 10^5\,\text{Pa}$.

**Analytical solution.** The boundary conditions require $\partial p/\partial x = 0$
at both walls, consistent with $\sin(kx)$ with $k = n\pi/L$ for integer $n$.
By d'Alembert's solution (Theorem 1.2) the wave decomposes into left- and
right-travelling pulses, producing a perfect standing wave:

$$
p(x, t) = p_0 \sin(kx)\cos(\omega t),
\qquad \omega = c_0 k.
\tag{1.30}
$$

This is the reference solution used in the `test_standing_wave_analytical` test in
`kwavers/tests/fdtd_pstd_comparison.rs`, which validates that the PSTD, FDTD,
Kuznetsov (linear limit), and Westervelt (linear limit) solvers all reproduce
(1.30) to machine precision at $t = 0$ and within numerical truncation error
for $t > 0$.

**Figure 1.1.** Standing-wave field at four instants $\omega t \in \{0, \pi/4, \pi/2, 3\pi/4\}$
for the analytical solution (1.30).

> *Generated by `pykwavers/examples/book/ch01_standing_wave.py`.*

---

## 1.14 Further reading

1. **Kinsler, L. E., Frey, A. R., Coppens, A. B., & Sanders, J. V.** (2000).
   *Fundamentals of Acoustics* (4th ed.). Wiley.  Chapters 2–6 for a detailed
   derivation of the wave equation from first principles.

2. **Duck, F. A.** (1990). *Physical Properties of Tissue: A Comprehensive
   Reference Book*. Academic Press.  The primary source for tissue acoustic
   properties (speed, attenuation, nonlinearity).

3. **Treeby, B. E., & Cox, B. T.** (2010). Modeling power law absorption and
   dispersion for acoustic propagation using the fractional Laplacian.
   *J. Acoust. Soc. Am.*, 127(5), 2741–2748.
   [doi:10.1121/1.3377056](https://doi.org/10.1121/1.3377056)

4. **Hamilton, M. F., & Blackstock, D. T.** (Eds.). (1998). *Nonlinear
   Acoustics: Theory and Applications*. Academic Press.  Chapters 1–3 for
   $B/A$, harmonic generation, and the Westervelt equation.

5. **Beyer, R. T.** (1997). *Nonlinear Acoustics*. Acoustical Society of
   America.

6. **Szabo, T. L.** (2004). *Diagnostic Ultrasound Imaging: Inside Out*
   (1st ed.). Academic Press.  Clinical context for all quantities above.

7. **Cobbold, R. S. C.** (2007). *Foundations of Biomedical Ultrasound*.
   Oxford University Press.

---

## Appendix 1A — Kramers–Kronig relations

For any causal linear system the real and imaginary parts of the frequency
response are related by Hilbert transforms:

$$
\alpha(\omega) = \frac{2\omega^2}{\pi c_0^2}
\int_0^\infty \frac{c(\omega') - c_0}{\omega'^2 - \omega^2} d\omega',
\tag{1A.1}
$$

where $c(\omega)$ is the phase velocity.  Any physically realizable absorption
law $\alpha(\omega)$ must be accompanied by a specific dispersion $c(\omega)$
determined by (1A.1).  The power-law model satisfies this constraint exactly
(O'Brien & Stern, 1983; Szabo, 1994).

---

## Appendix 1B — Notation index

| Symbol | Meaning | SI unit |
|--------|---------|---------|
| $p_0$ | Ambient pressure | Pa |
| $\rho_0$ | Ambient density | kg m⁻³ |
| $c_0$ | Small-signal sound speed | m s⁻¹ |
| $p'$, $\rho'$ | Acoustic perturbations | Pa, kg m⁻³ |
| $\mathbf{u}$ | Particle velocity | m s⁻¹ |
| $Z_0 = \rho_0 c_0$ | Specific acoustic impedance | MRayl |
| $\omega$ | Angular frequency | rad s⁻¹ |
| $f$ | Frequency | Hz |
| $k = \omega/c_0$ | Wavenumber | rad m⁻¹ |
| $\lambda = 2\pi/k$ | Wavelength | m |
| $\mathcal{R}$, $\mathcal{T}$ | Pressure reflection/transmission | — |
| $R_I$, $T_I$ | Intensity reflection/transmission | — |
| $\mathcal{E}$ | Acoustic energy density | J m⁻³ |
| $\mathbf{I}$ | Acoustic intensity | W m⁻² |
| $\alpha_0$ | Absorption coefficient at 1 MHz | Np m⁻¹ or dB cm⁻¹ |
| $y$ | Absorption power-law exponent | — |
| $B/A$ | Nonlinearity parameter | — |
| $\beta = 1+B/(2A)$ | Combined nonlinearity coefficient | — |
| $\delta$ | Acoustic diffusivity | m² s⁻¹ |
| $\eta$, $\eta_B$ | Shear, bulk viscosity | Pa s |
