// Fluid-Structure Interface (FSI) Coupling
//
// Implements acoustic-elastic coupling at fluid-structure interfaces following
// rigorous interface conditions that preserve physical conservation laws.
//
// ## Mathematical Foundation
//
// ### Interface Conditions
//
// At the fluid-structure interface Γ, the following conditions must be satisfied:
//
// **1. Pressure Continuity (Dynamic Condition)**
// ```text
// p_f = σ_ns   on Γ
// ```
// where p_f is fluid pressure and σ_ns is normal stress in the solid.
//
// **2. Velocity Continuity (Kinematic Condition)**
// ```text
// v_f · n = ∂u_s/∂t · n   on Γ
// ```
// where v_f is fluid particle velocity, u_s is solid displacement, and n is the interface normal.
//
// **3. Traction Balance**
// ```text
// t_s = -p_f n + τ_f · n   on Γ
// ```
// where t_s is solid traction and τ_f is fluid viscous stress.
//
// ### Energy Conservation Theorem
//
// **Theorem**: Interface Energy Conservation
// For a coupled fluid-structure system with interface Γ, the total energy satisfies:
//
// ```text
// d/dt(E_f + E_s) = ∫_Γ (p_f v_f · n - σ_s : ε_s) dS + P_ext
// ```
//
// where E_f and E_s are fluid and solid energies, σ_s is solid stress, and ε_s is solid strain.
//
// **Proof Sketch**:
// 1. Fluid energy evolution from acoustic wave equation
// 2. Solid energy evolution from elastodynamic equations
// 3. Interface terms cancel due to pressure/velocity matching
// 4. Result: Global energy conservation if P_ext = 0 and boundaries are lossless
//
// **Reference**: Fahy, F. (2007). "Sound and Structural Vibration", Academic Press. ISBN: 978-0080480734
//
// ### Transmission and Reflection Coefficients
//
// For plane waves incident on a planar interface between fluid (ρ₁, c₁) and solid (ρ₂, c_l, c_t):
//
// **Reflection Coefficient**:
// ```text
// R = [(ρ₁c_l/cos θ_l - ρ₂c₁/cos θ_i)(ρ₁c_t/cos θ_t + ρ₂c₁/cos θ_i) + ρ₂²c₁²tan² θ_t /cos² θ_i] /
//     [(ρ₁c_l/cos θ_l + ρ₂c₁/cos θ_i)(ρ₁c_t/cos θ_t + ρ₂c₁/cos θ_i) - ρ₂²c₁²tan² θ_t /cos² θ_i]
// ```
//
// **Transmission Coefficient** (refracted longitudinal):
// ```text
// T_l = 2ρ₂c₁cos θ_l / [(ρ₁c_l/cos θ_l + ρ₂c₁/cos θ_i)(ρ₁c_t/cos θ_t + ρ₂c₁/cos θ_i) - ρ₂²c₁²tan² θ_t /cos² θ_i]
// ```
//
// **References**:
// - Brekhovskikh, L. M., & Godin, O. A. (1990). "Acoustics of Layered Media I". Springer.
//   DOI: 10.1007/978-3-642-75129-8
// - de Hoop, A. T. (1995). "Handbook of Radiation and Scattering of Waves". Academic Press.
//   ISBN: 978-0122090521

pub mod coefficients;
pub mod interface;
pub mod solver;

pub use coefficients::ReflectionTransmissionCoefficients;
pub use interface::{FsiInterface, FsiInterfaceSpec};
pub use solver::FluidStructureSolver;
