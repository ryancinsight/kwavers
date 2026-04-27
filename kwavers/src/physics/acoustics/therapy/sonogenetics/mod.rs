//! Sonogenetics simulation components.
//!
//! Provides the core physical models required to simulate focused-ultrasound
//! neuromodulation via mechanosensitive ion channel gating (sonogenetics).
//!
//! # Physical pipeline
//!
//! ```text
//!  PSTD/FDTD simulation
//!  ─────────────────────────────────────────────────────────────────────────
//!  pressure field p(x, t)
//!       │
//!       ▼  [each time step]
//!  VolumetricArfField::accumulate(p)
//!       │
//!       ▼  [after ≥1 complete cycle]
//!  VolumetricArfField::finalize(α(x), c(x), ρ(x))
//!       │
//!       ├── intensity  I(x) = ⟨p²⟩/(ρ·c)   [W/m²]
//!       └── arf_density F(x) = 2·α·I/c      [N/m³]  ← body force for streaming/elasticity
//!
//!  Membrane mechanics
//!  ─────────────────────────────────────────────────────────────────────────
//!  radiation_pressure P_rad(x) = I/c         [Pa]
//!  membrane_tension   ΔT(x) = I·R/(2·c)      [N/m]   (Laplace thin-shell)
//!
//!  Channel gating
//!  ─────────────────────────────────────────────────────────────────────────
//!  Boltzmann (MscL-G22S, MscL-G22N, MscS, Piezo1, TRPC6):
//!    P_open(x) = 1 / (1 + exp(-A·(ΔT − T_half)/(k_B·T)))
//!
//!  Pressure-threshold (hsTRPA1):
//!    P_open(x) = 1 / (1 + exp(-(P_rad − P_half)/s))
//!
//!  Ion current
//!  ─────────────────────────────────────────────────────────────────────────
//!  I_ion = g_single · n_channels · P_open · (E_rev − V_m)
//!
//!  Neuron activation
//!  ─────────────────────────────────────────────────────────────────────────
//!  LifNeuron::step(I_ion, dt, t_now) → spike? → spike_times
//! ```
//!
//! # Supported ion channels
//!
//! | Channel    | Organism         | Gating model         | Reference                           |
//! |------------|------------------|----------------------|-------------------------------------|
//! | MscL-G22S  | E. coli (GOF)    | Boltzmann / tension  | Xian 2023; Li 2026                  |
//! | MscL-G22N  | E. coli (GOF)    | Boltzmann / tension  | Li 2026; Sawada 2015                |
//! | MscS       | E. coli          | Boltzmann / tension  | Li 2026; Nomura 2012                |
//! | Piezo1     | Mammalian        | Boltzmann / tension  | Cox 2016; Lewis 2017                |
//! | TRPC6      | Mammalian        | Boltzmann / tension  | Shimojo 2024; Matsushita 2024       |
//! | hsTRPA1    | H. salinarum     | Pressure threshold   | Ibsen 2015; Szablowski 2022         |
//!
//! # References
//!
//! - Ibsen, S. et al. (2015). Sonogenetics is a non-invasive approach to activating neurons
//!   in *C. elegans*. *Nature Nanotechnology*, 10(9), 810-815.
//! - Szablowski, J.O. et al. (2022). Sonogenetics: using ultrasound to program gene expression
//!   and neuromodulation. *Current Opinion in Neurobiology*, 73, 102515.
//! - Cox, C.D. et al. (2016). Removal of the mechanoprotective influence of the cytoskeleton
//!   reveals PIEZO1 is gated by bilayer tension. *Nature Communications*, 7, 10366.
//! - Duque, M. et al. (2023). Sonogenetic control of mammalian cells using exogenous
//!   transient receptor potential A1 channels. *Science*, 380(6649), 1084-1090.
//! - Xian, Q. et al. (2023). Modulation of deep neural circuits with sonogenetics.
//!   *PNAS*, 120(23), e2220575120.
//! - Li, X. et al. (2026). Channel-specific differential effects of bacterial
//!   mechanosensitive channels for ultrasound neuromodulation in precision
//!   sonogenetics. *Theranostics*, 16(5), 2447-2465.
//! - Shimojo, D. et al. (2024). TRPC6 is a mechanosensitive channel essential for
//!   ultrasound neuromodulation in the mammalian brain. *PNAS*, 121.
//! - Matsushita, S. et al. (2024). Selective sonogenetic activation through TRPC6.
//!   *PNAS*, 121(14), e2314729121.
//! - Hamill, O.P. & Martinac, B. (2001). Molecular basis of mechanotransduction in living cells.
//!   *Physiological Reviews*, 81(2), 685-740.
//! - Nyborg, W.L. (1965). Acoustic streaming. *Physical Acoustics*, 2B, 265-331.
//! - Sarvazyan, A.P. et al. (2010). Acoustic radiation force — a review.
//!   *Curr. Med. Imaging Rev.*, 6(1), 15-25.

pub mod arf_field;
pub mod channels;
pub mod membrane;
pub mod neuron;

pub use arf_field::VolumetricArfField;
pub use channels::{
    boltzmann_p_open, compute_p_open, ion_current, pressure_threshold_p_open,
    BoltzmannGatingParams, GatingModel, MechanoChannel, PressureThresholdParams, BODY_TEMP_K,
};
pub use membrane::{compute_membrane_tension, compute_radiation_pressure, CellMembraneParams};
pub use neuron::{LifNeuron, LifParams};
