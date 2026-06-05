//! Mechanosensitive ion channel gating models for sonogenetics.
//!
//! # Supported channels
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
//! # Boltzmann two-state gating theorem
//!
//! The equilibrium two-state channel model gives
//! `P_open = 1 / (1 + exp(-A_gate * (Delta T - T_half) / (k_B * T_temp)))`.
//! With `A_gate > 0`, `k_B > 0`, and `T_temp > 0`, the exponent is zero at
//! `Delta T = T_half`, therefore `P_open = 1/2`. Its derivative is positive
//! for finite tension, so the opening probability increases monotonically with
//! membrane tension.
//!
//! # Pressure-threshold gating theorem
//!
//! hsTRPA1 activation is modeled as
//! `P_open = 1 / (1 + exp(-(P_rad - P_half) / s))`. With `s > 0`, the
//! probability is `1/2` at `P_rad = P_half` and increases monotonically with
//! acoustic radiation pressure.
//!
//! # Ion-current theorem
//!
//! The injected-current convention used by the neuron model is
//! `I_inj = g_single * n_channels * P_open * (E_rev - V_m)`. Thus
//! `I_inj = 0` at reversal potential and positive current depolarizes when
//! `E_rev > V_m`.
//!
//! # References
//!
//! - Sukharev, S.I. et al. (1997). Mechanosensitive channel MscL in E. coli.
//!   *Biophysical Journal*, 72(1), 193-203.
//! - Hamill, O.P. & Martinac, B. (2001). *Physiological Reviews*, 81(2), 685-740.
//! - Ibsen, S. et al. (2015). Sonogenetics in C. elegans.
//!   *Nature Nanotechnology*, 10(9), 810-815.
//! - Cox, C.D. et al. (2016). Removal of the mechanoprotective influence of the cytoskeleton
//!   reveals PIEZO1 is gated by bilayer tension. *Nature Communications*, 7, 10366.
//! - Hille, B. (2001). *Ion Channels of Excitable Membranes*, 3rd ed. Sinauer.
//! - Suchyna, T.M. et al. (2000). Identification of a peptide toxin for a mechano-sensitive channel.
//!   *Journal of General Physiology*, 115(5), 583-598.
//! - Szablowski, J.O. et al. (2022). Sonogenetics. *Curr. Opin. Neurobiol.*, 73, 102515.
//! - Duque, M. et al. (2023). *Science*, 380(6649), 1084-1090.
//! - Xian, Q. et al. (2023). Modulation of deep neural circuits with sonogenetics.
//!   *PNAS*, 120(23), e2220575120.
//! - Li, X. et al. (2026). Channel-specific differential effects of bacterial
//!   mechanosensitive channels for ultrasound neuromodulation in precision
//!   sonogenetics. *Theranostics*, 16(5), 2447-2465.
//! - Shimojo, D. et al. (2024). TRPC6 is a mechanosensitive channel essential for
//!   ultrasound neuromodulation in the mammalian brain. *PNAS*, 121.
//! - Matsushita, S. et al. (2024). *PNAS*, 121(14), e2314729121.

mod constants;
mod current;
mod gating;
mod identity;
mod params;

#[cfg(test)]
mod tests;

pub use current::ion_current;
pub use gating::{boltzmann_p_open, compute_p_open, pressure_threshold_p_open};
pub use identity::MechanoChannel;
pub use params::{BoltzmannGatingParams, GatingModel, PressureThresholdParams};
