//! Constructor, nucleation detection, and coupling-interface factory for
//! [`CavitationCoupledDomain`].

use super::config::{CavitationCouplingConfig, CavitationCouplingType};
use super::domain::CavitationCoupledDomain;
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, PinnCouplingInterface, PinnPhysicsCouplingType,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_physics::bubble_dynamics::{BubbleState, KellerMiksisModel};
use std::collections::HashMap;

impl<B: AutodiffBackend> CavitationCoupledDomain<B> {
    /// Create a cavitation-coupled domain.
    pub fn new(
        config: CavitationCouplingConfig,
        coupling_type: CavitationCouplingType,
        domain_dims: Vec<f64>,
    ) -> Self {
        let bubble_model = KellerMiksisModel::new(config.bubble_params.clone());

        let n_points = config.bubbles_per_point * 100;
        let mut bubble_states = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            bubble_states.push(BubbleState::new(&config.bubble_params));
        }

        let coupling_interfaces = Self::create_coupling_interfaces(&config, &coupling_type);
        let bubble_locations = Self::initialize_bubble_locations(&domain_dims, n_points);

        Self {
            config,
            coupling_type,
            bubble_model,
            bubble_states,
            bubble_locations,
            coupling_interfaces,
            domain_dims,
            _backend: std::marker::PhantomData,
        }
    }

    /// Quasi-random uniform initial bubble placement across the domain volume.
    ///
    /// Actual nucleation sites are refined later by [`detect_nucleation_sites`].
    fn initialize_bubble_locations(domain_dims: &[f64], n_bubbles: usize) -> Vec<(f64, f64, f64)> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let lx = domain_dims.first().copied().unwrap_or(0.01);
        let ly = domain_dims.get(1).copied().unwrap_or(0.01);
        let lz = domain_dims.get(2).copied().unwrap_or(0.01);
        (0..n_bubbles)
            .map(|_| {
                (
                    rng.gen::<f64>() * lx,
                    rng.gen::<f64>() * ly,
                    rng.gen::<f64>() * lz,
                )
            })
            .collect()
    }

    /// Blake threshold pressure for cavitation nucleation.
    ///
    /// ## Theorem
    /// A nucleus of radius `R_n` and surface tension `σ` in a liquid at ambient
    /// pressure `P_0` nucleates when the local pressure drops below
    ///
    /// ```text
    /// P_Blake = P_0 + (2σ/R_n) · [√(2σ / (3 R_n P_0)) − 1]
    /// ```
    ///
    /// Derived from equating the unstable equilibrium condition of the Rayleigh–
    /// Plesset equation (Brennen 1995, *Cavitation and Bubble Dynamics*, §1.3).
    fn blake_threshold(r_nucleus: f64, surface_tension: f64, ambient_pressure: f64) -> f64 {
        let term1 = 2.0 * surface_tension / r_nucleus;
        let term2 = (2.0 * surface_tension / (3.0 * r_nucleus * ambient_pressure)).sqrt() - 1.0;
        ambient_pressure + term1 * term2
    }

    /// Detect cavitation nucleation sites from the acoustic pressure field.
    ///
    /// A voxel nucleates when the local pressure is negative and its magnitude
    /// exceeds the Blake threshold (Brennen 1995 §1.3).  New sites are appended
    /// to `self.bubble_locations`.
    ///
    /// # Arguments
    /// * `pressure_field` – acoustic pressure tensor `[N, 1]` (Pa)
    /// * `x`, `y`, `z`   – spatial coordinate tensors `[N, 1]` (m)
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn detect_nucleation_sites(
        &mut self,
        pressure_field: &Tensor<B, 2>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        z: Option<&Tensor<B, 2>>,
    ) -> Vec<(f64, f64, f64)> {
        // Blake threshold: R_n = 5 μm, σ = SURFACE_TENSION_WATER (water at 20°C), P_0 = 1 atm
        let p_blake = Self::blake_threshold(5e-6, SURFACE_TENSION_WATER, ATMOSPHERIC_PRESSURE);

        let pressure_data = pressure_field.clone().into_data();
        let pressure_slice = pressure_data.as_slice::<f32>().unwrap();

        let x_data = x.clone().into_data();
        let x_slice = x_data.as_slice::<f32>().unwrap();

        let y_data = y.clone().into_data();
        let y_slice = y_data.as_slice::<f32>().unwrap();

        let z_vec: Vec<f32> = if let Some(z_tensor) = z {
            let z_data = z_tensor.clone().into_data();
            z_data.as_slice::<f32>().unwrap().to_vec()
        } else {
            vec![0.0; pressure_slice.len()]
        };

        let mut sites = Vec::new();
        for i in 0..pressure_slice.len() {
            let p = pressure_slice[i] as f64;
            if p < 0.0 && p.abs() > p_blake.abs() {
                sites.push((x_slice[i] as f64, y_slice[i] as f64, z_vec[i] as f64));
            }
        }

        if !sites.is_empty() {
            self.bubble_locations.extend(sites.iter().copied());
        }
        sites
    }

    /// Build the coupling interfaces for the domain.
    ///
    /// Always produces an acoustic–bubble interface; appends a Bjerknes-force
    /// multi-bubble interface when `config.multi_bubble_effects` is set.
    fn create_coupling_interfaces(
        config: &CavitationCouplingConfig,
        coupling_type: &CavitationCouplingType,
    ) -> Vec<PinnCouplingInterface> {
        let mut interfaces = Vec::new();

        let rect = BoundaryPosition::CustomRectangular {
            x_min: 0.0,
            x_max: config.domain_size[0],
            y_min: 0.0,
            y_max: config.domain_size[1],
        };

        let acoustic_coupling_type = match coupling_type {
            CavitationCouplingType::Weak => PinnPhysicsCouplingType::FluxContinuity,
            CavitationCouplingType::Strong => PinnPhysicsCouplingType::Conjugate,
            CavitationCouplingType::MultiBubble => {
                PinnPhysicsCouplingType::Custom("multi_bubble".to_string())
            }
        };

        let mut acoustic_params = HashMap::new();
        acoustic_params.insert("coupling_strength".to_string(), config.coupling_strength);
        acoustic_params.insert(
            "bubbles_per_point".to_string(),
            config.bubbles_per_point as f64,
        );
        acoustic_params.insert(
            "nonlinear_acoustic".to_string(),
            if config.nonlinear_acoustic { 1.0 } else { 0.0 },
        );

        interfaces.push(PinnCouplingInterface {
            name: "acoustic_bubble_coupling".to_string(),
            position: rect.clone(),
            coupled_domains: vec!["acoustic".to_string(), "cavitation".to_string()],
            coupling_type: acoustic_coupling_type,
            coupling_params: acoustic_params,
        });

        if config.multi_bubble_effects {
            let mut mb_params = HashMap::new();
            mb_params.insert("enable_bjerknes".to_string(), 1.0);
            mb_params.insert("collective_effects".to_string(), 1.0);

            interfaces.push(PinnCouplingInterface {
                name: "multi_bubble_interactions".to_string(),
                position: rect,
                coupled_domains: vec!["cavitation".to_string()],
                coupling_type: PinnPhysicsCouplingType::Custom("bjerknes_forces".to_string()),
                coupling_params: mb_params,
            });
        }

        interfaces
    }
}
