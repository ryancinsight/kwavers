#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod audit;
pub mod board;
pub mod component_accuracy;
pub mod component_db;
pub mod cost;
pub mod dfm;
pub mod driver;
pub mod error;
pub mod fabrication;
pub mod five_level;
pub mod geom;
#[cfg(feature = "io")]
pub mod io;
#[cfg(feature = "io")]
pub mod kicad_cli;
pub mod manifest;
pub mod optim;
pub mod pipeline;
pub mod place;
pub mod pulse_skip;
pub mod render;
pub mod route;
pub mod rules;
pub mod stack;
pub mod tr_switch;
pub mod validate;
pub mod verify;

// Phase 0 vertical-slice placeholders (see docs/MIGRATION.md + docs/ARCHITECTURE.md). Each
// of these is a fresh `src/<slice>/` directory whose content will be migrated from the
// existing flat modules in Phase 1. Declaring them at Phase 0 puts the new vertical-slice
// tree into the compile graph + the docs target so the structure is locked-in for every
// follow-on phase.
//
// `pub mod units;` is canonical from Phase 1a — implements the length newtype (`Nm`) +
// the SI wrappers (`Hz`, `Ohm`, `Watt`, `Kelvin`, `Celsius`, `Volt`, `Amp`, `Henry`,
// `Farad`, `Coulomb`) with `From<f64>` conversions, scalar + same-unit + cross-product
// arithmetic, prefix factories, and SI-symbol `Display`. See its module docstring for the
// migration convention.

// Phase 1a prelude — the canonical `pub use` surface for downstream consumers.
// `use kwavers_driver::prelude::*;` brings the canonical unit newtypes, geometry types,
// board model, and physics facade into scope. Promoted from a Phase-0 doc-only file to
// a real entry point at Phase 1a; declared here as `pub mod prelude;` alongside the other
// top-level modules.
pub mod experiment;
#[cfg(feature = "kwavers")]
pub use experiment::KwaversSim;
pub use experiment::{
    artifact_key, build_beam_report, propagate_thermal, run_experiment, AcousticSimulator,
    DefaultStimulus, ExperimentMetrics, ExperimentRecord, ExperimentReport, InCrateAcousticSim,
    LaneBinding, PressureMap, Stimulus, ThermalState, TileDispatch,
};
pub mod geometry;
pub mod physics;
pub mod prelude;
pub mod ssot;
pub mod units;

pub use audit::{
    audit, charge_recycling_efficiency_audit, copper_area_per_layer, copper_imbalance,
    pulse_skip_interference_audit, weakness_field, ChargeRecyclingReport, FaultReport,
    PulseSkipInterferenceReport,
};
pub use board::{
    split_domain_from_name, Board, LayerId, Net, NetClassKind, NetId, Pad, SplitDomain, Track, Via,
    ViaKind,
};
pub use component_accuracy::{ComponentAccuracy, ComponentAccuracyReport};
pub use component_db::{
    available_pulsers, board_area_per_n_channels_mm2, compare_pulsers, decoupling_per_ch_uf,
    output_pin_capacitance_pf, pkg_area_mm2, recommend_96ch_architecture, signal_pins_per_ch,
    supply_pins_per_ch, PulserComparison, PulserIc, StockStatus,
};
pub use cost::{PhysicsCost, RoutingCost};
pub use dfm::{
    dedup_vias, ground_pour, merge_collinear, miter_right_angle_corners, quietest_layer,
    resolve_diagonal_via_clearance, teardrops, widen_for_ampacity,
};
pub use driver::{
    chip_power_rating_w, compare_driver_ics_at, damping_resistor_ohm, driver_efficiency,
    find_best_freq, load_quality_factor, max_safe_duty, power_rating_check, pulser_dissipation,
    reactive_drive_power_w, ringdown_cycles, sweep_driver_loss, switching_node_ringing_v,
    thermally_derated_efficiency, tuning_inductor_h, ComponentComparison, FreqSweepPoint,
    PowerRatingReport, PulserDissipation, PulserOp,
};
pub use error::{Error, Result};
pub use fabrication::{fabrication_readiness, is_exact_footprint, FabReadiness};
pub use five_level::{
    compare_drive_topologies, nlevel_dynamic_loss_w, nlevel_efficiency, nlevel_energy_per_cycle_j,
    nlevel_power_saving_w, nlevel_rails, typical_cr_efficiency, LevelComparison,
};
pub use geom::{
    convex_hull, dist_point_seg, dist_seg_seg, mechanical_features, GridSpec, MechFeature,
    MechKind, Point,
};
#[cfg(feature = "io")]
pub use io::{
    duplicate_pcb_uuids, parse_kicad_pcb, save_kicad_pcb, save_kicad_project, save_kicad_sch,
    write_kicad_pcb, write_kicad_sch,
};
#[cfg(feature = "io")]
pub use kicad_cli::{DrcOptions, DrcReport, FabBundle, KiCadCli};
pub use manifest::{
    hv_manifest_from_board, DriverManifest, EnergyBudgetInputs, EnergyBudgetReport,
    ResistorPackage, TileStimulationProfile,
};
pub use optim::{
    evaluate_design_point, hot_track_resistance, max_safe_duty_thermal, ringing_exceeds_breakdown,
    ArrayGeometry, DesignReport, EmiContext, PdnConfig, ThermalContext,
};
pub use physics::acoustic::{
    acoustic_intensity_w_per_m2, array_factor, bvd_anti_resonance_hz, bvd_series_resonance_hz,
    element_factor, f_number, focal_pressure_gain, focused_delay_profile_s, grating_lobe_angle_deg,
    isppa_w_per_m2, max_delay_quantization_error_s, max_grating_free_steer_deg, mechanical_index,
    near_field_distance_m, nonlinear_shock_parameter, pitch_from_aperture_m, pressure_derating,
    quantize_delays_s, round_trip_attenuation_db, tissue_attenuation_db, wavelength_m,
};
pub use physics::ampacity::{
    ac_resistance_factor, ampacity_check, annular_ring_mm, black_mttf_relative, copper_thickness_m,
    current_density_a_per_mm2, ipc2221_min_width, pth_aspect_ratio, skin_depth_m, track_resistance,
    AmpacityDeficit,
};
pub use physics::dielectric::{
    air_breakdown_possible, caf_ttf_relative, ipc2221_min_spacing_mm, paschen_breakdown_v,
    paschen_min_air,
};
pub use physics::emi::{
    capacitive_drive_current_a, commutation_loops, gate_drive_power_w, inductive_overshoot_v,
    loop_inductance_nh, radiated_emi_dbuv_m, reverse_recovery_loss_w, switching_loss_w,
    trace_partial_inductance_nh, CommutationLoop,
};
pub use physics::pdn::{
    anti_resonance_hz, holdup_capacitance_f, max_decoupling_distance_mm, pdn_impedance_at_freq,
    plane_resonance_hz, self_resonant_freq_hz, target_impedance_ohm,
};
pub use physics::si::{
    channel_operating_margin_db, crosstalk_coupling, differential_microstrip_impedance,
    impedance_target, microstrip_delay_s_per_m, microstrip_impedance, return_loss_db,
    risetime_degradation_ps_per_m, stripline_impedance, within_skew,
};
pub use physics::thermal::{
    ir_drop, junction_temperature_k, solve_board, solve_electrothermal,
    temperature_derated_resistance, thermal_time_constant_s, thermal_via_conductance,
    transient_rise_k, IrDrop, ThermalField,
};
pub use pipeline::{
    cooptimize, cooptimize_min_area, cooptimize_min_layers, place_to_board, CoOpt, CoOptResult,
    RoutingInputs,
};
pub use place::{
    anneal, component_clearance_violations, energy, import_kicad_mod, import_symbol_pinmap,
    AnnealParams, Axis, Component, ComponentClearanceViolation, CongestionField, DielectricGrade,
    EnergyTerms, FootprintDef, IsolationDomain, PackageFormFactor, PadDef, PinMap, PlaceConfig,
    PlaceWeights, Placement, Role, Rot, RotationPolicy,
};
pub use pulse_skip::{
    channel_skip_pattern, max_skip_spur_dbc, optimal_skip_fraction, optimize_skip,
    power_saving_fraction, rms_pressure_error_fraction, skip_induced_grating_lobe,
    skip_temperature_reduction_k, skipped_power_w, SkipConfig, SkipOptimization,
};
pub use render::{render_board_svg, save_board_svg};
pub use route::{NetTerminals, PathFinderParams, RouteOutcome, Router};
pub use rules::{CreepageRule, DesignRules, ViaPolicy};
pub use stack::{
    assemble_shield_stack, board_rise_k, optimize_shield_stack, optimize_stack,
    stack_board_manifest_from_board, verify_stack_pair, ShieldStackAssembly, ShieldStackPlan,
    StackBoardInstance, StackBoardManifest, StackBoardRole, StackCompatibility, StackConstraints,
    StackPlan, StackTileChannelMap,
};
pub use tr_switch::{
    tr_adequate, tr_area_saving_mm2, tr_clamp_dissipation_w, tr_noise_figure_db,
    tr_switch_profiles, TrSwitchConfig,
};
pub use units::{Amp, Celsius, Coulomb, Farad, Henry, Hz, Kelvin, Nm, Ohm, Volt, Watt};
pub use validate::{
    check_transmission_line_lengths, core_checks, group_skew_mm, manifest_to_kwavers_beam_step,
    microvia_aspect_check, min_hv_spacing_mm, net_length_mm, transmission_line_threshold_mm,
    validate_against_budget, via_census, worst_ampacity_margin_mm, Check, KwaversBeamStep,
    KwaversBeamValidation, PhysicsReport, TransmissionLineViolation, ViaCensus,
};
pub use verify::{
    assembly, bom, decoupling_proximity, erc, keepin, lvs, parasitic_ac_coupling_check,
    schematic_isolation_bfs, verify_all, AcCouplingReport, AcCouplingViolation, AssemblyReport,
    BomReport, DecouplingReport, ErcReport, IsolationReport, IsolationViolation, KeepinReport,
    LvsReport, Verification,
};
