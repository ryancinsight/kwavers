//! Composite medium trait for full-featured media
//!
//! This module provides a composite trait that combines all medium properties,
//! serving as a migration path from the monolithic Medium trait.

use crate::grid::Grid;
use crate::medium::{
    absorption::TissueType,
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{TemperatureState, ThermalProperties},
    viscous::ViscousProperties,
};
use ndarray::Array3;
use std::fmt::Debug;

/// Composite trait combining all medium properties
///
/// This trait is provided for backward compatibility and convenience when
/// a medium needs to support all physical properties. New code should prefer
/// using specific trait bounds for better modularity.
pub trait CompositeMedium:
    CoreMedium
    + ArrayAccess
    + AcousticProperties
    + ElasticProperties
    + ElasticArrayAccess
    + ThermalProperties
    + TemperatureState
    + OpticalProperties
    + ViscousProperties
    + BubbleProperties
    + BubbleState
    + Debug
    + Sync
    + Send
{
}

/// Blanket implementation for any type that implements all the required traits
impl<T> CompositeMedium for T where
    T: CoreMedium
        + ArrayAccess
        + AcousticProperties
        + ElasticProperties
        + ElasticArrayAccess
        + ThermalProperties
        + TemperatureState
        + OpticalProperties
        + ViscousProperties
        + BubbleProperties
        + BubbleState
        + Debug
        + Sync
        + Send
{
}

/// Legacy Medium trait for backward compatibility
///
/// This trait maintains the same interface as the original monolithic Medium trait
/// but is now implemented in terms of the new modular traits. All methods are
/// provided through the component traits or as default implementations here.
pub trait Medium: CompositeMedium {
    // These methods are provided by CoreMedium
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as CoreMedium>::density(self, x, y, z, grid)
    }

    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as CoreMedium>::sound_speed(self, x, y, z, grid)
    }

    fn is_homogeneous(&self) -> bool {
        <Self as CoreMedium>::is_homogeneous(self)
    }

    fn reference_frequency(&self) -> f64 {
        <Self as CoreMedium>::reference_frequency(self)
    }

    // These methods are provided by ArrayAccess
    fn density_array(&self) -> &Array3<f64> {
        <Self as ArrayAccess>::density_array(self)
    }

    fn sound_speed_array(&self) -> &Array3<f64> {
        <Self as ArrayAccess>::sound_speed_array(self)
    }

    // These methods are provided by AcousticProperties
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        <Self as AcousticProperties>::absorption_coefficient(self, x, y, z, grid, frequency)
    }

    fn attenuation(&self, x: f64, y: f64, z: f64, frequency: f64, grid: &Grid) -> f64 {
        <Self as AcousticProperties>::attenuation(self, x, y, z, frequency, grid)
    }

    fn nonlinearity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as AcousticProperties>::nonlinearity_parameter(self, x, y, z, grid)
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as AcousticProperties>::nonlinearity_parameter(self, x, y, z, grid)
    }

    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as AcousticProperties>::nonlinearity_coefficient(self, x, y, z, grid)
    }

    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as AcousticProperties>::acoustic_diffusivity(self, x, y, z, grid)
    }

    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
        <Self as AcousticProperties>::tissue_type(self, x, y, z, grid)
    }

    // These methods are provided by ElasticProperties
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ElasticProperties>::lame_lambda(self, x, y, z, grid)
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ElasticProperties>::lame_mu(self, x, y, z, grid)
    }

    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ElasticProperties>::shear_wave_speed(self, x, y, z, grid)
    }

    fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ElasticProperties>::compressional_wave_speed(self, x, y, z, grid)
    }

    // These methods are provided by ElasticArrayAccess
    fn lame_lambda_array(&self) -> Array3<f64> {
        <Self as ElasticArrayAccess>::lame_lambda_array(self)
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        <Self as ElasticArrayAccess>::lame_mu_array(self)
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        <Self as ElasticArrayAccess>::shear_sound_speed_array(self)
    }

    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        <Self as ElasticArrayAccess>::shear_viscosity_coeff_array(self)
    }

    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        <Self as ElasticArrayAccess>::bulk_viscosity_coeff_array(self)
    }

    // These methods are provided by ThermalProperties
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::specific_heat(self, x, y, z, grid)
    }

    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::specific_heat_capacity(self, x, y, z, grid)
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::thermal_conductivity(self, x, y, z, grid)
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::thermal_diffusivity(self, x, y, z, grid)
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::thermal_expansion(self, x, y, z, grid)
    }

    fn specific_heat_ratio(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::specific_heat_ratio(self, x, y, z, grid)
    }

    fn gamma(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ThermalProperties>::gamma(self, x, y, z, grid)
    }

    // These methods are provided by TemperatureState
    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        <Self as TemperatureState>::update_temperature(self, temperature)
    }

    fn temperature(&self) -> &Array3<f64> {
        <Self as TemperatureState>::temperature(self)
    }

    // These methods are provided by OpticalProperties
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as OpticalProperties>::optical_absorption_coefficient(self, x, y, z, grid)
    }

    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as OpticalProperties>::optical_scattering_coefficient(self, x, y, z, grid)
    }

    // These methods are provided by ViscousProperties
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ViscousProperties>::viscosity(self, x, y, z, grid)
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ViscousProperties>::shear_viscosity(self, x, y, z, grid)
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as ViscousProperties>::bulk_viscosity(self, x, y, z, grid)
    }

    // These methods are provided by BubbleProperties
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as BubbleProperties>::surface_tension(self, x, y, z, grid)
    }

    fn ambient_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as BubbleProperties>::ambient_pressure(self, x, y, z, grid)
    }

    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as BubbleProperties>::vapor_pressure(self, x, y, z, grid)
    }

    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as BubbleProperties>::polytropic_index(self, x, y, z, grid)
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        <Self as BubbleProperties>::gas_diffusion_coefficient(self, x, y, z, grid)
    }

    // These methods are provided by BubbleState
    fn bubble_radius(&self) -> &Array3<f64> {
        <Self as BubbleState>::bubble_radius(self)
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        <Self as BubbleState>::bubble_velocity(self)
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        <Self as BubbleState>::update_bubble_state(self, radius, velocity)
    }
}

/// Blanket implementation for backward compatibility
impl<T: CompositeMedium> Medium for T {}