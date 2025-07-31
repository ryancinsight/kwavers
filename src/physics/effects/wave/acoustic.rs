// src/physics/effects/wave/acoustic.rs
//! Acoustic wave effect implementation

use crate::error::KwaversResult;
use crate::physics::core::{PhysicsEffect, EffectCategory, EffectId, EffectContext};
use crate::physics::composable::FieldType;
use crate::physics::state::PhysicsState;

/// Acoustic wave effect
#[derive(Debug)]
pub struct AcousticWaveEffect {
    id: EffectId,
}

impl AcousticWaveEffect {
    pub fn new() -> Self {
        Self {
            id: EffectId::from("acoustic_wave"),
        }
    }
}

impl PhysicsEffect for AcousticWaveEffect {
    fn id(&self) -> &EffectId {
        &self.id
    }
    
    fn category(&self) -> EffectCategory {
        EffectCategory::Wave
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure, FieldType::Velocity]
    }
    
    fn update(&mut self, _state: &mut PhysicsState, _context: &EffectContext) -> KwaversResult<()> {
        // TODO: Implement acoustic wave physics
        Ok(())
    }
}