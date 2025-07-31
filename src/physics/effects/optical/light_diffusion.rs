// src/physics/effects/optical/light_diffusion.rs
//! Light diffusion effect implementation

use crate::error::KwaversResult;
use crate::physics::core::{PhysicsEffect, EffectCategory, EffectId, EffectContext};
use crate::physics::composable::FieldType;
use crate::physics::state::PhysicsState;

/// Light diffusion effect
#[derive(Debug)]
pub struct LightDiffusionEffect {
    id: EffectId,
}

impl LightDiffusionEffect {
    pub fn new() -> Self {
        Self {
            id: EffectId::from("light_diffusion"),
        }
    }
}

impl PhysicsEffect for LightDiffusionEffect {
    fn id(&self) -> &EffectId {
        &self.id
    }
    
    fn category(&self) -> EffectCategory {
        EffectCategory::Optical
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Light]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Light]
    }
    
    fn update(&mut self, _state: &mut PhysicsState, _context: &EffectContext) -> KwaversResult<()> {
        // TODO: Implement light diffusion physics
        Ok(())
    }
}