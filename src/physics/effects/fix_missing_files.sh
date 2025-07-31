#!/bin/bash

# Create missing effect files

# Wave effects
mkdir -p src/physics/effects/wave
cat > src/physics/effects/wave/acoustic.rs << 'EOF'
use crate::error::KwaversResult;
use crate::physics::core::{PhysicsEffect, EffectCategory, EffectId, EffectContext};
use crate::physics::composable::FieldType;
use crate::physics::state::PhysicsState;

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
        Ok(())
    }
}
EOF

cat > src/physics/effects/wave/elastic.rs << 'EOF'
use crate::error::KwaversResult;
use crate::physics::core::{PhysicsEffect, EffectCategory, EffectId, EffectContext};
use crate::physics::composable::FieldType;
use crate::physics::state::PhysicsState;

#[derive(Debug)]
pub struct ElasticWaveEffect {
    id: EffectId,
}

impl ElasticWaveEffect {
    pub fn new() -> Self {
        Self {
            id: EffectId::from("elastic_wave"),
        }
    }
}

impl PhysicsEffect for ElasticWaveEffect {
    fn id(&self) -> &EffectId {
        &self.id
    }
    
    fn category(&self) -> EffectCategory {
        EffectCategory::Wave
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Stress]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Stress, FieldType::Velocity]
    }
    
    fn update(&mut self, _state: &mut PhysicsState, _context: &EffectContext) -> KwaversResult<()> {
        Ok(())
    }
}
EOF

echo "Wave effects created"
