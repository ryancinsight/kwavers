// recorder/events.rs - Event structures for recording

use crate::domain::sensor::sonoluminescence::SonoluminescenceEvent;

/// Event representing cavitation activity
#[derive(Debug, Clone)]
pub struct CavitationEvent {
    pub time_step: usize,
    pub time: f64,
    pub position: [usize; 3],
    pub pressure: f64,
    pub bubble_radius: f64,
}

/// Event representing significant thermal activity
#[derive(Debug, Clone)]
pub struct ThermalEvent {
    pub time_step: usize,
    pub time: f64,
    pub position: [usize; 3],
    pub temperature: f64,
}

/// Collection of all event types
#[derive(Debug, Clone, Default)]
pub struct EventCollection {
    pub cavitation_events: Vec<CavitationEvent>,
    pub thermal_events: Vec<ThermalEvent>,
    pub sonoluminescence_events: Vec<SonoluminescenceEvent>,
}
