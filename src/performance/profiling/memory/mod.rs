//! Memory profiling infrastructure
//!
//! Tracks memory allocations, deallocations, and usage patterns.

use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Memory event type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryEventType {
    /// Memory allocation
    Allocation,
    /// Memory deallocation
    Deallocation,
    /// Memory reallocation
    Reallocation,
    /// Memory copy
    Copy,
}

/// Memory profiling event
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Allocation size in bytes
    pub size: usize,
    /// Event type
    pub event_type: MemoryEventType,
    /// Optional description
    pub description: Option<String>,
}

impl MemoryEvent {
    /// Create a new memory event
    #[must_use]
    pub fn new(size: usize, event_type: MemoryEventType) -> Self {
        Self {
            timestamp: Instant::now(),
            size,
            event_type,
            description: None,
        }
    }

    /// Create a memory event with description
    #[must_use]
    pub fn with_description(size: usize, event_type: MemoryEventType, description: String) -> Self {
        Self {
            timestamp: Instant::now(),
            size,
            event_type,
            description: Some(description),
        }
    }
}

/// Memory usage profile
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Total deallocations
    pub total_deallocations: usize,
    /// Allocation events
    pub events: Vec<MemoryEvent>,
}

impl MemoryProfile {
    /// Create a new memory profile
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            total_allocations: 0,
            total_deallocations: 0,
            events: Vec::new(),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.total_allocations += 1;
        self.events
            .push(MemoryEvent::new(size, MemoryEventType::Allocation));
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.total_deallocations += 1;
        self.events
            .push(MemoryEvent::new(size, MemoryEventType::Deallocation));
    }

    /// Get memory efficiency (deallocations / allocations)
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.total_allocations == 0 {
            1.0
        } else {
            self.total_deallocations as f64 / self.total_allocations as f64
        }
    }

    /// Get fragmentation estimate
    #[must_use]
    pub fn fragmentation(&self) -> f64 {
        if self.peak_usage == 0 {
            0.0
        } else {
            1.0 - (self.current_usage as f64 / self.peak_usage as f64)
        }
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory profiler for tracking memory usage
#[derive(Debug, Clone)]
pub struct MemoryProfiler {
    profile: Arc<Mutex<MemoryProfile>>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            profile: Arc::new(Mutex::new(MemoryProfile::new())),
        }
    }

    /// Record an allocation
    pub fn allocate(&self, size: usize) {
        if let Ok(mut profile) = self.profile.lock() {
            profile.record_allocation(size);
        }
    }

    /// Record a deallocation
    pub fn deallocate(&self, size: usize) {
        if let Ok(mut profile) = self.profile.lock() {
            profile.record_deallocation(size);
        }
    }

    /// Get current memory profile
    #[must_use]
    pub fn profile(&self) -> MemoryProfile {
        self.profile.lock().map(|p| p.clone()).unwrap_or_else(|_| MemoryProfile::new())
    }

    /// Clear all memory tracking data
    pub fn clear(&self) {
        if let Ok(mut profile) = self.profile.lock() {
            *profile = MemoryProfile::new();
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}
