//! `GpuPstdSolver` constructor and GPU pipeline initialisation.
//!
//! SRP split:
//! - `bgl`         — bind group layout builders (one per layout)
//! - `bind_groups` — bind group assembly from already-created buffers
//! - `construction`— `new()`: buffer allocation, shader, pipelines, Ok(Self)
//! - `auto_device` — `with_auto_device()`: adapter selection + delegates to `new()`

mod auto_device;
mod bgl;
mod bind_groups;
mod construction;
