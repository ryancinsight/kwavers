//! GPU PSTD time-marching loop and internal command-encoding helpers.
//!
//! Sub-module responsibilities (SRP boundaries):
//! - `buffer`   — packed source/sensor buffer helpers and run-cache management
//! - `dispatch` — low-level GPU dispatch helpers (single, absorb, 2-D, FFT/IFFT)
//! - `encode`   — per-step compute-pass encoding (velocity, source, density, pressure, record)
//! - `run`      — top-level `run` entry point: cache management, field zeroing, batch loop, sensor download

mod buffer;
mod dispatch;
mod encode;
mod run;
