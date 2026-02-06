//! Infrastructure Layer - Hardware Abstraction and System Services
//!
//! This layer provides foundational system services and hardware abstraction
//! necessary for production deployment:
//!
//! - **Device Management**: Hardware abstraction for ultrasound transducers
//! - **Data Persistence**: Patient records and treatment histories
//! - **System Monitoring**: Performance metrics and diagnostic logging
//! - **Configuration**: System-wide settings and device profiles
//! - **API Services**: RESTful API for model training and inference
//! - **Cloud Integration**: AWS, Azure, GCP deployment and orchestration
//! - **I/O Operations**: DICOM integration and data formats
//!
//! ## Layer Position
//!
//! ```
//! Layer 8: Infrastructure (Top layer)
//!         ↓
//! Layer 7: Analysis (Post-processing)
//!         ↓
//! Layer 6: Clinical (Medical applications)
//!         ↓
//! Layer 5: Simulation (Orchestration)
//!         ↓
//! Layer 4: Solver (Algorithms)
//!         ↓
//! Layer 3: Physics (Implementation)
//!         ↓
//! Layer 2: Domain (SSOT)
//!         ↓
//! Layer 1: Math (Primitives)
//!         ↓
//! Layer 0: Core (Foundation)
//! ```
//!
//! ## Architecture Principles
//!
//! - **Hardware Abstraction**: Decouple application from hardware details
//! - **Scalability**: Support multiple concurrent devices and operations
//! - **Reliability**: Graceful degradation and error recovery
//! - **Traceability**: Complete audit trail for regulatory compliance
//! - **Performance**: Minimal overhead for real-time operations
//! - **Service Orientation**: RESTful APIs for enterprise integration

#[cfg(feature = "api")]
pub mod api;

#[cfg(feature = "cloud")]
pub mod cloud;

pub mod device;
pub mod io;
pub mod runtime;
