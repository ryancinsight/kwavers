//! Infrastructure Layer - Hardware Abstraction and System Services
//!
//! This layer provides foundational system services and hardware abstraction
//! necessary for production deployment:
//!
//! - **Device Management**: Hardware abstraction for ultrasound transducers
//! - **Data Persistence**: Patient records and treatment histories
//! - **System Monitoring**: Performance metrics and diagnostic logging
//! - **Configuration**: System-wide settings and device profiles
//!
//! ## Layer Position
//!
//! ```
//! Application Layer (Clinical workflows, user interfaces)
//!         ↓
//! Infrastructure Layer (Hardware abstraction, data storage)
//!         ↓
//! Clinical Layer (Safety, reconstruction, therapy)
//!         ↓
//! Domain Layer (Abstractions, business rules)
//!         ↓
//! Solver Layer (Algorithms, numerical methods)
//!         ↓
//! Core Layer (Types, errors, utilities)
//! ```
//!
//! ## Architecture Principles
//!
//! - **Hardware Abstraction**: Decouple application from hardware details
//! - **Scalability**: Support multiple concurrent devices and operations
//! - **Reliability**: Graceful degradation and error recovery
//! - **Traceability**: Complete audit trail for regulatory compliance
//! - **Performance**: Minimal overhead for real-time operations

pub mod device;
