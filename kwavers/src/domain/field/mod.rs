pub mod bubble;
pub mod electromagnetic;
pub mod r#type;

pub use bubble::BubbleStateFields;
pub use electromagnetic::{EMFields, PoyntingVector};
pub use r#type::UnifiedFieldType;
pub mod operations;
pub use operations::{FieldOperations, FieldStatistics};
pub mod indices;
pub mod mapping;
pub mod wave;
