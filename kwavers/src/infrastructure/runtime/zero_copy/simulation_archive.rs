use rkyv::{
    check_archived_root,
    ser::{serializers::AllocSerializer, Serializer},
    Archive, Deserialize, Infallible, Serialize,
};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

use super::grid_archive::SerializableGrid;

/// Archived simulation snapshot.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct SimulationData {
    pub time: f64,
    pub pressure: Vec<f64>,
    pub grid: SerializableGrid,
}

impl SimulationData {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(time: f64, pressure: Vec<f64>, grid: &Grid) -> Self {
        Self {
            time,
            pressure,
            grid: SerializableGrid::from(grid),
        }
    }
    /// To bytes.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn to_bytes(&self) -> KwaversResult<Vec<u8>> {
        let mut serializer = AllocSerializer::<4096>::default();
        serializer.serialize_value(self).map_err(|error| {
            KwaversError::InvalidInput(format!("Serialization failed: {error}"))
        })?;
        Ok(serializer.into_serializer().into_inner().to_vec())
    }
    /// Access.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn access(bytes: &[u8]) -> KwaversResult<&ArchivedSimulationData> {
        check_archived_root::<SimulationData>(bytes)
            .map_err(|error| KwaversError::InvalidInput(format!("Invalid archived data: {error}")))
    }
    /// From bytes.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_bytes(bytes: &[u8]) -> KwaversResult<Self> {
        let archived = Self::access(bytes)?;
        archived
            .deserialize(&mut Infallible)
            .map_err(|_| KwaversError::InvalidInput("Deserialization failed".to_string()))
    }
}
