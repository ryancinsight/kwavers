use rkyv::{
    check_archived_root,
    ser::{serializers::AllocSerializer, Serializer},
    Archive, Deserialize, Infallible, Serialize,
};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

/// Serializable grid data for archive transfer.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct SerializableGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl From<&Grid> for SerializableGrid {
    fn from(grid: &Grid) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            nz: grid.nz,
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
        }
    }
}

impl TryFrom<SerializableGrid> for Grid {
    type Error = KwaversError;

    fn try_from(value: SerializableGrid) -> Result<Self, Self::Error> {
        Grid::new(value.nx, value.ny, value.nz, value.dx, value.dy, value.dz).map_err(|error| {
            KwaversError::InvalidInput(format!("Failed to create grid: {error:?}"))
        })
    }
}

pub fn serialize_grid(grid: &Grid) -> KwaversResult<Vec<u8>> {
    let serializable = SerializableGrid::from(grid);
    let mut serializer = AllocSerializer::<256>::default();
    serializer
        .serialize_value(&serializable)
        .map_err(|error| KwaversError::InvalidInput(format!("Serialization failed: {error}")))?;
    Ok(serializer.into_serializer().into_inner().to_vec())
}

pub fn access_serializable_grid(bytes: &[u8]) -> KwaversResult<&ArchivedSerializableGrid> {
    check_archived_root::<SerializableGrid>(bytes)
        .map_err(|error| KwaversError::InvalidInput(format!("Invalid archived grid: {error}")))
}

pub fn deserialize_grid(bytes: &[u8]) -> KwaversResult<Grid> {
    let archived = access_serializable_grid(bytes)?;
    let deserialized: SerializableGrid = archived
        .deserialize(&mut Infallible)
        .map_err(|_| KwaversError::InvalidInput("Deserialization failed".to_string()))?;
    Grid::try_from(deserialized)
}
