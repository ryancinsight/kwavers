use crate::error::DataError;
/// NIFTI file format support for medical imaging data
///
/// Implements NIFTI-1 and NIFTI-2 format readers using the `nifti` crate
/// for robust handling of medical imaging data formats.
use crate::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::path::Path;

// Re-export the nifti crate types for convenience
pub use nifti::{InMemNiftiObject, NiftiHeader, NiftiObject, ReaderOptions};

/// NIFTI file reader with proper endianness and format handling
#[derive(Debug)]
pub struct NiftiReader {
    /// Enable verbose logging
    verbose: bool,
}

impl Default for NiftiReader {
    fn default() -> Self {
        Self::new()
    }
}

impl NiftiReader {
    /// Create a new NIFTI reader
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Enable verbose logging during file operations
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Load NIFTI file as 3D array
    pub fn load<P: AsRef<Path>>(&self, path: P) -> KwaversResult<Array3<f64>> {
        let path = path.as_ref();

        if self.verbose {
            log::info!("Loading NIFTI file: {}", path.display());
        }

        // Load the NIFTI file
        let nifti_object = ReaderOptions::new().read_file(path).map_err(|e| {
            KwaversError::Data(DataError::IoError(format!(
                "Failed to load NIFTI file: {}",
                e
            )))
        })?;

        // Get header for dimensions
        let header = nifti_object.header();
        let dims = header.dim;
        let datatype = header.datatype;

        // Validate dimensions (NIFTI dim[0] is number of dimensions)
        if dims[0] < 3 {
            return Err(KwaversError::Data(DataError::IoError(format!(
                "NIFTI file must have at least 3 dimensions, got {}",
                dims[0]
            ))));
        }

        // Extract dimensions (dim[1], dim[2], dim[3] are x, y, z)
        let nx = dims[1] as usize;
        let ny = dims[2] as usize;
        let nz = dims[3] as usize;

        // Get the raw volume data
        let volume = nifti_object.into_volume();

        // Create output array
        let mut array_3d = Array3::zeros((nx, ny, nz));

        // Convert data based on header data type
        match datatype {
            16 => {
                // FLOAT32
                // Interpret raw data as f32
                let raw_data = volume.into_raw_data();
                let float_data: Vec<f32> = raw_data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                // Copy data into array
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let idx = i + j * nx + k * nx * ny;
                            if idx < float_data.len() {
                                array_3d[[i, j, k]] = float_data[idx] as f64;
                            }
                        }
                    }
                }
            }
            64 => {
                // FLOAT64
                // Interpret raw data as f64
                let raw_data = volume.into_raw_data();
                let float_data: Vec<f64> = raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let bytes = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        f64::from_ne_bytes(bytes)
                    })
                    .collect();

                // Copy data into array
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let idx = i + j * nx + k * nx * ny;
                            if idx < float_data.len() {
                                array_3d[[i, j, k]] = float_data[idx];
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(KwaversError::Data(DataError::IoError(format!(
                    "Unsupported NIFTI data type: {}. Only FLOAT32 (16) and FLOAT64 (64) are supported.",
                    datatype
                ))));
            }
        }

        if self.verbose {
            log::info!(
                "Successfully loaded NIFTI file with dimensions: {}x{}x{}",
                nx,
                ny,
                nz
            );
        }

        Ok(array_3d)
    }

    /// Load NIFTI file with header information
    pub fn load_with_header<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> KwaversResult<(Array3<f64>, NiftiHeader)> {
        let path = path.as_ref();

        if self.verbose {
            log::info!("Loading NIFTI file with header: {}", path.display());
        }

        // Load the NIFTI object
        let nifti_object = ReaderOptions::new().read_file(path).map_err(|e| {
            KwaversError::Data(DataError::IoError(format!(
                "Failed to load NIFTI file: {}",
                e
            )))
        })?;

        // Get header
        let header = nifti_object.header().clone();

        // Load the data using the same method
        let array_3d = self.load(path)?;

        Ok((array_3d, header))
    }

    /// Get basic information about a NIFTI file without loading the full volume
    pub fn get_info<P: AsRef<Path>>(&self, path: P) -> KwaversResult<NiftiInfo> {
        let path = path.as_ref();

        // Load only the header
        let nifti_object = ReaderOptions::new().read_file(path).map_err(|e| {
            KwaversError::Data(DataError::IoError(format!(
                "Failed to load NIFTI file: {}",
                e
            )))
        })?;

        let header = nifti_object.header();

        // Convert description from Vec<u8> to String
        let description = String::from_utf8(header.descrip.clone())
            .unwrap_or_else(|_| String::from("(invalid UTF-8)"));

        Ok(NiftiInfo {
            dimensions: [
                header.dim[1] as usize,
                header.dim[2] as usize,
                header.dim[3] as usize,
            ],
            voxel_dimensions: [
                header.pixdim[1] as f64,
                header.pixdim[2] as f64,
                header.pixdim[3] as f64,
            ],
            datatype: header.datatype,
            description,
        })
    }

    /// Save a 3D array as a NIFTI file (placeholder - full implementation requires API update)
    pub fn save<P: AsRef<Path>>(&self, _path: P, _data: &Array3<f64>) -> KwaversResult<()> {
        // Note: The nifti 0.17 API doesn't support direct creation of NIFTI objects
        // from raw data. This would require either:
        // 1. Updating to a newer version of the nifti crate
        // 2. Using a different approach to save NIFTI files
        // 3. Implementing raw NIFTI file writing

        Err(KwaversError::NotImplemented(
            "NIFTI saving is not yet implemented for nifti crate 0.17".to_string(),
        ))
    }
}

/// Information about a NIFTI file
#[derive(Debug, Clone)]
pub struct NiftiInfo {
    /// Dimensions of the volume [x, y, z]
    pub dimensions: [usize; 3],
    /// Voxel dimensions in mm [x, y, z]
    pub voxel_dimensions: [f64; 3],
    /// Data type code
    pub datatype: i16,
    /// Description field
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_nifti_reader_creation() {
        let reader = NiftiReader::new();
        assert!(!reader.verbose);

        let verbose_reader = NiftiReader::new().with_verbose(true);
        assert!(verbose_reader.verbose);
    }
}
