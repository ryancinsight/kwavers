/// NIFTI file format support for medical imaging data
///
/// Implements NIFTI-1 and NIFTI-2 format readers using the mature `nifti` crate
/// for robust, correct handling of endianness, data types, and file formats.
use crate::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::path::Path;

// Re-export the nifti crate types for convenience
pub use nifti::{InMemNiftiObject, NiftiHeader, NiftiObject, ReaderOptions};

/// NIFTI file reader with proper endianness and format handling
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
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to load NIFTI file: {}", e)))?;

        // Get header for dimensions
        let header = nifti_object.header();
        let dims = header.dim;
        let datatype = header.datatype;

        // Validate dimensions (NIFTI dim[0] is number of dimensions)
        if dims[0] < 3 {
            return Err(KwaversError::Io(format!(
                "NIFTI file must have at least 3 dimensions, got {}",
                dims[0]
            )));
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
                return Err(KwaversError::Io(format!(
                    "Unsupported NIFTI data type: {}. Only FLOAT32 (16) and FLOAT64 (64) are supported.",
                    datatype
                )));
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
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to load NIFTI file: {}", e)))?;

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
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to load NIFTI file: {}", e)))?;

        let header = nifti_object.header();

        // Convert description from Vec<u8> to String
        let description = String::from_utf8(header.descrip.clone())
            .unwrap_or_else(|_| "Invalid UTF-8 description".to_string());

        Ok(NiftiInfo {
            dimensions: [
                header.dim[1] as usize,
                header.dim[2] as usize,
                if header.dim[0] >= 3 {
                    header.dim[3] as usize
                } else {
                    1
                },
            ],
            voxel_size: [header.pixdim[1], header.pixdim[2], header.pixdim[3]],
            data_type: header.datatype,
            description,
        })
    }
}

/// Basic information about a NIFTI file
#[derive(Debug, Clone)]
pub struct NiftiInfo {
    /// Volume dimensions [nx, ny, nz]
    pub dimensions: [usize; 3],
    /// Voxel size [dx, dy, dz] in mm
    pub voxel_size: [f32; 3],
    /// NIFTI data type code
    pub data_type: i16,
    /// File description
    pub description: String,
}

/// Convenience function to load a NIFTI file
pub fn load_nifti<P: AsRef<Path>>(path: P) -> KwaversResult<Array3<f64>> {
    NiftiReader::new().load(path)
}

/// Convenience function to load a NIFTI file with header
pub fn load_nifti_with_header<P: AsRef<Path>>(
    path: P,
) -> KwaversResult<(Array3<f64>, NiftiHeader)> {
    NiftiReader::new().load_with_header(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nifti_reader_creation() {
        let reader = NiftiReader::new();
        assert!(!reader.verbose);

        let reader_verbose = NiftiReader::new().with_verbose(true);
        assert!(reader_verbose.verbose);
    }
}
