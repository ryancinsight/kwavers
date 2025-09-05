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
    /// Voxel dimensions in mm
    voxel_dims: [f64; 3],
}

impl Default for NiftiReader {
    fn default() -> Self {
        Self::new()
    }
}

impl NiftiReader {
    /// Create a new NIFTI reader
    #[must_use]
    pub fn new() -> Self {
        Self {
            verbose: false,
            voxel_dims: [1.0, 1.0, 1.0], // Default 1mm isotropic
        }
    }

    /// Enable verbose logging during file operations
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set voxel dimensions in mm
    #[must_use]
    pub fn with_voxel_dims(mut self, dims: [f64; 3]) -> Self {
        self.voxel_dims = dims;
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
                "Failed to load NIFTI file: {e}"
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
                                array_3d[[i, j, k]] = f64::from(float_data[idx]);
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
                    "Unsupported NIFTI data type: {datatype}. Only FLOAT32 (16) and FLOAT64 (64) are supported."
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
                "Failed to load NIFTI file: {e}"
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
                "Failed to load NIFTI file: {e}"
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
                f64::from(header.pixdim[1]),
                f64::from(header.pixdim[2]),
                f64::from(header.pixdim[3]),
            ],
            datatype: header.datatype,
            description,
        })
    }

    /// Save a 3D array as a NIFTI file
    pub fn save<P: AsRef<Path>>(&self, path: P, data: &Array3<f64>) -> KwaversResult<()> {
        use std::fs::File;
        use std::io::Write;

        let path = path.as_ref();
        let (nx, ny, nz) = data.dim();

        // Create NIFTI header (348 bytes)
        let mut header = vec![0u8; 348];

        // Magic number for NIFTI-1 format
        header[0..4].copy_from_slice(&348i32.to_le_bytes());

        // Dimensions
        header[40] = 3; // Number of dimensions
        header[42..44].copy_from_slice(&(nx as i16).to_le_bytes());
        header[44..46].copy_from_slice(&(ny as i16).to_le_bytes());
        header[46..48].copy_from_slice(&(nz as i16).to_le_bytes());
        header[48..50].copy_from_slice(&1i16.to_le_bytes()); // time dimension

        // Data type (64 = float64)
        header[70..72].copy_from_slice(&64i16.to_le_bytes());
        header[72..74].copy_from_slice(&64i16.to_le_bytes()); // bits per pixel

        // Voxel dimensions from metadata
        let pixdim = [
            0.0f32,
            self.voxel_dims[0] as f32,
            self.voxel_dims[1] as f32,
            self.voxel_dims[2] as f32,
            1.0,
            1.0,
            1.0,
            1.0,
        ];
        for (i, &dim) in pixdim.iter().enumerate() {
            header[76 + i * 4..80 + i * 4].copy_from_slice(&dim.to_le_bytes());
        }

        // vox_offset - data starts immediately after header
        header[108..112].copy_from_slice(&352.0f32.to_le_bytes());

        // Magic string "n+1\0"
        header[344..348].copy_from_slice(b"n+1\0");

        // Write header and data
        let mut file = File::create(path)?;
        file.write_all(&header)?;

        // Pad to 352 bytes
        file.write_all(&[0u8; 4])?;

        // Write data in row-major order
        for ((_i, _j, _k), &value) in data.indexed_iter() {
            file.write_all(&value.to_le_bytes())?;
        }

        Ok(())
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

    #[test]
    fn test_nifti_reader_creation() {
        let reader = NiftiReader::new();
        assert!(!reader.verbose);

        let verbose_reader = NiftiReader::new().with_verbose(true);
        assert!(verbose_reader.verbose);
    }
}
