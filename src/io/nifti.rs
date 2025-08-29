/// NIFTI file format support for medical imaging data
///
/// Implements NIFTI-1 and NIFTI-2 format readers using the mature `nifti` crate
/// for robust, correct handling of endianness, data types, and file formats.
use crate::{KwaversError, KwaversResult};
use ndarray::Array3;
use nifti::volume::NiftiVolume;
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
    /// 
    /// This method delegates to the nifti crate's built-in conversion,
    /// which properly handles:
    /// - Endianness conversion
    /// - Data type casting
    /// - Scaling factors (scl_slope, scl_inter)
    /// - All NIFTI data types (not just float32/float64)
    pub fn load<P: AsRef<Path>>(&self, path: P) -> KwaversResult<Array3<f64>> {
        let path = path.as_ref();

        if self.verbose {
            log::info!("Loading NIFTI file: {}", path.display());
        }

        // Load the NIFTI file
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to load NIFTI file: {}", e)))?;

        // Get header for validation
        let header = nifti_object.header();
        let dims = header.dim;

        // Validate dimensions (NIFTI dim[0] is number of dimensions)
        if dims[0] < 3 {
            return Err(KwaversError::Io(format!(
                "NIFTI file must have at least 3 dimensions, got {}",
                dims[0]
            )));
        }

        // Convert to volume and use the nifti crate's built-in ndarray conversion
        let volume = nifti_object.into_volume();
        
        // Use the nifti crate's built-in conversion which handles:
        // - All data types (int8, int16, int32, float32, float64, etc.)
        // - Endianness conversion
        // - Scaling factors (scl_slope and scl_inter)
        // - Proper memory layout
        let array_3d = volume
            .to_ndarray::<f64>()
            .map_err(|e| KwaversError::Io(format!("Failed to convert NIFTI volume to array: {}", e)))?;

        if self.verbose {
            let shape = array_3d.shape();
            log::info!(
                "Successfully loaded NIFTI file with dimensions: {}x{}x{}",
                shape[0], shape[1], shape[2]
            );
        }

        Ok(array_3d)
    }

    /// Load NIFTI file with header information
    /// 
    /// This method now loads the file only once, extracting both
    /// the header and the volume data from a single read operation.
    pub fn load_with_header<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> KwaversResult<(Array3<f64>, NiftiHeader)> {
        let path = path.as_ref();

        if self.verbose {
            log::info!("Loading NIFTI file with header: {}", path.display());
        }

        // Load the NIFTI object ONCE
        let nifti_object = ReaderOptions::new()
            .read_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to load NIFTI file: {}", e)))?;

        // Get header before consuming the object for the volume
        let header = nifti_object.header().clone();

        // Validate dimensions
        let dims = header.dim;
        if dims[0] < 3 {
            return Err(KwaversError::Io(format!(
                "NIFTI file must have at least 3 dimensions, got {}",
                dims[0]
            )));
        }

        // Get the volume and convert it to an array
        let volume = nifti_object.into_volume();
        let array_3d = volume
            .to_ndarray::<f64>()
            .map_err(|e| KwaversError::Io(format!("Failed to convert NIFTI volume: {}", e)))?;

        if self.verbose {
            let shape = array_3d.shape();
            log::info!(
                "Successfully loaded NIFTI file with dimensions: {}x{}x{} and header",
                shape[0], shape[1], shape[2]
            );
        }

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
            .unwrap_or_else(|_| String::from("(invalid UTF-8)"));

        Ok(NiftiInfo {
            dimensions: [
                header.dim[1] as usize,
                header.dim[2] as usize,
                header.dim[3] as usize,
            ],
            voxel_dimensions: [header.pixdim[1], header.pixdim[2], header.pixdim[3]],
            datatype: header.datatype,
            description,
        })
    }

    /// Save a 3D array as a NIFTI file
    pub fn save<P: AsRef<Path>>(&self, path: P, data: &Array3<f64>) -> KwaversResult<()> {
        let path = path.as_ref();

        if self.verbose {
            log::info!("Saving NIFTI file: {}", path.display());
        }

        // Get dimensions
        let shape = data.shape();
        let nx = shape[0];
        let ny = shape[1];
        let nz = shape[2];

        // Create header with proper dimensions
        let mut header = NiftiHeader::default();
        header.dim[0] = 3; // 3D volume
        header.dim[1] = nx as u16;
        header.dim[2] = ny as u16;
        header.dim[3] = nz as u16;
        header.datatype = 64; // FLOAT64
        header.bitpix = 64; // 64 bits per voxel

        // Default voxel dimensions (can be customized)
        header.pixdim[1] = 1.0;
        header.pixdim[2] = 1.0;
        header.pixdim[3] = 1.0;

        // Convert data to raw bytes
        let mut raw_data = Vec::with_capacity(nx * ny * nz * 8);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let value = data[[i, j, k]];
                    raw_data.extend_from_slice(&value.to_ne_bytes());
                }
            }
        }

        // Create NIFTI object
        let nifti_object = InMemNiftiObject::from_raw_data(&header, raw_data)
            .map_err(|e| KwaversError::Io(format!("Failed to create NIFTI object: {}", e)))?;

        // Write to file
        nifti_object
            .write_file(path)
            .map_err(|e| KwaversError::Io(format!("Failed to write NIFTI file: {}", e)))?;

        if self.verbose {
            log::info!(
                "Successfully saved NIFTI file with dimensions: {}x{}x{}",
                nx, ny, nz
            );
        }

        Ok(())
    }
}

/// Basic information about a NIFTI file
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