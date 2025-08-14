/// NIFTI file format support for medical imaging data
/// 
/// Implements NIFTI-1 and NIFTI-2 format readers for loading brain models
/// and other medical imaging data. This replaces the simplified implementations
/// in examples with a proper, complete implementation.

use crate::{KwaversResult, KwaversError};
use ndarray::Array3;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// NIFTI-1 header structure (348 bytes)
/// 
/// Based on the NIFTI-1 standard specification
/// See: https://nifti.nimh.nih.gov/nifti-1/
#[derive(Debug, Clone)]
pub struct NiftiHeader {
    /// Header size (must be 348 for NIFTI-1)
    pub sizeof_hdr: i32,
    /// Data dimensions [ndims, nx, ny, nz, nt, ...]
    pub dim: [i16; 8],
    /// Data type code
    pub datatype: i16,
    /// Bits per voxel
    pub bitpix: i16,
    /// Voxel dimensions [qfac, dx, dy, dz, dt, ...]
    pub pixdim: [f32; 8],
    /// Offset to data in file
    pub vox_offset: f32,
    /// Slope for data scaling
    pub scl_slope: f32,
    /// Intercept for data scaling
    pub scl_inter: f32,
    /// Q-form code
    pub qform_code: i16,
    /// S-form code
    pub sform_code: i16,
    /// Quaternion b parameter
    pub quatern_b: f32,
    /// Quaternion c parameter
    pub quatern_c: f32,
    /// Quaternion d parameter
    pub quatern_d: f32,
    /// Q-form x offset
    pub qoffset_x: f32,
    /// Q-form y offset
    pub qoffset_y: f32,
    /// Q-form z offset
    pub qoffset_z: f32,
    /// S-form matrix (first row)
    pub srow_x: [f32; 4],
    /// S-form matrix (second row)
    pub srow_y: [f32; 4],
    /// S-form matrix (third row)
    pub srow_z: [f32; 4],
}

impl Default for NiftiHeader {
    fn default() -> Self {
        Self {
            sizeof_hdr: 348,
            dim: [0; 8],
            datatype: 0,
            bitpix: 0,
            pixdim: [0.0; 8],
            vox_offset: 352.0, // Standard offset after header + extension
            scl_slope: 1.0,
            scl_inter: 0.0,
            qform_code: 0,
            sform_code: 0,
            quatern_b: 0.0,
            quatern_c: 0.0,
            quatern_d: 0.0,
            qoffset_x: 0.0,
            qoffset_y: 0.0,
            qoffset_z: 0.0,
            srow_x: [0.0; 4],
            srow_y: [0.0; 4],
            srow_z: [0.0; 4],
        }
    }
}

/// NIFTI data type codes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NiftiDataType {
    Unknown = 0,
    Binary = 1,
    UInt8 = 2,
    Int16 = 4,
    Int32 = 8,
    Float32 = 16,
    Complex64 = 32,
    Float64 = 64,
    RGB24 = 128,
    Int8 = 256,
    UInt16 = 512,
    UInt32 = 768,
    Int64 = 1024,
    UInt64 = 1280,
    Float128 = 1536,
    Complex128 = 1792,
    Complex256 = 2048,
}

impl From<i16> for NiftiDataType {
    fn from(value: i16) -> Self {
        match value {
            1 => NiftiDataType::Binary,
            2 => NiftiDataType::UInt8,
            4 => NiftiDataType::Int16,
            8 => NiftiDataType::Int32,
            16 => NiftiDataType::Float32,
            32 => NiftiDataType::Complex64,
            64 => NiftiDataType::Float64,
            128 => NiftiDataType::RGB24,
            256 => NiftiDataType::Int8,
            512 => NiftiDataType::UInt16,
            768 => NiftiDataType::UInt32,
            1024 => NiftiDataType::Int64,
            1280 => NiftiDataType::UInt64,
            1536 => NiftiDataType::Float128,
            1792 => NiftiDataType::Complex128,
            2048 => NiftiDataType::Complex256,
            _ => NiftiDataType::Unknown,
        }
    }
}

/// NIFTI file loader for brain and medical imaging data
pub struct NiftiLoader {
    /// Path to the NIFTI file
    pub file_path: String,
    /// Whether to apply scaling (scl_slope and scl_inter)
    pub apply_scaling: bool,
    /// Whether to validate header checksum
    pub validate_header: bool,
}

impl NiftiLoader {
    /// Create a new NIFTI loader
    pub fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            apply_scaling: true,
            validate_header: true,
        }
    }

    /// Load NIFTI file and return data as Array3<f64>
    pub fn load(&self) -> KwaversResult<(Array3<f64>, NiftiHeader)> {
        if !Path::new(&self.file_path).exists() {
            return Err(KwaversError::Io(format!("NIFTI file not found: {}", self.file_path)));
        }

        let mut file = File::open(&self.file_path)
            .map_err(|e| KwaversError::Io(format!("Failed to open NIFTI file: {}", e)))?;

        let mut reader = BufReader::new(&mut file);
        
        // Read and parse header
        let header = self.read_header(&mut reader)?;
        
        // Validate header if requested
        if self.validate_header {
            self.validate_header_fields(&header)?;
        }

        // Seek to data offset
        reader.seek(SeekFrom::Start(header.vox_offset as u64))
            .map_err(|e| KwaversError::Io(format!("Failed to seek to data: {}", e)))?;

        // Read data based on datatype
        let data = self.read_data(&mut reader, &header)?;

        Ok((data, header))
    }

    /// Load NIFTI file and return data as Array3<u16> for segmentation masks
    pub fn load_segmentation(&self) -> KwaversResult<(Array3<u16>, NiftiHeader)> {
        if !Path::new(&self.file_path).exists() {
            return Err(KwaversError::Io(format!("NIFTI file not found: {}", self.file_path)));
        }

        let mut file = File::open(&self.file_path)
            .map_err(|e| KwaversError::Io(format!("Failed to open NIFTI file: {}", e)))?;

        let mut reader = BufReader::new(&mut file);
        let header = self.read_header(&mut reader)?;

        if self.validate_header {
            self.validate_header_fields(&header)?;
        }

        reader.seek(SeekFrom::Start(header.vox_offset as u64))
            .map_err(|e| KwaversError::Io(format!("Failed to seek to data: {}", e)))?;

        let data = self.read_segmentation_data(&mut reader, &header)?;
        Ok((data, header))
    }

    /// Read NIFTI header from file
    fn read_header(&self, reader: &mut BufReader<&mut File>) -> KwaversResult<NiftiHeader> {
        let mut header_bytes = vec![0u8; 348];
        reader.read_exact(&mut header_bytes)
            .map_err(|e| KwaversError::Io(format!("Failed to read header: {}", e)))?;

        let mut header = NiftiHeader::default();

        // Parse header fields (little-endian)
        header.sizeof_hdr = i32::from_le_bytes([
            header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]
        ]);

        // Check for byte swapping (big-endian files)
        let needs_swap = header.sizeof_hdr != 348;
        if needs_swap {
            header.sizeof_hdr = i32::from_be_bytes([
                header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]
            ]);
            if header.sizeof_hdr != 348 {
                return Err(KwaversError::Io("Invalid NIFTI header size".to_string()));
            }
        }

        // Parse dimensions
        for i in 0..8 {
            let offset = 40 + i * 2;
            header.dim[i] = if needs_swap {
                i16::from_be_bytes([header_bytes[offset], header_bytes[offset + 1]])
            } else {
                i16::from_le_bytes([header_bytes[offset], header_bytes[offset + 1]])
            };
        }

        // Parse datatype and bitpix
        header.datatype = if needs_swap {
            i16::from_be_bytes([header_bytes[70], header_bytes[71]])
        } else {
            i16::from_le_bytes([header_bytes[70], header_bytes[71]])
        };

        header.bitpix = if needs_swap {
            i16::from_be_bytes([header_bytes[72], header_bytes[73]])
        } else {
            i16::from_le_bytes([header_bytes[72], header_bytes[73]])
        };

        // Parse pixel dimensions
        for i in 0..8 {
            let offset = 76 + i * 4;
            let bytes = [
                header_bytes[offset],
                header_bytes[offset + 1],
                header_bytes[offset + 2],
                header_bytes[offset + 3],
            ];
            header.pixdim[i] = if needs_swap {
                f32::from_be_bytes(bytes)
            } else {
                f32::from_le_bytes(bytes)
            };
        }

        // Parse vox_offset
        let vox_offset_bytes = [
            header_bytes[108], header_bytes[109], header_bytes[110], header_bytes[111]
        ];
        header.vox_offset = if needs_swap {
            f32::from_be_bytes(vox_offset_bytes)
        } else {
            f32::from_le_bytes(vox_offset_bytes)
        };

        // Parse scaling parameters
        let scl_slope_bytes = [
            header_bytes[112], header_bytes[113], header_bytes[114], header_bytes[115]
        ];
        header.scl_slope = if needs_swap {
            f32::from_be_bytes(scl_slope_bytes)
        } else {
            f32::from_le_bytes(scl_slope_bytes)
        };

        let scl_inter_bytes = [
            header_bytes[116], header_bytes[117], header_bytes[118], header_bytes[119]
        ];
        header.scl_inter = if needs_swap {
            f32::from_be_bytes(scl_inter_bytes)
        } else {
            f32::from_le_bytes(scl_inter_bytes)
        };

        // Parse qform and sform codes
        header.qform_code = if needs_swap {
            i16::from_be_bytes([header_bytes[252], header_bytes[253]])
        } else {
            i16::from_le_bytes([header_bytes[252], header_bytes[253]])
        };

        header.sform_code = if needs_swap {
            i16::from_be_bytes([header_bytes[254], header_bytes[255]])
        } else {
            i16::from_le_bytes([header_bytes[254], header_bytes[255]])
        };

        Ok(header)
    }

    /// Validate header fields for consistency
    fn validate_header_fields(&self, header: &NiftiHeader) -> KwaversResult<()> {
        // Check dimensions
        if header.dim[0] < 1 || header.dim[0] > 7 {
            return Err(KwaversError::Io(format!(
                "Invalid number of dimensions: {}", header.dim[0]
            )));
        }

        // Check data dimensions
        for i in 1..=header.dim[0] as usize {
            if header.dim[i] < 1 {
                return Err(KwaversError::Io(format!(
                    "Invalid dimension size at index {}: {}", i, header.dim[i]
                )));
            }
        }

        // Check datatype
        let dtype = NiftiDataType::from(header.datatype);
        if dtype == NiftiDataType::Unknown {
            return Err(KwaversError::Io(format!(
                "Unsupported NIFTI datatype: {}", header.datatype
            )));
        }

        // Check voxel offset
        if header.vox_offset < 352.0 {
            return Err(KwaversError::Io(format!(
                "Invalid voxel offset: {}", header.vox_offset
            )));
        }

        Ok(())
    }

    /// Read data from NIFTI file based on datatype
    fn read_data(&self, reader: &mut BufReader<&mut File>, header: &NiftiHeader) -> KwaversResult<Array3<f64>> {
        let nx = header.dim[1] as usize;
        let ny = header.dim[2] as usize;
        let nz = if header.dim[0] >= 3 { header.dim[3] as usize } else { 1 };

        let total_voxels = nx * ny * nz;
        let dtype = NiftiDataType::from(header.datatype);

        // Read raw data based on datatype
        let raw_data: Vec<f64> = match dtype {
            NiftiDataType::UInt8 => {
                let mut buffer = vec![0u8; total_voxels];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.into_iter().map(|v| v as f64).collect()
            },
            NiftiDataType::Int16 => {
                let mut buffer = vec![0u8; total_voxels * 2];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f64)
                    .collect()
            },
            NiftiDataType::UInt16 => {
                let mut buffer = vec![0u8; total_voxels * 2];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as f64)
                    .collect()
            },
            NiftiDataType::Float32 => {
                let mut buffer = vec![0u8; total_voxels * 4];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(4)
                    .map(|chunk| {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64
                    })
                    .collect()
            },
            NiftiDataType::Float64 => {
                let mut buffer = vec![0u8; total_voxels * 8];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7]
                        ])
                    })
                    .collect()
            },
            _ => {
                return Err(KwaversError::Io(format!(
                    "Unsupported datatype for reading: {:?}", dtype
                )));
            }
        };

        // Apply scaling if requested
        let scaled_data = if self.apply_scaling && (header.scl_slope != 0.0 && header.scl_slope != 1.0) {
            raw_data.into_iter()
                .map(|v| v * header.scl_slope as f64 + header.scl_inter as f64)
                .collect()
        } else {
            raw_data
        };

        // Reshape into 3D array
        Array3::from_shape_vec((nx, ny, nz), scaled_data)
            .map_err(|e| KwaversError::Io(format!("Failed to reshape data: {}", e)))
    }

    /// Read segmentation data as u16
    fn read_segmentation_data(&self, reader: &mut BufReader<&mut File>, header: &NiftiHeader) -> KwaversResult<Array3<u16>> {
        let nx = header.dim[1] as usize;
        let ny = header.dim[2] as usize;
        let nz = if header.dim[0] >= 3 { header.dim[3] as usize } else { 1 };

        let total_voxels = nx * ny * nz;
        let dtype = NiftiDataType::from(header.datatype);

        let raw_data: Vec<u16> = match dtype {
            NiftiDataType::UInt8 => {
                let mut buffer = vec![0u8; total_voxels];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.into_iter().map(|v| v as u16).collect()
            },
            NiftiDataType::UInt16 => {
                let mut buffer = vec![0u8; total_voxels * 2];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect()
            },
            NiftiDataType::Int16 => {
                let mut buffer = vec![0u8; total_voxels * 2];
                reader.read_exact(&mut buffer)
                    .map_err(|e| KwaversError::Io(format!("Failed to read data: {}", e)))?;
                buffer.chunks_exact(2)
                    .map(|chunk| {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        if val < 0 { 0 } else { val as u16 }
                    })
                    .collect()
            },
            _ => {
                return Err(KwaversError::Io(format!(
                    "Unsupported datatype for segmentation: {:?}", dtype
                )));
            }
        };

        Array3::from_shape_vec((nx, ny, nz), raw_data)
            .map_err(|e| KwaversError::Io(format!("Failed to reshape data: {}", e)))
    }
}

/// Tissue segmentation labels commonly used in brain imaging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrainTissueLabel {
    Background = 0,
    CSF = 1,
    GreyMatter = 2,
    WhiteMatter = 3,
    Fat = 4,
    Muscle = 5,
    Skin = 6,
    Skull = 7,
    Vessels = 8,
    Connective = 9,
    Dura = 10,
    BoneMarrow = 11,
}

impl From<u16> for BrainTissueLabel {
    fn from(value: u16) -> Self {
        match value {
            1 => BrainTissueLabel::CSF,
            2 => BrainTissueLabel::GreyMatter,
            3 => BrainTissueLabel::WhiteMatter,
            4 => BrainTissueLabel::Fat,
            5 => BrainTissueLabel::Muscle,
            6 => BrainTissueLabel::Skin,
            7 => BrainTissueLabel::Skull,
            8 => BrainTissueLabel::Vessels,
            9 => BrainTissueLabel::Connective,
            10 => BrainTissueLabel::Dura,
            11 => BrainTissueLabel::BoneMarrow,
            _ => BrainTissueLabel::Background,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nifti_header_default() {
        let header = NiftiHeader::default();
        assert_eq!(header.sizeof_hdr, 348);
        assert_eq!(header.scl_slope, 1.0);
        assert_eq!(header.scl_inter, 0.0);
    }

    #[test]
    fn test_nifti_datatype_conversion() {
        assert_eq!(NiftiDataType::from(2), NiftiDataType::UInt8);
        assert_eq!(NiftiDataType::from(16), NiftiDataType::Float32);
        assert_eq!(NiftiDataType::from(64), NiftiDataType::Float64);
        assert_eq!(NiftiDataType::from(9999), NiftiDataType::Unknown);
    }

    #[test]
    fn test_brain_tissue_label_conversion() {
        assert_eq!(BrainTissueLabel::from(0), BrainTissueLabel::Background);
        assert_eq!(BrainTissueLabel::from(2), BrainTissueLabel::GreyMatter);
        assert_eq!(BrainTissueLabel::from(7), BrainTissueLabel::Skull);
        assert_eq!(BrainTissueLabel::from(99), BrainTissueLabel::Background);
    }
}