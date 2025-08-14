//! Brain Data Loader Example for k-Wave BrainUltrasoundSimulation
//!
//! This example demonstrates loading brain model data using the proper
//! NIFTI loader implementation from the kwavers library.

use kwavers::{KwaversResult, io::{NiftiLoader, BrainTissueLabel}};
use ndarray::Array3;

/// Example of loading brain data using the library's NIFTI loader
fn load_brain_data_example() -> KwaversResult<()> {
    println!("=== Brain Data Loading Example ===");
    println!();
    
    // Example 1: Load brain model from NIFTI file
    let brain_model_path = "BrainUltrasoundSimulation/brain_model.nii";
    let loader = NiftiLoader::new(brain_model_path);
    
    println!("Attempting to load brain model from: {}", brain_model_path);
    
    // Try to load the actual file, or create synthetic data for demonstration
    match loader.load() {
        Ok((data, header)) => {
            println!("Successfully loaded NIFTI file!");
            println!("  Dimensions: {}x{}x{}", 
                header.dim[1], header.dim[2], header.dim[3]);
            println!("  Voxel size: {:.2}x{:.2}x{:.2} mm", 
                header.pixdim[1], header.pixdim[2], header.pixdim[3]);
            println!("  Data range: {:.2} to {:.2}", 
                data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        },
        Err(_) => {
            println!("NIFTI file not found. Creating synthetic brain model for demonstration...");
            let synthetic_model = create_synthetic_brain_model()?;
            println!("Created synthetic brain model with dimensions: {:?}", synthetic_model.dim());
            demonstrate_tissue_segmentation(&synthetic_model);
        }
    }
    
    // Example 2: Load segmentation mask
    let skull_model_path = "BrainUltrasoundSimulation/brain_model_skull.nii";
    let skull_loader = NiftiLoader::new(skull_model_path);
    
    println!("\nAttempting to load skull segmentation from: {}", skull_model_path);
    
    match skull_loader.load_segmentation() {
        Ok((segmentation, header)) => {
            println!("Successfully loaded segmentation!");
            println!("  Unique labels: {:?}", get_unique_labels(&segmentation));
        },
        Err(_) => {
            println!("Segmentation file not found. Would require actual NIFTI data.");
        }
    }
    
    Ok(())
}

/// Create a synthetic brain model for demonstration
fn create_synthetic_brain_model() -> KwaversResult<Array3<f64>> {
    // Create realistic brain dimensions (256x320x256 voxels)
    let nx = 256;
    let ny = 320;
    let nz = 256;
    
    let mut model = Array3::zeros((nx, ny, nz));
    
    // Create anatomically-inspired brain structure
    let center_x = nx / 2;
    let center_y = ny / 2;
    let center_z = nz / 2;
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 - center_x as f64;
                let y = j as f64 - center_y as f64;
                let z = k as f64 - center_z as f64;
                
                // Ellipsoidal brain shape
                let brain_radius_x = 100.0;
                let brain_radius_y = 120.0;
                let brain_radius_z = 100.0;
                
                let normalized_r = (x*x)/(brain_radius_x*brain_radius_x) + 
                                  (y*y)/(brain_radius_y*brain_radius_y) + 
                                  (z*z)/(brain_radius_z*brain_radius_z);
                
                if normalized_r <= 1.0 {
                    // Inside brain - assign tissue values
                    let r = normalized_r.sqrt();
                    
                    model[[i, j, k]] = if r > 0.95 {
                        // Skull
                        BrainTissueLabel::Skull as u16 as f64
                    } else if r > 0.90 {
                        // CSF
                        BrainTissueLabel::CSF as u16 as f64
                    } else if r > 0.7 {
                        // Grey matter
                        BrainTissueLabel::GreyMatter as u16 as f64
                    } else {
                        // White matter
                        BrainTissueLabel::WhiteMatter as u16 as f64
                    };
                }
            }
        }
    }
    
    Ok(model)
}

/// Demonstrate tissue segmentation
fn demonstrate_tissue_segmentation(model: &Array3<f64>) {
    println!("\nTissue Segmentation Analysis:");
    
    let mut tissue_counts = std::collections::HashMap::new();
    
    for &value in model.iter() {
        let label = BrainTissueLabel::from(value as u16);
        *tissue_counts.entry(label).or_insert(0) += 1;
    }
    
    for (tissue, count) in tissue_counts.iter() {
        if *count > 0 {
            let percentage = (*count as f64 / model.len() as f64) * 100.0;
            println!("  {:?}: {} voxels ({:.1}%)", tissue, count, percentage);
        }
    }
}

/// Get unique labels from segmentation
fn get_unique_labels(segmentation: &Array3<u16>) -> Vec<u16> {
    let mut labels = std::collections::HashSet::new();
    for &value in segmentation.iter() {
        labels.insert(value);
    }
    let mut sorted_labels: Vec<u16> = labels.into_iter().collect();
    sorted_labels.sort();
    sorted_labels
}

fn main() -> KwaversResult<()> {
    env_logger::init();
    
    println!("Brain Data Loader - Using Proper NIFTI Implementation");
    println!("======================================================");
    println!();
    
    load_brain_data_example()?;
    
    println!("\nNote: This example now uses the complete NIFTI loader implementation");
    println!("from src/io/nifti.rs which supports:");
    println!("  - NIFTI-1 format with proper header parsing");
    println!("  - Multiple data types (uint8, int16, float32, etc.)");
    println!("  - Scaling and transformation support");
    println!("  - Endianness handling");
    println!("  - Segmentation mask loading");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_synthetic_brain_model() {
        let model = create_synthetic_brain_model().unwrap();
        let (nx, ny, nz) = model.dim();
        
        assert_eq!(nx, 256);
        assert_eq!(ny, 320);
        assert_eq!(nz, 256);
        
        let mut tissue_counts = std::collections::HashMap::new();
        for &value in model.iter() {
            let label = BrainTissueLabel::from(value as u16);
            *tissue_counts.entry(label).or_insert(0) += 1;
        }
        
        assert!(tissue_counts.len() > 3); // Should have multiple tissue types
    }
    
    #[test]
    fn test_segmentation_loading() {
        let skull_model_path = "BrainUltrasoundSimulation/brain_model_skull.nii";
        let skull_loader = NiftiLoader::new(skull_model_path);
        
        match skull_loader.load_segmentation() {
            Ok((segmentation, _)) => {
                let unique_labels = get_unique_labels(&segmentation);
                assert!(unique_labels.contains(&BrainTissueLabel::Skull as u16));
                assert!(unique_labels.contains(&BrainTissueLabel::CSF as u16));
                assert!(unique_labels.contains(&BrainTissueLabel::GreyMatter as u16));
                assert!(unique_labels.contains(&BrainTissueLabel::WhiteMatter as u16));
            },
            Err(_) => {
                assert!(false, "Skull segmentation file not found for test");
            }
        }
    }
}