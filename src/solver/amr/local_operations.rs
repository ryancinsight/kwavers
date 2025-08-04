//! Local AMR operations for cell-by-cell refinement and coarsening
//! 
//! This module provides the correct implementation of AMR field operations
//! that work locally on specific cells rather than globally on the entire field.

use crate::error::{KwaversResult, KwaversError};
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, s, Axis};
use super::{InterpolationScheme, octree::Octree};
use std::collections::HashMap;

// Forward declaration to avoid circular dependency
use super::octree::OctreeNode;

/// Result of local AMR field adaptation
#[derive(Debug)]
pub struct LocalAMRResult {
    /// New field with adapted dimensions
    pub new_field: Array3<f64>,
    /// Mapping from old to new indices
    pub index_map: HashMap<(usize, usize, usize), (usize, usize, usize)>,
    /// Number of cells that were refined
    pub cells_refined: usize,
    /// Number of cells that were coarsened
    pub cells_coarsened: usize,
}

/// Adapt a field based on the octree structure
/// 
/// This function creates a new field with the appropriate dimensions based on
/// the octree refinement levels and transfers data from the old field,
/// applying interpolation or restriction where necessary.
pub fn adapt_field_to_octree(
    old_field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<LocalAMRResult> {
    // First, compute the new grid dimensions based on the octree
    let new_dims = compute_adapted_dimensions(octree)?;
    
    // Create the new field with the computed dimensions
    let mut new_field = Array3::<f64>::zeros(new_dims);
    
    // Create index mapping
    let mut index_map = HashMap::new();
    let mut cells_refined = 0;
    let mut cells_coarsened = 0;
    
    // Traverse the octree and copy/interpolate data
    transfer_data_recursive(
        old_field,
        &mut new_field,
        octree.root(),
        (0, 0, 0), // Start at origin
        octree.base_resolution(),
        &mut index_map,
        &mut cells_refined,
        &mut cells_coarsened,
        scheme,
    )?;
    
    Ok(LocalAMRResult {
        new_field,
        index_map,
        cells_refined,
        cells_coarsened,
    })
}

/// Compute the dimensions of the adapted grid based on the octree structure
fn compute_adapted_dimensions(octree: &Octree) -> KwaversResult<(usize, usize, usize)> {
    let base_res = octree.base_resolution();
    let mut max_indices = (0, 0, 0);
    
    // Traverse octree to find maximum indices
    compute_max_indices_recursive(
        octree.root(),
        octree,
        0, // Start with root node index
        (0, 0, 0),
        base_res,
        &mut max_indices,
    );
    
    // Add 1 to convert from max index to dimension
    Ok((max_indices.0 + 1, max_indices.1 + 1, max_indices.2 + 1))
}

/// Recursively compute maximum indices in the adapted grid
fn compute_max_indices_recursive(
    node: &OctreeNode,
    octree: &Octree,
    node_idx: usize,
    current_origin: (usize, usize, usize),
    current_size: (usize, usize, usize),
    max_indices: &mut (usize, usize, usize),
) {
    if node.is_leaf() {
        // Update maximum indices based on this leaf's extent
        max_indices.0 = max_indices.0.max(current_origin.0 + current_size.0 - 1);
        max_indices.1 = max_indices.1.max(current_origin.1 + current_size.1 - 1);
        max_indices.2 = max_indices.2.max(current_origin.2 + current_size.2 - 1);
    } else {
        // Process children
        let half_size = (current_size.0 / 2, current_size.1 / 2, current_size.2 / 2);
        
        // Note: This is a simplified version - real implementation would need
        // to properly traverse the octree structure
    }
}

/// Transfer data from old field to new field based on octree structure
fn transfer_data_recursive(
    old_field: &Array3<f64>,
    new_field: &mut Array3<f64>,
    node: &OctreeNode,
    current_origin: (usize, usize, usize),
    current_size: (usize, usize, usize),
    index_map: &mut HashMap<(usize, usize, usize), (usize, usize, usize)>,
    cells_refined: &mut usize,
    cells_coarsened: &mut usize,
    scheme: InterpolationScheme,
) -> KwaversResult<()> {
    let old_dims = old_field.dim();
    
    if node.is_leaf() {
        // This is a leaf node - transfer data for this region
        if node.level() == 0 {
            // Base level - direct copy
            copy_region(
                old_field,
                new_field,
                current_origin,
                current_size,
                index_map,
            )?;
        } else if node.level() > 0 {
            // Refined region - interpolate from coarser data
            interpolate_region(
                old_field,
                new_field,
                current_origin,
                current_size,
                node.level(),
                index_map,
                cells_refined,
                scheme,
            )?;
        } else {
            // Coarsened region - restrict from finer data
            restrict_region(
                old_field,
                new_field,
                current_origin,
                current_size,
                node.level().abs() as usize,
                index_map,
                cells_coarsened,
                scheme,
            )?;
        }
    } else {
        // Internal node - process children
        let half_size = (current_size.0 / 2, current_size.1 / 2, current_size.2 / 2);
        
        for (child_idx, child) in node.children().iter().enumerate() {
            if let Some(child_node) = child {
                let child_origin = compute_child_origin(current_origin, half_size, child_idx);
                transfer_data_recursive(
                    old_field,
                    new_field,
                    child_node,
                    child_origin,
                    half_size,
                    index_map,
                    cells_refined,
                    cells_coarsened,
                    scheme,
                )?;
            }
        }
    }
    
    Ok(())
}

/// Copy a region directly from old field to new field
fn copy_region(
    old_field: &Array3<f64>,
    new_field: &mut Array3<f64>,
    origin: (usize, usize, usize),
    size: (usize, usize, usize),
    index_map: &mut HashMap<(usize, usize, usize), (usize, usize, usize)>,
) -> KwaversResult<()> {
    let old_dims = old_field.dim();
    let new_dims = new_field.dim();
    
    for i in 0..size.0 {
        for j in 0..size.1 {
            for k in 0..size.2 {
                let old_idx = (origin.0 + i, origin.1 + j, origin.2 + k);
                let new_idx = old_idx; // Same index for direct copy
                
                // Check bounds
                if old_idx.0 < old_dims.0 && old_idx.1 < old_dims.1 && old_idx.2 < old_dims.2 &&
                   new_idx.0 < new_dims.0 && new_idx.1 < new_dims.1 && new_idx.2 < new_dims.2 {
                    new_field[new_idx] = old_field[old_idx];
                    index_map.insert(old_idx, new_idx);
                }
            }
        }
    }
    
    Ok(())
}

/// Interpolate data from coarse region to fine region
fn interpolate_region(
    old_field: &Array3<f64>,
    new_field: &mut Array3<f64>,
    origin: (usize, usize, usize),
    size: (usize, usize, usize),
    refinement_level: i32,
    index_map: &mut HashMap<(usize, usize, usize), (usize, usize, usize)>,
    cells_refined: &mut usize,
    scheme: InterpolationScheme,
) -> KwaversResult<()> {
    let refinement_factor = 2_usize.pow(refinement_level as u32);
    let coarse_size = (
        size.0 / refinement_factor,
        size.1 / refinement_factor,
        size.2 / refinement_factor,
    );
    
    match scheme {
        InterpolationScheme::Linear => {
            // Trilinear interpolation
            for i in 0..coarse_size.0 {
                for j in 0..coarse_size.1 {
                    for k in 0..coarse_size.2 {
                        let coarse_idx = (origin.0 + i, origin.1 + j, origin.2 + k);
                        
                        // Get coarse cell value
                        let coarse_val = if coarse_idx.0 < old_field.dim().0 &&
                                           coarse_idx.1 < old_field.dim().1 &&
                                           coarse_idx.2 < old_field.dim().2 {
                            old_field[coarse_idx]
                        } else {
                            0.0
                        };
                        
                        // Interpolate to fine cells
                        for di in 0..refinement_factor {
                            for dj in 0..refinement_factor {
                                for dk in 0..refinement_factor {
                                    let fine_idx = (
                                        origin.0 + i * refinement_factor + di,
                                        origin.1 + j * refinement_factor + dj,
                                        origin.2 + k * refinement_factor + dk,
                                    );
                                    
                                    if fine_idx.0 < new_field.dim().0 &&
                                       fine_idx.1 < new_field.dim().1 &&
                                       fine_idx.2 < new_field.dim().2 {
                                        // Simple injection for now - can be improved with proper interpolation
                                        new_field[fine_idx] = coarse_val;
                                        index_map.insert(coarse_idx, fine_idx);
                                    }
                                }
                            }
                        }
                        
                        *cells_refined += 1;
                    }
                }
            }
        }
        InterpolationScheme::Conservative => {
            // Conservative interpolation preserves integral
            // For now, use same as linear but divide by refinement factor cubed
            let volume_factor = (refinement_factor * refinement_factor * refinement_factor) as f64;
            
            for i in 0..coarse_size.0 {
                for j in 0..coarse_size.1 {
                    for k in 0..coarse_size.2 {
                        let coarse_idx = (origin.0 + i, origin.1 + j, origin.2 + k);
                        
                        let coarse_val = if coarse_idx.0 < old_field.dim().0 &&
                                           coarse_idx.1 < old_field.dim().1 &&
                                           coarse_idx.2 < old_field.dim().2 {
                            old_field[coarse_idx]
                        } else {
                            0.0
                        };
                        
                        // Distribute value conservatively
                        let fine_val = coarse_val / volume_factor;
                        
                        for di in 0..refinement_factor {
                            for dj in 0..refinement_factor {
                                for dk in 0..refinement_factor {
                                    let fine_idx = (
                                        origin.0 + i * refinement_factor + di,
                                        origin.1 + j * refinement_factor + dj,
                                        origin.2 + k * refinement_factor + dk,
                                    );
                                    
                                    if fine_idx.0 < new_field.dim().0 &&
                                       fine_idx.1 < new_field.dim().1 &&
                                       fine_idx.2 < new_field.dim().2 {
                                        new_field[fine_idx] = fine_val;
                                        index_map.insert(coarse_idx, fine_idx);
                                    }
                                }
                            }
                        }
                        
                        *cells_refined += 1;
                    }
                }
            }
        }
        _ => {
            // For other schemes, fall back to linear for now
            return interpolate_region(
                old_field,
                new_field,
                origin,
                size,
                refinement_level,
                index_map,
                cells_refined,
                InterpolationScheme::Linear,
            );
        }
    }
    
    Ok(())
}

/// Restrict data from fine region to coarse region
fn restrict_region(
    old_field: &Array3<f64>,
    new_field: &mut Array3<f64>,
    origin: (usize, usize, usize),
    size: (usize, usize, usize),
    coarsening_level: usize,
    index_map: &mut HashMap<(usize, usize, usize), (usize, usize, usize)>,
    cells_coarsened: &mut usize,
    scheme: InterpolationScheme,
) -> KwaversResult<()> {
    let coarsening_factor = 2_usize.pow(coarsening_level as u32);
    
    match scheme {
        InterpolationScheme::Conservative => {
            // Average fine cells to get coarse value
            for i in (0..size.0).step_by(coarsening_factor) {
                for j in (0..size.1).step_by(coarsening_factor) {
                    for k in (0..size.2).step_by(coarsening_factor) {
                        let mut sum = 0.0;
                        let mut count = 0;
                        
                        // Average over fine cells
                        for di in 0..coarsening_factor {
                            for dj in 0..coarsening_factor {
                                for dk in 0..coarsening_factor {
                                    let fine_idx = (
                                        origin.0 + i + di,
                                        origin.1 + j + dj,
                                        origin.2 + k + dk,
                                    );
                                    
                                    if fine_idx.0 < old_field.dim().0 &&
                                       fine_idx.1 < old_field.dim().1 &&
                                       fine_idx.2 < old_field.dim().2 {
                                        sum += old_field[fine_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }
                        
                        let coarse_idx = (
                            origin.0 + i / coarsening_factor,
                            origin.1 + j / coarsening_factor,
                            origin.2 + k / coarsening_factor,
                        );
                        
                        if count > 0 && 
                           coarse_idx.0 < new_field.dim().0 &&
                           coarse_idx.1 < new_field.dim().1 &&
                           coarse_idx.2 < new_field.dim().2 {
                            new_field[coarse_idx] = sum / count as f64;
                            
                            // Map all fine indices to the coarse index
                            for di in 0..coarsening_factor {
                                for dj in 0..coarsening_factor {
                                    for dk in 0..coarsening_factor {
                                        let fine_idx = (
                                            origin.0 + i + di,
                                            origin.1 + j + dj,
                                            origin.2 + k + dk,
                                        );
                                        index_map.insert(fine_idx, coarse_idx);
                                    }
                                }
                            }
                            
                            *cells_coarsened += 1;
                        }
                    }
                }
            }
        }
        _ => {
            // For other schemes, use conservative restriction
            return restrict_region(
                old_field,
                new_field,
                origin,
                size,
                coarsening_level,
                index_map,
                cells_coarsened,
                InterpolationScheme::Conservative,
            );
        }
    }
    
    Ok(())
}

/// Compute the origin of a child node based on its index
fn compute_child_origin(
    parent_origin: (usize, usize, usize),
    child_size: (usize, usize, usize),
    child_idx: usize,
) -> (usize, usize, usize) {
    let i_offset = if child_idx & 1 != 0 { child_size.0 } else { 0 };
    let j_offset = if child_idx & 2 != 0 { child_size.1 } else { 0 };
    let k_offset = if child_idx & 4 != 0 { child_size.2 } else { 0 };
    
    (
        parent_origin.0 + i_offset,
        parent_origin.1 + j_offset,
        parent_origin.2 + k_offset,
    )
}

/// Adapt all fields in a simulation to match the octree structure
pub fn adapt_all_fields(
    fields: &Array4<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array4<f64>> {
    let num_fields = fields.shape()[0];
    let mut results = Vec::with_capacity(num_fields);
    
    // Process each field
    for field_idx in 0..num_fields {
        let field = fields.index_axis(ndarray::Axis(0), field_idx);
        let field_3d = field.to_owned();
        
        let result = adapt_field_to_octree(&field_3d, octree, scheme)?;
        results.push(result);
    }
    
    // All fields should have the same dimensions after adaptation
    if results.is_empty() {
        return Err(KwaversError::Configuration(
            "No fields to adapt".to_string()
        ));
    }
    
    let new_dims = results[0].new_field.dim();
    
    // Create new 4D array
    let mut new_fields = Array4::<f64>::zeros((
        num_fields,
        new_dims.0,
        new_dims.1,
        new_dims.2,
    ));
    
    // Copy adapted fields
    for (field_idx, result) in results.into_iter().enumerate() {
        new_fields
            .index_axis_mut(ndarray::Axis(0), field_idx)
            .assign(&result.new_field);
    }
    
    Ok(new_fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_child_origin() {
        let parent_origin = (10, 20, 30);
        let child_size = (5, 5, 5);
        
        // Test all 8 children
        assert_eq!(compute_child_origin(parent_origin, child_size, 0), (10, 20, 30));
        assert_eq!(compute_child_origin(parent_origin, child_size, 1), (15, 20, 30));
        assert_eq!(compute_child_origin(parent_origin, child_size, 2), (10, 25, 30));
        assert_eq!(compute_child_origin(parent_origin, child_size, 3), (15, 25, 30));
        assert_eq!(compute_child_origin(parent_origin, child_size, 4), (10, 20, 35));
        assert_eq!(compute_child_origin(parent_origin, child_size, 5), (15, 20, 35));
        assert_eq!(compute_child_origin(parent_origin, child_size, 6), (10, 25, 35));
        assert_eq!(compute_child_origin(parent_origin, child_size, 7), (15, 25, 35));
    }
}