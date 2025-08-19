#!/bin/bash
echo "Fixing compilation errors..."

# Fix density_array return type
sed -i 's/fn density_array(&self) -> Array3<f64>/fn density_array(\&self) -> \&Array3<f64>/' /workspace/src/medium/heterogeneous/tissue.rs

# Fix sound_speed_array return type  
sed -i 's/fn sound_speed_array(&self) -> Array3<f64>/fn sound_speed_array(\&self) -> \&Array3<f64>/' /workspace/src/medium/heterogeneous/tissue.rs

# Remove .clone() calls
sed -i 's/}).clone()/})/' /workspace/src/medium/heterogeneous/tissue.rs

echo "Done"
