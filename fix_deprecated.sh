#!/bin/bash
files=(
  "src/solver/pstd/mod.rs"
  "src/solver/hybrid/mod.rs"
  "src/sensor/passive_acoustic_mapping.rs"
  "src/solver/thermal_diffusion/mod.rs"
  "src/physics/plugin/acoustic_wave_plugin.rs"
)

for file in "${files[@]}"; do
  echo "Processing $file"
  # Remove clone_plugin, as_any, and as_any_mut methods
  sed -i '/fn clone_plugin/,/^    }/d' "$file"
  sed -i '/fn as_any(/,/^    }/d' "$file"
  sed -i '/fn as_any_mut/,/^    }/d' "$file"
done
