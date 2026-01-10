#!/bin/bash
# Phase 2: Math Extraction Import Update Script
# Updates all imports from domain::math to new locations:
# - domain::math::{fft,geometry,linear_algebra,numerics} → math::*
# - domain::math::ml → analysis::ml

set -e

echo "=== Phase 2: Math Import Update Script ==="
echo "Updating imports from domain::math to math:: and analysis::ml"

# Find all Rust source files
find src -name "*.rs" -type f | while read -r file; do
    # Skip the compatibility facade itself
    if [[ "$file" == "src/domain/math/mod.rs" ]]; then
        continue
    fi

    # Update pure math module imports
    sed -i 's/use crate::domain::math::fft/use crate::math::fft/g' "$file"
    sed -i 's/use crate::domain::math::geometry/use crate::math::geometry/g' "$file"
    sed -i 's/use crate::domain::math::linear_algebra/use crate::math::linear_algebra/g' "$file"
    sed -i 's/use crate::domain::math::numerics/use crate::math::numerics/g' "$file"

    # Update ML imports to analysis layer
    sed -i 's/use crate::domain::math::ml/use crate::analysis::ml/g' "$file"
    sed -i 's/crate::domain::math::ml::/crate::analysis::ml::/g' "$file"

    # Handle pub use re-exports
    sed -i 's/pub use crate::domain::math::fft/pub use crate::math::fft/g' "$file"
    sed -i 's/pub use crate::domain::math::geometry/pub use crate::math::geometry/g' "$file"
    sed -i 's/pub use crate::domain::math::linear_algebra/pub use crate::math::linear_algebra/g' "$file"
    sed -i 's/pub use crate::domain::math::numerics/pub use crate::math::numerics/g' "$file"
    sed -i 's/pub use crate::domain::math::ml/pub use crate::analysis::ml/g' "$file"

    # Handle use statements with multiple paths
    sed -i 's/domain::math::fft/math::fft/g' "$file"
    sed -i 's/domain::math::geometry/math::geometry/g' "$file"
    sed -i 's/domain::math::linear_algebra/math::linear_algebra/g' "$file"
    sed -i 's/domain::math::numerics/math::numerics/g' "$file"
    sed -i 's/domain::math::ml/analysis::ml/g' "$file"
done

echo "✓ Import updates complete"
echo ""
echo "Files updated: $(find src -name "*.rs" -type f | wc -l)"
echo ""
echo "Next steps:"
echo "1. Run: cargo check --all-features"
echo "2. Run: cargo test --all-features"
echo "3. Review and commit changes"
