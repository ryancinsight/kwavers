#!/bin/bash
# Update all imports from domain::core:: to core::

echo "🔄 Updating core module imports..."

# Find all Rust files and update the import statements
find src -name "*.rs" -type f -exec sed -i 's/use crate::domain::core::/use crate::core::/g' {} \;

echo "✅ Updated all imports from domain::core:: to core::"
