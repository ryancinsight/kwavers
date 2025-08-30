#!/bin/bash

# Fix all Plugin implementations with missing methods

FILES=(
    "/workspace/src/sensor/passive_acoustic_mapping/plugin.rs"
    "/workspace/src/solver/fdtd/plugin.rs"
    "/workspace/src/solver/hybrid/plugin.rs"
    "/workspace/src/solver/pstd/plugin.rs"
    "/workspace/src/solver/pstd_implementation.rs"
    "/workspace/src/solver/thermal_diffusion/mod.rs"
)

for file in "${FILES[@]}"; do
    echo "Fixing $file"
    
    # Check if file has the missing methods already
    if ! grep -q "fn set_state" "$file"; then
        # Find the last closing brace of the impl block
        # Add the missing methods before it
        sed -i '/^impl.*Plugin.*{/,/^}$/ {
            /^}$/ i\
\
    fn set_state(&mut self, state: PluginState) {\
        self.state = state;\
    }\
\
    fn as_any(&self) -> &dyn std::any::Any {\
        self\
    }\
\
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {\
        self\
    }
        }' "$file"
    fi
done

echo "Done fixing Plugin implementations"