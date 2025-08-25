#!/bin/bash
# Fix chemistry function
sed -i '189a\        Ok(())' src/physics/chemistry/ros_plasma/plasma_reactions.rs
sed -i 's/chemistry.initialize_concentrations();/let _ = chemistry.initialize_concentrations();/' src/physics/chemistry/ros_plasma/plasma_reactions.rs

# Fix state.rs deref
sed -i 's|// Cannot implement - use view() method|unimplemented!("Use view() instead")|' src/physics/state.rs
sed -i 's|// Cannot implement - use view_mut() method|unimplemented!("Use view_mut() instead")|' src/physics/state.rs

# Fix solver Arc deref
sed -i 's|&\*\*self.medium,  // Deref Arc<dyn Medium> to &dyn Medium|self.medium.as_ref(),  // Get &dyn Medium from Arc|' src/solver/plugin_based/solver.rs

echo "Fixes applied"
