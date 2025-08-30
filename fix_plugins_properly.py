#!/usr/bin/env python3
import re
import sys

def fix_plugin_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the Plugin trait impl block
    plugin_impl_pattern = r'impl\s+(?:crate::physics::plugin::)?Plugin\s+for\s+\w+\s*\{'
    match = re.search(plugin_impl_pattern, content)
    
    if not match:
        print(f"No Plugin impl found in {filepath}")
        return False
    
    # Check if the required methods are already in the Plugin impl
    if 'fn set_state(' in content[match.end():] and \
       'fn as_any(' in content[match.end():] and \
       'fn as_any_mut(' in content[match.end():]:
        print(f"{filepath} already has all methods")
        return True
    
    # Find the closing brace of the Plugin impl block
    impl_start = match.end()
    brace_count = 1
    pos = impl_start
    
    while brace_count > 0 and pos < len(content):
        if content[pos] == '{':
            brace_count += 1
        elif content[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        print(f"Could not find end of Plugin impl in {filepath}")
        return False
    
    impl_end = pos - 1  # Position of the closing brace
    
    # Check what methods are missing
    impl_content = content[impl_start:impl_end]
    methods_to_add = []
    
    if 'fn set_state(' not in impl_content:
        methods_to_add.append("""
    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }""")
    
    if 'fn as_any(' not in impl_content:
        methods_to_add.append("""
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }""")
    
    if 'fn as_any_mut(' not in impl_content:
        methods_to_add.append("""
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }""")
    
    if methods_to_add:
        # Insert the methods before the closing brace
        new_content = content[:impl_end] + ''.join(methods_to_add) + '\n' + content[impl_end:]
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"Added {len(methods_to_add)} methods to {filepath}")
        return True
    
    return True

# Files to fix
files = [
    '/workspace/src/solver/hybrid/plugin.rs',
    '/workspace/src/solver/pstd/plugin.rs',
    '/workspace/src/solver/pstd_implementation.rs',
    '/workspace/src/solver/thermal_diffusion/mod.rs'
]

for f in files:
    fix_plugin_file(f)