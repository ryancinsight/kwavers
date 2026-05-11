//! Topological dependency resolution for plugin execution ordering.

use super::PluginManager;
use crate::core::error::{KwaversResult, ValidationError};
use crate::domain::field::mapping::UnifiedFieldType;
use std::collections::{HashMap, HashSet};

impl PluginManager {
    /// Resolve plugin dependencies and compute a topological execution order.
    ///
    /// Uses DFS-based topological sort with cycle detection.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if a duplicate field provider or circular dependency is detected.
    ///
    pub(super) fn resolve_dependencies(&mut self) -> KwaversResult<()> {
        let n = self.plugins.len();
        if n == 0 {
            self.execution_order.clear();
            return Ok(());
        }

        let mut provides: HashMap<UnifiedFieldType, usize> = HashMap::new();
        let mut requires: Vec<HashSet<UnifiedFieldType>> = Vec::with_capacity(n);

        for (i, plugin) in self.plugins.iter().enumerate() {
            for field in plugin.provided_fields() {
                if let Some(&other) = provides.get(&field) {
                    return Err(ValidationError::FieldValidation {
                        field: "plugin_dependencies".to_owned(),
                        value: format!("{field:?}"),
                        constraint: format!(
                            "Field {:?} provided by multiple plugins: {} and {}",
                            field,
                            self.plugins[other].metadata().id,
                            plugin.metadata().id
                        ),
                    }
                    .into());
                }
                provides.insert(field, i);
            }

            let deps: HashSet<UnifiedFieldType> = plugin.required_fields().into_iter().collect();
            requires.push(deps);
        }

        let mut order = Vec::new();
        let mut state = vec![0u8; n];

        fn visit(
            node: usize,
            state: &mut [u8],
            requires: &[HashSet<UnifiedFieldType>],
            provides: &HashMap<UnifiedFieldType, usize>,
            order: &mut Vec<usize>,
        ) -> Result<(), String> {
            if state[node] == 2 {
                return Ok(());
            }
            if state[node] == 1 {
                return Err(format!("Circular dependency detected at plugin {node}"));
            }
            state[node] = 1;
            for field in &requires[node] {
                if let Some(&dep) = provides.get(field) {
                    if dep != node {
                        visit(dep, state, requires, provides, order)?;
                    }
                }
            }
            state[node] = 2;
            order.push(node);
            Ok(())
        }

        for i in 0..n {
            if state[i] == 0 {
                visit(i, &mut state, &requires, &provides, &mut order).map_err(|msg| {
                    ValidationError::FieldValidation {
                        field: "plugin_dependencies".to_owned(),
                        value: "invalid".to_owned(),
                        constraint: msg,
                    }
                })?;
            }
        }

        self.execution_order = order;
        Ok(())
    }
}
