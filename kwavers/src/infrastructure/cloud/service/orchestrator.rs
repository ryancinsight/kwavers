//! `CloudPINNService` orchestrator for cloud PINN deployment management.
//!
//! # Architecture
//!
//! Follows the Application Layer pattern in Clean Architecture:
//! - Orchestrates use cases (deploy, scale, terminate).
//! - Delegates to provider-specific infrastructure implementations.
//! - Maintains deployment state and handles.
//! - Validates configurations before operations.
//!
//! # Design Patterns
//!
//! - **Facade**: Unified interface to multiple provider implementations.
//! - **Strategy**: Selects provider-specific strategy based on configuration.
//! - **Repository**: Manages deployment handles as domain entities.

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
#[cfg(feature = "pinn")]
use std::sync::Arc;

#[cfg(feature = "pinn")]
use super::super::config::DeploymentConfig;
use super::super::{
    types::{CloudProvider, DeploymentHandle, DeploymentStatus},
    utilities,
};

/// Cloud PINN service orchestrator for deployment lifecycle management.
///
/// Provides a unified interface for deploying, scaling, and terminating PINN models
/// across multiple cloud providers (AWS, GCP, Azure).
#[derive(Debug)]
pub struct CloudPINNService {
    /// Cloud provider for this service instance.
    pub(crate) provider: CloudProvider,
    /// Provider-specific configuration (credentials, regions, etc.).
    #[cfg(feature = "pinn")]
    config: HashMap<String, String>,
    /// Active deployment handles.
    deployments: HashMap<String, DeploymentHandle>,
}

impl CloudPINNService {
    /// Create a new cloud PINN service.
    ///
    /// Initializes the service with provider-specific configuration loaded
    /// from environment variables.
    ///
    /// # Errors
    /// Returns error if configuration loading fails.
    pub async fn new(provider: CloudProvider) -> KwaversResult<Self> {
        let config = utilities::load_provider_config(&provider).await?;
        #[cfg(not(feature = "pinn"))]
        let _ = config;

        Ok(Self {
            provider,
            #[cfg(feature = "pinn")]
            config,
            deployments: HashMap::new(),
        })
    }

    /// Return the cloud provider selected for this service instance.
    pub fn provider(&self) -> CloudProvider {
        self.provider
    }

    /// Deploy a PINN model to the cloud.
    ///
    /// Workflow:
    /// 1. Validates deployment configuration.
    /// 2. Serializes model for cloud storage.
    /// 3. Delegates to provider-specific deployment implementation.
    /// 4. Stores deployment handle for lifecycle management.
    #[cfg(feature = "pinn")]
    pub async fn deploy_model<B: burn::tensor::backend::AutodiffBackend>(
        &mut self,
        model: &crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
        deployment_config: DeploymentConfig,
    ) -> KwaversResult<Arc<DeploymentHandle>> {
        deployment_config.validate()?;
        self.validate_provider_match(&deployment_config)?;

        let model_data = utilities::serialize_model_for_deployment(model, &self.provider).await?;

        let handle = match self.provider {
            #[cfg(feature = "api")]
            CloudProvider::AWS => {
                super::super::providers::aws::deploy_to_aws(
                    model,
                    &self.config,
                    &deployment_config,
                    &model_data,
                )
                .await?
            }
            #[cfg(not(feature = "api"))]
            CloudProvider::AWS => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "AWS deployment".to_string(),
                        reason: "Requires 'api' feature flag".to_string(),
                    },
                ))
            }
            CloudProvider::GCP => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "GCP deployment".to_string(),
                        reason: "GCP provider removed per ADR-011".to_string(),
                    },
                ))
            }
            CloudProvider::Azure => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "Azure deployment".to_string(),
                        reason: "Azure provider removed per ADR-011".to_string(),
                    },
                ))
            }
        };

        let handle = Arc::new(handle);
        self.deployments
            .insert(handle.id.clone(), (*handle).clone());
        Ok(handle)
    }

    /// Retrieve the current status of a deployment.
    ///
    /// # Errors
    /// Returns `ResourceUnavailable` if the deployment ID is not found.
    pub async fn get_deployment_status(
        &self,
        deployment_id: &str,
    ) -> KwaversResult<DeploymentStatus> {
        match self.deployments.get(deployment_id) {
            Some(handle) => Ok(handle.status.clone()),
            None => Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("deployment {}", deployment_id),
                },
            )),
        }
    }

    /// Adjust the number of instances for an active deployment.
    ///
    /// # Errors
    /// Returns error if the deployment is not found, `target_instances` is zero,
    /// or the provider scaling operation fails.
    pub async fn scale_deployment(
        &mut self,
        deployment_id: &str,
        target_instances: usize,
    ) -> KwaversResult<()> {
        if !self.deployments.contains_key(deployment_id) {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("deployment {}", deployment_id),
                },
            ));
        }

        if target_instances == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "target_instances".to_string(),
                    reason: "Must specify at least 1 instance".to_string(),
                },
            ));
        }

        if let Some(handle) = self.deployments.get_mut(deployment_id) {
            handle.status = DeploymentStatus::Scaling;
        }

        #[cfg(feature = "pinn")]
        {
            let config = self.config.clone();
            if let Some(handle) = self.deployments.get_mut(deployment_id) {
                match self.provider {
                    #[cfg(feature = "api")]
                    CloudProvider::AWS => {
                        super::super::providers::aws::scale_aws_deployment(
                            &config,
                            handle,
                            target_instances,
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "api"))]
                    CloudProvider::AWS => {
                        return Err(KwaversError::System(
                            crate::core::error::SystemError::FeatureNotAvailable {
                                feature: "AWS scaling".to_string(),
                                reason: "Requires 'api' feature flag".to_string(),
                            },
                        ))
                    }
                    CloudProvider::GCP => {
                        return Err(KwaversError::System(
                            crate::core::error::SystemError::FeatureNotAvailable {
                                feature: "GCP scaling".to_string(),
                                reason: "GCP provider removed per ADR-011".to_string(),
                            },
                        ))
                    }
                    CloudProvider::Azure => {
                        return Err(KwaversError::System(
                            crate::core::error::SystemError::FeatureNotAvailable {
                                feature: "Azure scaling".to_string(),
                                reason: "Azure provider removed per ADR-011".to_string(),
                            },
                        ))
                    }
                }
            }
        }

        if let Some(handle) = self.deployments.get_mut(deployment_id) {
            handle.status = DeploymentStatus::Active;
        }

        Ok(())
    }

    /// Delete all cloud resources associated with a deployment.
    ///
    /// # Errors
    /// Returns error if the deployment is not found or provider termination fails.
    pub async fn terminate_deployment(&mut self, deployment_id: &str) -> KwaversResult<()> {
        let mut handle = self.deployments.remove(deployment_id).ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("deployment {}", deployment_id),
            })
        })?;

        handle.status = DeploymentStatus::Terminating;

        #[cfg(feature = "pinn")]
        match self.provider {
            #[cfg(feature = "api")]
            CloudProvider::AWS => {
                super::super::providers::aws::terminate_aws_deployment(&self.config, &handle)
                    .await?;
            }
            #[cfg(not(feature = "api"))]
            CloudProvider::AWS => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "AWS termination".to_string(),
                        reason: "Requires 'api' feature flag".to_string(),
                    },
                ))
            }
            CloudProvider::GCP => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "GCP termination".to_string(),
                        reason: "GCP provider removed per ADR-011".to_string(),
                    },
                ))
            }
            CloudProvider::Azure => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::FeatureNotAvailable {
                        feature: "Azure termination".to_string(),
                        reason: "Azure provider removed per ADR-011".to_string(),
                    },
                ))
            }
        }

        Ok(())
    }

    /// Validate that `config.provider` matches the service provider.
    #[cfg(feature = "pinn")]
    fn validate_provider_match(&self, config: &DeploymentConfig) -> KwaversResult<()> {
        if config.provider != self.provider {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "provider".to_string(),
                    reason: format!(
                        "Deployment config provider ({:?}) doesn't match service provider ({:?})",
                        config.provider, self.provider
                    ),
                },
            ));
        }
        Ok(())
    }

    /// Return the number of currently active deployments.
    pub fn deployment_count(&self) -> usize {
        self.deployments.len()
    }

    /// Return all deployment IDs managed by this service.
    pub fn deployment_ids(&self) -> Vec<String> {
        self.deployments.keys().cloned().collect()
    }
}
