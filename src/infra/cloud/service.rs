//! Cloud PINN Service Orchestrator
//!
//! This module provides the main service orchestrator for cloud PINN deployments,
//! coordinating provider-specific operations and managing deployment state.
//!
//! # Architecture
//!
//! The service follows the Application Layer pattern in Clean Architecture:
//! - Orchestrates use cases (deploy, scale, terminate)
//! - Delegates to provider-specific infrastructure implementations
//! - Maintains deployment state and handles
//! - Validates configurations before operations
//!
//! # Design Patterns
//!
//! - **Facade**: Provides unified interface to multiple provider implementations
//! - **Strategy**: Selects provider-specific strategy based on configuration
//! - **Repository**: Manages deployment handles as domain entities
//!
//! # Literature References
//!
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166
//! - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software.
//!   Addison-Wesley. ISBN: 978-0321125217

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
#[cfg(feature = "pinn")]
use std::sync::Arc;

use super::{
    config::DeploymentConfig,
    types::{CloudProvider, DeploymentHandle, DeploymentStatus},
    utilities,
};

/// Cloud PINN service for deployment management
///
/// Provides a unified interface for deploying, scaling, and managing PINN models
/// across multiple cloud providers (AWS, GCP, Azure).
///
/// # Example
///
/// ```
/// use kwavers::infra::cloud::{CloudPINNService, CloudProvider};
///
/// # tokio_test::block_on(async {
/// let service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
/// # });
/// ```
#[derive(Debug)]
pub struct CloudPINNService {
    /// Cloud provider for this service instance
    provider: CloudProvider,
    /// Provider-specific configuration (credentials, regions, etc.)
    config: HashMap<String, String>,
    /// Active deployment handles
    deployments: HashMap<String, DeploymentHandle>,
}

impl CloudPINNService {
    /// Create a new cloud PINN service
    ///
    /// Initializes the service with provider-specific configuration loaded
    /// from environment variables.
    ///
    /// # Arguments
    ///
    /// - `provider`: Cloud provider to use (AWS, GCP, or Azure)
    ///
    /// # Returns
    ///
    /// Service instance ready for deployment operations
    ///
    /// # Errors
    ///
    /// Returns error if configuration loading fails
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::infra::cloud::{CloudPINNService, CloudProvider};
    ///
    /// # tokio_test::block_on(async {
    /// let aws_service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    /// let gcp_service = CloudPINNService::new(CloudProvider::GCP).await.unwrap();
    /// let azure_service = CloudPINNService::new(CloudProvider::Azure).await.unwrap();
    /// # });
    /// ```
    pub async fn new(provider: CloudProvider) -> KwaversResult<Self> {
        let config = utilities::load_provider_config(&provider).await?;

        Ok(Self {
            provider,
            config,
            deployments: HashMap::new(),
        })
    }

    /// Deploy a PINN model to the cloud
    ///
    /// Orchestrates the complete deployment workflow:
    /// 1. Validates deployment configuration
    /// 2. Serializes model for cloud storage
    /// 3. Delegates to provider-specific deployment implementation
    /// 4. Stores deployment handle for lifecycle management
    ///
    /// # Arguments
    ///
    /// - `model`: PINN model to deploy
    /// - `deployment_config`: Deployment configuration (instances, scaling, monitoring)
    ///
    /// # Returns
    ///
    /// Deployment handle with endpoint URL and initial metrics
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Configuration validation fails
    /// - Model serialization fails
    /// - Provider deployment fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::infra::cloud::{CloudPINNService, CloudProvider, DeploymentConfig};
    /// use kwavers::ml::pinn::BurnPINN2DWave;
    ///
    /// # #[cfg(feature = "pinn")]
    /// # async fn example<B: burn::tensor::backend::AutodiffBackend>(
    /// #     model: &BurnPINN2DWave<B>,
    /// #     config: DeploymentConfig,
    /// # ) {
    /// let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
    /// let handle = service.deploy_model(model, config).await.unwrap();
    /// println!("Deployed to: {}", handle.endpoint);
    /// # }
    /// ```
    #[cfg(feature = "pinn")]
    pub async fn deploy_model<B: burn::tensor::backend::AutodiffBackend>(
        &mut self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        deployment_config: DeploymentConfig,
    ) -> KwaversResult<Arc<DeploymentHandle>> {
        // Validate configuration
        deployment_config.validate()?;
        self.validate_provider_match(&deployment_config)?;

        // Serialize model for deployment
        let model_data = utilities::serialize_model_for_deployment(model, &self.provider).await?;

        // Create deployment based on provider
        let handle = match self.provider {
            #[cfg(feature = "api")]
            CloudProvider::AWS => {
                super::providers::aws::deploy_to_aws(
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
                super::providers::gcp::deploy_to_gcp(
                    model,
                    &self.config,
                    &deployment_config,
                    &model_data,
                )
                .await?
            }
            CloudProvider::Azure => {
                super::providers::azure::deploy_to_azure(
                    model,
                    &self.config,
                    &deployment_config,
                    &model_data,
                )
                .await?
            }
        };

        let handle = Arc::new(handle);
        self.deployments
            .insert(handle.id.clone(), (*handle).clone());

        Ok(handle)
    }

    /// Get deployment status
    ///
    /// Retrieves the current status of a deployment.
    ///
    /// # Arguments
    ///
    /// - `deployment_id`: Unique deployment identifier
    ///
    /// # Returns
    ///
    /// Current deployment status
    ///
    /// # Errors
    ///
    /// Returns `ResourceUnavailable` if deployment ID is not found
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use kwavers::infra::cloud::CloudPINNService;
    /// # async fn example(service: &CloudPINNService, deployment_id: &str) {
    /// let status = service.get_deployment_status(deployment_id).await.unwrap();
    /// println!("Status: {:?}", status);
    /// # }
    /// ```
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

    /// Scale deployment
    ///
    /// Adjusts the number of instances for a deployment.
    ///
    /// # Arguments
    ///
    /// - `deployment_id`: Unique deployment identifier
    /// - `target_instances`: Desired instance count (must be ≥ 1)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Deployment not found
    /// - Target instances is zero
    /// - Provider scaling operation fails
    ///
    /// # Algorithm
    ///
    /// 1. Validate deployment exists
    /// 2. Validate target instance count
    /// 3. Update status to Scaling
    /// 4. Delegate to provider-specific scaling implementation
    /// 5. Update status back to Active
    /// 6. Update metrics
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use kwavers::infra::cloud::CloudPINNService;
    /// # async fn example(service: &mut CloudPINNService, deployment_id: &str) {
    /// service.scale_deployment(deployment_id, 5).await.unwrap();
    /// # }
    /// ```
    pub async fn scale_deployment(
        &mut self,
        deployment_id: &str,
        target_instances: usize,
    ) -> KwaversResult<()> {
        // Check if deployment exists first
        if !self.deployments.contains_key(deployment_id) {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("deployment {}", deployment_id),
                },
            ));
        }

        // Validate target instances
        if target_instances == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "target_instances".to_string(),
                    reason: "Must specify at least 1 instance".to_string(),
                },
            ));
        }

        // Update status to scaling
        if let Some(handle) = self.deployments.get_mut(deployment_id) {
            handle.status = DeploymentStatus::Scaling;
        }

        // Perform provider-specific scaling for cloud deployment
        #[cfg(feature = "pinn")]
        {
            let config = self.config.clone();
            if let Some(handle) = self.deployments.get_mut(deployment_id) {
                match self.provider {
                    #[cfg(feature = "api")]
                    CloudProvider::AWS => {
                        super::providers::aws::scale_aws_deployment(
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
                        super::providers::gcp::scale_gcp_deployment(
                            &config,
                            handle,
                            target_instances,
                        )
                        .await?;
                    }
                    CloudProvider::Azure => {
                        super::providers::azure::scale_azure_deployment(
                            &config,
                            handle,
                            target_instances,
                        )
                        .await?;
                    }
                }
            }
        }

        // Update status back to active
        if let Some(handle) = self.deployments.get_mut(deployment_id) {
            handle.status = DeploymentStatus::Active;
        }

        Ok(())
    }

    /// Terminate deployment
    ///
    /// Deletes all cloud resources associated with a deployment.
    ///
    /// # Arguments
    ///
    /// - `deployment_id`: Unique deployment identifier
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Deployment not found
    /// - Provider termination operation fails
    ///
    /// # Algorithm
    ///
    /// 1. Remove deployment from service registry
    /// 2. Update status to Terminating
    /// 3. Delegate to provider-specific termination implementation
    /// 4. Clean up all associated resources
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use kwavers::infra::cloud::CloudPINNService;
    /// # async fn example(service: &mut CloudPINNService, deployment_id: &str) {
    /// service.terminate_deployment(deployment_id).await.unwrap();
    /// # }
    /// ```
    pub async fn terminate_deployment(&mut self, deployment_id: &str) -> KwaversResult<()> {
        let mut handle = self.deployments.remove(deployment_id).ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("deployment {}", deployment_id),
            })
        })?;

        // Update status to terminating
        handle.status = DeploymentStatus::Terminating;

        // Terminate based on provider
        #[cfg(feature = "pinn")]
        match self.provider {
            #[cfg(feature = "api")]
            CloudProvider::AWS => {
                super::providers::aws::terminate_aws_deployment(&self.config, &handle).await?;
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
                super::providers::gcp::terminate_gcp_deployment(&self.config, &handle).await?;
            }
            CloudProvider::Azure => {
                super::providers::azure::terminate_azure_deployment(&self.config, &handle).await?;
            }
        }

        Ok(())
    }

    /// Validate provider matches deployment configuration
    ///
    /// Ensures the deployment configuration specifies the same provider
    /// as the service instance.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfiguration` if providers don't match
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

    /// Get active deployment count
    ///
    /// Returns the number of currently active deployments managed by this service.
    pub fn deployment_count(&self) -> usize {
        self.deployments.len()
    }

    /// Get all deployment IDs
    ///
    /// Returns a vector of all deployment identifiers managed by this service.
    pub fn deployment_ids(&self) -> Vec<String> {
        self.deployments.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cloud_service_creation() {
        let service = CloudPINNService::new(CloudProvider::AWS).await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_cloud_service_provider_types() {
        let aws = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        assert_eq!(aws.provider, CloudProvider::AWS);

        let gcp = CloudPINNService::new(CloudProvider::GCP).await.unwrap();
        assert_eq!(gcp.provider, CloudProvider::GCP);

        let azure = CloudPINNService::new(CloudProvider::Azure).await.unwrap();
        assert_eq!(azure.provider, CloudProvider::Azure);
    }

    #[tokio::test]
    async fn test_deployment_count_initially_zero() {
        let service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        assert_eq!(service.deployment_count(), 0);
    }

    #[tokio::test]
    async fn test_get_nonexistent_deployment_status() {
        let service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        let result = service.get_deployment_status("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scale_nonexistent_deployment() {
        let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        let result = service.scale_deployment("nonexistent", 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scale_zero_instances() {
        let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        let result = service.scale_deployment("any-id", 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_terminate_nonexistent_deployment() {
        let mut service = CloudPINNService::new(CloudProvider::AWS).await.unwrap();
        let result = service.terminate_deployment("nonexistent").await;
        assert!(result.is_err());
    }
}
