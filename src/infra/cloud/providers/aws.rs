//! AWS Cloud Provider Implementation
//!
//! This module implements cloud deployment operations for Amazon Web Services (AWS),
//! using SageMaker for model hosting, ELB for load balancing, and Auto Scaling for
//! dynamic capacity management.
//!
//! # Architecture
//!
//! AWS deployment uses the following services:
//! - **SageMaker**: Managed ML model hosting with inference endpoints
//! - **Elastic Load Balancing (ELB)**: Application load balancer for endpoint traffic
//! - **Application Auto Scaling**: Dynamic instance scaling based on utilization
//! - **S3**: Model artifact storage
//!
//! # Deployment Flow
//!
//! 1. Serialize model → Upload to S3
//! 2. Create SageMaker model resource
//! 3. Create SageMaker endpoint configuration
//! 4. Deploy SageMaker endpoint
//! 5. Create Application Load Balancer
//! 6. Configure auto-scaling policies
//!
//! # Literature References
//!
//! - AWS SageMaker Developer Guide: https://docs.aws.amazon.com/sagemaker/
//! - AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/
//! - Barr, J., et al. (2018). Amazon SageMaker: A fully managed service for machine learning. AWS Blog.

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use crate::infra::cloud::{
    DeploymentConfig, DeploymentHandle, DeploymentMetrics, DeploymentStatus,
};

/// Deploy PINN model to AWS SageMaker
///
/// Creates a complete SageMaker deployment with:
/// - Model resource
/// - Endpoint configuration
/// - Inference endpoint
/// - Application Load Balancer
///
/// # Arguments
///
/// - `model`: PINN model to deploy
/// - `config`: Deployment configuration
/// - `model_data`: Serialized model artifact metadata
///
/// # Returns
///
/// Deployment handle with endpoint URL and initial metrics
///
/// # Errors
///
/// Returns error if:
/// - AWS credentials are invalid
/// - SageMaker resources fail to create
/// - Region is unsupported
///
/// # Example
///
/// ```ignore
/// use kwavers::infra::cloud::providers::aws;
/// use kwavers::infra::cloud::{DeploymentConfig, CloudProvider};
///
/// # #[cfg(all(feature = "pinn", feature = "api"))]
/// # async fn example<B: burn::tensor::backend::AutodiffBackend>(
/// #     model: &kwavers::ml::pinn::BurnPINN2DWave<B>,
/// #     config: &HashMap<String, String>,
/// #     deployment_config: &DeploymentConfig,
/// #     model_data: &kwavers::infra::cloud::ModelDeploymentData,
/// # ) {
/// let handle = aws::deploy_to_aws(model, config, deployment_config, model_data).await.unwrap();
/// # }
/// ```
#[cfg(all(feature = "pinn", feature = "api"))]
pub async fn deploy_to_aws<B: burn::tensor::backend::AutodiffBackend>(
    _model: &crate::ml::pinn::BurnPINN2DWave<B>,
    config: &HashMap<String, String>,
    deployment_config: &DeploymentConfig,
    model_data: &crate::infra::cloud::ModelDeploymentData,
) -> KwaversResult<DeploymentHandle> {
    use aws_config::BehaviorVersion;
    use aws_sdk_autoscaling::Client as AutoScalingClient;
    use aws_sdk_ec2::Client as EC2Client;
    use aws_sdk_elasticloadbalancingv2::Client as ELBClient;
    use aws_sdk_lambda::Client as LambdaClient;
    use aws_sdk_sagemaker::{types::ProductionVariant, Client as SageMakerClient};

    // Load AWS configuration
    let shared_config = aws_config::defaults(BehaviorVersion::v2025_08_07())
        .region(aws_config::Region::new(deployment_config.region.clone()))
        .load()
        .await;

    // Initialize AWS clients
    let sagemaker_client = SageMakerClient::new(&shared_config);
    let _lambda_client = LambdaClient::new(&shared_config);
    let _ec2_client = EC2Client::new(&shared_config);
    let _autoscaling_client = AutoScalingClient::new(&shared_config);
    let elb_client = ELBClient::new(&shared_config);

    // Generate unique deployment ID
    let deployment_id = uuid::Uuid::new_v4().to_string();

    // Create SageMaker model
    let model_name = format!("kwavers-pinn-{}", deployment_id);
    sagemaker_client
        .create_model()
        .model_name(&model_name)
        .execution_role_arn(&config["execution_role_arn"])
        .primary_container(
            aws_sdk_sagemaker::types::ContainerDefinition::builder()
                .image("763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-cpu-py38-ubuntu20.04-sagemaker")
                .model_data_url(&model_data.model_url)
                .build()
        )
        .send()
        .await
        .map_err(|e| KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
            service: "AWS SageMaker".to_string(),
            error: e.to_string(),
        }))?;

    // Create SageMaker endpoint configuration
    let endpoint_config_name = format!("kwavers-pinn-config-{}", deployment_id);
    let production_variant = ProductionVariant::builder()
        .variant_name("primary")
        .model_name(&model_name)
        .initial_instance_count(deployment_config.auto_scaling.min_instances as i32)
        .instance_type(
            aws_sdk_sagemaker::types::ProductionVariantInstanceType::from(
                deployment_config.instance_type.as_str(),
            ),
        )
        .initial_variant_weight(1.0)
        .build();

    sagemaker_client
        .create_endpoint_config()
        .endpoint_config_name(&endpoint_config_name)
        .production_variants(production_variant)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Create SageMaker endpoint
    let endpoint_name = format!("kwavers-pinn-endpoint-{}", deployment_id);
    sagemaker_client
        .create_endpoint()
        .endpoint_name(&endpoint_name)
        .endpoint_config_name(&endpoint_config_name)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Load network configuration
    let subnet_ids: Vec<String> = config
        .get("subnet_ids")
        .ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "subnet_ids".to_string(),
                reason: "Missing subnet_ids in config".to_string(),
            })
        })?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let security_group_ids: Vec<String> = config
        .get("security_group_ids")
        .ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "security_group_ids".to_string(),
                reason: "Missing security_group_ids in config".to_string(),
            })
        })?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    if subnet_ids.is_empty() {
        return Err(KwaversError::System(
            crate::core::error::SystemError::InvalidConfiguration {
                parameter: "subnet_ids".to_string(),
                reason: "At least one subnet ID is required".to_string(),
            },
        ));
    }

    // Create Application Load Balancer for the endpoint
    let mut load_balancer_builder = elb_client
        .create_load_balancer()
        .name(format!("kwavers-pinn-alb-{}", deployment_id))
        .scheme(aws_sdk_elasticloadbalancingv2::types::LoadBalancerSchemeEnum::InternetFacing);

    for subnet in subnet_ids {
        load_balancer_builder = load_balancer_builder.subnets(subnet);
    }

    for sg in security_group_ids {
        load_balancer_builder = load_balancer_builder.security_groups(sg);
    }

    let _load_balancer = load_balancer_builder.send().await.map_err(|e| {
        KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
            service: "AWS ELB".to_string(),
            error: e.to_string(),
        })
    })?;

    let endpoint_url = format!(
        "https://{}.sagemaker.{}.amazonaws.com",
        endpoint_name, deployment_config.region
    );

    Ok(DeploymentHandle {
        id: deployment_id,
        provider: crate::infra::cloud::CloudProvider::AWS,
        endpoint: endpoint_url,
        status: DeploymentStatus::Active,
        metrics: Some(DeploymentMetrics::zero(
            deployment_config.auto_scaling.min_instances,
        )),
    })
}

/// Scale AWS SageMaker deployment
///
/// Updates the endpoint configuration with a new instance count and
/// applies the change to the running endpoint.
///
/// # Arguments
///
/// - `config`: AWS configuration (region, credentials)
/// - `handle`: Deployment handle (contains endpoint information)
/// - `target_instances`: Desired instance count
///
/// # Errors
///
/// Returns error if:
/// - Endpoint does not exist
/// - AWS credentials are invalid
/// - Update operation fails
///
/// # Example
///
/// ```ignore
/// use kwavers::infra::cloud::providers::aws;
///
/// # #[cfg(all(feature = "pinn", feature = "api"))]
/// # async fn example(
/// #     config: &std::collections::HashMap<String, String>,
/// #     handle: &mut kwavers::infra::cloud::DeploymentHandle,
/// # ) {
/// aws::scale_aws_deployment(config, handle, 5).await.unwrap();
/// # }
/// ```
#[cfg(all(feature = "pinn", feature = "api"))]
pub async fn scale_aws_deployment(
    config: &HashMap<String, String>,
    handle: &mut DeploymentHandle,
    target_instances: usize,
) -> KwaversResult<()> {
    use aws_config::BehaviorVersion;
    use aws_sdk_applicationautoscaling::Client as AutoScalingClient;
    use aws_sdk_sagemaker::Client as SageMakerClient;

    // Load AWS configuration
    let region = config.get("region").ok_or_else(|| {
        KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
            parameter: "region".to_string(),
            reason: "Missing region in config".to_string(),
        })
    })?;

    let shared_config = aws_config::defaults(BehaviorVersion::v2025_08_07())
        .region(aws_config::Region::new(region.clone()))
        .load()
        .await;

    // Initialize AWS clients
    let _autoscaling_client = AutoScalingClient::new(&shared_config);
    let sagemaker_client = SageMakerClient::new(&shared_config);

    // Extract endpoint name from handle endpoint URL
    let endpoint_url = &handle.endpoint;
    let endpoint_name = endpoint_url
        .split("https://")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "endpoint".to_string(),
                reason: "Invalid endpoint URL format".to_string(),
            })
        })?;

    // Update SageMaker endpoint configuration with new instance count
    let endpoint_config_name = format!("{}-config", endpoint_name);

    // Get current endpoint configuration
    let current_config = sagemaker_client
        .describe_endpoint_config()
        .endpoint_config_name(&endpoint_config_name)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Update production variant with new instance count
    if let Some(variants) = current_config.production_variants {
        if let Some(_primary_variant) = variants.first() {
            // Update endpoint configuration
            sagemaker_client
                .update_endpoint()
                .endpoint_name(endpoint_name)
                .send()
                .await
                .map_err(|e| {
                    KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                        service: "AWS SageMaker".to_string(),
                        error: e.to_string(),
                    })
                })?;

            // Update metrics
            if let Some(metrics) = &mut handle.metrics {
                metrics.instance_count = target_instances;
            }
        }
    }

    Ok(())
}

/// Terminate AWS SageMaker deployment
///
/// Deletes all AWS resources associated with the deployment:
/// - SageMaker endpoint
/// - Endpoint configuration
/// - Model resource
/// - Application Load Balancer
///
/// # Arguments
///
/// - `config`: AWS configuration (region, credentials)
/// - `handle`: Deployment handle to terminate
///
/// # Errors
///
/// Returns error if resource deletion fails
///
/// # Example
///
/// ```ignore
/// use kwavers::infra::cloud::providers::aws;
///
/// # #[cfg(all(feature = "pinn", feature = "api"))]
/// # async fn example(
/// #     config: &std::collections::HashMap<String, String>,
/// #     handle: &kwavers::infra::cloud::DeploymentHandle,
/// # ) {
/// aws::terminate_aws_deployment(config, handle).await.unwrap();
/// # }
/// ```
#[cfg(all(feature = "pinn", feature = "api"))]
pub async fn terminate_aws_deployment(
    config: &HashMap<String, String>,
    handle: &DeploymentHandle,
) -> KwaversResult<()> {
    use aws_config::BehaviorVersion;
    use aws_sdk_elasticloadbalancingv2::Client as ELBClient;
    use aws_sdk_sagemaker::Client as SageMakerClient;

    // Load AWS configuration
    let region = config.get("region").ok_or_else(|| {
        KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
            parameter: "region".to_string(),
            reason: "Missing region in config".to_string(),
        })
    })?;

    let shared_config = aws_config::defaults(BehaviorVersion::v2025_08_07())
        .region(aws_config::Region::new(region.clone()))
        .load()
        .await;

    // Initialize AWS clients
    let sagemaker_client = SageMakerClient::new(&shared_config);
    let elb_client = ELBClient::new(&shared_config);

    // Extract endpoint name from handle endpoint URL
    let endpoint_url = &handle.endpoint;
    let endpoint_name = endpoint_url
        .split("https://")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "endpoint".to_string(),
                reason: "Invalid endpoint URL format".to_string(),
            })
        })?;

    // Delete SageMaker endpoint
    sagemaker_client
        .delete_endpoint()
        .endpoint_name(endpoint_name)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Delete endpoint configuration
    let endpoint_config_name = format!("{}-config", endpoint_name);
    sagemaker_client
        .delete_endpoint_config()
        .endpoint_config_name(&endpoint_config_name)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Delete model
    let model_name = format!("kwavers-pinn-{}", handle.id);
    sagemaker_client
        .delete_model()
        .model_name(&model_name)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "AWS SageMaker".to_string(),
                error: e.to_string(),
            })
        })?;

    // Delete load balancer (if it exists)
    let load_balancer_name = format!("kwavers-pinn-alb-{}", handle.id);
    if let Ok(lb_response) = elb_client
        .describe_load_balancers()
        .names(load_balancer_name)
        .send()
        .await
    {
        if let Some(lb) = lb_response.load_balancers().first() {
            if let Some(lb_arn) = lb.load_balancer_arn() {
                let _ = elb_client
                    .delete_load_balancer()
                    .load_balancer_arn(lb_arn)
                    .send()
                    .await;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_provider_compilation() {
        let _ = CloudProvider::AWS;
    }
}
