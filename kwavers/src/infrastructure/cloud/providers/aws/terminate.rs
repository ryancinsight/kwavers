//! AWS SageMaker termination function.

use crate::core::error::{KwaversError, KwaversResult};
use crate::infrastructure::cloud::DeploymentHandle;
use std::collections::HashMap;

/// Terminate AWS SageMaker deployment and delete all associated resources.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub async fn terminate_aws_deployment(
    config: &HashMap<String, String>,
    handle: &DeploymentHandle,
) -> KwaversResult<()> {
    use aws_config::BehaviorVersion;
    use aws_sdk_elasticloadbalancingv2::Client as ELBClient;
    use aws_sdk_sagemaker::Client as SageMakerClient;

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

    let sagemaker_client = SageMakerClient::new(&shared_config);
    let elb_client = ELBClient::new(&shared_config);

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
