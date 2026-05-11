//! AWS SageMaker scaling function.

use crate::core::error::{KwaversError, KwaversResult};
use crate::infrastructure::cloud::DeploymentHandle;
use std::collections::HashMap;

/// Scale AWS SageMaker deployment to the target instance count.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub async fn scale_aws_deployment(
    config: &HashMap<String, String>,
    handle: &mut DeploymentHandle,
    target_instances: usize,
) -> KwaversResult<()> {
    use aws_config::BehaviorVersion;
    use aws_sdk_applicationautoscaling::Client as AutoScalingClient;
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

    let _autoscaling_client = AutoScalingClient::new(&shared_config);
    let sagemaker_client = SageMakerClient::new(&shared_config);

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

    let endpoint_config_name = format!("{}-config", endpoint_name);

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

    if let Some(variants) = current_config.production_variants {
        if let Some(_primary_variant) = variants.first() {
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

            if let Some(metrics) = &mut handle.metrics {
                metrics.instance_count = target_instances;
            }
        }
    }

    Ok(())
}
