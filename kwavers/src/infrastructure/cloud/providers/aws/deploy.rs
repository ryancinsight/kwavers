//! AWS SageMaker deployment function.

use crate::core::error::{KwaversError, KwaversResult};
use crate::infrastructure::cloud::{
    DeploymentConfig, DeploymentHandle, DeploymentMetrics, DeploymentStatus,
};
use std::collections::HashMap;

/// Deploy PINN model to AWS SageMaker.
///
/// Creates a SageMaker endpoint, endpoint configuration, model resource, and
/// Application Load Balancer.
#[cfg(all(feature = "pinn", feature = "cloud-aws"))]
pub async fn deploy_to_aws<B: burn::tensor::backend::AutodiffBackend>(
    _model: &crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
    config: &HashMap<String, String>,
    deployment_config: &DeploymentConfig,
    model_data: &crate::infrastructure::cloud::ModelDeploymentData,
) -> KwaversResult<DeploymentHandle> {
    use aws_config::BehaviorVersion;
    use aws_sdk_autoscaling::Client as AutoScalingClient;
    use aws_sdk_ec2::Client as EC2Client;
    use aws_sdk_elasticloadbalancingv2::Client as ELBClient;
    use aws_sdk_lambda::Client as LambdaClient;
    use aws_sdk_sagemaker::{types::ProductionVariant, Client as SageMakerClient};

    let shared_config = aws_config::defaults(BehaviorVersion::v2025_08_07())
        .region(aws_config::Region::new(deployment_config.region.clone()))
        .load()
        .await;

    let sagemaker_client = SageMakerClient::new(&shared_config);
    let _lambda_client = LambdaClient::new(&shared_config);
    let _ec2_client = EC2Client::new(&shared_config);
    let _autoscaling_client = AutoScalingClient::new(&shared_config);
    let elb_client = ELBClient::new(&shared_config);

    let deployment_id = uuid::Uuid::new_v4().to_string();

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
        provider: crate::infrastructure::cloud::CloudProvider::AWS,
        endpoint: endpoint_url,
        status: DeploymentStatus::Active,
        metrics: Some(DeploymentMetrics::zero(
            deployment_config.auto_scaling.min_instances,
        )),
    })
}
