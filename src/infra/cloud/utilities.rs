//! Cloud Deployment Utilities
//!
//! This module provides shared utilities for cloud deployments including
//! configuration loading from environment and model serialization.
//!
//! # Architecture
//!
//! Utilities follow the Infrastructure layer pattern:
//! - Configuration loading: External configuration sources (environment, files)
//! - Model serialization: PINN model → cloud storage format
//!
//! # Literature References
//!
//! - 12-Factor App methodology for configuration management
//!   https://12factor.net/config
//! - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
//!   Prentice Hall. ISBN: 978-0134494166

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

use super::types::{CloudProvider, ModelDeploymentData};

/// Load provider-specific configuration from environment
///
/// Reads cloud provider credentials and settings from environment variables
/// following the 12-Factor App configuration methodology.
///
/// # Environment Variables
///
/// ## AWS
/// - `AWS_ACCESS_KEY_ID`: Access key credential
/// - `AWS_SECRET_ACCESS_KEY`: Secret key credential
/// - `AWS_REGION`: Deployment region (default: us-east-1)
/// - `AWS_SAGEMAKER_EXECUTION_ROLE_ARN`: SageMaker execution role
///
/// ## GCP
/// - `GOOGLE_CLOUD_PROJECT`: Project ID
/// - `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
///
/// ## Azure
/// - `AZURE_SUBSCRIPTION_ID`: Azure subscription identifier
/// - `AZURE_CLIENT_ID`: Service principal client ID
/// - `AZURE_CLIENT_SECRET`: Service principal secret
/// - `AZURE_TENANT_ID`: Azure AD tenant ID
///
/// # Returns
///
/// Configuration map with provider-specific keys
///
/// # Example
///
/// ```
/// use kwavers::infra::cloud::{CloudProvider, utilities};
///
/// # tokio_test::block_on(async {
/// let config = utilities::load_provider_config(&CloudProvider::AWS).await;
/// # });
/// ```
pub async fn load_provider_config(
    provider: &CloudProvider,
) -> KwaversResult<HashMap<String, String>> {
    let mut config = HashMap::new();

    match provider {
        CloudProvider::AWS => {
            config.insert(
                "access_key".to_string(),
                std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default(),
            );
            config.insert(
                "secret_key".to_string(),
                std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default(),
            );
            config.insert(
                "region".to_string(),
                std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
            );
            config.insert(
                "execution_role_arn".to_string(),
                std::env::var("AWS_SAGEMAKER_EXECUTION_ROLE_ARN").unwrap_or_else(|_| {
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole".to_string()
                }),
            );
        }
        CloudProvider::GCP => {
            config.insert(
                "project_id".to_string(),
                std::env::var("GOOGLE_CLOUD_PROJECT").unwrap_or_default(),
            );
            config.insert(
                "credentials_path".to_string(),
                std::env::var("GOOGLE_APPLICATION_CREDENTIALS").unwrap_or_default(),
            );
            config.insert(
                "access_token".to_string(),
                std::env::var("GOOGLE_ACCESS_TOKEN").unwrap_or_default(),
            );
        }
        CloudProvider::Azure => {
            config.insert(
                "subscription_id".to_string(),
                std::env::var("AZURE_SUBSCRIPTION_ID").unwrap_or_default(),
            );
            config.insert(
                "client_id".to_string(),
                std::env::var("AZURE_CLIENT_ID").unwrap_or_default(),
            );
            config.insert(
                "client_secret".to_string(),
                std::env::var("AZURE_CLIENT_SECRET").unwrap_or_default(),
            );
            config.insert(
                "tenant_id".to_string(),
                std::env::var("AZURE_TENANT_ID").unwrap_or_default(),
            );
            config.insert(
                "resource_group".to_string(),
                std::env::var("AZURE_RESOURCE_GROUP").unwrap_or_else(|_| "kwavers-rg".to_string()),
            );
            config.insert(
                "workspace_name".to_string(),
                std::env::var("AZURE_ML_WORKSPACE")
                    .unwrap_or_else(|_| "kwavers-ml-workspace".to_string()),
            );
            config.insert(
                "azure_access_token".to_string(),
                std::env::var("AZURE_ACCESS_TOKEN").unwrap_or_default(),
            );
        }
    }

    Ok(config)
}

/// Serialize PINN model for cloud deployment
///
/// Converts a Burn PINN model to binary format and prepares metadata
/// for cloud storage upload.
///
/// # Algorithm
///
/// 1. Clone model to ensure ownership
/// 2. Convert to Burn record format
/// 3. Serialize using BinBytesRecorder with full precision
/// 4. Generate unique model ID (UUID)
/// 5. Construct cloud storage URL based on provider
///
/// # Arguments
///
/// - `model`: PINN model to serialize
/// - `provider`: Target cloud provider (affects URL format)
///
/// # Returns
///
/// Model deployment data with storage URL and size
///
/// # Errors
///
/// Returns error if serialization fails
///
/// # Example
///
/// ```ignore
/// use kwavers::infra::cloud::{CloudProvider, utilities};
/// use kwavers::ml::pinn::BurnPINN2DWave;
///
/// # #[cfg(feature = "pinn")]
/// # async fn example<B: burn::tensor::backend::AutodiffBackend>(model: &BurnPINN2DWave<B>) {
/// let data = utilities::serialize_model_for_deployment(
///     model,
///     &CloudProvider::AWS
/// ).await.unwrap();
/// # }
/// ```
#[cfg(feature = "pinn")]
pub async fn serialize_model_for_deployment<B: burn::tensor::backend::AutodiffBackend>(
    model: &crate::ml::pinn::BurnPINN2DWave<B>,
    provider: &CloudProvider,
) -> KwaversResult<ModelDeploymentData> {
    use burn::module::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    // Generate unique model ID for this deployment
    let model_id = uuid::Uuid::new_v4().to_string();
    let model_key = format!("kwavers-pinn-{}.bin", model_id);

    // Serialize model to binary format
    let recorder = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default();
    let model_record = model.clone().into_record();
    let buffer = recorder.record(model_record, ()).map_err(|e| {
        KwaversError::Data(crate::core::error::DataError::FormatError {
            format: "burn model".to_string(),
            reason: format!("Failed to serialize model: {e}"),
        })
    })?;

    let model_size_bytes = buffer.len();

    // Construct cloud storage URL based on provider
    let model_url = match provider {
        CloudProvider::AWS => {
            let bucket = "kwavers-models";
            format!("s3://{}/{}", bucket, model_key)
        }
        CloudProvider::GCP => {
            let bucket = "kwavers-models";
            format!("gs://{}/{}", bucket, model_key)
        }
        CloudProvider::Azure => {
            let account = "kwaversstorage";
            let container = "models";
            format!(
                "https://{}.blob.core.windows.net/{}/{}",
                account, container, model_key
            )
        }
    };

    Ok(ModelDeploymentData {
        model_url,
        model_size_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_aws_config() {
        let config = load_provider_config(&CloudProvider::AWS).await.unwrap();
        assert!(config.contains_key("access_key"));
        assert!(config.contains_key("secret_key"));
        assert!(config.contains_key("region"));
        assert!(config.contains_key("execution_role_arn"));
    }

    #[tokio::test]
    async fn test_load_gcp_config() {
        let config = load_provider_config(&CloudProvider::GCP).await.unwrap();
        assert!(config.contains_key("project_id"));
        assert!(config.contains_key("credentials_path"));
        assert!(config.contains_key("access_token"));
    }

    #[tokio::test]
    async fn test_load_azure_config() {
        let config = load_provider_config(&CloudProvider::Azure).await.unwrap();
        assert!(config.contains_key("subscription_id"));
        assert!(config.contains_key("client_id"));
        assert!(config.contains_key("client_secret"));
        assert!(config.contains_key("tenant_id"));
        assert!(config.contains_key("resource_group"));
        assert!(config.contains_key("workspace_name"));
    }

    #[test]
    fn test_config_keys_present() {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let aws_config = runtime
            .block_on(load_provider_config(&CloudProvider::AWS))
            .unwrap();
        assert_eq!(aws_config.len(), 4);

        let gcp_config = runtime
            .block_on(load_provider_config(&CloudProvider::GCP))
            .unwrap();
        assert_eq!(gcp_config.len(), 3);

        let azure_config = runtime
            .block_on(load_provider_config(&CloudProvider::Azure))
            .unwrap();
        assert_eq!(azure_config.len(), 7);
    }
}
