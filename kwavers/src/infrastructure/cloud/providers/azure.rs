//! Azure ML Cloud Provider Implementation
//!
//! Implements PINN model deployment using the Azure Machine Learning service
//! via the Azure ML REST API v2.0 (management.azure.com).
//!
//! # Architecture
//!
//! Azure ML deployment uses the following services:
//! - **Azure ML Managed Online Endpoint**: Scalable, real-time inference hosting
//! - **Azure Blob Storage**: Model artifact storage (uploaded via SAS URL)
//! - **Azure Load Balancer**: Integrated within Managed Online Endpoint
//! - **Application Insights**: Monitoring and telemetry
//!
//! # Deployment Flow
//!
//! 1. Register model in Azure ML workspace (PUT /models/{name}/versions/{ver})
//! 2. Create Managed Online Endpoint (PUT /onlineEndpoints/{endpoint-name})
//! 3. Create deployment within the endpoint (PUT /onlineEndpoints/{name}/deployments/{dep})
//! 4. Configure traffic allocation to 100% on new deployment
//!
//! # Authentication
//!
//! Uses OAuth 2.0 client credentials flow with Azure AD:
//! ```text
//! POST https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
//! grant_type=client_credentials
//! &client_id={app_id}
//! &client_secret={secret}
//! &scope=https://management.azure.com/.default
//! ```
//!
//! # API Reference
//!
//! - Azure ML REST API 2024-10-01:
//!   https://learn.microsoft.com/en-us/rest/api/azureml/
//! - Managed Online Endpoints:
//!   https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online
//!
//! # Literature References
//!
//! - Microsoft Azure (2024). Azure Machine Learning documentation.
//! - Lakshmanan, V., et al. (2020). *Machine Learning Design Patterns*. O'Reilly.
//! - Martin, R. C. (2017). *Clean Architecture*. Prentice Hall.

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
use crate::core::error::{KwaversError, KwaversResult};
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
use crate::infrastructure::cloud::{
    DeploymentConfig, DeploymentHandle, DeploymentMetrics, DeploymentStatus,
};
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
use std::collections::HashMap;

// ─── API version for Azure ML REST API ────────────────────────────────────────
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
const API_VERSION: &str = "2024-10-01";

// ─── Azure ML Management URL ──────────────────────────────────────────────────
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
const MANAGEMENT_URL: &str = "https://management.azure.com";

// ─── Azure AD token endpoint ──────────────────────────────────────────────────
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
const AZURE_AD_URL: &str = "https://login.microsoftonline.com";

/// Acquire an Azure AD bearer token via client credentials OAuth 2.0 flow.
///
/// ## Algorithm
///
/// POST to `https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`
/// with `grant_type=client_credentials`, returning the `access_token` field.
///
/// ## Arguments
///
/// * `http`        — shared reqwest HTTP client
/// * `tenant_id`   — Azure AD tenant ID (UUID)
/// * `client_id`   — Azure application (service principal) client ID
/// * `client_secret` — Service principal secret
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
async fn acquire_azure_token(
    http: &reqwest::Client,
    tenant_id: &str,
    client_id: &str,
    client_secret: &str,
) -> KwaversResult<String> {
    let token_url = format!("{}/{}/oauth2/v2.0/token", AZURE_AD_URL, tenant_id);
    let params = [
        ("grant_type", "client_credentials"),
        ("client_id", client_id),
        ("client_secret", client_secret),
        ("scope", "https://management.azure.com/.default"),
    ];

    let response = http
        .post(&token_url)
        .form(&params)
        .send()
        .await
        .map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "Azure AD".to_string(),
                error: e.to_string(),
            })
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(KwaversError::System(
            crate::core::error::SystemError::ExternalServiceError {
                service: "Azure AD".to_string(),
                error: format!("Token acquisition failed (HTTP {}): {}", status, body),
            },
        ));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| {
        KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
            service: "Azure AD".to_string(),
            error: format!("Failed to parse token response: {e}"),
        })
    })?;

    json["access_token"]
        .as_str()
        .map(|s| s.to_owned())
        .ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
                service: "Azure AD".to_string(),
                error: "access_token field missing from token response".to_string(),
            })
        })
}

/// Deploy a PINN model to Azure ML as a Managed Online Endpoint.
///
/// ## Deployment Steps
///
/// 1. Acquire Azure AD bearer token (client credentials)
/// 2. Register model in Azure ML workspace
/// 3. Create Managed Online Endpoint (PUT)
/// 4. Create online deployment within the endpoint (PUT)
/// 5. Set 100% traffic to the new deployment (PATCH)
///
/// ## Required Config Keys
///
/// | Key | Description |
/// |-----|-------------|
/// | `tenant_id`       | Azure AD tenant ID |
/// | `client_id`       | Service principal client ID |
/// | `client_secret`   | Service principal secret |
/// | `subscription_id` | Azure subscription ID |
/// | `resource_group`  | Azure resource group name |
/// | `workspace_name`  | Azure ML workspace name |
///
/// ## Arguments
///
/// * `_model`             — PINN model (serialised artifact URL in `model_data`)
/// * `config`             — Azure credentials and workspace parameters
/// * `deployment_config`  — Deployment settings (instance type, scaling)
/// * `model_data`         — Model artifact URL (Azure Blob Storage SAS URL)
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
pub async fn deploy_to_azure<B: burn::tensor::backend::AutodiffBackend>(
    _model: &crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
    config: &HashMap<String, String>,
    deployment_config: &DeploymentConfig,
    model_data: &crate::infrastructure::cloud::ModelDeploymentData,
) -> KwaversResult<DeploymentHandle> {
    let tenant_id = required_config(config, "tenant_id")?;
    let client_id = required_config(config, "client_id")?;
    let client_secret = required_config(config, "client_secret")?;
    let subscription_id = required_config(config, "subscription_id")?;
    let resource_group = required_config(config, "resource_group")?;
    let workspace_name = required_config(config, "workspace_name")?;

    let http = reqwest::Client::new();
    let token = acquire_azure_token(&http, tenant_id, client_id, client_secret).await?;
    let auth = format!("Bearer {token}");

    let deployment_id = uuid::Uuid::new_v4().to_string();
    let endpoint_name = format!("kwavers-{}", &deployment_id[..8]);
    let deployment_name = "primary";

    // Base URL: subscriptions/.../workspaces/{ws}/...
    let ws_base = format!(
        "{}/subscriptions/{}/resourceGroups/{}/providers\
         /Microsoft.MachineLearningServices/workspaces/{}",
        MANAGEMENT_URL, subscription_id, resource_group, workspace_name
    );

    // ── Step 1: Register model artifact ──────────────────────────────────────
    let model_body = serde_json::json!({
        "properties": {
            "modelType": "CustomModel",
            "modelUri": model_data.model_url,
            "description": "kwavers PINN acoustic model"
        }
    });
    let model_url = format!(
        "{}/models/kwavers-pinn/versions/{}?api-version={}",
        ws_base, deployment_id, API_VERSION
    );
    put_json(&http, &model_url, &auth, &model_body).await?;

    // ── Step 2: Create Managed Online Endpoint ────────────────────────────────
    let endpoint_body = serde_json::json!({
        "location": deployment_config.region,
        "kind": "Managed",
        "properties": {
            "authMode": "Key",
            "description": "kwavers PINN inference endpoint"
        }
    });
    let endpoint_url = format!(
        "{}/onlineEndpoints/{}?api-version={}",
        ws_base, endpoint_name, API_VERSION
    );
    put_json(&http, &endpoint_url, &auth, &endpoint_body).await?;

    // ── Step 3: Create online deployment ─────────────────────────────────────
    let dep_body = serde_json::json!({
        "location": deployment_config.region,
        "kind": "Managed",
        "properties": {
            "model": format!("azureml:kwavers-pinn:{}",  deployment_id),
            "instanceType": deployment_config.instance_type,
            "instanceCount": deployment_config.auto_scaling.min_instances,
            "scaleSettings": {
                "scaleType": "TargetUtilization",
                "minInstances": deployment_config.auto_scaling.min_instances,
                "maxInstances": deployment_config.auto_scaling.max_instances,
                "targetUtilizationPercentage": 70
            }
        }
    });
    let dep_url = format!(
        "{}/onlineEndpoints/{}/deployments/{}?api-version={}",
        ws_base, endpoint_name, deployment_name, API_VERSION
    );
    put_json(&http, &dep_url, &auth, &dep_body).await?;

    // ── Step 4: Route 100% traffic to new deployment ──────────────────────────
    let traffic_body = serde_json::json!({
        "properties": {
            "traffic": { deployment_name: 100 }
        }
    });
    let traffic_url = format!(
        "{}/onlineEndpoints/{}?api-version={}",
        ws_base, endpoint_name, API_VERSION
    );
    patch_json(&http, &traffic_url, &auth, &traffic_body).await?;

    let endpoint_uri = format!(
        "https://{}.{}.inference.ml.azure.com/score",
        endpoint_name, deployment_config.region
    );

    Ok(DeploymentHandle {
        id: deployment_id,
        provider: crate::infrastructure::cloud::CloudProvider::Azure,
        endpoint: endpoint_uri,
        status: DeploymentStatus::Active,
        metrics: Some(DeploymentMetrics::zero(
            deployment_config.auto_scaling.min_instances,
        )),
    })
}

/// Scale an Azure ML Managed Online Endpoint by patching the deployment.
///
/// ## Algorithm
///
/// PATCH `…/onlineEndpoints/{ep}/deployments/{dep}` with updated `instanceCount`.
///
/// ## Arguments
///
/// * `config`            — Azure credentials and workspace parameters
/// * `handle`            — Deployment handle (encodes endpoint + deployment names)
/// * `target_instances`  — Desired instance count
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
pub async fn scale_azure_deployment(
    config: &HashMap<String, String>,
    handle: &mut DeploymentHandle,
    target_instances: usize,
) -> KwaversResult<()> {
    let tenant_id = required_config(config, "tenant_id")?;
    let client_id = required_config(config, "client_id")?;
    let client_secret = required_config(config, "client_secret")?;
    let subscription_id = required_config(config, "subscription_id")?;
    let resource_group = required_config(config, "resource_group")?;
    let workspace_name = required_config(config, "workspace_name")?;

    let http = reqwest::Client::new();
    let token = acquire_azure_token(&http, tenant_id, client_id, client_secret).await?;
    let auth = format!("Bearer {token}");

    // Extract endpoint name from URI: https://{ep-name}.{region}.inference.ml.azure.com/score
    let endpoint_name = handle
        .endpoint
        .strip_prefix("https://")
        .and_then(|s| s.split('.').next())
        .ok_or_else(|| azure_config_error("endpoint", "Invalid Azure endpoint URI format"))?;

    let ws_base = format!(
        "{}/subscriptions/{}/resourceGroups/{}/providers\
         /Microsoft.MachineLearningServices/workspaces/{}",
        MANAGEMENT_URL, subscription_id, resource_group, workspace_name
    );

    let patch_body = serde_json::json!({
        "properties": {
            "instanceCount": target_instances,
            "scaleSettings": {
                "scaleType": "Default",
                "minInstances": target_instances,
                "maxInstances": target_instances
            }
        }
    });
    let dep_url = format!(
        "{}/onlineEndpoints/{}/deployments/primary?api-version={}",
        ws_base, endpoint_name, API_VERSION
    );
    patch_json(&http, &dep_url, &auth, &patch_body).await?;

    if let Some(metrics) = &mut handle.metrics {
        metrics.instance_count = target_instances;
    }
    Ok(())
}

/// Terminate an Azure ML Managed Online Endpoint and all associated resources.
///
/// ## Algorithm
///
/// 1. DELETE the deployment (`primary`)
/// 2. DELETE the endpoint
/// 3. DELETE the model version
///
/// ## Arguments
///
/// * `config`  — Azure credentials and workspace parameters
/// * `handle`  — Deployment handle
#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
pub async fn terminate_azure_deployment(
    config: &HashMap<String, String>,
    handle: &DeploymentHandle,
) -> KwaversResult<()> {
    let tenant_id = required_config(config, "tenant_id")?;
    let client_id = required_config(config, "client_id")?;
    let client_secret = required_config(config, "client_secret")?;
    let subscription_id = required_config(config, "subscription_id")?;
    let resource_group = required_config(config, "resource_group")?;
    let workspace_name = required_config(config, "workspace_name")?;

    let http = reqwest::Client::new();
    let token = acquire_azure_token(&http, tenant_id, client_id, client_secret).await?;
    let auth = format!("Bearer {token}");

    let endpoint_name = handle
        .endpoint
        .strip_prefix("https://")
        .and_then(|s| s.split('.').next())
        .ok_or_else(|| azure_config_error("endpoint", "Invalid Azure endpoint URI format"))?;

    let ws_base = format!(
        "{}/subscriptions/{}/resourceGroups/{}/providers\
         /Microsoft.MachineLearningServices/workspaces/{}",
        MANAGEMENT_URL, subscription_id, resource_group, workspace_name
    );

    // Delete deployment first (must be deleted before endpoint)
    let dep_url = format!(
        "{}/onlineEndpoints/{}/deployments/primary?api-version={}",
        ws_base, endpoint_name, API_VERSION
    );
    delete_resource(&http, &dep_url, &auth).await?;

    // Delete endpoint
    let ep_url = format!(
        "{}/onlineEndpoints/{}?api-version={}",
        ws_base, endpoint_name, API_VERSION
    );
    delete_resource(&http, &ep_url, &auth).await?;

    // Delete model version
    let model_url = format!(
        "{}/models/kwavers-pinn/versions/{}?api-version={}",
        ws_base, handle.id, API_VERSION
    );
    delete_resource(&http, &model_url, &auth).await?;

    Ok(())
}

// ─── HTTP helpers ──────────────────────────────────────────────────────────────

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
async fn put_json(
    http: &reqwest::Client,
    url: &str,
    auth: &str,
    body: &serde_json::Value,
) -> KwaversResult<()> {
    let resp = http
        .put(url)
        .header("Authorization", auth)
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| azure_service_error(e.to_string()))?;
    check_status(resp, url).await
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
async fn patch_json(
    http: &reqwest::Client,
    url: &str,
    auth: &str,
    body: &serde_json::Value,
) -> KwaversResult<()> {
    let resp = http
        .patch(url)
        .header("Authorization", auth)
        .header("Content-Type", "application/merge-patch+json")
        .json(body)
        .send()
        .await
        .map_err(|e| azure_service_error(e.to_string()))?;
    check_status(resp, url).await
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
async fn delete_resource(http: &reqwest::Client, url: &str, auth: &str) -> KwaversResult<()> {
    let resp = http
        .delete(url)
        .header("Authorization", auth)
        .send()
        .await
        .map_err(|e| azure_service_error(e.to_string()))?;
    // 200, 202 (accepted async), 204 (no content) are all success
    if resp.status().is_success() || resp.status().as_u16() == 202 {
        return Ok(());
    }
    check_status(resp, url).await
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
async fn check_status(resp: reqwest::Response, url: &str) -> KwaversResult<()> {
    if resp.status().is_success() || resp.status().as_u16() == 202 {
        return Ok(());
    }
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    Err(KwaversError::System(
        crate::core::error::SystemError::ExternalServiceError {
            service: "Azure ML".to_string(),
            error: format!("HTTP {status} at {url}: {body}"),
        },
    ))
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
fn required_config<'a>(config: &'a HashMap<String, String>, key: &str) -> KwaversResult<&'a str> {
    config.get(key).map(|s| s.as_str()).ok_or_else(|| {
        KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
            parameter: key.to_string(),
            reason: format!("Missing required Azure config key: {key}"),
        })
    })
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
fn azure_service_error(msg: impl Into<String>) -> KwaversError {
    KwaversError::System(crate::core::error::SystemError::ExternalServiceError {
        service: "Azure ML".to_string(),
        error: msg.into(),
    })
}

#[cfg(all(feature = "pinn", feature = "cloud-azure"))]
fn azure_config_error(param: &str, reason: &str) -> KwaversError {
    KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
        parameter: param.to_string(),
        reason: reason.to_string(),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_azure_provider_compilation() {
        let _ = crate::infrastructure::cloud::CloudProvider::Azure;
    }
}
