use super::*;

#[test]
fn test_jwt_token_generation_and_validation() {
    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let user_id = "test-user";
    let roles = vec!["user".to_string()];
    let permissions = vec!["pinn:infer".to_string()];

    let token = auth
        .generate_token(user_id, &roles, &permissions)
        .expect("token generation must succeed in tests");

    let user = auth
        .authenticate_jwt(&token)
        .expect("token authentication must succeed in tests");

    assert_eq!(user.user_id, user_id);
    assert_eq!(user.roles, roles);
    assert_eq!(user.permissions, permissions);
    assert!(matches!(user.auth_method, AuthMethod::JWT));
}

#[test]
fn test_api_key_generation_and_validation() {
    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let user_id = "test-user";
    let permissions = vec!["pinn:infer".to_string()];

    let (raw_key, api_key_info) = auth.generate_api_key(user_id, "test-key", &permissions, None);

    assert_eq!(api_key_info.user_id, user_id);
    assert!(AuthMiddleware::validate_api_key_format(&raw_key));

    let api_keys = vec![api_key_info];
    let user = auth
        .authenticate_api_key(&raw_key, &api_keys)
        .expect("API key authentication must succeed in tests");

    assert_eq!(user.user_id, user_id);
    assert!(matches!(user.auth_method, AuthMethod::APIKey));
}

#[test]
fn test_authorization_success() {
    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let user = AuthenticatedUser {
        user_id: "test-user".to_string(),
        roles: vec![],
        permissions: vec!["pinn:infer".to_string()],
        auth_method: AuthMethod::JWT,
    };

    assert!(auth.authorize(&user, "pinn", "infer").is_ok());
}

#[test]
fn test_authorization_failure() {
    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let user = AuthenticatedUser {
        user_id: "test-user".to_string(),
        roles: vec![],
        permissions: vec!["pinn:infer".to_string()],
        auth_method: AuthMethod::JWT,
    };

    assert!(auth.authorize(&user, "pinn", "train").is_err());
}

#[test]
fn test_wildcard_permissions() {
    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let user = AuthenticatedUser {
        user_id: "test-user".to_string(),
        roles: vec![],
        permissions: vec!["pinn:*".to_string()],
        auth_method: AuthMethod::JWT,
    };

    assert!(auth.authorize(&user, "pinn", "train").is_ok());
    assert!(auth.authorize(&user, "pinn", "infer").is_ok());
}

#[test]
fn test_api_key_format_validation() {
    assert!(AuthMiddleware::validate_api_key_format(
        "valid-api-key-with-reasonable-length"
    ));
    assert!(!AuthMiddleware::validate_api_key_format(""));
    assert!(!AuthMiddleware::validate_api_key_format("short"));
    assert!(!AuthMiddleware::validate_api_key_format(&"x".repeat(200)));
}

#[test]
fn test_expired_api_key() {
    use chrono::Duration;

    let auth = AuthMiddleware::new("test-secret-do-not-use-in-production", JWTConfig::default())
        .expect("test auth middleware construction must succeed");

    let expired_key = APIKey {
        key_id: "test".to_string(),
        user_id: "user".to_string(),
        name: "test".to_string(),
        key_hash: "dummy".to_string(),
        permissions: vec![],
        created_at: chrono::Utc::now(),
        expires_at: Some(chrono::Utc::now() - Duration::hours(1)),
        status: APIKeyStatus::Active,
    };

    let api_keys = vec![expired_key];
    let result = auth.authenticate_api_key("dummy", &api_keys);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err().error,
        APIErrorType::AuthenticationFailed
    ));
}
