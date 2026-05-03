//! Tests for AWS cloud provider.

#[test]
fn test_aws_provider_compilation() {
    let _ = crate::infrastructure::cloud::CloudProvider::AWS;
}
