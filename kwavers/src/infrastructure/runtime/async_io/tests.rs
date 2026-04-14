use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array3;

use super::{AsyncFileReader, AsyncFileWriter};

fn temp_path(name: &str) -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("kwavers-{name}-{}-{stamp}.bin", std::process::id()))
}

#[tokio::test]
async fn async_file_roundtrip_uses_platform_temp_dir() {
    let path = temp_path("roundtrip");
    let original = Array3::from_shape_fn((10, 20, 30), |(i, j, k)| (i * 600 + j * 30 + k) as f64);

    AsyncFileWriter::new(&path)
        .unwrap()
        .write_array3_f64(&original)
        .await
        .unwrap();

    let loaded = AsyncFileReader::new(&path)
        .unwrap()
        .read_array3_f64()
        .await
        .unwrap();
    assert_eq!(original, loaded);

    let _ = tokio::fs::remove_file(&path).await;
}
