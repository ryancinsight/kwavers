//! Golden-image isolation and comparison for integration-test figures.

use std::borrow::Cow;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Component, Path};
use std::sync::OnceLock;

const REGENERATE_ENV: &str = "KWAVERS_REGENERATE_TEST_FIGURES";

/// One 8-bit quantization code value per channel. This admits only the rounding
/// uncertainty of quantizing a continuous channel value to an 8-bit code.
const MAXIMUM_CHANNEL_ERROR_BOUND: u8 = 1;
const TEST_FONT_NAME: &str = "Ubuntu-Light";

static FONT_REGISTRATION: OnceLock<Result<(), String>> = OnceLock::new();

#[derive(Debug)]
struct DecodedPng {
    width: u32,
    height: u32,
    color_type: png::ColorType,
    bit_depth: png::BitDepth,
    channels: Vec<u8>,
}

/// Render a test figure and compare it with its committed PNG golden.
///
/// Ordinary test runs render into a unique temporary directory and leave the
/// working tree untouched. Setting `KWAVERS_REGENERATE_TEST_FIGURES=1` is the
/// only regeneration authority and renders directly to the committed golden.
///
/// # Errors
///
/// Returns an error for an invalid file name, an unsupported regeneration
/// environment value, rendering or PNG decoding failure, incompatible image
/// metadata, or channel error above the documented quantization bound.
pub fn render_and_compare<F>(file_name: &str, render: F) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(&Path) -> Result<(), Box<dyn Error>>,
{
    validate_file_name(file_name)?;
    register_test_font().map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
    let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-figures")
        .join(file_name);

    match std::env::var_os(REGENERATE_ENV) {
        None => {
            let output_directory = tempfile::Builder::new()
                .prefix("kwavers-test-figure-")
                .tempdir()?;
            let generated = output_directory.path().join(file_name);
            render(&generated)?;
            compare_pngs(&generated, &golden)
        }
        Some(value) if value == OsStr::new("1") => render(&golden),
        Some(value) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{REGENERATE_ENV} must be unset or exactly `1`, got `{}`",
                value.to_string_lossy()
            ),
        )
        .into()),
    }
}

fn register_test_font() -> Result<(), String> {
    FONT_REGISTRATION
        .get_or_init(|| {
            let mut definitions = epaint::text::FontDefinitions::default();
            let font = definitions
                .font_data
                .remove(TEST_FONT_NAME)
                .ok_or_else(|| format!("embedded test font `{TEST_FONT_NAME}` is unavailable"))?;
            let bytes = match font.font {
                Cow::Borrowed(bytes) => bytes,
                Cow::Owned(_) => {
                    return Err(format!(
                        "embedded test font `{TEST_FONT_NAME}` must have static storage"
                    ));
                }
            };

            plotters::style::register_font("sans-serif", plotters::style::FontStyle::Normal, bytes)
                .map_err(|_| format!("failed to register embedded test font `{TEST_FONT_NAME}`"))
        })
        .clone()
}

fn validate_file_name(file_name: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(file_name);
    let mut components = path.components();
    let is_single_file = matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && path.extension() == Some(OsStr::new("png"));
    if is_single_file {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("test figure name must be one relative `.png` file, got `{file_name}`"),
    )
    .into())
}

fn compare_pngs(generated_path: &Path, golden_path: &Path) -> Result<(), Box<dyn Error>> {
    let generated = decode_png(generated_path)?;
    let golden = decode_png(golden_path)?;

    if (generated.width, generated.height) != (golden.width, golden.height) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure dimensions differ for `{}`: generated {}x{}, golden {}x{}",
                golden_path.display(),
                generated.width,
                generated.height,
                golden.width,
                golden.height
            ),
        )
        .into());
    }
    if generated.color_type != golden.color_type {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure color type differs for `{}`: generated {:?}, golden {:?}",
                golden_path.display(),
                generated.color_type,
                golden.color_type
            ),
        )
        .into());
    }
    if generated.bit_depth != golden.bit_depth {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure bit depth differs for `{}`: generated {:?}, golden {:?}",
                golden_path.display(),
                generated.bit_depth,
                golden.bit_depth
            ),
        )
        .into());
    }
    if generated.bit_depth != png::BitDepth::Eight {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure `{}` uses {:?}; the golden metric is defined for 8-bit channels",
                golden_path.display(),
                generated.bit_depth
            ),
        )
        .into());
    }
    if generated.channels.len() != golden.channels.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "decoded channel counts differ for `{}`: generated {}, golden {}",
                golden_path.display(),
                generated.channels.len(),
                golden.channels.len()
            ),
        )
        .into());
    }
    if generated.channels.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure `{}` decoded to no channel data",
                golden_path.display()
            ),
        )
        .into());
    }

    let maximum_channel_error = generated
        .channels
        .iter()
        .zip(&golden.channels)
        .map(|(&actual, &expected)| actual.abs_diff(expected))
        .max()
        .expect("invariant: non-empty channel buffers were validated above");
    if maximum_channel_error > MAXIMUM_CHANNEL_ERROR_BOUND {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "figure `{}` maximum channel error {} exceeds {} quantization code",
                golden_path.display(),
                maximum_channel_error,
                MAXIMUM_CHANNEL_ERROR_BOUND
            ),
        )
        .into());
    }

    Ok(())
}

fn decode_png(path: &Path) -> Result<DecodedPng, Box<dyn Error>> {
    let file = File::open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to open PNG `{}`: {error}", path.display()),
        )
    })?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut channels = vec![0_u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut channels)?;
    channels.truncate(info.buffer_size());

    Ok(DecodedPng {
        width: info.width,
        height: info.height,
        color_type: info.color_type,
        bit_depth: info.bit_depth,
        channels,
    })
}
