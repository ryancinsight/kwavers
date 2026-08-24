//! Shared DICOM-series selection for seismic CT workflows.

use ritk_io::DicomSeriesInfo;

const DEFAULT_MEDIMODEL_SERIES_UID: &str =
    "1.3.6.1.4.1.5962.99.1.1761388472.1291962045.1616669124536.2634.0";

/// Select the canonical CT series, merging one-file-per-series datasets.
pub(crate) fn select_series(mut series: Vec<DicomSeriesInfo>) -> DicomSeriesInfo {
    let maximum_file_count = series
        .iter()
        .map(|candidate| candidate.file_paths.len())
        .max()
        .unwrap_or(0);

    if let Some(index) = series
        .iter()
        .position(|candidate| candidate.series_instance_uid() == DEFAULT_MEDIMODEL_SERIES_UID)
    {
        return series.swap_remove(index);
    }

    if maximum_file_count <= 1 {
        let has_ct_series = series.iter().any(|candidate| candidate.modality() == "CT");
        let all_paths: Vec<_> = series
            .iter_mut()
            .filter(|candidate| !has_ct_series || candidate.modality() == "CT")
            .flat_map(|candidate| candidate.file_paths.drain(..))
            .collect();
        let count = all_paths.len();
        println!(
            "  Note: each DICOM slice has a unique SeriesInstanceUID; \
                  merging {count} files into one logical series for spatial sort."
        );
        return DicomSeriesInfo::new(
            "merged",
            format!("merged-{count}-slices"),
            "CT",
            String::new(),
            all_paths,
        );
    }

    let ct_indices: Vec<usize> = series
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.modality() == "CT")
        .map(|(index, _)| index)
        .collect();
    let candidates = if ct_indices.is_empty() {
        (0..series.len()).collect()
    } else {
        ct_indices
    };
    let best = candidates
        .into_iter()
        .max_by_key(|&index| series[index].file_paths.len())
        .unwrap_or(0);
    series.swap_remove(best)
}
