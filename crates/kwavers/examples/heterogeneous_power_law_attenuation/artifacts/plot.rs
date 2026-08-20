use std::path::Path;

use anyhow::Result;
use plotters::prelude::*;

use super::super::configuration::{alpha_np_m, GAMMAS};
use super::super::model::SweepRow;

/// Plot prescribed power laws and simulated measurements on log-log axes.
pub(crate) fn write_plot(path: &Path, rows: &[SweepRow]) -> Result<()> {
    let shown: Vec<&SweepRow> = rows.iter().filter(|row| row.alpha0_db == 0.5).collect();
    let alpha0 = alpha_np_m(0.5);

    let root = BitMapBackend::new(path, (900, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Power-law attenuation, alpha0 = 0.5 dB/cm/MHz^gamma",
            ("sans-serif", 22),
        )
        .margin(16)
        .x_label_area_size(52)
        .y_label_area_size(64)
        .build_cartesian_2d((0.5f64..5.0f64).log_scale(), (1.0f64..120.0f64).log_scale())?;
    chart
        .configure_mesh()
        .x_desc("frequency [MHz]")
        .y_desc("alpha [Np/m]")
        .draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &BLACK];
    for (index, &gamma) in GAMMAS.iter().enumerate() {
        let colour = palette[index % palette.len()];
        let curve = (0..=100).map(|sample| {
            let frequency_mhz = 0.5 * (10.0f64).powf(sample as f64 / 100.0);
            (frequency_mhz, alpha0 * frequency_mhz.powf(gamma))
        });
        chart
            .draw_series(LineSeries::new(curve, colour.stroke_width(2)))?
            .label(format!("gamma = {gamma} (prescribed)"))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], colour.stroke_width(2))
            });
        chart.draw_series(
            shown
                .iter()
                .filter(|row| (row.gamma - gamma).abs() < f64::EPSILON)
                .map(|row| {
                    Circle::new(
                        (row.frequency_hz / 1.0e6, row.measured_np_m),
                        4,
                        colour.filled(),
                    )
                }),
        )?;
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
