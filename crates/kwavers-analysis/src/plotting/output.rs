use anyhow::{Context, Result};
use iris::render::RenderBackend;
use iris_plot_html::{HtmlPlotBackend, Plot};

pub(super) fn save(filename: &str, plot: Result<Plot<'_>>) {
    let result = plot.and_then(|plot| {
        let mut backend = HtmlPlotBackend::default();
        let document = backend.render(&plot)?;
        std::fs::write(filename, document).context("write the rendered HTML")
    });
    match result {
        Ok(()) => log::info!("Plot saved to {filename}"),
        Err(error) => log::error!("Plot {filename}: {error:#}"),
    }
}
