//! SVG figure generation via `plotters`．
//!
//! Only the `svg_backend` + `line_series` plotters features are enabled, so
//! text renders as native SVG `<text>` (no font rasterization)．Every figure
//! plots **non-PII** axis labels only: channel names (H3) or month strings
//! (H5) and aggregate counts (summary)．User IDs never reach a figure．Each
//! function returns `Ok(false)` when there is too little data to plot a
//! meaningful chart (no file is written) and `Ok(true)` after writing a
//! valid SVG．

use std::path::Path;

use plotters::prelude::*;

use crate::error::{CommError, Result};
use crate::models::ReportSummary;

const PALETTE: [RGBColor; 5] = [
    RGBColor(0x44, 0x01, 0x54),
    RGBColor(0x3B, 0x52, 0x8B),
    RGBColor(0x21, 0x90, 0x8C),
    RGBColor(0x5D, 0xC8, 0x63),
    RGBColor(0xFD, 0xE7, 0x25),
];

const W: u32 = 960;
const H: u32 = 540;

fn map_err<E: std::fmt::Display>(e: E) -> CommError {
    CommError::Config(e.to_string())
}

/// Horizontal-style bar chart of per-channel normalized concentration．
pub fn entropy_by_channel_svg(path: &Path, data: &[(String, f64)], top: usize) -> Result<bool> {
    if data.is_empty() || top == 0 {
        return Ok(false);
    }
    let mut rows: Vec<(String, f64)> = data.to_vec();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(top);
    if rows.is_empty() {
        return Ok(false);
    }
    rows.reverse();
    let n = rows.len();

    let root = SVGBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;

    let labels: Vec<String> = rows.iter().map(|(c, _)| c.clone()).collect();
    let mut chart = ChartBuilder::on(&root)
        .caption("チャネル別 発話集中度 (0=均等, 1=独占)", ("sans-serif", 22))
        .margin(16)
        .x_label_area_size(40)
        .y_label_area_size(180)
        .build_cartesian_2d(0f64..1.0f64, 0usize..n)
        .map_err(map_err)?;

    chart
        .configure_mesh()
        .disable_y_mesh()
        .y_labels(n)
        .y_label_formatter(&|y: &usize| labels.get(*y).cloned().unwrap_or_default())
        .x_desc("正規化集中度 η")
        .draw()
        .map_err(map_err)?;

    let color = PALETTE[2];
    chart
        .draw_series(rows.iter().enumerate().map(|(i, (_, v))| {
            let v = v.clamp(0.0, 1.0);
            Rectangle::new([(0.0, i), (v, i + 1)], color.filled())
        }))
        .map_err(map_err)?;

    root.present().map_err(map_err)?;
    Ok(true)
}

/// Line chart of the monthly novel-vocabulary rate over time．
pub fn novel_vocab_svg(path: &Path, monthly: &[(String, f64)]) -> Result<bool> {
    if monthly.len() < 2 {
        return Ok(false);
    }
    let n = monthly.len();
    let root = SVGBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;

    let labels: Vec<String> = monthly.iter().map(|(m, _)| m.clone()).collect();
    let mut chart = ChartBuilder::on(&root)
        .caption("月次 新規語彙率", ("sans-serif", 22))
        .margin(16)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..((n - 1) as f64), 0f64..1.0f64)
        .map_err(map_err)?;

    chart
        .configure_mesh()
        .x_labels(n.min(12))
        .x_label_formatter(&|x: &f64| {
            let i = x.round() as usize;
            labels.get(i).cloned().unwrap_or_default()
        })
        .y_desc("新規語彙率")
        .draw()
        .map_err(map_err)?;

    let series: Vec<(f64, f64)> = monthly
        .iter()
        .enumerate()
        .map(|(i, (_, v))| (i as f64, v.clamp(0.0, 1.0)))
        .collect();
    chart
        .draw_series(LineSeries::new(series.clone(), PALETTE[1].stroke_width(2)))
        .map_err(map_err)?;
    chart
        .draw_series(
            series
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, PALETTE[0].filled())),
        )
        .map_err(map_err)?;

    root.present().map_err(map_err)?;
    Ok(true)
}

/// Compact bar chart summarizing organizational health．
pub fn health_summary_svg(path: &Path, s: &ReportSummary) -> Result<bool> {
    let red = s.red_flags.len();
    let yellow = s.yellow_flags.len();
    let green = s.green_signals.len();
    let max_flag = red.max(yellow).max(green).max(1) as f64;

    let bars: [(&str, f64, Option<usize>); 4] = [
        ("健全度", s.overall_health_score.clamp(0.0, 1.0), None),
        ("赤信号", red as f64 / max_flag, Some(red)),
        ("黄信号", yellow as f64 / max_flag, Some(yellow)),
        ("緑信号", green as f64 / max_flag, Some(green)),
    ];

    let root = SVGBackend::new(path, (W, H)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("健全度サマリ", ("sans-serif", 22))
        .margin(16)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0usize..bars.len(), 0f64..1.05f64)
        .map_err(map_err)?;

    let labels: Vec<&str> = bars.iter().map(|(l, _, _)| *l).collect();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .x_labels(bars.len())
        .x_label_formatter(&|x: &usize| labels.get(*x).map(|s| s.to_string()).unwrap_or_default())
        .y_desc("スコア / 正規化件数")
        .draw()
        .map_err(map_err)?;

    let colors = [PALETTE[2], PALETTE[0], PALETTE[4], PALETTE[3]];
    chart
        .draw_series(bars.iter().enumerate().map(|(i, (_, v, _))| {
            Rectangle::new([(i, 0.0), (i + 1, *v)], colors[i % colors.len()].filled())
        }))
        .map_err(map_err)?;

    root.present().map_err(map_err)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    fn is_svg(path: &Path) -> bool {
        let body = std::fs::read_to_string(path).unwrap();
        (body.starts_with("<?xml") || body.contains("<svg")) && {
            let re = Regex::new(r"USER_[A-Z]+").unwrap();
            !re.is_match(&body)
        }
    }

    #[test]
    fn test_entropy_by_channel_svg() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h3.svg");
        let data = vec![
            ("general".to_string(), 0.2),
            ("random".to_string(), 0.8),
            ("dev".to_string(), 0.5),
        ];
        assert!(entropy_by_channel_svg(&p, &data, 20).unwrap());
        assert!(p.exists());
        assert!(is_svg(&p));

        let p2 = dir.path().join("h3empty.svg");
        assert!(!entropy_by_channel_svg(&p2, &[], 20).unwrap());
        assert!(!p2.exists());
    }

    #[test]
    fn test_novel_vocab_svg() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h5.svg");
        let m = vec![
            ("2025-01".to_string(), 1.0),
            ("2025-02".to_string(), 0.6),
            ("2025-03".to_string(), 0.3),
        ];
        assert!(novel_vocab_svg(&p, &m).unwrap());
        assert!(p.exists());
        assert!(is_svg(&p));

        let p2 = dir.path().join("h5one.svg");
        assert!(!novel_vocab_svg(&p2, &[("2025-01".to_string(), 1.0)]).unwrap());
    }

    #[test]
    fn test_health_summary_svg() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sum.svg");
        let s = ReportSummary {
            overall_health_score: 0.42,
            red_flags: vec!["a".into(), "b".into()],
            yellow_flags: vec!["c".into()],
            green_signals: vec![],
        };
        assert!(health_summary_svg(&p, &s).unwrap());
        assert!(p.exists());
        assert!(is_svg(&p));
    }
}
