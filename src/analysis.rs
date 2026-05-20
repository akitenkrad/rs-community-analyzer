//! Shared period / channel scope resolution, thread grouping, output-directory helpers.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Duration, NaiveDate, NaiveTime, TimeZone, Utc};

use crate::error::{CommError, Result};
use crate::types::Message;

/// Convert a `DateTime<Utc>` to `f64` epoch seconds with sub-second precision．
///
/// Used by the metrics layer wherever the source crate used the Slack-specific
/// `ts_secs("<epoch>.<micros>")` helper．
pub fn timestamp_secs_f64(t: DateTime<Utc>) -> f64 {
    t.timestamp() as f64 + t.timestamp_subsec_micros() as f64 / 1_000_000.0
}

/// Convert a `chrono::Duration` to `f64` seconds (sign preserved)．
pub fn duration_secs_f64(d: Duration) -> f64 {
    d.num_seconds() as f64 + (d.subsec_nanos() as f64 / 1_000_000_000.0)
}

/// A group of messages sharing one `(channel_id, thread_key)`．
///
/// `thread_key` is the thread root identifier: `thread_root_id` when present
/// and not equal to `id`, otherwise the message's own `id`. `messages` is
/// sorted ascending by `timestamp`.
pub struct ThreadGroup {
    pub channel_id: String,
    pub thread_key: String,
    pub messages: Vec<Message>,
}

/// Group `messages` into threads keyed by `(channel_id, thread_root_id || id)`．
///
/// Within each [`ThreadGroup`], messages are sorted ascending by `timestamp`．
/// Final group ordering is `(channel_id, thread_key)` for deterministic
/// downstream iteration．
///
/// Adapters may encode thread roots as either `thread_root_id = None` or
/// `thread_root_id = Some(self.id)`; both shapes produce identical groupings．
pub fn group_threads(messages: &[Message]) -> Vec<ThreadGroup> {
    let mut buckets: HashMap<(String, String), Vec<Message>> = HashMap::new();
    for m in messages {
        let thread_key = m.thread_root_id.clone().unwrap_or_else(|| m.id.clone());
        buckets
            .entry((m.channel_id.clone(), thread_key))
            .or_default()
            .push(m.clone());
    }
    let mut groups: Vec<ThreadGroup> = buckets
        .into_iter()
        .map(|((channel_id, thread_key), mut msgs)| {
            msgs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            ThreadGroup {
                channel_id,
                thread_key,
                messages: msgs,
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        (a.channel_id.as_str(), a.thread_key.as_str())
            .cmp(&(b.channel_id.as_str(), b.thread_key.as_str()))
    });
    groups
}

fn day_start(date: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&date.and_time(NaiveTime::from_hms_opt(0, 0, 0).expect("00:00:00")))
}

fn day_end(date: NaiveDate) -> DateTime<Utc> {
    Utc.from_utc_datetime(&date.and_time(NaiveTime::from_hms_opt(23, 59, 59).expect("23:59:59")))
}

fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
        .map_err(|e| CommError::Config(format!("invalid date '{s}': {e}")))
}

fn last_day_of_month(year: i32, month: u32) -> NaiveDate {
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    NaiveDate::from_ymd_opt(ny, nm, 1)
        .expect("valid first-of-next-month")
        .pred_opt()
        .expect("valid last day")
}

/// Parse a quarter spec into `(q, fy)` where `q in 1..=4`．
///
/// Accepted forms: `Q<q>-<yyyy>`, `<yyyy>-Q<q>`, `FY<yyyy>-Q<q>`, `<yyyy>Q<q>`．
fn parse_quarter(s: &str) -> Result<(u32, i32)> {
    let up = s.trim().to_ascii_uppercase();
    let err = || CommError::Config(format!("invalid --quarter '{s}'"));

    let (q_str, y_str): (String, String) = if let Some(rest) = up.strip_prefix("FY") {
        let (y, q) = rest.split_once("-Q").ok_or_else(err)?;
        (q.to_string(), y.to_string())
    } else if let Some(rest) = up.strip_prefix('Q') {
        let (q, y) = rest.split_once('-').ok_or_else(err)?;
        (q.to_string(), y.to_string())
    } else if let Some((y, q)) = up.split_once("-Q") {
        (q.to_string(), y.to_string())
    } else if let Some((y, q)) = up.split_once('Q') {
        (q.to_string(), y.to_string())
    } else {
        return Err(err());
    };

    let q: u32 = q_str.trim().parse().map_err(|_| err())?;
    let y: i32 = y_str.trim().parse().map_err(|_| err())?;
    if !(1..=4).contains(&q) {
        return Err(err());
    }
    Ok((q, y))
}

fn quarter_dates(q: u32, fy: i32, fiscal_start_month: u32) -> (NaiveDate, NaiveDate) {
    let offset = (q - 1) * 3;
    let raw_month = fiscal_start_month + offset;
    let years_forward = (raw_month - 1) / 12;
    let start_month = ((raw_month - 1) % 12) + 1;
    let start_year = fy + years_forward as i32;
    let start =
        NaiveDate::from_ymd_opt(start_year, start_month, 1).expect("valid quarter-start date");

    let raw_end_month = start_month + 2;
    let end_years_forward = (raw_end_month - 1) / 12;
    let end_month = ((raw_end_month - 1) % 12) + 1;
    let end_year = start_year + end_years_forward as i32;
    let end = last_day_of_month(end_year, end_month);
    (start, end)
}

/// A period filter (UTC half-inclusive: `from <= t <= to`)．
///
/// The library deliberately does NOT take CLI-layer concepts (e.g. `--last`,
/// `--quarter`)．Use the builder constructors below to translate CLI strings
/// into a [`PeriodFilter`]，then call [`apply_filters`]．
#[derive(Debug, Clone, Default)]
pub struct PeriodFilter {
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
}

impl PeriodFilter {
    /// Build a `--last` style filter relative to `anchor`．Spec is e.g．
    /// `"6m"`, `"30d"`, `"1y"`, `"2w"`．Approximates months as 30 days and
    /// years as 365 days (no calendar arithmetic — matches the source crate)．
    pub fn last(spec: &str, anchor: DateTime<Utc>) -> Result<Self> {
        if spec.is_empty() {
            return Err(CommError::Config("empty --last spec".into()));
        }
        let (num_str, unit) = spec.split_at(spec.len() - 1);
        let n: i64 = num_str
            .parse()
            .map_err(|_| CommError::Config(format!("invalid --last '{spec}'")))?;
        let days = match unit {
            "d" => n,
            "w" => n * 7,
            "m" => n * 30,
            "y" => n * 365,
            _ => {
                return Err(CommError::Config(format!(
                    "invalid --last unit in '{spec}' (expected d|w|m|y)"
                )))
            }
        };
        Ok(Self {
            from: Some(anchor - Duration::days(days)),
            to: Some(anchor),
        })
    }

    /// Build a quarter filter (`"FY2025-Q4"`, `"2025-Q4"`, …) given the
    /// fiscal start month (1..=12)．
    pub fn quarter(spec: &str, fiscal_start_month: u32) -> Result<Self> {
        let (q, fy) = parse_quarter(spec)?;
        let (start, end) = quarter_dates(q, fy, fiscal_start_month);
        Ok(Self {
            from: Some(day_start(start)),
            to: Some(day_end(end)),
        })
    }

    /// Build from `YYYY-MM-DD` date strings (either or both may be `None`)．
    pub fn from_dates(from: Option<&str>, to: Option<&str>) -> Result<Self> {
        let from = match from {
            Some(s) => Some(day_start(parse_date(s)?)),
            None => None,
        };
        let to = match to {
            Some(s) => Some(day_end(parse_date(s)?)),
            None => None,
        };
        Ok(Self { from, to })
    }

    /// `true` iff `timestamp` falls within `[from, to]` (inclusive both sides)．
    /// Unbounded sides match anything．
    pub fn contains(&self, ts: DateTime<Utc>) -> bool {
        if let Some(f) = self.from {
            if ts < f {
                return false;
            }
        }
        if let Some(t) = self.to {
            if ts > t {
                return false;
            }
        }
        true
    }
}

/// Filter `messages` by `period` and `channel_ids`．
///
/// * `period` — applied via [`PeriodFilter::contains`]．Unbounded sides match
///   anything．
/// * `channel_ids` — if non-empty, only messages whose `channel_id` matches
///   any element pass．Empty => no channel restriction．
pub fn apply_filters(
    messages: &[Message],
    period: &PeriodFilter,
    channel_ids: &[String],
) -> Vec<Message> {
    let allowed: Option<std::collections::HashSet<&str>> = if channel_ids.is_empty() {
        None
    } else {
        Some(channel_ids.iter().map(|s| s.as_str()).collect())
    };
    messages
        .iter()
        .filter(|m| {
            if let Some(set) = &allowed {
                if !set.contains(m.channel_id.as_str()) {
                    return false;
                }
            }
            period.contains(m.timestamp)
        })
        .cloned()
        .collect()
}

/// Resolve (or create) the output directory for a subcommand．
pub fn make_output_dir(subcommand: &str, override_dir: Option<&str>) -> Result<PathBuf> {
    let dir = match override_dir {
        Some(d) => PathBuf::from(d),
        None => {
            let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
            PathBuf::from(format!("output/comm/{}_{}", subcommand, stamp))
        }
    };
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).single().unwrap()
    }

    fn msg(ch: &str, id: &str, ts: DateTime<Utc>, root: Option<&str>) -> Message {
        Message {
            id: id.to_string(),
            channel_id: ch.to_string(),
            author_id: "U".to_string(),
            text: String::new(),
            timestamp: ts,
            thread_root_id: root.map(String::from),
            reaction_count: 0,
        }
    }

    #[test]
    fn test_timestamp_secs_roundtrip() {
        let t = ts(1_700_000_000);
        assert!((timestamp_secs_f64(t) - 1_700_000_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_group_threads_groups_and_orders() {
        let root = "M1";
        let msgs = vec![
            msg("C1", "M1", ts(100), None),
            msg("C1", "M2", ts(120), Some(root)),
            msg("C1", "M3", ts(110), Some(root)),
            msg("C2", "M4", ts(300), None),
            msg("C2", "M5", ts(310), Some("M4")),
        ];
        let groups = group_threads(&msgs);
        assert_eq!(groups.len(), 2);
        let g1 = groups
            .iter()
            .find(|g| g.channel_id == "C1" && g.thread_key == "M1")
            .expect("C1 thread");
        assert_eq!(g1.messages.len(), 3);
        assert_eq!(g1.messages[0].id, "M1");
        assert_eq!(g1.messages[1].id, "M3");
        assert_eq!(g1.messages[2].id, "M2");
    }

    #[test]
    fn test_group_threads_root_as_self_id() {
        // Adapter encodes a root as `thread_root_id = Some(self.id)`.
        let msgs = vec![
            msg("C1", "M1", ts(100), Some("M1")),
            msg("C1", "M2", ts(110), Some("M1")),
        ];
        let groups = group_threads(&msgs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].thread_key, "M1");
        assert_eq!(groups[0].messages.len(), 2);
    }

    #[test]
    fn test_quarter_fy_q4_with_april_start() {
        let pf = PeriodFilter::quarter("FY2025-Q4", 4).unwrap();
        let from = pf.from.unwrap();
        let to = pf.to.unwrap();
        assert_eq!(from.format("%Y-%m-%d").to_string(), "2026-01-01");
        assert_eq!(to.format("%Y-%m-%d").to_string(), "2026-03-31");
    }

    #[test]
    fn test_quarter_other_forms() {
        for spec in ["Q4-2025", "2025-Q4", "2025Q4", "FY2025-Q4"] {
            let pf = PeriodFilter::quarter(spec, 4).unwrap();
            assert!(pf.from.is_some());
        }
    }

    #[test]
    fn test_last_relative_to_anchor() {
        let anchor = ts(1_700_000_000);
        let pf = PeriodFilter::last("3m", anchor).unwrap();
        assert_eq!(pf.to.unwrap(), anchor);
        let expected_from = anchor - Duration::days(90);
        assert_eq!(pf.from.unwrap(), expected_from);
    }

    #[test]
    fn test_invalid_date_is_err() {
        assert!(PeriodFilter::from_dates(Some("2025-13-99"), None).is_err());
    }

    #[test]
    fn test_apply_filters_basic() {
        let msgs = vec![
            msg("C1", "a", ts(100), None),
            msg("C1", "b", ts(200), None),
            msg("C2", "c", ts(300), None),
        ];
        let pf = PeriodFilter {
            from: Some(ts(150)),
            to: Some(ts(250)),
        };
        let filtered = apply_filters(&msgs, &pf, &[]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "b");

        let filtered = apply_filters(&msgs, &PeriodFilter::default(), &["C2".to_string()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].channel_id, "C2");
    }
}
