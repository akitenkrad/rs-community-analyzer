//! Platform-agnostic input types and the [`AnalysisInput`] bundle.
//!
//! The library is stateless: all `compute_h*` functions receive an
//! [`AnalysisInput`] of in-memory slices.  Callers building adapters from
//! Slack / Discord / Teams / GitHub etc. map their platform records into the
//! types here once and pass them through unchanged.
//!
//! ## Identifiers
//!
//! All IDs (`Message::id`, `channel_id`, `author_id`, `User::id`,
//! `Channel::id`, `Reaction::user_id`, `Reaction::message_id`) are opaque
//! strings.  The library never parses or assumes their shape.  Slack-style
//! IDs (`"U0..."`, `"C0..."`) work; so do UUIDs, integer-as-string IDs,
//! email addresses, etc.
//!
//! ## Threads
//!
//! [`Message::thread_root_id`] follows the convention:
//! * `None` — top-level message (and, equivalently, a thread root).
//! * `Some(other_id)` — reply pointing to the thread root with `id == other_id`.
//!
//! Adapters may also encode roots as `Some(self.id)`; [`crate::analysis::group_threads`]
//! accepts both shapes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::models::{ChannelCategory, Role};

/// A message in any chat / collaboration platform.  Platform-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Platform-agnostic message identifier (opaque string).
    pub id: String,
    /// Identifier of the channel this message belongs to.
    pub channel_id: String,
    /// Identifier of the user who authored this message.
    pub author_id: String,
    /// Raw message text (no markdown preprocessing assumed).
    pub text: String,
    /// Send time (UTC).  Replaces Slack's `"<epoch>.<micros>"` string.
    pub timestamp: DateTime<Utc>,
    /// `None` => top-level / thread root; `Some(root_id)` => reply.
    /// A root may equivalently be encoded as `Some(self.id)`; both forms are
    /// accepted (see module docs).
    pub thread_root_id: Option<String>,
    /// Aggregate reaction count if known; `0` when unused.
    pub reaction_count: usize,
}

/// A channel / room / forum.  Platform-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub id: String,
    pub name: String,
    /// Pre-resolved by the caller via config or platform metadata.
    pub category: Option<ChannelCategory>,
    /// Decision / leadership channel (set by the caller).
    pub is_decision_channel: bool,
}

/// A user / member.  Platform-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub display_name: String,
    /// Pre-resolved by the caller; if unknown set [`Role::Unknown`].
    pub role: Role,
}

/// A reaction (e.g. emoji on a message).  First-class — no JSON parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reaction {
    pub message_id: String,
    pub channel_id: String,
    pub user_id: String,
    /// Canonical short name (e.g. `"thumbsup"`, `"+1"`).  Callers strip the
    /// surrounding `':'` and lowercase it.
    pub emoji_name: String,
}

/// Bundle of in-memory slices that every `compute_h*` consumes.
///
/// The bundle borrows its caller's data; no ownership transfer is performed.
pub struct AnalysisInput<'a> {
    pub messages: &'a [Message],
    pub channels: &'a [Channel],
    pub users: &'a [User],
    pub reactions: &'a [Reaction],
    pub config: &'a crate::CommConfig,
}
