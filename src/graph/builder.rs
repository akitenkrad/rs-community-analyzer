//! Directed reply graph: `replier -> thread-root author`．

use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};

use crate::config::CommConfig;
use crate::models::Role;
use crate::types::Message;

/// A user vertex in the reply graph．
pub struct UserNode {
    pub user_id: String,
    pub role: Role,
    pub message_count: usize,
}

/// Aggregated edge weight from `replier` to `root author`．
#[derive(Default, Clone)]
pub struct EdgeWeight {
    pub reply_count: u32,
    /// Reserved for future use (manager-reaction is computed separately in H3)．
    pub reaction_count: u32,
}

/// Directed graph of reply interactions．
pub struct ReplyGraph {
    pub(crate) graph: DiGraph<UserNode, EdgeWeight>,
    idx: HashMap<String, NodeIndex>,
}

impl ReplyGraph {
    /// Build the reply graph from an already period/channel-filtered slice．
    ///
    /// A message is a *reply* iff `thread_root_id.is_some()` and
    /// `thread_root_id != Some(self.id)`. Its parent author is resolved from
    /// the thread-root message (`id == thread_root_id` in the same channel)
    /// within `messages`; if the root is not present in the slice the edge is
    /// skipped. Self-edges are skipped．
    pub fn build(messages: &[Message], cfg: &CommConfig) -> Self {
        let mut msg_count: HashMap<&str, usize> = HashMap::new();
        for m in messages {
            *msg_count.entry(m.author_id.as_str()).or_insert(0) += 1;
        }

        // Map (channel_id, id) -> root author.
        let mut root_author: HashMap<(&str, &str), &str> = HashMap::new();
        for m in messages {
            root_author.insert((m.channel_id.as_str(), m.id.as_str()), m.author_id.as_str());
        }

        let mut graph: DiGraph<UserNode, EdgeWeight> = DiGraph::new();
        let mut idx: HashMap<String, NodeIndex> = HashMap::new();

        let ensure_node = |graph: &mut DiGraph<UserNode, EdgeWeight>,
                           idx: &mut HashMap<String, NodeIndex>,
                           user_id: &str|
         -> NodeIndex {
            if let Some(ni) = idx.get(user_id) {
                return *ni;
            }
            let ni = graph.add_node(UserNode {
                user_id: user_id.to_string(),
                role: cfg.role_for(user_id),
                message_count: msg_count.get(user_id).copied().unwrap_or(0),
            });
            idx.insert(user_id.to_string(), ni);
            ni
        };

        for m in messages {
            ensure_node(&mut graph, &mut idx, &m.author_id);
        }

        for m in messages {
            let Some(root_id) = m.thread_root_id.as_deref() else {
                continue;
            };
            if root_id == m.id {
                continue; // thread root, not a reply
            }
            let Some(&root) = root_author.get(&(m.channel_id.as_str(), root_id)) else {
                continue; // root not in slice
            };
            if root == m.author_id {
                continue; // self-edge
            }
            let from = ensure_node(&mut graph, &mut idx, &m.author_id);
            let to = ensure_node(&mut graph, &mut idx, root);
            if let Some(eid) = graph.find_edge(from, to) {
                let w = graph.edge_weight_mut(eid).expect("edge exists");
                w.reply_count += 1;
            } else {
                graph.add_edge(
                    from,
                    to,
                    EdgeWeight {
                        reply_count: 1,
                        reaction_count: 0,
                    },
                );
            }
        }

        Self { graph, idx }
    }

    /// Number of user nodes．
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Role of a user node, if present．
    pub fn node_role(&self, user_id: &str) -> Option<Role> {
        self.idx.get(user_id).map(|ni| self.graph[*ni].role)
    }

    /// Sum of incoming reply weight per user (for Gini), sorted desc．
    pub fn replies_received(&self) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = self
            .graph
            .node_indices()
            .map(|ni| {
                let total: u32 = self
                    .graph
                    .edges_directed(ni, petgraph::Direction::Incoming)
                    .map(|e| e.weight().reply_count)
                    .sum();
                (self.graph[ni].user_id.clone(), total as f64)
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}
