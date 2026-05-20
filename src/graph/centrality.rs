//! Centrality measures over the reply graph．

use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use super::builder::ReplyGraph;

impl ReplyGraph {
    /// Weighted PageRank via power iteration．
    pub fn pagerank(&self, damping: f64, max_iter: usize) -> Vec<(String, f64)> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }
        let nf = n as f64;
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();

        let out_w: HashMap<NodeIndex, f64> = nodes
            .iter()
            .map(|&ni| {
                let w: f64 = self
                    .graph
                    .edges_directed(ni, petgraph::Direction::Outgoing)
                    .map(|e| e.weight().reply_count as f64)
                    .sum();
                (ni, w)
            })
            .collect();

        let mut pr: HashMap<NodeIndex, f64> = nodes.iter().map(|&ni| (ni, 1.0 / nf)).collect();

        for _ in 0..max_iter {
            let dangling: f64 = nodes
                .iter()
                .filter(|&&ni| out_w[&ni] == 0.0)
                .map(|&ni| pr[&ni])
                .sum();

            let mut next: HashMap<NodeIndex, f64> = nodes
                .iter()
                .map(|&ni| (ni, (1.0 - damping) / nf + damping * dangling / nf))
                .collect();

            for &u in &nodes {
                let ow = out_w[&u];
                if ow == 0.0 {
                    continue;
                }
                let pu = pr[&u];
                for e in self.graph.edges_directed(u, petgraph::Direction::Outgoing) {
                    let v = e.target();
                    let contrib = damping * pu * (e.weight().reply_count as f64) / ow;
                    *next.get_mut(&v).expect("node exists") += contrib;
                }
            }

            let delta: f64 = nodes.iter().map(|&ni| (next[&ni] - pr[&ni]).abs()).sum();
            pr = next;
            if delta < 1e-9 {
                break;
            }
        }

        let mut out: Vec<(String, f64)> = nodes
            .iter()
            .map(|&ni| (self.graph[ni].user_id.clone(), pr[&ni]))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Weighted in-degree centrality: incoming weight / (N-1). N<2 -> all 0.0．
    pub fn in_degree_centrality(&self) -> Vec<(String, f64)> {
        self.degree_centrality(petgraph::Direction::Incoming)
    }

    /// Weighted out-degree centrality: outgoing weight / (N-1). N<2 -> all 0.0．
    pub fn out_degree_centrality(&self) -> Vec<(String, f64)> {
        self.degree_centrality(petgraph::Direction::Outgoing)
    }

    fn degree_centrality(&self, dir: petgraph::Direction) -> Vec<(String, f64)> {
        let n = self.graph.node_count();
        let denom = if n < 2 { 0.0 } else { (n - 1) as f64 };
        let mut out: Vec<(String, f64)> = self
            .graph
            .node_indices()
            .map(|ni| {
                let w: f64 = self
                    .graph
                    .edges_directed(ni, dir)
                    .map(|e| e.weight().reply_count as f64)
                    .sum();
                let c = if denom == 0.0 { 0.0 } else { w / denom };
                (self.graph[ni].user_id.clone(), c)
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}
