//! Hierarchy score via strongly-connected-component analysis．

use petgraph::algo::tarjan_scc;

use super::builder::ReplyGraph;

impl ReplyGraph {
    /// Hierarchy score in `[0, 1]`．
    ///
    /// `1.0` => a perfect DAG (purely hierarchical)．Lower values => more
    /// reciprocal / flat interaction (mutual reply cycles)．
    pub fn hierarchy_score(&self) -> f64 {
        let n = self.graph.node_count();
        if n == 0 {
            return 0.0;
        }
        let cyclic_nodes: usize = tarjan_scc(&self.graph)
            .into_iter()
            .filter(|scc| scc.len() > 1)
            .map(|scc| scc.len())
            .sum();
        1.0 - cyclic_nodes as f64 / n as f64
    }
}
