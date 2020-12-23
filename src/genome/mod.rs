use std::collections::{HashMap, HashSet, VecDeque};

use crate::activation::ActivationKind;
use crate::node::NodeKind;
pub use crossover::*;
pub use genes::{ConnectionGene, NodeGene};
use mutation::MutationKind;

pub mod crossover;
pub mod genes;
pub mod mutation;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Genome {
    inputs: usize,
    outputs: usize,
    connection_genes: Vec<ConnectionGene>,
    node_genes: Vec<NodeGene>,
}

impl Genome {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut node_genes = vec![];

        (0..inputs).for_each(|_| node_genes.push(NodeGene::new(NodeKind::Input)));
        (0..outputs).for_each(|_| node_genes.push(NodeGene::new(NodeKind::Output)));

        let connection_genes: Vec<ConnectionGene> = (0..inputs)
            .flat_map(|i| {
                (inputs..inputs + outputs)
                    .map(|o| ConnectionGene::new(i, o))
                    .collect::<Vec<ConnectionGene>>()
            })
            .collect();

        Genome {
            inputs,
            outputs,
            connection_genes,
            node_genes,
        }
    }

    fn empty(inputs: usize, outputs: usize) -> Self {
        Genome {
            inputs,
            outputs,
            connection_genes: vec![],
            node_genes: vec![],
        }
    }

    pub fn input_count(&self) -> usize {
        self.inputs
    }

    pub fn output_count(&self) -> usize {
        self.outputs
    }

    pub fn nodes(&self) -> &[NodeGene] {
        &self.node_genes
    }

    pub fn connections(&self) -> &[ConnectionGene] {
        &self.connection_genes
    }

    fn calculate_node_order(
        &self,
        additional_connections: Option<Vec<ConnectionGene>>,
    ) -> Option<Vec<usize>> {
        let mut connections: Vec<ConnectionGene> = self
            .connection_genes
            .iter()
            .filter(|c| !c.disabled)
            .cloned()
            .collect();

        if let Some(mut conns) = additional_connections {
            connections.append(&mut conns);
        }

        if connections.is_empty() {
            return None;
        }

        let mut visited: Vec<usize> = vec![];

        // Input nodes are automatically visited as they get their values from inputs
        self.node_genes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
            .for_each(|(i, _)| {
                visited.push(i);
            });

        let mut newly_visited = 1;
        while newly_visited != 0 {
            newly_visited = 0;

            let mut nodes_to_visit: Vec<usize> = self
                .node_genes
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    // The node is not visited but all prerequisite nodes are visited
                    !visited.contains(i)
                        && connections
                            .iter()
                            .filter(|c| c.to == *i)
                            .map(|c| c.from)
                            .all(|node_index| visited.contains(&node_index))
                })
                .map(|(i, _)| i)
                .collect();

            newly_visited += nodes_to_visit.len();
            visited.append(&mut nodes_to_visit);
        }

        if visited.len() != self.node_genes.len() {
            return None;
        }

        Some(visited)
    }

    pub fn node_order(&self) -> Option<Vec<usize>> {
        self.calculate_node_order(None)
    }

    pub fn node_order_with(
        &self,
        additional_connections: Vec<ConnectionGene>,
    ) -> Option<Vec<usize>> {
        self.calculate_node_order(Some(additional_connections))
    }

    fn calculate_node_distance_from_inputs(&self) -> HashMap<usize, usize> {
        // Inputs are immediately added with distance of 0
        let mut distances: HashMap<usize, usize> = self
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
            .map(|(i, _)| (i, 0))
            .collect();

        // Inputs need to be visited first
        let mut to_visit: VecDeque<usize> = self
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
            .map(|(i, _)| i)
            .collect();

        while let Some(i) = to_visit.pop_front() {
            let source_distance = *distances.get(&i).unwrap_or(&0);

            self.connections()
                .iter()
                .filter(|c| c.from == i)
                .for_each(|c| {
                    let node_index = c.to;
                    let potential_distance = source_distance + 1;

                    let maybe_change = if let Some(distance) = distances.get(&node_index) {
                        if potential_distance > *distance {
                            to_visit.push_back(node_index);
                            Some(potential_distance)
                        } else {
                            None
                        }
                    } else {
                        to_visit.push_back(node_index);
                        Some(potential_distance)
                    };

                    if let Some(new_distance) = maybe_change {
                        distances.insert(node_index, new_distance);
                    }
                });
        }

        distances
    }

    fn is_projecting_directly(&self, source: usize, target: usize) -> bool {
        self.connection_genes
            .iter()
            .filter(|c| !c.disabled)
            .any(|c| c.from == source && c.to == target)
    }

    fn is_projected_directly(&self, target: usize, source: usize) -> bool {
        self.is_projecting_directly(source, target)
    }

    fn is_projecting(&self, source: usize, target: usize) -> bool {
        let mut visited_nodes: HashSet<usize> = HashSet::new();
        let mut nodes_to_visit: VecDeque<usize> = VecDeque::new();

        nodes_to_visit.push_back(source);

        let mut projecting = false;
        while let Some(i) = nodes_to_visit.pop_front() {
            visited_nodes.insert(i);
            if self.is_projecting_directly(i, target) {
                projecting = true;
                break;
            } else {
                self.connection_genes
                    .iter()
                    .filter(|c| c.from == i && !c.disabled && !visited_nodes.contains(&i))
                    .for_each(|c| nodes_to_visit.push_back(c.to));
            }
        }

        projecting
    }

    fn is_projected(&self, target: usize, source: usize) -> bool {
        self.is_projecting(source, target)
    }

    fn can_connect(&self, from: usize, to: usize) -> bool {
        let from_node = self.node_genes.get(from).unwrap();
        let to_node = self.node_genes.get(to).unwrap();

        let is_from_output = matches!(from_node.kind, NodeKind::Output);
        let is_to_input = matches!(to_node.kind, NodeKind::Input);

        let distances = self.calculate_node_distance_from_inputs();
        let from_distance = distances.get(&from).unwrap();
        let to_distance = distances.get(&to).unwrap_or(&usize::MAX);
        let is_recurrent = from_distance > to_distance;

        if is_from_output || is_to_input || is_recurrent {
            false
        } else {
            !self.is_projecting(from, to)
        }
    }

    fn add_connection(&mut self, from: usize, to: usize) -> Result<usize, ()> {
        if !self.can_connect(from, to) {
            return Err(());
        }

        let maybe_connection = self
            .connection_genes
            .iter_mut()
            .find(|c| c.from == from && c.to == to);

        if let Some(mut conn) = maybe_connection {
            conn.disabled = false;
        } else {
            self.connection_genes.push(ConnectionGene::new(from, to));
        }

        Ok(self.connection_genes.len() - 1)
    }

    fn add_many_connections(&mut self, params: &[(usize, usize)]) -> Vec<Result<usize, ()>> {
        let results = params
            .iter()
            .map(|(from, to)| self.add_connection(*from, *to))
            .collect();

        results
    }

    fn disable_connection(&mut self, index: usize) {
        self.connection_genes.get_mut(index).unwrap().disabled = true;
    }

    fn disable_many_connections(&mut self, indexes: &[usize]) {
        indexes.iter().for_each(|i| self.disable_connection(*i));
    }

    /// Add a new hidden node to the genome
    fn add_node(&mut self) -> usize {
        let index = self.node_genes.len();
        self.node_genes.push(NodeGene::new(NodeKind::Hidden));

        index
    }

    pub fn mutate(&mut self, kind: &MutationKind) {
        mutation::mutate(kind, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize() {
        Genome::new(2, 2);
    }

    #[test]
    fn add_node_does_not_change_connections() {
        let mut g = Genome::new(1, 2);

        g.add_node();

        let first_connection = g.connection_genes.get(0).unwrap();
        assert_eq!(first_connection.from, 0);
        assert_eq!(first_connection.to, 1);

        let second_connection = g.connection_genes.get(1).unwrap();
        assert_eq!(second_connection.from, 0);
        assert_eq!(second_connection.to, 2);
    }

    #[test]
    fn is_projecting_directly() {
        let g = Genome::new(2, 2);

        assert!(g.is_projecting_directly(0, 2));
        assert!(g.is_projecting_directly(0, 3));
        assert!(g.is_projecting_directly(1, 2));
        assert!(g.is_projecting_directly(1, 3));

        assert!(!g.is_projecting_directly(2, 0));
        assert!(!g.is_projecting_directly(3, 0));
        assert!(!g.is_projecting_directly(2, 1));
        assert!(!g.is_projecting_directly(3, 1));
    }

    #[test]
    fn is_projected_directly() {
        let g = Genome::new(2, 2);

        assert!(g.is_projected_directly(2, 0));
        assert!(g.is_projected_directly(3, 0));
        assert!(g.is_projected_directly(2, 1));
        assert!(g.is_projected_directly(3, 1));

        assert!(!g.is_projected_directly(0, 2));
        assert!(!g.is_projected_directly(0, 3));
        assert!(!g.is_projected_directly(1, 2));
        assert!(!g.is_projected_directly(1, 3));
    }

    // TODO rewrite the tests or both the implementation and tests

    // #[test]
    // fn is_projecting() {
    //     let mut g = Genome::empty(1, 1);

    //     g.node_genes.push(NodeGene::new(NodeKind::Input));
    //     g.node_genes.push(NodeGene::new(NodeKind::Hidden));
    //     g.node_genes.push(NodeGene::new(NodeKind::Hidden));
    //     g.node_genes.push(NodeGene::new(NodeKind::Output));

    //     g.connection_genes.push(ConnectionGene::new(0, 1));
    //     g.connection_genes.push(ConnectionGene::new(1, 2));
    //     g.connection_genes.push(ConnectionGene::new(2, 3));

    //     assert!(g.is_projecting(0, 3));
    //     assert!(g.is_projecting(1, 3));
    //     assert!(g.is_projecting(2, 3));

    //     assert!(!g.is_projecting(3, 0));
    //     assert!(!g.is_projecting(3, 1));
    //     assert!(!g.is_projecting(3, 2));
    // }

    // #[test]
    // fn is_projected() {
    //     let mut g = Genome::empty(1, 1);

    //     g.node_genes.push(NodeGene::new(NodeKind::Input));
    //     g.node_genes.push(NodeGene::new(NodeKind::Hidden));
    //     g.node_genes.push(NodeGene::new(NodeKind::Hidden));
    //     g.node_genes.push(NodeGene::new(NodeKind::Output));

    //     g.connection_genes.push(ConnectionGene::new(0, 1));
    //     g.connection_genes.push(ConnectionGene::new(1, 2));
    //     g.connection_genes.push(ConnectionGene::new(2, 3));

    //     assert!(g.is_projected(3, 0));
    //     assert!(g.is_projected(3, 1));
    //     assert!(g.is_projected(3, 2));

    //     assert!(!g.is_projected(0, 3));
    //     assert!(!g.is_projected(1, 3));
    //     assert!(!g.is_projected(2, 3));
    // }

    #[test]
    fn can_connect() {
        let mut g = Genome::empty(1, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Output));

        g.connection_genes.push(ConnectionGene::new(0, 1));
        g.connection_genes.push(ConnectionGene::new(0, 2));
        g.connection_genes.push(ConnectionGene::new(1, 3));
        g.connection_genes.push(ConnectionGene::new(2, 3));
        g.connection_genes.push(ConnectionGene::new(3, 4));

        assert!(g.can_connect(1, 2));
        assert!(g.can_connect(2, 1));

        assert!(!g.can_connect(3, 1));
        assert!(!g.can_connect(3, 2));
        assert!(!g.can_connect(4, 1));
        assert!(!g.can_connect(4, 2));
    }

    #[test]
    fn get_node_order() {
        let mut g = Genome::empty(2, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Output));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));

        g.add_connection(0, 2).unwrap();
        g.add_connection(1, 3).unwrap();
        g.add_connection(1, 4).unwrap();
        g.add_connection(1, 5).unwrap();
        g.add_connection(3, 2).unwrap();
        g.add_connection(4, 3).unwrap();
        g.add_connection(5, 4).unwrap();

        assert!(g.node_order().is_some());
        assert!(g.node_order_with(vec![ConnectionGene::new(3, 5)]).is_none());
    }

    #[test]
    fn no_recurrent_connections() {
        let mut g = Genome::empty(2, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Output));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));

        g.add_connection(0, 2).unwrap();
        g.add_connection(1, 3).unwrap();
        g.add_connection(1, 4).unwrap();
        g.add_connection(1, 5).unwrap();
        g.add_connection(3, 2).unwrap();
        g.add_connection(4, 3).unwrap();
        g.add_connection(5, 4).unwrap();

        assert!(g.add_connection(3, 5).is_err());
    }

    #[test]
    fn node_distances_simple() {
        let g = Genome::new(2, 1);

        dbg!(g.calculate_node_distance_from_inputs());
    }

    #[test]
    fn node_distances_block_recurrent_connections() {
        let mut g = Genome::empty(2, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Output));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));

        g.add_connection(0, 3).unwrap();
        g.add_connection(1, 3).unwrap();
        g.add_connection(3, 4).unwrap();
        g.add_connection(4, 5).unwrap();
        g.add_connection(4, 2).unwrap();
        g.add_connection(5, 2).unwrap();

        assert!(g.add_connection(5, 3).is_err());
    }
}
