use rand::distributions::{Distribution, Standard};
use rand::{random, Rng};
use std::collections::{HashMap, VecDeque};

use crate::activation::ActivationKind;
use crate::node::NodeKind;

#[derive(Debug, Clone)]
struct ConnectionGene {
    from: usize,
    to: usize,
    weight: f64,
    disabled: bool,
}

impl ConnectionGene {
    pub fn new(from: usize, to: usize) -> Self {
        ConnectionGene {
            from,
            to,
            weight: random::<f64>() - 0.5,
            disabled: false,
        }
    }

    pub fn innovation_number(&self) -> usize {
        let a = self.from;
        let b = self.to;

        let first_part = (a + b) * (a + b + 1);
        let second_part = b;

        first_part.checked_div(2).unwrap() + second_part
    }
}

#[derive(Debug, Clone)]
struct NodeGene {
    kind: NodeKind,
    activation: ActivationKind,
    bias: f64,
}

impl NodeGene {
    pub fn new(kind: NodeKind) -> Self {
        let activation: ActivationKind = match kind {
            NodeKind::Input => ActivationKind::Input,
            _ => rand::random(),
        };
        let bias: f64 = match kind {
            NodeKind::Input => 0.,
            _ => rand::random::<f64>() - 0.5,
        };

        NodeGene {
            kind,
            activation,
            bias,
        }
    }
}

enum MutationKind {
    AddConnection,
    AddNode,
}

impl Distribution<MutationKind> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MutationKind {
        match rng.gen_range(0, 2) {
            0 => MutationKind::AddConnection,
            _ => MutationKind::AddNode,
        }
    }
}

#[derive(Debug)]
struct Genome {
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

    pub fn crossover(a: (Self, f64), b: (Self, f64)) -> Self {
        let inputs_count_not_equal = a.0.inputs != b.0.inputs;
        let outputs_count_not_equal = a.0.outputs != b.0.outputs;

        if inputs_count_not_equal || outputs_count_not_equal {
            panic!("Cannot cross genomes with different inputs or outputs");
        }

        let genome_a = a.0;
        let fitness_a: f64 = a.1;

        let genome_b = b.0;
        let fitness_b: f64 = b.1;

        let fitnesses_equal = (fitness_a - fitness_b).abs() < f64::EPSILON;

        let node_count = if fitnesses_equal {
            let node_max = usize::max(genome_a.node_genes.len(), genome_b.node_genes.len());
            let node_min = usize::min(genome_a.node_genes.len(), genome_b.node_genes.len());

            node_min + (random::<usize>() % (node_max - node_min))
        } else if fitness_a > fitness_b {
            genome_a.node_genes.len()
        } else {
            genome_b.node_genes.len()
        };

        let mut genome = Genome::empty(genome_a.inputs, genome_a.outputs);

        // Copy the input nodes
        (0..genome_a.inputs).for_each(|i| {
            genome
                .node_genes
                .push(genome_a.node_genes.get(i).unwrap().clone())
        });

        // Pick hidden nodes
        (genome_a.inputs..node_count - genome_a.outputs).for_each(|i| {
            let node_a = genome_a.node_genes.get(i);
            let node_b = genome_b.node_genes.get(i);

            // If one of the genomes is much shorter, this index might be out of bounds
            if node_a.is_none() || node_b.is_none() {
                genome.node_genes.push(node_a.or(node_b).unwrap().clone());
                return;
            }

            let node_a = node_a.unwrap();
            let node_b = node_b.unwrap();

            let picked_node = match (
                matches!(node_a.kind, NodeKind::Hidden),
                matches!(node_b.kind, NodeKind::Hidden),
            ) {
                (true, false) => node_a.clone(),
                (false, true) => node_b.clone(),
                (true, true) => {
                    if random::<f64>() < 0.5 {
                        node_a.clone()
                    } else {
                        node_b.clone()
                    }
                }
                _ => panic!("Both nodes are not of kind hidden"),
            };

            genome.node_genes.push(picked_node);
        });

        // Pick output nodes
        (node_count - genome_a.outputs..node_count).for_each(|i| {
            genome.node_genes.push(if random::<f64>() < 0.5 {
                genome_a.node_genes.get(i).unwrap().clone()
            } else {
                genome_b.node_genes.get(i).unwrap().clone()
            });
        });

        // TODO do connections
        let mut is_gene_common: HashMap<usize, bool> = HashMap::new();

        genome_a.connection_genes.iter().for_each(|c| {
            let num = c.innovation_number();

            match is_gene_common.get(&num) {
                None => {
                    is_gene_common.insert(num, false);
                }
                Some(false) => {
                    is_gene_common.insert(num, true);
                }
                _ => {}
            }
        });

        genome_b.connection_genes.iter().for_each(|c| {
            let num = c.innovation_number();

            match is_gene_common.get(&num) {
                None => {
                    is_gene_common.insert(num, false);
                }
                Some(false) => {
                    is_gene_common.insert(num, true);
                }
                _ => {}
            }
        });

        is_gene_common.iter().for_each(|(num, is_common)| {
            let a = genome_a
                .connection_genes
                .iter()
                .find(|c| c.innovation_number() == *num);
            let b = genome_b
                .connection_genes
                .iter()
                .find(|c| c.innovation_number() == *num);

            let picked = if *is_common {
                if random::<f64>() < 0.5 {
                    a.cloned()
                } else {
                    b.cloned()
                }
            } else {
                match (fitness_a > fitness_b, a.is_some()) {
                    (true, true) => a.cloned(),
                    (false, false) => b.cloned(),
                    (true, false) => None,
                    (false, true) => None,
                }
            };

            if let Some(conn) = picked {
                genome.connection_genes.push(conn);
            }
        });

        genome.connection_genes.sort_by(|a, b| {
            if a.from == b.from {
                a.to.cmp(&b.to)
            } else {
                a.from.cmp(&b.from)
            }
        });

        genome
    }

    fn is_projecting_directly(&self, source: usize, target: usize) -> bool {
        self.connection_genes
            .iter()
            .any(|c| c.from == source && c.to == target)
    }

    fn is_projected_directly(&self, target: usize, source: usize) -> bool {
        self.is_projecting_directly(source, target)
    }

    fn is_projecting(&self, source: usize, target: usize) -> bool {
        let mut nodes_to_visit: VecDeque<usize> = VecDeque::new();

        nodes_to_visit.push_back(source);

        let mut projecting = false;
        while let Some(i) = nodes_to_visit.pop_front() {
            if self.is_projecting_directly(i, target) {
                projecting = true;
                break;
            } else {
                self.connection_genes
                    .iter()
                    .filter(|c| c.from == i)
                    .for_each(|c| nodes_to_visit.push_back(c.to));
            }
        }

        projecting
    }

    fn is_projected(&self, target: usize, source: usize) -> bool {
        self.is_projecting(source, target)
    }

    fn can_connect(&self, source: usize, target: usize) -> bool {
        let source_node = self.node_genes.get(source).unwrap();
        let target_node = self.node_genes.get(source).unwrap();

        if matches!(source_node.kind, NodeKind::Output)
            || matches!(target_node.kind, NodeKind::Input)
        {
            false
        } else {
            !self.is_projecting(target, source)
        }
    }

    pub fn mutate(&mut self) {
        match random::<MutationKind>() {
            MutationKind::AddConnection => {
                // TODO
                // let possible_connections = (0..self.node_genes.len()).
            }
            MutationKind::AddNode => {
                //
            }
        }
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
    fn crossover() {
        let a = Genome::new(2, 2);
        let b = Genome::new(2, 2);

        Genome::crossover((a, 1.), (b, 2.));
    }

    #[test]
    #[should_panic]
    fn crossover_fail_1() {
        let a = Genome::new(2, 3);
        let b = Genome::new(2, 2);

        Genome::crossover((a, 1.), (b, 2.));
    }

    #[test]
    #[should_panic]
    fn crossover_fail_2() {
        let a = Genome::new(3, 2);
        let b = Genome::new(2, 2);

        Genome::crossover((a, 1.), (b, 2.));
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

    #[test]
    fn is_projecting() {
        let mut g = Genome::empty(1, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Output));

        g.connection_genes.push(ConnectionGene::new(0, 1));
        g.connection_genes.push(ConnectionGene::new(1, 2));
        g.connection_genes.push(ConnectionGene::new(2, 3));

        assert!(g.is_projecting(0, 3));
        assert!(g.is_projecting(1, 3));
        assert!(g.is_projecting(2, 3));

        assert!(!g.is_projecting(3, 0));
        assert!(!g.is_projecting(3, 1));
        assert!(!g.is_projecting(3, 2));
    }

    #[test]
    fn is_projected() {
        let mut g = Genome::empty(1, 1);

        g.node_genes.push(NodeGene::new(NodeKind::Input));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Hidden));
        g.node_genes.push(NodeGene::new(NodeKind::Output));

        g.connection_genes.push(ConnectionGene::new(0, 1));
        g.connection_genes.push(ConnectionGene::new(1, 2));
        g.connection_genes.push(ConnectionGene::new(2, 3));

        assert!(g.is_projected(3, 0));
        assert!(g.is_projected(3, 1));
        assert!(g.is_projected(3, 2));

        assert!(!g.is_projected(0, 3));
        assert!(!g.is_projected(1, 3));
        assert!(!g.is_projected(2, 3));
    }

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
}
