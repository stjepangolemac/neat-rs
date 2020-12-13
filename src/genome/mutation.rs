use rand::distributions::{Distribution, Standard};
use rand::random;
use rand::Rng;

use super::{ActivationKind, Genome, NodeKind};

pub fn mutate(kind: MutationKind, g: &mut Genome) {
    match kind {
        MutationKind::AddConnection => add_connection(g),
        MutationKind::RemoveConnection => remove_connection(g),
        MutationKind::AddNode => add_node(g),
        MutationKind::RemoveNode => remove_node(g),
        MutationKind::ModifyWeight => change_weight(g),
        MutationKind::ModifyBias => change_bias(g),
        MutationKind::ModifyActivation => change_activation(g),
        _ => panic!("Mutation kind is unknown"),
    };
}

#[derive(Debug)]
pub enum MutationKind {
    AddConnection,
    RemoveConnection,
    AddNode,
    RemoveNode,
    ModifyWeight,
    ModifyBias,
    ModifyActivation,
}

impl Distribution<MutationKind> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MutationKind {
        match rng.gen_range(0, 7) {
            0 => MutationKind::AddConnection,
            1 => MutationKind::RemoveConnection,
            2 => MutationKind::AddNode,
            // 3 => MutationKind::RemoveNode, // This one is problematic
            4 => MutationKind::ModifyWeight,
            5 => MutationKind::ModifyBias,
            _ => MutationKind::ModifyActivation,
        }
    }
}

/// Adds a new random connection
fn add_connection(g: &mut Genome) {
    let existing_connections: Vec<(usize, usize)> =
        g.connection_genes.iter().map(|c| (c.from, c.to)).collect();

    let mut possible_connections: Vec<(usize, usize)> = (0..g.node_genes.len())
        .flat_map(|i| {
            let mut inner = vec![];

            (0..g.node_genes.len()).for_each(|j| {
                if i != j {
                    if !existing_connections.contains(&(i, j)) {
                        inner.push((i, j))
                    };
                    if !existing_connections.contains(&(j, i)) {
                        inner.push((j, i))
                    };
                }
            });

            inner
        })
        .filter(|(i, j)| g.can_connect(*i, *j))
        .collect();

    possible_connections.sort_unstable();
    possible_connections.dedup();

    if possible_connections.is_empty() {
        return;
    }

    let picked_connection = possible_connections
        .get(random::<usize>() % possible_connections.len())
        .unwrap();

    g.add_connection(picked_connection.0, picked_connection.1)
        .unwrap();
}

/// Removes a random connection if it's not the only one
fn remove_connection(g: &mut Genome) {
    let deletable_indexes: Vec<usize> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            let from_index = c.from;
            let to_index = c.to;

            // Number of outgoing connections for the `from` node
            let from_connections_count = g
                .connection_genes
                .iter()
                .filter(|c| c.from == from_index)
                .count();
            // Number of incoming connections for the `to` node
            let to_connections_count = g
                .connection_genes
                .iter()
                .filter(|c| c.to == to_index)
                .count();

            from_connections_count > 1 && to_connections_count > 1
        })
        .map(|(i, _)| i)
        .collect();

    if deletable_indexes.is_empty() {
        return;
    }

    let index = deletable_indexes
        .get(random::<usize>() % deletable_indexes.len())
        .unwrap();

    g.connection_genes.remove(*index);
}

/// Adds a random hidden node to the genome and its connections
fn add_node(g: &mut Genome) {
    let new_node_index = g.add_node();

    let random_connection_index = random::<usize>() % g.connection_genes.len();
    let (picked_from, picked_to, picked_weight) = {
        let picked = g.connection_genes.get(random_connection_index).unwrap();

        (picked.from, picked.to, picked.weight)
    };

    g.connection_genes.remove(random_connection_index);

    let connection_index = g.add_connection(picked_from, new_node_index).unwrap();
    g.add_connection(new_node_index, picked_to).unwrap();

    // Reuse the weight from the removed connection
    g.connection_genes.get_mut(connection_index).unwrap().weight = picked_weight;
}

/// Removes a random hidden node from the genome and rewires connected nodes
fn remove_node(g: &mut Genome) {
    let hidden_nodes: Vec<usize> = g
        .node_genes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Hidden))
        .map(|(i, _)| i)
        .collect();

    if hidden_nodes.is_empty() {
        return;
    }

    let picked_node_index = hidden_nodes
        .get(random::<usize>() % hidden_nodes.len())
        .unwrap();

    let incoming_connections_and_from_indexes: Vec<(usize, usize)> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.to == *picked_node_index)
        .map(|(i, c)| (i, c.from))
        .collect();
    let outgoing_connections_and_to_indexes: Vec<(usize, usize)> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.from == *picked_node_index)
        .map(|(i, c)| (i, c.to))
        .collect();

    let new_from_to_pairs: Vec<(usize, usize)> = incoming_connections_and_from_indexes
        .iter()
        .flat_map(|(_, from)| {
            outgoing_connections_and_to_indexes
                .iter()
                .map(|(_, to)| (*from, *to))
                .collect::<Vec<(usize, usize)>>()
        })
        .filter(|(from, to)| {
            g.connection_genes
                .iter()
                .all(|c| c.from != *from || c.to != *to)
        })
        .collect();

    g.add_many_connections(&new_from_to_pairs);

    let connection_indexes_to_delete: Vec<usize> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.from == *picked_node_index || c.to == *picked_node_index)
        .map(|(i, _)| i)
        .collect();

    g.remove_many_connections(&connection_indexes_to_delete);
    g.remove_node(*picked_node_index);
}

/// Changes the weight of a random connection
fn change_weight(g: &mut Genome) {
    let index = random::<usize>() % g.connection_genes.len();
    let picked_connection = g.connection_genes.get_mut(index).unwrap();

    picked_connection.weight = random::<f64>() - 0.5;
}

/// Changes the bias of a random non input node
fn change_bias(g: &mut Genome) {
    let eligible_indexes: Vec<usize> = g
        .node_genes
        .iter()
        .enumerate()
        .filter(|(_, n)| !matches!(n.kind, NodeKind::Input))
        .map(|(i, _)| i)
        .collect();

    let index = eligible_indexes
        .get(random::<usize>() % eligible_indexes.len())
        .unwrap();
    let picked_node = g.node_genes.get_mut(*index).unwrap();

    picked_node.bias = random::<f64>() - 0.5;
}

/// Changes the activation function of a random non input node
fn change_activation(g: &mut Genome) {
    let eligible_indexes: Vec<usize> = g
        .node_genes
        .iter()
        .enumerate()
        .filter(|(_, n)| !matches!(n.kind, NodeKind::Input))
        .map(|(i, _)| i)
        .collect();

    let index = eligible_indexes
        .get(random::<usize>() % eligible_indexes.len())
        .unwrap();
    let picked_node = g.node_genes.get_mut(*index).unwrap();

    picked_node.activation = random::<ActivationKind>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_connection_adds_missing_connection() {
        let mut g = Genome::new(1, 2);

        g.add_node();
        g.add_connection(0, 3).unwrap();
        g.add_connection(3, 2).unwrap();

        assert!(!g.connection_genes.iter().any(|c| c.from == 3 && c.to == 1));
        add_connection(&mut g);
        assert!(g.connection_genes.iter().any(|c| c.from == 3 && c.to == 1));
    }

    #[test]
    fn add_connection_doesnt_add_unecessary_connections() {
        let mut g = Genome::new(1, 2);

        g.add_node();
        g.add_connection(0, 3).unwrap();
        g.add_connection(3, 2).unwrap();

        // This will add the last missing connection
        assert_eq!(g.connection_genes.len(), 4);
        add_connection(&mut g);
        assert_eq!(g.connection_genes.len(), 5);

        // There should be no new connections
        add_connection(&mut g);
        assert_eq!(g.connection_genes.len(), 5);
    }

    #[test]
    fn remove_connection_doesnt_remove_last_connection_of_a_node() {
        let mut g = Genome::new(1, 2);
        assert_eq!(g.connection_genes.len(), 2);

        remove_connection(&mut g);
        assert_eq!(g.connection_genes.len(), 2);

        g.add_node();
        g.add_connection(0, 3).unwrap();
        g.add_connection(3, 2).unwrap();
        assert_eq!(g.connection_genes.len(), 4);

        remove_connection(&mut g);
        assert_eq!(g.connection_genes.len(), 3);

        remove_connection(&mut g);
        assert_eq!(g.connection_genes.len(), 3);
    }

    #[test]
    fn add_node_doesnt_change_existing_connections() {
        let mut g = Genome::new(1, 1);
        let original_connections = g.connection_genes.clone();

        add_node(&mut g);

        let original_nodes_not_modified = original_connections
            .iter()
            .filter(|oc| {
                g.connection_genes
                    .iter()
                    .any(|c| c.from == oc.from && c.to == oc.to)
            })
            .count();

        // When adding a node, a connection is selected to be replaced with a new node and two new
        // connections
        assert_eq!(original_nodes_not_modified, original_connections.len() - 1);
    }

    #[test]
    fn remove_node_doesnt_mess_up_the_connections() {
        let mut g = Genome::new(1, 1);

        add_node(&mut g);
        let original_connections = g.connection_genes.clone();

        add_node(&mut g);
        remove_node(&mut g);

        let node_count = g.node_genes.len();
        g.connection_genes
            .iter()
            .for_each(|c| assert!(c.from < node_count && c.to < node_count));
    }

    #[test]
    fn change_bias_doesnt_change_input_nodes() {
        let mut g = Genome::new(1, 1);

        let input_bias = g.node_genes.get(0).unwrap().bias;
        let output_bias = g.node_genes.get(1).unwrap().bias;

        for _ in 0..10 {
            change_bias(&mut g);
        }

        let new_input_bias = g.node_genes.get(0).unwrap().bias;
        let new_output_bias = g.node_genes.get(1).unwrap().bias;

        assert!((input_bias - new_input_bias).abs() < f64::EPSILON);
        assert!((output_bias - new_output_bias).abs() > f64::EPSILON);
    }

    #[test]
    fn change_activation_doesnt_change_input_nodes() {
        let mut g = Genome::new(1, 1);

        let i_activation = g.node_genes.get(0).unwrap().activation.clone();
        let o_activation = g.node_genes.get(1).unwrap().activation.clone();

        let mut new_i_activations = vec![];
        let mut new_o_activations = vec![];

        for _ in 0..10 {
            change_activation(&mut g);

            new_i_activations.push(g.node_genes.get(0).unwrap().activation.clone());
            new_o_activations.push(g.node_genes.get(1).unwrap().activation.clone());
        }

        assert!(new_i_activations.iter().all(|a| *a == i_activation));
        assert!(new_o_activations.iter().any(|a| *a == o_activation));
    }

    #[test]
    fn mutate_genome() {
        let mut g = Genome::new(1, 1);

        for _ in 0..100 {
            let kind: MutationKind = random();
            mutate(kind, &mut g);
        }
    }
}
