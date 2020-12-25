use rand::distributions::{Distribution, Standard};
use rand::random;
use rand::thread_rng;
use rand::Rng;
use rand_distr::StandardNormal;

use super::{ActivationKind, Genome, NodeKind};

pub fn mutate(kind: &MutationKind, g: &mut Genome) {
    match kind {
        MutationKind::AddConnection => add_connection(g),
        MutationKind::RemoveConnection => disable_connection(g),
        MutationKind::AddNode => add_node(g),
        MutationKind::RemoveNode => remove_node(g),
        MutationKind::ModifyWeight => change_weight(g),
        MutationKind::ModifyBias => change_bias(g),
        MutationKind::ModifyActivation => change_activation(g),
        _ => panic!("Mutation kind is unknown"),
    };
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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
            3 => MutationKind::RemoveNode,
            4 => MutationKind::ModifyWeight,
            5 => MutationKind::ModifyBias,
            _ => MutationKind::ModifyActivation,
        }
    }
}

/// Adds a new random connection
pub fn add_connection(g: &mut Genome) {
    let existing_connections: Vec<(usize, usize, bool)> = g
        .connection_genes
        .iter()
        .map(|c| (c.from, c.to, c.disabled))
        .collect();

    let mut possible_connections: Vec<(usize, usize)> = (0..g.node_genes.len())
        .flat_map(|i| {
            let mut inner = vec![];

            (0..g.node_genes.len()).for_each(|j| {
                if i != j {
                    if !existing_connections.contains(&(i, j, false)) {
                        inner.push((i, j));
                    };
                    if !existing_connections.contains(&(j, i, false)) {
                        inner.push((j, i));
                    };
                }
            });

            inner
        })
        .collect();

    possible_connections.sort_unstable();
    possible_connections.dedup();

    possible_connections = possible_connections
        .into_iter()
        .filter(|(i, j)| g.can_connect(*i, *j))
        .collect();

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
fn disable_connection(g: &mut Genome) {
    let eligible_indexes: Vec<usize> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            if c.disabled {
                return false;
            }

            let from_index = c.from;
            let to_index = c.to;

            // Number of outgoing connections for the `from` node
            let from_connections_count = g
                .connection_genes
                .iter()
                .filter(|c| c.from == from_index && !c.disabled)
                .count();
            // Number of incoming connections for the `to` node
            let to_connections_count = g
                .connection_genes
                .iter()
                .filter(|c| c.to == to_index && !c.disabled)
                .count();

            from_connections_count > 1 && to_connections_count > 1
        })
        .map(|(i, _)| i)
        .collect();

    if eligible_indexes.is_empty() {
        return;
    }

    let index = eligible_indexes
        .get(random::<usize>() % eligible_indexes.len())
        .unwrap();

    g.disable_connection(*index);
}

/// Adds a random hidden node to the genome and its connections
pub fn add_node(g: &mut Genome) {
    let new_node_index = g.add_node();

    // Only enabled connections can be disabled
    let enabled_connections: Vec<usize> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.disabled)
        .map(|(i, _)| i)
        .collect();

    let (picked_index, picked_from, picked_to, picked_weight) = {
        let random_enabled_connection_index = random::<usize>() % enabled_connections.len();
        let picked_index = enabled_connections
            .get(random_enabled_connection_index)
            .unwrap();
        let picked_connection = g.connection_genes.get(*picked_index).unwrap();

        (
            picked_index,
            picked_connection.from,
            picked_connection.to,
            picked_connection.weight,
        )
    };

    g.disable_connection(*picked_index);

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
        .filter(|(i, n)| {
            let incoming_count = g
                .connection_genes
                .iter()
                .filter(|c| c.to == *i && !c.disabled)
                .count();
            let outgoing_count = g
                .connection_genes
                .iter()
                .filter(|c| c.from == *i && !c.disabled)
                .count();

            matches!(n.kind, NodeKind::Hidden) && incoming_count > 0 && outgoing_count > 0
        })
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
        .filter(|(_, c)| c.to == *picked_node_index && !c.disabled)
        .map(|(i, c)| (i, c.from))
        .collect();
    let outgoing_connections_and_to_indexes: Vec<(usize, usize)> = g
        .connection_genes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.from == *picked_node_index && !c.disabled)
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
                .find(|c| c.from == *from && c.to == *to && !c.disabled)
                .is_none()
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

    g.disable_many_connections(&connection_indexes_to_delete);
}

/// Changes the weight of a random connection
fn change_weight(g: &mut Genome) {
    let index = random::<usize>() % g.connection_genes.len();
    let picked_connection = g.connection_genes.get_mut(index).unwrap();

    let new_weight = if random::<f64>() < 0.1 {
        picked_connection.weight + thread_rng().sample::<f64, StandardNormal>(StandardNormal)
    } else {
        random::<f64>() * 2. - 1.
    };

    picked_connection.weight = new_weight.max(-1.).min(1.);
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

    let new_bias = if random::<f64>() < 0.1 {
        picked_node.bias + thread_rng().sample::<f64, StandardNormal>(StandardNormal)
    } else {
        random::<f64>() * 2. - 1.
    };

    picked_node.bias = new_bias.max(-1.).min(1.);
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
        assert_eq!(g.connection_genes.iter().filter(|c| !c.disabled).count(), 2);

        disable_connection(&mut g);
        assert_eq!(g.connection_genes.iter().filter(|c| !c.disabled).count(), 2);
    }

    #[test]
    fn add_node_doesnt_change_existing_connections() {
        let mut g = Genome::new(1, 1);
        let original_connections = g.connection_genes.clone();

        add_node(&mut g);

        let original_connections_not_modified = original_connections
            .iter()
            .filter(|oc| {
                g.connection_genes
                    .iter()
                    .any(|c| c.from == oc.from && c.to == oc.to && c.disabled == oc.disabled)
            })
            .count();

        // When adding a node, a connection is selected to be disabled and replaced with a new node and two new
        // connections
        assert_eq!(
            original_connections_not_modified,
            original_connections.len() - 1
        );
    }

    #[test]
    fn remove_node_doesnt_mess_up_the_connections() {
        let mut g = Genome::new(1, 1);
        let connection_enabled_initially = !g.connection_genes.first().unwrap().disabled;

        add_node(&mut g);
        let connection_disabled_after_add = g.connection_genes.first().unwrap().disabled;

        remove_node(&mut g);
        let connection_enabled_after_remove = !g.connection_genes.first().unwrap().disabled;

        assert!(connection_enabled_initially);
        assert!(connection_disabled_after_add);
        assert!(connection_enabled_after_remove);
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
        assert!(new_o_activations.iter().any(|a| *a != o_activation));
    }

    #[test]
    fn mutate_genome() {
        use std::collections::HashMap;
        use std::convert::TryFrom;
        use std::time;

        let mut times: HashMap<MutationKind, Vec<time::Duration>> = HashMap::new();
        let mut g = Genome::new(1, 1);

        let limit = 50;
        for i in 1..=limit {
            let kind: MutationKind = random();

            let before = std::time::Instant::now();
            mutate(&kind, &mut g);
            let after = std::time::Instant::now();
            let duration = after.duration_since(before);

            if times.get(&kind).is_none() {
                times.insert(kind.clone(), vec![]);
            }

            times.get_mut(&kind).unwrap().push(duration);

            if g.connection_genes.iter().all(|c| c.disabled) {
                panic!("All connections disabled, happened after {:?}", kind);
            }

            println!("mutation {}/{}", i, limit);
        }

        let mut kind_average_times: Vec<(MutationKind, time::Duration)> = times
            .iter()
            .map(|(k, t)| {
                let sum: u128 = t.iter().map(|d| d.as_micros()).sum();
                let avg: u128 = sum.div_euclid(u128::try_from(t.len()).unwrap());

                let duration = time::Duration::from_micros(u64::try_from(avg).unwrap());

                (k.clone(), duration)
            })
            .collect();

        kind_average_times.sort_by(|(_, duration1), (_, duration2)| duration1.cmp(duration2));

        kind_average_times.iter().for_each(|(k, duration)| {
            println!("{:?} on avg took {:?}", k, duration);
        });

        println!(
            "Genome had {} nodes and {} connections, of which {} were active",
            g.nodes().len(),
            g.connections().len(),
            g.connections().iter().filter(|c| !c.disabled).count(),
        );
    }
}
