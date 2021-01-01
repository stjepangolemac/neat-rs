use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::Configuration;
use crate::{ConnectionGene, Genome};

type DistanceKey = String;
pub struct GenomicDistanceCache {
    configuration: Rc<RefCell<Configuration>>,
    cache: HashMap<DistanceKey, f64>,
}

impl GenomicDistanceCache {
    pub fn new(configuration: Rc<RefCell<Configuration>>) -> Self {
        GenomicDistanceCache {
            configuration,
            cache: HashMap::new(),
        }
    }

    pub fn get(&mut self, a: &Genome, b: &Genome) -> f64 {
        let distance_key = GenomicDistanceCache::make_key(a, b);

        if let Some(distance) = self.cache.get(&distance_key) {
            *distance
        } else {
            0.
        }
    }

    fn distance(&self, a: &Genome, b: &Genome) -> f64 {
        let (
            distance_connection_disjoint_coefficient,
            distance_connection_weight_coeficcient,
            distance_connection_disabled_coefficient,
            distance_node_bias_coefficient,
            distance_node_activation_coefficient,
            distance_node_aggregation_coefficient,
        ) = {
            let conf = self.configuration.borrow();

            (
                conf.distance_connection_disjoint_coefficient,
                conf.distance_connection_weight_coeficcient,
                conf.distance_connection_disabled_coefficient,
                conf.distance_node_bias_coefficient,
                conf.distance_node_activation_coefficient,
                conf.distance_node_aggregation_coefficient,
            )
        };

        let mut distance = 0.;

        let max_connection_genes = usize::max(a.connections().len(), b.connections().len());
        let max_node_genes = usize::max(a.nodes().len(), b.nodes().len());

        let mut disjoint_connections: Vec<&ConnectionGene> = vec![];
        let mut common_connections: Vec<(&ConnectionGene, &ConnectionGene)> = vec![];

        let mut disjoint_map: HashMap<usize, bool> = HashMap::new();
        a.connections()
            .iter()
            .chain(b.connections().iter())
            .map(|connection| connection.innovation_number())
            .for_each(|innovation_number| {
                if let Some(is_disjoint) = disjoint_map.get_mut(&innovation_number) {
                    *is_disjoint = false;
                } else {
                    disjoint_map.insert(innovation_number, true);
                }
            });

        disjoint_map
            .into_iter()
            .for_each(|(innovation_number, is_disjoint)| {
                if is_disjoint {
                    let disjoint_connection = a
                        .connections()
                        .iter()
                        .chain(b.connections().iter())
                        .find(|connection| connection.innovation_number() == innovation_number)
                        .unwrap();

                    disjoint_connections.push(disjoint_connection);
                } else {
                    let common_connection_a = a
                        .connections()
                        .iter()
                        .find(|connection| connection.innovation_number() == innovation_number)
                        .unwrap();
                    let common_connection_b = b
                        .connections()
                        .iter()
                        .find(|connection| connection.innovation_number() == innovation_number)
                        .unwrap();

                    common_connections.push((common_connection_a, common_connection_b));
                }
            });

        let disjoint_factor =
            disjoint_connections.len() as f64 * distance_connection_disjoint_coefficient;

        let connections_difference_factor: f64 = common_connections
            .iter()
            .map(|(connection_a, connection_b)| {
                let mut connection_distance = 0.;

                if connection_a.disabled != connection_b.disabled {
                    connection_distance += 1. * distance_connection_disabled_coefficient;
                }

                connection_distance += (connection_a.weight - connection_b.weight).abs()
                    * distance_connection_weight_coeficcient;

                connection_distance
            })
            .sum::<f64>();

        let nodes_difference_factor: f64 = a
            .nodes()
            .iter()
            .zip(b.nodes())
            .map(|(node_a, node_b)| {
                let mut node_distance = 0.;

                if node_a.activation != node_b.activation {
                    node_distance += 1. * distance_node_activation_coefficient;
                }

                if node_a.aggregation != node_b.aggregation {
                    node_distance += 1. * distance_node_aggregation_coefficient;
                }

                node_distance += (node_a.bias - node_b.bias).abs() * distance_node_bias_coefficient;

                node_distance
            })
            .sum();

        distance += nodes_difference_factor;
        distance += (connections_difference_factor + disjoint_factor) / max_connection_genes as f64;

        distance
    }

    fn make_key<'o>(a: &'o Genome, b: &'o Genome) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_a = {
            let mut hasher = DefaultHasher::new();
            a.hash(&mut hasher);
            hasher.finish()
        };

        let hash_b = {
            let mut hasher = DefaultHasher::new();
            b.hash(&mut hasher);
            hasher.finish()
        };

        hash_a.to_string();

        if hash_a > hash_b {
            hash_a.to_string() + &hash_b.to_string()
        } else {
            hash_b.to_string() + &hash_a.to_string()
        }
    }
}
