use std::default::Default;

use crate::mutations::MutationKind;

/// Holds configuration options of the whole NEAT process
#[derive(Debug)]
pub struct Configuration {
    /// The generations limit of for the evolution process
    pub max_generations: usize,

    /// The maximum number of genomes in each generation
    pub population_size: usize,

    /// The ratio of champion individuals that are copied to the next generation
    pub elitism: f64,

    /// The minimum amount of species that need to exist after the removal of stagnated ones
    pub elitism_species: usize,

    /// How many generations of not making progress is considered stagnation
    pub stagnation_after: usize,

    /// The fitness cost of every node in the gene
    pub node_cost: f64,

    /// The fitness cost of every connection in the gene
    pub connection_cost: f64,

    /// The mutation rate of offspring
    pub mutation_rate: f64,

    /// The ratio of genomes that will survive to the next generation
    pub survival_ratio: f64,

    /// The types of mutations available and their sampling weights
    pub mutation_kinds: Vec<(MutationKind, usize)>,

    /// The process will stop if the fitness goal is reached
    pub fitness_goal: Option<f64>,

    /*
     * Genomic distance during speciation
     */
    /// Controls how much connections can affect distance
    pub distance_connection_disjoint_coefficient: f64,
    pub distance_connection_weight_coeficcient: f64,
    pub distance_connection_disabled_coefficient: f64,

    /// Controls how much nodes can affect distance
    pub distance_node_bias_coefficient: f64,
    pub distance_node_activation_coefficient: f64,
    pub distance_node_aggregation_coefficient: f64,

    /// A limit on how distant two genomes can be to belong to the same species
    pub compatibility_threshold: f64,
}

impl Default for Configuration {
    fn default() -> Self {
        Configuration {
            max_generations: 1000,
            population_size: 150,
            elitism: 0.1,
            elitism_species: 3,
            stagnation_after: 50,
            node_cost: 0.,
            connection_cost: 0.,
            mutation_rate: 0.5,
            survival_ratio: 0.5,
            mutation_kinds: default_mutation_kinds(),
            fitness_goal: None,
            distance_connection_disjoint_coefficient: 1.,
            distance_connection_weight_coeficcient: 0.5,
            distance_connection_disabled_coefficient: 0.5,
            distance_node_bias_coefficient: 0.33,
            distance_node_activation_coefficient: 0.33,
            distance_node_aggregation_coefficient: 0.33,
            compatibility_threshold: 3.,
        }
    }
}

pub fn default_mutation_kinds() -> Vec<(MutationKind, usize)> {
    use MutationKind::*;

    vec![
        (AddConnection, 10),
        (RemoveConnection, 10),
        (AddNode, 10),
        (RemoveNode, 10),
        (ModifyWeight, 10),
        (ModifyBias, 10),
        (ModifyActivation, 10),
        (ModifyAggregation, 10),
    ]
}
