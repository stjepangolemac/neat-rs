use std::default::Default;

use crate::genome::mutation::MutationKind;

/// Holds configuration options of the whole NEAT process
#[derive(Debug)]
pub struct Configuration {
    /// The generations limit of for the evolution process
    pub max_generations: usize,

    /// The maximum number of genomes in each generation
    pub population_size: usize,

    /// The ratio of champion individuals that are copied to the next generation
    pub elitism: f64,

    /// The fitness cost of every node in the gene
    pub node_cost: f64,

    /// The fitness cost of every connection in the gene
    pub connection_cost: f64,

    /// The ratio of reproductions by crossover
    pub crossover_ratio: f64,

    /// The types of mutations available and their sampling weights
    pub mutation_kinds: Vec<(MutationKind, usize)>,

    /// The process will stop if the fitness goal is reached
    pub fitness_goal: Option<f64>,

    /// Controls how non shared genes affect speciation
    pub speciation_disjoint_coefficient: f64,

    /// Controls how shared gene weights affect speciation
    pub speciation_weight_coeficcient: f64,

    /// A limit on how distant two genomes can be to belong to the same species
    pub compatibility_threshold: f64,
}

impl Default for Configuration {
    fn default() -> Self {
        Configuration {
            max_generations: 1000,
            population_size: 150,
            elitism: 0.1,
            node_cost: 0.,
            connection_cost: 0.,
            crossover_ratio: 0.5,
            mutation_kinds: default_mutation_kinds(),
            fitness_goal: None,
            speciation_disjoint_coefficient: 1.,
            speciation_weight_coeficcient: 0.1,
            compatibility_threshold: 2.,
        }
    }
}

pub fn default_mutation_kinds() -> Vec<(MutationKind, usize)> {
    use MutationKind::*;

    vec![
        (AddConnection, 2),
        (RemoveConnection, 2),
        (AddNode, 1),
        (RemoveNode, 1),
        (ModifyWeight, 3),
        (ModifyBias, 3),
        (ModifyActivation, 3),
    ]
}