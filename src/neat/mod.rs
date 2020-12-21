use rand::random;
use std::collections::HashMap;

use crate::genome::mutation::MutationKind;
use crate::genome::{crossover, Genome};
use crate::network::Network;
use configuration::Configuration;

mod configuration;

pub struct NEAT {
    inputs: usize,
    outputs: usize,
    fitness_fn: fn(&mut Network) -> f64,
    genomes: Vec<Genome>,
    fitnesses: HashMap<Genome, f64>,
    configuration: Configuration,
}

impl NEAT {
    pub fn new(inputs: usize, outputs: usize, fitness_fn: fn(&mut Network) -> f64) -> Self {
        NEAT {
            inputs,
            outputs,
            fitness_fn,
            genomes: vec![],
            fitnesses: HashMap::new(),
            configuration: Default::default(),
        }
    }

    pub fn set_configuration(&mut self, config: Configuration) {
        self.configuration = config;
    }

    pub fn start(&mut self) -> (Network, f64) {
        self.genomes = (0..self.configuration.population_size)
            .map(|_| Genome::new(self.inputs, self.outputs))
            .collect();

        self.test_fitness();

        for i in 0..self.configuration.max_generations {
            println!("Generation {}", i);

            let elites_count = (self.genomes.len()
                * (self.configuration.elitism * 100.).round() as usize)
                .div_euclid(100);
            let mut elites: Vec<Genome> = self.genomes.iter().take(elites_count).cloned().collect();

            let mut offspring = vec![];

            while elites.len() + offspring.len() < self.configuration.population_size {
                let maybe_child = if random::<f64>() < self.configuration.crossover_ratio {
                    // Crossover
                    let parent_index_a = random::<usize>() % elites.len();
                    let parent_a = elites.get(parent_index_a).unwrap();
                    let parent_fitness_a = self.fitnesses.get(&parent_a).unwrap();

                    let parent_index_b = random::<usize>() % elites.len();
                    let parent_b = elites.get(parent_index_b).unwrap();
                    let parent_fitness_b = self.fitnesses.get(&parent_b).unwrap();

                    crossover((parent_a, *parent_fitness_a), (parent_b, *parent_fitness_b))
                } else {
                    // Mutation
                    let parent_index = random::<usize>() % elites.len();
                    let parent = elites.get(parent_index).unwrap();

                    let mut child = parent.clone();
                    child.mutate(self.pick_mutation());

                    Some(child)
                };

                if let Some(child) = maybe_child {
                    offspring.push(child);
                }
            }

            elites.append(&mut offspring);

            self.genomes = elites;
            self.test_fitness();
        }

        let best_genome = self.genomes.first().unwrap();
        let best_fitness = self.fitnesses.get(&best_genome).unwrap();

        (Network::from(best_genome), *best_fitness)
    }

    fn test_fitness(&mut self) {
        let ids_and_networks: Vec<(Genome, Network)> = self
            .genomes
            .iter()
            .map(|g| (g.clone(), Network::from(g)))
            .collect();

        ids_and_networks.into_iter().for_each(|(g, mut n)| {
            let mut fitness = (self.fitness_fn)(&mut n);
            fitness -= self.configuration.node_cost * n.nodes.len() as f64;
            fitness -= self.configuration.connection_cost * n.connections.len() as f64;

            self.fitnesses.insert(g, fitness);
        });

        self.sort_by_fitness();
    }

    fn sort_by_fitness(&mut self) {
        let mut copy = self.genomes.clone();

        copy.sort_by(|a, b| {
            let fitness_a = self.fitnesses.get(&a).unwrap();
            let fitness_b = self.fitnesses.get(&b).unwrap();

            if (fitness_a - fitness_b).abs() < f64::EPSILON {
                std::cmp::Ordering::Equal
            } else if fitness_a > fitness_b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        self.genomes = copy;
    }

    fn pick_mutation(&self) -> &MutationKind {
        use rand::{distributions::Distribution, thread_rng};
        use rand_distr::weighted_alias::WeightedAliasIndex;

        let dist = WeightedAliasIndex::new(
            self.configuration
                .mutation_kinds
                .iter()
                .map(|k| k.1)
                .collect(),
        )
        .unwrap();

        let mut rng = thread_rng();

        &self
            .configuration
            .mutation_kinds
            .get(dist.sample(&mut rng))
            .unwrap()
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor() {
        let mut system = NEAT::new(2, 1, |n| {
            let inputs: Vec<Vec<f64>> =
                vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
            let outputs: Vec<f64> = vec![0., 1., 1., 0.];

            let mut error = 0.;

            for (i, o) in inputs.iter().zip(outputs) {
                let results = n.forward_pass(i.clone());
                let result = results.first().unwrap();

                error += (o - *result).powi(2);
            }

            1. / (1. + error)
        });

        system.configuration.max_generations = 50;

        let (mut network, fitness) = system.start();

        let inputs: Vec<Vec<f64>> = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        for i in inputs {
            let o = network.forward_pass(i.clone());
            dbg!(i, o);
        }

        dbg!(network, fitness);
    }
}
