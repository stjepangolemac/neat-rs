use rand::random;
use std::collections::HashMap;

use crate::genome::mutation::MutationKind;
use crate::genome::{crossover, Genome};
use crate::network::Network;
use configuration::Configuration;
use reporter::Reporter;

mod configuration;
mod reporter;
mod speciation;

pub struct NEAT {
    inputs: usize,
    outputs: usize,
    fitness_fn: fn(&mut Network) -> f64,
    genomes: Vec<Genome>,
    fitnesses: HashMap<Genome, f64>,
    configuration: Configuration,
    reporter: Reporter,
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
            reporter: Reporter::new(),
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

            self.reporter.report(i, &self);

            let goal_reached = {
                let (_, best_fitness) = self.get_best();

                if let Some(goal) = self.configuration.fitness_goal {
                    best_fitness >= goal
                } else {
                    false
                }
            };

            if goal_reached {
                break;
            }
        }

        let (best_genome, best_fitness) = self.get_best();
        (Network::from(best_genome), best_fitness)
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

    fn get_best(&self) -> (&Genome, f64) {
        let best_genome = self.genomes.first().unwrap();
        let best_fitness = self.fitnesses.get(&best_genome).unwrap();

        (best_genome, *best_fitness)
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

    pub fn add_hook(&mut self, every: usize, hook: reporter::Hook) {
        self.reporter.register(every, hook);
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

        system.set_configuration(Configuration {
            max_generations: 50,
            fitness_goal: Some(1.0),
            ..Default::default()
        });
        system.add_hook(5, |i, system| {
            println!("Generation {}, fitness at {}", i, system.get_best().1)
        });

        let (mut network, fitness) = system.start();

        let inputs: Vec<Vec<f64>> = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        for i in inputs {
            let o = network.forward_pass(i.clone());
            dbg!(i, o);
        }

        dbg!(network, fitness);
    }
}
