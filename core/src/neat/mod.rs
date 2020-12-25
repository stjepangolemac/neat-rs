use rand::random;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::genome::mutation::MutationKind;
use crate::genome::{crossover, Genome};
use crate::network::Network;
pub use configuration::Configuration;
use reporter::Reporter;
use speciation::GenomeBank;

mod configuration;
mod reporter;
mod speciation;

pub struct NEAT {
    inputs: usize,
    outputs: usize,
    fitness_fn: fn(&mut Network) -> f64,
    genomes: GenomeBank,
    fitnesses: HashMap<Genome, f64>,
    configuration: Rc<RefCell<Configuration>>,
    reporter: Reporter,
}

impl NEAT {
    pub fn new(inputs: usize, outputs: usize, fitness_fn: fn(&mut Network) -> f64) -> Self {
        let configuration: Rc<RefCell<Configuration>> = Default::default();

        NEAT {
            inputs,
            outputs,
            fitness_fn,
            genomes: GenomeBank::new(configuration.clone()),
            fitnesses: HashMap::new(),
            configuration,
            reporter: Reporter::new(),
        }
    }

    pub fn set_configuration(&mut self, config: Configuration) {
        *self.configuration.borrow_mut() = config;
    }

    pub fn start(&mut self) -> (Network, f64) {
        let initial_genomes = (0..self.configuration.borrow().population_size)
            .map(|_| Genome::new(self.inputs, self.outputs))
            .collect();

        self.genomes.replace_genomes(initial_genomes);

        let max_generations = self.configuration.borrow().max_generations;

        for i in 1..=max_generations {
            self.genomes.speciate();
            self.test_fitness();

            let elites_count = (self.genomes.genomes().len()
                * (self.configuration.borrow().elitism * 100.).round() as usize)
                .div_euclid(100);

            let all_genomes: Vec<&Genome> = self
                .genomes_by_adjusted_fitness()
                .iter()
                .map(|(genome, _)| *genome)
                .collect();
            let mut elites: Vec<Genome> = all_genomes
                .iter()
                .take(elites_count)
                .cloned()
                .cloned()
                .collect();
            let non_elites = all_genomes;
            // let non_elites: Vec<&Genome> = all_genomes.iter().skip(elites_count).cloned().collect();

            let mut offspring = vec![];

            while (elites.len() + offspring.len()) < self.configuration.borrow().population_size {
                let maybe_child = if random::<f64>() < self.configuration.borrow().crossover_ratio {
                    // Crossover
                    let parent_index_a = random::<usize>() % non_elites.len();
                    let parent_a = non_elites.get(parent_index_a).unwrap();

                    let parent_fitness_a = self.genomes.fitnesses().get(&parent_index_a).unwrap();

                    let parent_index_b = random::<usize>() % non_elites.len();
                    let parent_b = non_elites.get(parent_index_b).unwrap();
                    let parent_fitness_b = self.genomes.fitnesses().get(&parent_index_b).unwrap();

                    crossover((parent_a, *parent_fitness_a), (parent_b, *parent_fitness_b))
                } else {
                    // Mutation
                    let parent_index = random::<usize>() % non_elites.len();
                    let parent: &Genome = non_elites.get(parent_index).unwrap();

                    let mut child = parent.clone();
                    child.mutate(&self.pick_mutation());

                    Some(child)
                };

                if let Some(child) = maybe_child {
                    offspring.push(child);
                }
            }

            let mut new_genomes = vec![];
            new_genomes.append(&mut elites);
            new_genomes.append(&mut offspring);

            self.genomes.replace_genomes(new_genomes);
            self.test_fitness();

            self.reporter.report(i, &self);

            let goal_reached = {
                if let Some(goal) = self.configuration.borrow().fitness_goal {
                    let (_, _, best_fitness) = self.get_best();

                    best_fitness >= goal
                } else {
                    false
                }
            };

            if goal_reached {
                break;
            }
        }

        let (_, best_genome, best_fitness) = self.get_best();
        (Network::from(best_genome), best_fitness)
    }

    fn genomes_by_adjusted_fitness(&self) -> Vec<(&Genome, f64)> {
        let mut genomes: Vec<(usize, &Genome)> =
            self.genomes.genomes().iter().enumerate().collect();
        let adjusted_fitnesses = self.genomes.adjusted_fitnesses();

        genomes.sort_by(|a, b| {
            let fitness_a = adjusted_fitnesses.get(a.0).unwrap();
            let fitness_b = adjusted_fitnesses.get(b.0).unwrap();

            if (fitness_a - fitness_b).abs() < f64::EPSILON {
                std::cmp::Ordering::Equal
            } else if fitness_a > fitness_b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        genomes
            .into_iter()
            .map(|(index, genome)| (genome, *adjusted_fitnesses.get(index).unwrap()))
            .collect()
    }

    fn test_fitness(&mut self) {
        let indexes_and_networks: Vec<(usize, Network)> = self
            .genomes
            .genomes()
            .iter()
            .enumerate()
            .map(|(index, genome)| (index, Network::from(genome)))
            .collect();

        indexes_and_networks
            .into_iter()
            .for_each(|(index, mut network)| {
                let mut fitness = (self.fitness_fn)(&mut network);
                fitness -= self.configuration.borrow().node_cost * network.nodes.len() as f64;
                fitness -=
                    self.configuration.borrow().connection_cost * network.connections.len() as f64;

                self.genomes.mark_fitness(index, fitness);
            });
    }

    pub fn get_best(&self) -> (usize, &Genome, f64) {
        let (best_genome_index, best_fitness) = self.genomes.fitnesses().iter().fold(
            (0, 0.),
            |(best_index, best_fitness), (genome_index, genome_fitness)| {
                if *genome_fitness > best_fitness {
                    (*genome_index, *genome_fitness)
                } else {
                    (best_index, best_fitness)
                }
            },
        );
        let best_genome = self.genomes.genomes().get(best_genome_index).unwrap();

        (best_genome_index, best_genome, best_fitness)
    }

    fn pick_mutation(&self) -> MutationKind {
        use rand::{distributions::Distribution, thread_rng};
        use rand_distr::weighted_alias::WeightedAliasIndex;

        let dist = WeightedAliasIndex::new(
            self.configuration
                .borrow()
                .mutation_kinds
                .iter()
                .map(|k| k.1)
                .collect(),
        )
        .unwrap();

        let mut rng = thread_rng();

        self.configuration
            .borrow()
            .mutation_kinds
            .get(dist.sample(&mut rng))
            .cloned()
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
            population_size: 150,
            max_generations: 500,
            fitness_goal: Some(0.934), // Perfect score for these settings
            node_cost: 0.01,
            connection_cost: 0.005,
            compatibility_threshold: 0.85,
            ..Default::default()
        });
        system.add_hook(1, |i, system| {
            let (best_genome_index, _, fitness) = system.get_best();

            println!("Generation {}, best fitness is {}", i, fitness);
        });

        let (mut network, fitness) = system.start();

        // let inputs: Vec<Vec<f64>> = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        // for i in inputs {
        //     let o = network.forward_pass(i.clone());
        //     dbg!(i, o);
        // }

        // dbg!(&network, &fitness);

        println!(
            "Found network with {} nodes and {} connections, of fitness {}",
            network.nodes.len(),
            network.connections.len(),
            fitness
        );
    }
}
