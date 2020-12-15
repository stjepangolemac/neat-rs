use crate::genome::mutation::MutationKind;
use crate::genome::Genome;
use crate::network::Network;
use rand::random;
use std::collections::HashMap;
use uuid::Uuid;

pub struct NEAT {
    inputs: usize,
    outputs: usize,
    population_size: usize,
    fitness_fn: fn(&mut Network) -> f64,
    mutation_kinds: Vec<MutationKind>,
    genomes: Vec<Genome>,
    fitnesses: HashMap<Uuid, f64>,
}

impl NEAT {
    pub fn new(inputs: usize, outputs: usize, fitness_fn: fn(&mut Network) -> f64) -> Self {
        use MutationKind::*;

        NEAT {
            inputs,
            outputs,
            population_size: 10,
            fitness_fn,
            mutation_kinds: vec![
                AddConnection,
                RemoveConnection,
                AddNode,
                RemoveNode,
                ModifyWeight,
                ModifyBias,
                ModifyActivation,
            ],
            genomes: vec![],
            fitnesses: HashMap::new(),
        }
    }

    pub fn set_mutation_kinds(&mut self, kinds: Vec<MutationKind>) {
        self.mutation_kinds = kinds;
    }

    pub fn set_population_size(&mut self, size: usize) {
        self.population_size = size;
    }

    pub fn start(&mut self) -> (Genome, f64) {
        self.genomes = (0..self.population_size)
            .map(|_| Genome::new(self.inputs, self.outputs))
            .collect();

        self.test_fitness();

        for i in 0..99 {
            println!("Iteration: {}", i);

            let mut new_genomes = self.genomes.clone();

            // new_genomes.sort_by(|a, b| {
            //     let fitness_a = self.fitnesses.get(&a.id()).unwrap();
            //     let fitness_b = self.fitnesses.get(&b.id()).unwrap();

            //     if (fitness_a - fitness_b).abs() < f64::EPSILON {
            //         std::cmp::Ordering::Equal
            //     } else if fitness_a > fitness_b {
            //         std::cmp::Ordering::Greater
            //     } else {
            //         std::cmp::Ordering::Less
            //     }
            // });

            // new_genomes = new_genomes.into_iter().take(10).collect();
            // let mut offspring = vec![];

            // while new_genomes.len() + offspring.len() < self.population_size {
            //     let index_a = random::<usize>() % new_genomes.len();
            //     let index_b = random::<usize>() % new_genomes.len();

            //     if index_a != index_b {
            //         let genome_a = new_genomes.get(index_a).unwrap();
            //         let genome_b = new_genomes.get(index_b).unwrap();

            //         let mut g = Genome::crossover(
            //             (genome_a, *self.fitnesses.get(&genome_a.id()).unwrap()),
            //             (genome_b, *self.fitnesses.get(&genome_b.id()).unwrap()),
            //         );

            //         if random::<f64>() < 0.5 {
            //             g.mutate();
            //         }

            //         offspring.push(g);
            //     }
            // }

            new_genomes.iter_mut().for_each(|g| g.mutate());
            // new_genomes.append(&mut offspring);

            self.genomes = new_genomes;
            self.test_fitness();
        }

        let (id, fitness) = self
            .fitnesses
            .iter()
            .max_by(|(_, f1), (_, f2)| {
                if (**f1 - **f2).abs() < f64::EPSILON {
                    std::cmp::Ordering::Equal
                } else if f1 > f2 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .unwrap();

        (
            self.genomes.iter().find(|g| g.id() == *id).unwrap().clone(),
            *fitness,
        )
    }

    fn test_fitness(&mut self) {
        let ids_and_networks: Vec<(Uuid, Network)> = self
            .genomes
            .iter()
            .map(|g| (g.id(), Network::from(g)))
            .collect();

        ids_and_networks.into_iter().for_each(|(id, mut n)| {
            self.fitnesses.insert(id, (self.fitness_fn)(&mut n));
        });
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

            1. / error
        });

        system.set_population_size(15);
        let (genome, fitness) = system.start();

        dbg!(genome, fitness);
    }
}
