use crate::genome::mutation::MutationKind;
use crate::genome::Genome;
use crate::network::Network;
use std::collections::HashMap;
use uuid::Uuid;

pub struct NEAT {
    inputs: usize,
    outputs: usize,
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

    pub fn start(&mut self) {
        let genomes: Vec<Genome> = (0..2)
            .map(|_| Genome::new(self.inputs, self.outputs))
            .collect();
        self.genomes = genomes;

        self.test_fitness();
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

        system.start();

        for (id, f) in system.fitnesses.iter() {
            let genome = system.genomes.iter().find(|g| g.id() == *id).unwrap();
            dbg!(genome, f);
        }
    }
}
