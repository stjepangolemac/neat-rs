use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::configuration::Configuration;
use crate::genome::ConnectionGene;
use crate::genome::Genome;

/// Holds all genomes and species, does the process of speciation
#[derive(Debug)]
pub struct GenomeBank {
    configuration: Rc<RefCell<Configuration>>,
    genomes: Vec<Genome>,
    fitnesses: HashMap<usize, f64>,
    species: HashMap<usize, Vec<usize>>,
}

impl GenomeBank {
    pub fn new(configuration: Rc<RefCell<Configuration>>) -> Self {
        GenomeBank {
            configuration,
            genomes: vec![],
            fitnesses: HashMap::new(),
            species: HashMap::new(),
        }
    }

    /// Adds a new genome
    pub fn add_genome(&mut self, genome: Genome) {
        self.genomes.push(genome);
    }

    pub fn replace_genomes(&mut self, genomes: Vec<Genome>) {
        self.genomes = genomes;
    }

    /// Returns a reference to the genomes
    pub fn genomes(&self) -> &[Genome] {
        &self.genomes
    }

    /// Tracks the fitness of a particular genome
    pub fn mark_fitness(&mut self, tested_genome_index: usize, fitness: f64) {
        self.fitnesses.insert(tested_genome_index, fitness);
    }

    /// Returns a reference to the fitnesses
    pub fn fitnesses(&self) -> &HashMap<usize, f64> {
        &self.fitnesses
    }

    /// Checks that all genomes have had their fitness measured
    fn all_genomes_tested(&self) -> bool {
        (0..self.genomes.len()).all(|index| self.fitnesses.get(&index).is_some())
    }

    pub fn species(&self) -> &HashMap<usize, Vec<usize>> {
        &self.species
    }

    /// Classifies genomes into their respective species
    pub fn speciate(&mut self) {
        self.species.clear();

        for (genome_id, genome) in self.genomes.iter().enumerate() {
            let maybe_species = self
                .species
                .iter()
                .find(|(_, species_genome_ids)| {
                    // Paper says checking the first one is enough
                    let maybe_other_genome = species_genome_ids
                        .first()
                        .and_then(|other_genome_index| self.genomes.get(*other_genome_index));

                    if let Some(other_genome) = maybe_other_genome {
                        self.are_genomes_related(genome, other_genome)
                    } else {
                        false
                    }
                })
                .map(|species| species.0)
                .cloned();

            if let Some(species_id) = maybe_species {
                self.species.get_mut(&species_id).unwrap().push(genome_id);
            } else {
                self.species.insert(self.species.len(), vec![genome_id]);
            }
        }
    }

    fn are_genomes_related(&self, a: &Genome, b: &Genome) -> bool {
        let (
            distance_connection_disjoint_coefficient,
            distance_connection_weight_coeficcient,
            distance_connection_disabled_coefficient,
            distance_node_bias_coefficient,
            distance_node_activation_coefficient,
            distance_node_aggregation_coefficient,
            compatibility_threshold,
        ) = {
            let conf = self.configuration.borrow();

            (
                conf.distance_connection_disjoint_coefficient,
                conf.distance_connection_weight_coeficcient,
                conf.distance_connection_disabled_coefficient,
                conf.distance_node_bias_coefficient,
                conf.distance_node_activation_coefficient,
                conf.distance_node_aggregation_coefficient,
                conf.compatibility_threshold,
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

        distance <= compatibility_threshold
    }

    pub fn species_size_for(&self, genome_index: usize) -> usize {
        self.species
            .iter()
            .find(|(_, genome_indexes)| genome_indexes.contains(&genome_index))
            .map(|(_, genome_indexes)| genome_indexes.len())
            .unwrap()
    }

    pub fn adjusted_fitnesses(&self) -> Vec<f64> {
        let (node_cost, connection_cost) = {
            let conf = self.configuration.borrow();

            (conf.node_cost, conf.connection_cost)
        };

        self.genomes
            .iter()
            .enumerate()
            .map(|(index, genome)| {
                let fitness = self
                    .fitnesses
                    .get(&index)
                    .expect("Fitness of genome not marked");

                let genome_node_cost = genome.nodes().len() as f64 * node_cost;
                let genome_connection_cost = genome.nodes().len() as f64 * connection_cost;

                let related_genome_count = self
                    .species
                    .iter()
                    .map(|(_, species_genome_ids)| species_genome_ids)
                    .find(|species_genome_ids| species_genome_ids.contains(&index))
                    .unwrap()
                    .len();

                (fitness - genome_node_cost - genome_connection_cost) / related_genome_count as f64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_add_genome() {
        let configuration: Rc<RefCell<Configuration>> = Default::default();
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);
        bank.add_genome(genome);
    }

    #[test]
    fn can_mark_fitness() {
        let configuration: Rc<RefCell<Configuration>> = Default::default();
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);
        bank.add_genome(genome);

        bank.mark_fitness(0, 1337.);
    }

    #[test]
    fn checks_all_have_fitness_measured() {
        let configuration: Rc<RefCell<Configuration>> = Default::default();
        let mut bank = GenomeBank::new(configuration);

        bank.add_genome(Genome::new(1, 1));
        bank.add_genome(Genome::new(1, 1));

        bank.mark_fitness(0, 1337.);
        assert!(!bank.all_genomes_tested());

        bank.mark_fitness(1, 1338.);
        assert!(bank.all_genomes_tested());
    }

    #[test]
    fn identical_genomes_are_related() {
        let configuration: Rc<RefCell<Configuration>> = Default::default();
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);

        bank.add_genome(genome.clone());
        bank.add_genome(genome);

        assert_eq!(
            bank.are_genomes_related(
                bank.genomes().get(0).unwrap(),
                bank.genomes().get(1).unwrap()
            ),
            true
        );
    }

    #[test]
    fn different_genomes_are_not_related() {
        let configuration: Rc<RefCell<Configuration>> = Rc::new(RefCell::new(Configuration {
            compatibility_threshold: 0.,
            ..Default::default()
        }));
        let mut bank = GenomeBank::new(configuration);

        bank.add_genome(Genome::new(1, 1));
        bank.add_genome(Genome::new(1, 1));

        assert_eq!(
            bank.are_genomes_related(
                bank.genomes().get(0).unwrap(),
                bank.genomes().get(1).unwrap()
            ),
            false
        );
    }

    #[test]
    fn identical_genomes_are_same_species() {
        let configuration: Rc<RefCell<Configuration>> = Default::default();
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);

        bank.add_genome(genome.clone());
        bank.add_genome(genome);

        bank.speciate();

        assert_eq!(bank.species.get(&0).unwrap().len(), 2);
    }

    #[test]
    fn different_genomes_are_different_species() {
        let configuration: Rc<RefCell<Configuration>> = Rc::new(RefCell::new(Configuration {
            compatibility_threshold: 0.,
            ..Default::default()
        }));
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);

        bank.add_genome(genome.clone());
        bank.add_genome(genome);
        bank.add_genome(Genome::new(1, 1));

        bank.speciate();

        assert_eq!(bank.species.get(&0).unwrap().len(), 2);
        assert_eq!(bank.species.get(&1).unwrap().len(), 1);
    }

    #[test]
    fn exports_adjusted_fitness() {
        let first_fitness = 10.;
        let second_fitness = 5.;
        let third_fitness = 5.;

        let configuration: Rc<RefCell<Configuration>> = Rc::new(RefCell::new(Configuration {
            compatibility_threshold: 0.,
            ..Default::default()
        }));
        let mut bank = GenomeBank::new(configuration);

        let genome = Genome::new(1, 1);

        bank.add_genome(genome.clone());
        bank.add_genome(genome);

        bank.add_genome(Genome::new(1, 1));

        bank.mark_fitness(0, first_fitness);
        bank.mark_fitness(1, second_fitness);

        bank.mark_fitness(2, third_fitness);

        bank.speciate();

        let adjusted_fitnesses = bank.adjusted_fitnesses();

        assert!(*adjusted_fitnesses.get(0).unwrap() < first_fitness);
        assert!(*adjusted_fitnesses.get(1).unwrap() < second_fitness);
        assert!((*adjusted_fitnesses.get(2).unwrap() - third_fitness).abs() < f64::EPSILON);
    }
}
