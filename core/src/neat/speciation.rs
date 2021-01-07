use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::configuration::Configuration;
use crate::genome::{Genome, GenomeId};

/// Holds all genomes and species, does the process of speciation
#[derive(Debug)]
pub struct GenomeBank {
    configuration: Rc<RefCell<Configuration>>,
    genomes: HashMap<GenomeId, Genome>,
    previous_genomes: HashMap<GenomeId, Genome>,
    fitnesses: HashMap<GenomeId, f64>,
}

impl GenomeBank {
    pub fn new(configuration: Rc<RefCell<Configuration>>) -> Self {
        GenomeBank {
            configuration,
            genomes: HashMap::new(),
            previous_genomes: HashMap::new(),
            fitnesses: HashMap::new(),
        }
    }

    /// Adds a new genome
    pub fn add_genome(&mut self, genome: Genome) {
        self.genomes.insert(genome.id(), genome);
    }

    /// Clear genomes
    pub fn clear(&mut self) {
        let mut new_bank = GenomeBank::new(self.configuration.clone());
        new_bank.previous_genomes = self.genomes.clone();

        *self = new_bank;
    }

    /// Returns a reference to the genomes
    pub fn genomes(&self) -> &HashMap<GenomeId, Genome> {
        &self.genomes
    }

    pub fn previous_genomes(&self) -> &HashMap<GenomeId, Genome> {
        &self.previous_genomes
    }

    /// Tracks the fitness of a particular genome
    pub fn mark_fitness(&mut self, genome_id: GenomeId, fitness: f64) {
        self.fitnesses.insert(genome_id, fitness);
    }

    /// Returns a reference to the fitnesses
    pub fn fitnesses(&self) -> &HashMap<GenomeId, f64> {
        &self.fitnesses
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
        bank.add_genome(genome.clone());

        bank.mark_fitness(genome.id(), 1337.);
    }
}
