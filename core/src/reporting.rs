use std::collections::HashMap;
use std::time::Instant;

use crate::genome::Genome;

// TODO
pub type Species = usize;
pub struct Generation {
    number: usize,
    started: Instant,
}
pub struct Population<'system> {
    genomes: Vec<Genome>,
    fitnesses: HashMap<&'system Genome, f64>,
}

pub trait Reporter {
    fn on_generation_start(generation: Generation) {}
    fn on_generation_end(generation: Generation, population: Population, species: &[Species]) {}
    fn on_evaluation_end(population: Population, species: &[Species], best_genome: &Genome) {}
    fn on_reproduction_end(population: Population, species: &[Species]) {}
    fn on_extinction() {}
    fn on_solution_found(generation: Generation, population: Population, best_genome: &Genome) {}
    fn on_species_stagnant(species_id: usize, species: &[Species]) {}
}

struct StdoutReporter;

impl Reporter for StdoutReporter {
    fn on_generation_start(generation: Generation) {
        println!("Running generation {}", generation.number);
    }

    fn on_generation_end(generation: Generation, population: Population, species: &[Species]) {
        println!(
            "Generation {} done in {} seconds with {} members in {} species",
            generation.number,
            generation.started.elapsed().as_secs(),
            population.genomes.len(),
            species.len()
        );
    }

    fn on_evaluation_end(population: Population, species: &[Species], best_genome: &Genome) {
        let average_fitness = population
            .fitnesses
            .iter()
            .map(|(_, fitness)| fitness)
            .sum::<f64>()
            / population.fitnesses.len() as f64;

        println!(
            "Evaluated members have an average fitness of {}, best genome has {}",
            average_fitness,
            population.fitnesses.get(best_genome).unwrap()
        );
    }

    fn on_extinction() {
        println!("All species are extinct");
    }

    fn on_solution_found(generation: Generation, population: Population, best_genome: &Genome) {
        println!(
            "Best genome found in generation {} and has fitness {}",
            generation.number,
            population.fitnesses.get(best_genome).unwrap()
        );
    }

    fn on_species_stagnant(species_id: usize, species: &[Species]) {
        println!("Removing stagnant species {}", species_id);
    }
}
