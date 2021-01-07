use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::Configuration;
use crate::{Genome, GenomeId};

use distance::GenomicDistanceCache;

mod distance;

pub struct SpeciesSet {
    configuration: Rc<RefCell<Configuration>>,
    last_index: Option<usize>,
    species: HashMap<usize, Species>,
}

impl SpeciesSet {
    pub fn new(configuration: Rc<RefCell<Configuration>>) -> Self {
        SpeciesSet {
            configuration,
            last_index: None,
            species: HashMap::new(),
        }
    }

    pub fn species(&self) -> &HashMap<usize, Species> {
        &self.species
    }

    pub fn speciate(
        &mut self,
        generation: usize,
        current_genomes: &[GenomeId],
        all_genomes: &HashMap<GenomeId, Genome>,
        fitnesses: &HashMap<GenomeId, f64>,
    ) {
        let (compatibility_threshold, stagnation_after, elitism_species) = {
            let config = self.configuration.borrow();

            (
                config.compatibility_threshold,
                config.stagnation_after,
                config.elitism_species,
            )
        };

        let mut distances = GenomicDistanceCache::new(self.configuration.clone());

        let mut unspeciated_genomes: HashSet<GenomeId> = current_genomes.iter().cloned().collect();
        let mut new_species: HashMap<usize, Species> = self.species.clone();

        // Find new representatives for existing species
        self.species.iter().for_each(|(species_id, species)| {
            let genome_representative = all_genomes.get(&species.representative).unwrap();

            let (maybe_new_representative_id, _) = current_genomes
                .iter()
                .map(|genome_id| {
                    let genome = all_genomes.get(genome_id).unwrap();
                    (genome_id, distances.get(genome, genome_representative))
                })
                .filter(|(_, distance)| *distance < compatibility_threshold)
                .fold(
                    (None, f64::MAX),
                    |(maybe_closest_genome_id, closest_genome_distance),
                     (genome_id, genome_distance)| {
                        if maybe_closest_genome_id.is_some() {
                            if genome_distance < closest_genome_distance {
                                return (Some(genome_id), genome_distance);
                            }
                        } else {
                            return (Some(genome_id), genome_distance);
                        }

                        (maybe_closest_genome_id, closest_genome_distance)
                    },
                );

            if let Some(new_representative_id) = maybe_new_representative_id {
                let species = new_species.get_mut(species_id).unwrap();
                species.representative = *new_representative_id;
                species.members = vec![*new_representative_id];

                unspeciated_genomes.remove(&new_representative_id);
            } else {
                new_species.remove(species_id);
            }
        });

        // Put unspeciated genomes into species
        unspeciated_genomes.iter().for_each(|genome_id| {
            let genome = all_genomes.get(genome_id).unwrap();

            let (maybe_closest_species_id, _) = {
                new_species
                    .iter()
                    .map(|(species_id, species)| {
                        let species_representative_genome =
                            all_genomes.get(&species.representative).unwrap();

                        (
                            species_id,
                            distances.get(genome, species_representative_genome),
                        )
                    })
                    .filter(|(_, distance)| *distance < compatibility_threshold)
                    .fold(
                        (None, f64::MAX),
                        |(maybe_closest_species_id, closest_representative_distance),
                         (species_id, representative_distance)| {
                            if maybe_closest_species_id.is_some() {
                                if representative_distance < closest_representative_distance {
                                    return (Some(*species_id), representative_distance);
                                }
                            } else {
                                return (Some(*species_id), representative_distance);
                            }

                            (maybe_closest_species_id, closest_representative_distance)
                        },
                    )
            };

            if let Some(closest_species_id) = maybe_closest_species_id {
                // Fits into an existing species
                new_species
                    .get_mut(&closest_species_id)
                    .unwrap()
                    .members
                    .push(*genome_id);
            } else {
                // Needs to go in a brand new species
                let species = Species::new(generation, *genome_id, vec![*genome_id]);
                let next_species_id = new_species.keys().max().or(Some(&0)).cloned().unwrap();

                new_species.insert(next_species_id + 1, species);
            }
        });

        // Calculate fitness for every species
        new_species.iter_mut().for_each(|(_, mut species)| {
            let member_fitnesses: Vec<f64> = species
                .members
                .iter()
                .map(|member_genome_id| *fitnesses.get(member_genome_id).unwrap())
                .collect();

            let species_mean_fitness =
                member_fitnesses.iter().sum::<f64>() / member_fitnesses.len() as f64;
            let best_previous_fitness = species
                .fitness_history
                .iter()
                .cloned()
                .fold(f64::MIN, f64::max);

            if species_mean_fitness > best_previous_fitness {
                species.last_improved = generation;
            }

            species.fitness = Some(species_mean_fitness);
            species.fitness_history.push(species_mean_fitness);
        });

        // Calculate adjusted fitness for every species
        let species_fitnesses: Vec<f64> = new_species
            .iter()
            .map(|(_, species)| species.fitness.unwrap())
            .collect();

        new_species.iter_mut().for_each(|(_, mut species)| {
            let own_exp = species.fitness.unwrap().exp();
            let exp_sum: f64 = species_fitnesses.iter().map(|fitness| fitness.exp()).sum();

            let adjusted_fitness = own_exp / exp_sum;

            species.adjusted_fitness = Some(adjusted_fitness);
        });

        // Remove stagnated species
        let mut stagnated_ids_and_adjusted_fitnesses: Vec<(usize, f64)> = new_species
            .iter()
            .filter(|(_, species)| generation - species.last_improved >= stagnation_after)
            .map(|(id, species)| (*id, species.adjusted_fitness.unwrap()))
            .collect();

        stagnated_ids_and_adjusted_fitnesses.sort_by(|a, b| {
            use std::cmp::Ordering::*;

            if a.1 > b.1 {
                Less
            } else {
                Greater
            }
        });

        stagnated_ids_and_adjusted_fitnesses
            .iter()
            .take(usize::max(new_species.len() - elitism_species, 0))
            .for_each(|(id, _)| {
                new_species.remove(id).unwrap();
            });

        // Finally replace old species
        self.species = new_species;
    }
}

#[derive(Debug, Clone)]
pub struct Species {
    created: usize,

    last_improved: usize,
    representative: GenomeId,
    pub members: Vec<GenomeId>,

    fitness: Option<f64>,
    pub adjusted_fitness: Option<f64>,
    fitness_history: Vec<f64>,
}

impl Species {
    pub fn new(generation: usize, representative: GenomeId, members: Vec<GenomeId>) -> Self {
        Species {
            created: generation,
            last_improved: generation,
            representative,
            members,
            fitness: None,
            adjusted_fitness: None,
            fitness_history: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let genome = Genome::new(5, 3);

        let first_hash = {
            let mut hasher = DefaultHasher::new();

            genome.hash(&mut hasher);
            hasher.finish().to_string()
        };

        let second_hash = {
            let genome_clone = genome.clone();
            let mut hasher = DefaultHasher::new();

            genome_clone.hash(&mut hasher);
            hasher.finish().to_string()
        };

        dbg!(&first_hash, &second_hash);

        assert_eq!(first_hash, second_hash);
    }
}
