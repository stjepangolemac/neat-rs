use neat_core::{Configuration, NEAT};
use neat_environment_cart_pole::{CartPole, Environment};
use neat_export::to_file;

mod gui;

fn train() {
    let mut system = NEAT::new(4, 1, |network| {
        let num_simulations = 10;
        let max_steps = 1000;
        let mut env = CartPole::new_single();

        let mut steps_done = 0;
        let mut fitness = 0.;

        for _ in 0..num_simulations {
            env.reset();

            for _ in 0..max_steps {
                if env.done() {
                    break;
                }

                let state = env.state();
                let network_output = network.forward_pass(state.to_vec());
                let env_input = f64::max(-1., f64::min(1., *network_output.first().unwrap()));

                env.step(env_input).unwrap();
                steps_done += 1;
            }

            fitness += env.fitness();
        }

        fitness / num_simulations as f64
    });

    system.set_configuration(Configuration {
        population_size: 100,
        max_generations: 500,
        stagnation_after: 50,
        node_cost: 1.,
        connection_cost: 1.,
        compatibility_threshold: 2.,
        ..Default::default()
    });

    system.add_hook(10, |generation, system| {
        println!(
            "Generation {}, best fitness is {}, {} species alive",
            generation,
            system.get_best().2,
            system.species_set.species().len()
        );
    });

    let (network, fitness) = system.start();

    // println!(
    //     "Found network with {} nodes and {} connections, fitness is {}",
    //     network.nodes.len(),
    //     network.connections.len(),
    //     fitness
    // );

    to_file("network.bin", &network);
}

fn main() {
    let param: String = std::env::args().skip(1).take(1).collect();

    if param == "train" {
        train();
    };

    if param == "visualize" {
        gui::visualize();
    };
}
