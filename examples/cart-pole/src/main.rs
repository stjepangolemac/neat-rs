use neat_core::{Configuration, NEAT};
use neat_environment_cart_pole::{CartPole, Environment};

fn main() {
    let mut system = NEAT::new(4, 1, |network| {
        let num_simulations = 10;
        let max_steps = 200;
        let mut env = CartPole::new();

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

        // dbg!(steps_done / num_simulations);

        fitness / num_simulations as f64
    });

    system.set_configuration(Configuration {
        population_size: 150,
        node_cost: 1.,
        connection_cost: 1.,
        compatibility_threshold: 1.5,
        ..Default::default()
    });

    system.add_hook(1, |generation, system| {
        println!(
            "Generation {}, best fitness is {}",
            generation,
            system.get_best().2
        );
    });

    let (network, fitness) = system.start();

    println!(
        "Found network with {} nodes and {} connections, fitness is {}",
        network.nodes.len(),
        network.connections.len(),
        fitness
    );
}
