# neat-rs

A neuroevolution framework written in rust.

## How to use

Here is how to train a cart pole balancing neural network, available in the
`examples/` dir.

```rust
let mut system = NEAT::new(4, 1, |network| {
    let num_simulations = 10;
    let max_steps = 1000;
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
```

To start the training go to the `examples/cart-pole/` dir and run the
command below. It will produce a `network.bin` that contains the neural network
"recipe".

```bash
cargo run --release -- train
```

After training, you can see the neural network balancing the pole by running
the command below, then dragging the `network.bin` into the window. That will
load and instantiate the neural network. You can apply "wind" with arrow keys.

```bash
cargo run --release -- visualize
```

## Things I'd like to add (but probably won't due to the lack of time)

- Two pole balancing task (started it in a different branch)
- Recurrent connections
- Extend the `system` so it works with both `f32` and `f64` (might improve performance)
- HyperNEAT
- FS NEAT (feature selection)

## Is this useful?

If somebody finds this code useful, or is even willing to fund further
development, I'd be happy to talk to you. You can reach me by [sending me a message
on Twitter](https://twitter.com/SGolemac).
