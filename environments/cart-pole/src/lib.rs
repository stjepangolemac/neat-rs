use rand::{random, thread_rng, Rng};

pub use neat_environment::Environment;
use utils::*;

mod utils;

pub struct CartPoleConfiguration {
    pub gravity: f64,
    pub mass_cart: f64,
    pub mass_pole: f64,
    pub mass_pole2: f64,
    pub length_pole: f64,
    pub length_pole2: f64,
    pub time_step: f64,

    pub limit_position: f64,
    pub limit_angle_radians: f64,
}

impl Default for CartPoleConfiguration {
    fn default() -> Self {
        CartPoleConfiguration {
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            mass_pole2: 0.02,
            length_pole: 1.,
            length_pole2: 0.2,
            time_step: 1. / 60.,

            limit_position: 2.4,
            limit_angle_radians: to_radians(45.),
        }
    }
}

pub struct CartPole {
    pub configuration: CartPoleConfiguration,

    x: f64,
    theta: f64,
    theta2: f64,

    dx: f64,
    dtheta: f64,
    dtheta2: f64,

    xacc: f64,
    tacc: f64,
    tacc2: f64,

    fitness: f64,
    t: f64,

    finished: bool,
}

impl CartPole {
    pub fn new_single() -> Self {
        let configuration: CartPoleConfiguration = Default::default();
        let mut rng = thread_rng();

        let x =
            rng.gen_range(-0.5 * configuration.limit_position..0.5 * configuration.limit_position);
        let theta = rng.gen_range(
            -0.5 * configuration.limit_angle_radians..0.5 * configuration.limit_angle_radians,
        );
        let dx = rng.gen_range(-1f64..1f64);
        let dtheta = rng.gen_range(-1f64..1f64);

        CartPole {
            configuration,

            x,
            theta,
            theta2: 0.,

            dx,
            dtheta,
            dtheta2: 0.,

            xacc: 0.,
            tacc: 0.,
            tacc2: 0.,

            t: 0.,
            fitness: 0.,

            finished: false,
        }
    }

    fn continuous_actuator_force(input: f64) -> f64 {
        input * 10.
    }

    fn continuous_noisy_actuator_force(input: f64) -> f64 {
        (input + random::<f64>() * 0.75) * 10.
    }

    fn measure_fitness(&mut self) {
        let x_component = f64::max(0., self.configuration.limit_position - self.x.abs());
        let theta_component = f64::max(
            0.,
            self.configuration.limit_angle_radians - self.theta.abs(),
        );

        let step_fitness = 1. - x_component * theta_component;

        self.fitness += step_fitness.powi(2);
    }

    fn check_finished(&mut self) {
        if self.x.abs() > self.configuration.limit_position
            || self.theta.abs() > self.configuration.limit_angle_radians
        {
            self.finished = true;
        }
    }

    pub fn apply_force_to_pole(&mut self, force: f64) {
        self.dtheta += force;
    }
}

impl Environment for CartPole {
    type State = [f64; 4];
    type Input = f64;

    fn state(&self) -> Self::State {
        [self.x, self.dx, self.theta, self.dtheta]
    }

    fn step(&mut self, input: Self::Input) -> Result<(), ()> {
        if input > 1. || input < -1. {
            panic!("Input must be between 1 and -1");
        }
        if self.done() {
            return Err(());
        }

        utils::step_single_pole(self, input);

        Ok(())
    }

    fn done(&self) -> bool {
        self.finished
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn reset(&mut self) {
        *self = CartPole::new_single();
    }

    fn render(&self) {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misc() {
        let mut env = CartPole::new_single();

        for _ in 0..5 {
            env.step(1.).unwrap();

            let state = env.state();
            dbg!(state);
        }

        let fitness = env.fitness();

        dbg!(fitness);
    }
}
