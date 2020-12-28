use rand::{thread_rng, Rng};

pub use neat_environment::Environment;
use utils::*;

mod utils;

pub struct CartPoleConfiguration {
    gravity: f64,
    mass_cart: f64,
    mass_pole: f64,
    length_pole: f64,
    time_step: f64,

    limit_position: f64,
    limit_angle_radians: f64,
}

impl Default for CartPoleConfiguration {
    fn default() -> Self {
        CartPoleConfiguration {
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            length_pole: 0.5,
            time_step: 0.01,

            limit_position: 2.4,
            limit_angle_radians: to_radians(45.),
        }
    }
}

pub struct CartPole {
    configuration: CartPoleConfiguration,

    x: f64,
    theta: f64,
    dx: f64,
    dtheta: f64,
    t: f64,
    xacc: f64,
    tacc: f64,
    fitness: f64,

    finished: bool,
}

impl CartPole {
    pub fn new() -> Self {
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
            dx,
            dtheta,
            t: 0.,
            xacc: 0.,
            tacc: 0.,
            fitness: 0.,

            finished: false,
        }
    }

    fn continuous_actuator_force(input: f64) -> f64 {
        input * 10.
    }

    fn measure_fitness(&mut self) {
        let x_component = f64::max(0., self.configuration.limit_position - self.x.abs());
        let theta_component = f64::max(
            0.,
            self.configuration.limit_angle_radians - self.theta.abs(),
        );

        self.fitness += 1. - x_component * theta_component;
    }

    fn check_finished(&mut self) {
        if self.x.abs() > self.configuration.limit_position
            || self.theta.abs() > self.configuration.limit_angle_radians
        {
            self.finished = true;
        }
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

        let force = CartPole::continuous_actuator_force(input);
        let xacc_current = self.xacc;
        let tacc_current = self.tacc;
        let mass_all = self.configuration.mass_pole + self.configuration.mass_cart;

        self.x += self.configuration.time_step * self.dx
            + 0.5 * xacc_current * self.configuration.time_step.powi(2);
        self.theta += self.configuration.time_step * self.dtheta
            + 0.5 * tacc_current * self.configuration.time_step.powi(2);

        let theta_sin = self.theta.sin();
        let theta_cos = self.theta.cos();

        self.tacc = (self.configuration.gravity * theta_sin
            + theta_cos
                * (-force
                    - self.configuration.mass_pole
                        * self.configuration.length_pole
                        * self.dtheta.powi(2)
                        * theta_sin)
                / mass_all)
            / (self.configuration.length_pole
                * (4. / 3. - self.configuration.mass_pole * theta_cos.powi(2) / mass_all));
        self.xacc = (force
            + self.configuration.mass_pole
                * self.configuration.length_pole
                * (self.dtheta.powi(2) * theta_sin - self.tacc * theta_cos))
            / mass_all;

        self.dx += 0.5 * (xacc_current + self.xacc) * self.configuration.time_step;
        self.dtheta += 0.5 * (tacc_current + self.tacc) * self.configuration.time_step;

        self.t += self.configuration.time_step;

        self.measure_fitness();
        self.check_finished();

        Ok(())
    }

    fn done(&self) -> bool {
        self.finished
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn reset(&mut self) {
        *self = CartPole::new();
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
        let mut env = CartPole::new();

        for _ in 0..5 {
            env.step(1.).unwrap();

            let state = env.state();
            dbg!(state);
        }

        let fitness = env.fitness();

        dbg!(fitness);
    }
}
