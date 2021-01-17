use std::f64::consts::PI;

use crate::{CartPole, Environment};

pub fn to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.
}

pub(crate) fn step_single_pole(env: &mut CartPole, input: f64) {
    let force = CartPole::continuous_actuator_force(input);
    let xacc_current = env.xacc;
    let tacc_current = env.tacc;

    let mass_all = env.configuration.mass_pole + env.configuration.mass_cart;

    env.x += env.configuration.time_step * env.dx
        + 0.5 * xacc_current * env.configuration.time_step.powi(2);
    env.theta += env.configuration.time_step * env.dtheta
        + 0.5 * tacc_current * env.configuration.time_step.powi(2);

    let theta_sin = env.theta.sin();
    let theta_cos = env.theta.cos();

    env.tacc = (env.configuration.gravity * theta_sin
        + theta_cos
            * (-force
                - env.configuration.mass_pole * env.configuration.length_pole / 2.
                    * env.dtheta.powi(2)
                    * theta_sin)
            / mass_all)
        / (env.configuration.length_pole / 2.
            * (4. / 3. - env.configuration.mass_pole * theta_cos.powi(2) / mass_all));
    env.xacc = (force
        + env.configuration.mass_pole * env.configuration.length_pole / 2.
            * (env.dtheta.powi(2) * theta_sin - env.tacc * theta_cos))
        / mass_all;

    env.dx += 0.5 * (xacc_current + env.xacc) * env.configuration.time_step;
    env.dtheta += 0.5 * (tacc_current + env.tacc) * env.configuration.time_step;

    env.t += env.configuration.time_step;

    env.measure_fitness();
    env.check_finished();
}

// TODO Figure out the formulas for two poles
pub(crate) fn step_double_pole(env: &mut CartPole, input: f64) {
    let force = CartPole::continuous_actuator_force(input);
    let xacc_current = env.xacc;
    let tacc_current = env.tacc;

    let mass_all = env.configuration.mass_pole + env.configuration.mass_cart;

    env.x += env.configuration.time_step * env.dx
        + 0.5 * xacc_current * env.configuration.time_step.powi(2);
    env.theta += env.configuration.time_step * env.dtheta
        + 0.5 * tacc_current * env.configuration.time_step.powi(2);

    let theta_sin = env.theta.sin();
    let theta_cos = env.theta.cos();

    env.tacc = (env.configuration.gravity * theta_sin
        + theta_cos
            * (-force
                - env.configuration.mass_pole * env.configuration.length_pole / 2.
                    * env.dtheta.powi(2)
                    * theta_sin)
            / mass_all)
        / (env.configuration.length_pole / 2.
            * (4. / 3. - env.configuration.mass_pole * theta_cos.powi(2) / mass_all));
    env.xacc = (force
        + env.configuration.mass_pole * env.configuration.length_pole / 2.
            * (env.dtheta.powi(2) * theta_sin - env.tacc * theta_cos))
        / mass_all;

    env.dx += 0.5 * (xacc_current + env.xacc) * env.configuration.time_step;
    env.dtheta += 0.5 * (tacc_current + env.tacc) * env.configuration.time_step;

    env.t += env.configuration.time_step;

    env.measure_fitness();
    env.check_finished();
}
