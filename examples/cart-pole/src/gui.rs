use nannou::prelude::*;
use nannou::ui::prelude::*;

use neat_core::Network;
use neat_environment_cart_pole::{CartPole, CartPoleConfiguration, Environment};
use neat_export::from_file;

pub fn visualize() {
    nannou::app(model).update(update).view(view).run();
}

struct Model {
    network: Option<Network>,
    env: CartPole,
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(480, 320)
        .dropped_file(dropped_file)
        .key_released(key_released)
        .build()
        .unwrap();

    Model {
        network: None,
        env: CartPole::new_single(),
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {
    if let Some(ref mut network) = model.network {
        let state = model.env.state();
        let network_output = network.forward_pass(state.to_vec());
        let env_input = f64::max(-1., f64::min(1., *network_output.first().unwrap()));

        if model.env.step(env_input).is_err() {
            model.env.reset();
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let CartPoleConfiguration { length_pole, .. } = model.env.configuration;
    let [x, _, theta, _] = model.env.state();

    let cart_x = 0. + x as f32 * 100.;
    let cart_width = 20.;
    let cart_height = 10.;
    let pole_height = length_pole as f32 * 100.;
    let pole_rotational_dx = theta.sin() as f32 * pole_height / 2.;
    let pole_rotational_dy = theta.cos() as f32 * pole_height / 2.;

    let draw = app.draw();

    draw.background().rgb(0., 0., 0.);

    draw.rect()
        .x(cart_x)
        .y(0. - cart_height / 2.)
        .w_h(cart_width, cart_height);
    draw.rect()
        .x(cart_x + pole_rotational_dx)
        .y(pole_rotational_dy)
        .w_h(1., pole_height)
        .rotate(-theta as f32);

    // Write the result of our drawing to the window's frame.
    draw.to_frame(app, &frame).unwrap();
}

fn dropped_file(_app: &App, model: &mut Model, path: std::path::PathBuf) {
    model.network = Some(from_file(path));
}

fn key_released(_app: &App, model: &mut Model, key: Key) {
    let force = match key {
        Key::Left => -1.,
        Key::Right => 1.,
        _ => 0.,
    };

    model.env.apply_force_to_pole(force);
}
