use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub enum ActivationKind {
    Input,
    Tanh,
    Relu,
}

impl Distribution<ActivationKind> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ActivationKind {
        match rng.gen_range(0, 2) {
            0 => ActivationKind::Tanh,
            _ => ActivationKind::Relu,
        }
    }
}

pub fn activate(x: f64, kind: &ActivationKind) -> f64 {
    match kind {
        ActivationKind::Tanh => x.tanh(),
        ActivationKind::Relu => {
            if x > 0. {
                x
            } else {
                0.
            }
        }
        _ => x,
    }
}
