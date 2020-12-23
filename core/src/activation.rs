use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ActivationKind {
    Input,
    Tanh,
    Relu,
    Step,
}

impl Distribution<ActivationKind> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ActivationKind {
        match rng.gen_range(0, 3) {
            0 => ActivationKind::Tanh,
            1 => ActivationKind::Relu,
            _ => ActivationKind::Step,
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
        ActivationKind::Step => {
            if x > 0. {
                1.
            } else {
                0.
            }
        }
        _ => x,
    }
}
