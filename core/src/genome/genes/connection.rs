use rand::random;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub from: usize,
    pub to: usize,
    pub disabled: bool,
    pub weight: f64,
}

impl ConnectionGene {
    pub fn new(from: usize, to: usize) -> Self {
        ConnectionGene {
            from,
            to,
            weight: random::<f64>() * 2. - 1.,
            disabled: false,
        }
    }

    pub fn innovation_number(&self) -> usize {
        let a = self.from;
        let b = self.to;

        let first_part = (a + b) * (a + b + 1);
        let second_part = b;

        first_part.checked_div(2).unwrap() + second_part
    }
}

impl PartialEq for ConnectionGene {
    fn eq(&self, other: &Self) -> bool {
        self.from == other.from
            && self.to == other.to
            && self.disabled == other.disabled
            && (self.weight - other.weight).abs() < f64::EPSILON
    }
}

impl Eq for ConnectionGene {}

impl Hash for ConnectionGene {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.from.hash(state);
        self.to.hash(state);
        self.disabled.hash(state);
        self.weight.to_bits().hash(state);
    }
}
