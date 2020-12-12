use crate::activation::ActivationKind;
use crate::node::NodeKind;
use rand::random;

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub disabled: bool,
}

impl ConnectionGene {
    pub fn new(from: usize, to: usize) -> Self {
        ConnectionGene {
            from,
            to,
            weight: random::<f64>() - 0.5,
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

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub kind: NodeKind,
    pub activation: ActivationKind,
    pub bias: f64,
}

impl NodeGene {
    pub fn new(kind: NodeKind) -> Self {
        let activation: ActivationKind = match kind {
            NodeKind::Input => ActivationKind::Input,
            _ => rand::random(),
        };
        let bias: f64 = match kind {
            NodeKind::Input => 0.,
            _ => rand::random::<f64>() - 0.5,
        };

        NodeGene {
            kind,
            activation,
            bias,
        }
    }
}
