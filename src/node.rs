use crate::activation::ActivationKind;

#[derive(Debug, Clone)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
    Constant,
}

#[derive(Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub activation: ActivationKind,
    pub bias: f64,
    pub value: Option<f64>,
}

impl Node {
    pub fn new(kind: NodeKind) -> Self {
        let activation: ActivationKind = match kind {
            NodeKind::Input => ActivationKind::Input,
            _ => rand::random(),
        };
        let bias: f64 = match kind {
            NodeKind::Input => 0.,
            _ => rand::random::<f64>() - 0.5,
        };

        Node {
            kind,
            activation,
            bias,
            value: None,
        }
    }

    pub fn new_input() -> Self {
        Node::new(NodeKind::Input)
    }

    pub fn new_output() -> Self {
        Node::new(NodeKind::Output)
    }

    pub fn new_hidden() -> Self {
        Node::new(NodeKind::Hidden)
    }
}
