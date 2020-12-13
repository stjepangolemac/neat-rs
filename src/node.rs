use crate::activation::ActivationKind;
use crate::genome::genes::NodeGene;

#[derive(Debug, Clone, PartialEq, Hash)]
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

impl From<&NodeGene> for Node {
    fn from(g: &NodeGene) -> Self {
        Node {
            kind: g.kind.clone(),
            activation: g.activation.clone(),
            bias: g.bias,
            value: None,
        }
    }
}
