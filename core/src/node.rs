use crate::activation::ActivationKind;
use crate::aggregations::Aggregation;
use crate::genome::node::NodeGene;

#[derive(Debug, Clone, PartialEq, Hash)]
#[cfg_attr(
    feature = "network-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
    Constant,
}

#[derive(Debug)]
#[cfg_attr(
    feature = "network-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Node {
    pub kind: NodeKind,
    pub aggregation: Aggregation,
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
            aggregation: g.aggregation.clone(),
        }
    }
}
