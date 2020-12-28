use crate::genome::connection::ConnectionGene;

#[derive(Debug)]
#[cfg_attr(
    feature = "network-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

impl From<&ConnectionGene> for Connection {
    fn from(g: &ConnectionGene) -> Self {
        Connection {
            from: g.from,
            to: g.to,
            weight: g.weight,
        }
    }
}
