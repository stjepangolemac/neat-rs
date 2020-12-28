use crate::genome::connection::ConnectionGene;

#[derive(Debug)]
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
