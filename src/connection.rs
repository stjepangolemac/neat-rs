#[derive(Debug)]
pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

impl Connection {
    pub fn new(from: usize, to: usize) -> Self {
        Connection {
            from,
            to,
            weight: rand::random::<f64>() - 0.5,
        }
    }
}
