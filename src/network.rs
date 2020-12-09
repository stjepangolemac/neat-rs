use crate::activation::*;
use crate::connection::*;
use crate::node::*;

#[derive(Debug)]
pub struct Network {
    input_count: usize,
    output_count: usize,
    nodes: Vec<Node>,
    connections: Vec<Connection>,
}

impl Network {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        let mut nodes: Vec<Node> = vec![];
        (0..input_count).for_each(|_| nodes.push(Node::new_input()));
        (0..output_count).for_each(|_| nodes.push(Node::new_output()));

        let mut connections: Vec<Connection> = vec![];
        nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
            .for_each(|(input_i, _)| {
                nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| matches!(n.kind, NodeKind::Output))
                    .for_each(|(output_i, _)| {
                        connections.push(Connection::new(input_i, output_i));
                    });
            });

        Network {
            input_count,
            output_count,
            nodes,
            connections,
        }
    }

    fn is_node_ready(&self, index: usize) -> bool {
        let node = self.nodes.get(index).unwrap();

        let requirements_fullfilled = self.connections.iter().filter(|c| c.to == index).all(|c| {
            let from_index = c.from;
            let from_node = &self.nodes[from_index];

            from_node.value.is_some()
        });
        let has_no_value = node.value.is_none();

        requirements_fullfilled && has_no_value
    }

    pub fn forward_pass(&mut self, inputs: Vec<f64>) {
        let mut inputs_updated = false;
        let mut nodes_changed = -1;

        while nodes_changed != 0 {
            nodes_changed = 0;

            // First pass, update inputs
            if !inputs_updated {
                self.nodes
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
                    .for_each(|(i, n)| {
                        let input_value = *inputs.get(i).expect(
                            "Inputs need to be of the same length as the number of input nodes",
                        );

                        n.value = Some(input_value);
                        nodes_changed += 1;
                    });

                inputs_updated = true;
            }

            // Other passes, update non input nodes
            let mut node_updates: Vec<(usize, f64)> = vec![];
            self.nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| {
                    let is_not_input = !matches!(n.kind, NodeKind::Input);
                    let is_ready = self.is_node_ready(*i);

                    is_not_input && is_ready
                })
                .for_each(|(i, n)| {
                    let incoming_connections: Vec<&Connection> =
                        self.connections.iter().filter(|c| c.to == i).collect();

                    let mut value = 0.;

                    for c in incoming_connections {
                        let from_node = self.nodes.get(c.from).unwrap();
                        value += from_node.value.unwrap() * c.weight;
                    }

                    value += n.bias;

                    node_updates.push((i, value));
                });

            node_updates.iter().for_each(|(i, v)| {
                let n = self.nodes.get_mut(*i).unwrap();

                n.value = Some(activate(*v, &n.activation));

                nodes_changed += 1;
            });
        }
    }

    fn clear_values(&mut self) {
        self.nodes.iter_mut().for_each(|n| n.value = None);
    }

    /// Used for the add node mutation
    pub fn insert_node(&mut self, connection: usize) {
        // 1. Create a new hidden node
        // 2. Connect from connection.from to it
        // 3. Connect from it to the connection.to
        // 4. Remove the original connection

        // PROBLEM TODO When a new node is added all subsequent nodes change their index
        // if you're not adding at the end of the vector

        // let next_index = self.nodes.len();
        // self.nodes.push(Node::new_hidden());

        // let conn = self.connections.get(connection).unwrap();
        // self.connections.push(Connection::new(conn.from, next_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_network() {
        Network::new(3, 3);
    }

    #[test]
    fn forward_pass() {
        let input_count = 3;
        let mut network = Network::new(input_count, 3);

        let inputs: Vec<f64> = (0..input_count).map(|_| rand::random::<f64>()).collect();
        network.forward_pass(inputs);
    }
}
