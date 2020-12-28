use crate::activation::*;
use crate::aggregations::aggregate;
use crate::connection::*;
use crate::genome::Genome;
use crate::node::*;

#[derive(Debug)]
pub struct Network {
    pub input_count: usize,
    pub output_count: usize,
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    node_calculation_order: Vec<usize>,
}

impl Network {
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

    pub fn forward_pass(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        for i in &self.node_calculation_order {
            let node = self.nodes.get(*i).unwrap();

            if matches!(node.kind, NodeKind::Input) {
                self.nodes.get_mut(*i).unwrap().value = Some(*inputs.get(*i).unwrap());
            } else {
                let components: Vec<f64> = self
                    .connections
                    .iter()
                    .filter(|c| c.to == *i)
                    .map(|c| {
                        let incoming_value = self.nodes.get(c.from).unwrap().value.unwrap();
                        incoming_value * c.weight
                    })
                    .collect();

                let aggregated = aggregate(&node.aggregation, &components);
                let aggregated_with_bias = aggregated + node.bias;

                self.nodes.get_mut(*i).unwrap().value =
                    Some(activate(aggregated_with_bias, &node.activation));
            }
        }

        self.nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Output))
            .map(|n| n.value.unwrap())
            .collect()

        // let mut inputs_updated = false;
        // let mut nodes_changed = -1;
        // let mut nodes_changed_sum = 0;

        // while nodes_changed != 0 {
        //     nodes_changed = 0;

        //     // First pass, update inputs
        //     if !inputs_updated {
        //         self.nodes
        //             .iter_mut()
        //             .enumerate()
        //             .filter(|(_, n)| matches!(n.kind, NodeKind::Input))
        //             .for_each(|(i, n)| {
        //                 let input_value = *inputs.get(i).expect(
        //                     "Inputs need to be of the same length as the number of input nodes",
        //                 );

        //                 n.value = Some(input_value);
        //                 nodes_changed += 1;
        //             });

        //         inputs_updated = true;
        //     }

        //     // Other passes, update non input nodes
        //     let mut node_updates: Vec<(usize, f64)> = vec![];
        //     self.nodes
        //         .iter()
        //         .enumerate()
        //         .filter(|(i, n)| {
        //             let is_not_input = !matches!(n.kind, NodeKind::Input);
        //             let is_ready = self.is_node_ready(*i);

        //             is_not_input && is_ready
        //         })
        //         .for_each(|(i, n)| {
        //             let incoming_connections: Vec<&Connection> =
        //                 self.connections.iter().filter(|c| c.to == i).collect();

        //             let mut value = 0.;

        //             for c in incoming_connections {
        //                 let from_node = self.nodes.get(c.from).unwrap();
        //                 value += from_node.value.unwrap() * c.weight;
        //             }

        //             value += n.bias;

        //             node_updates.push((i, value));
        //         });

        //     node_updates.iter().for_each(|(i, v)| {
        //         let n = self.nodes.get_mut(*i).unwrap();

        //         n.value = Some(activate(*v, &n.activation));

        //         nodes_changed += 1;
        //     });

        //     nodes_changed_sum += nodes_changed;
        // }

        // let outputs = self
        //     .nodes
        //     .iter()
        //     .filter(|n| matches!(n.kind, NodeKind::Output))
        //     .map(|n| n.value.unwrap())
        //     .collect();

        // // Very important, I forgot this initially :facepalm:
        // self.clear_values();

        // outputs
    }

    fn clear_values(&mut self) {
        self.nodes.iter_mut().for_each(|n| n.value = None);
    }
}

impl From<&Genome> for Network {
    fn from(g: &Genome) -> Self {
        let nodes: Vec<Node> = g.nodes().iter().map(From::from).collect();
        let connections: Vec<Connection> = g
            .connections()
            .iter()
            .filter(|c| !c.disabled)
            .map(From::from)
            .collect();

        Network {
            input_count: g.input_count(),
            output_count: g.output_count(),
            nodes,
            connections,
            node_calculation_order: g.node_order().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_network() {
        let g = Genome::new(1, 1);
        Network::from(&g);
    }

    #[test]
    fn forward_pass() {
        let g = Genome::new(2, 1);
        let mut n = Network::from(&g);

        let inputs: Vec<Vec<f64>> = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];

        for i in inputs {
            let o = n.forward_pass(i.clone());

            dbg!(i, o);
        }
    }
}
