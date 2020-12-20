use rand::random;

use super::{ConnectionGene, Genome, NodeGene};

pub fn crossover(a: (&Genome, f64), b: (&Genome, f64)) -> Option<Genome> {
    let mut parent_a = a.0.clone();
    let mut fitness_a = a.1;

    let mut parent_b = b.0.clone();
    let mut fitness_b = b.1;

    // Parent A will always be the fitter one
    if fitness_a < fitness_b {
        std::mem::swap(&mut parent_a, &mut parent_b);
        std::mem::swap(&mut fitness_a, &mut fitness_b);
    }

    let mut child = Genome::empty(parent_a.inputs, parent_a.outputs);

    let child_connection_genes: Vec<ConnectionGene> = parent_a
        .connection_genes
        .iter()
        .map(|connection| {
            let maybe_counterpart_connection = parent_b
                .connection_genes
                .iter()
                .find(|cb| cb.innovation_number() == connection.innovation_number());

            if let Some(counterpart_connection) = maybe_counterpart_connection {
                if random::<f64>() < 0.5 {
                    connection
                } else {
                    counterpart_connection
                }
            } else {
                connection
            }
        })
        .cloned()
        .collect();

    let required_node_count = 1 + child_connection_genes
        .iter()
        .fold(0, |max, c| usize::max(usize::max(max, c.from), c.to));

    let child_node_genes: Vec<NodeGene> = (0..required_node_count)
        .map(
            |i| match (parent_a.node_genes.get(i), parent_b.node_genes.get(i)) {
                (Some(a), Some(b)) => {
                    if random::<f64>() < 0.5 {
                        a
                    } else {
                        b
                    }
                }
                (Some(a), None) => a,
                (None, Some(b)) => b,
                _ => panic!("Node selection out of bounds"),
            },
        )
        .cloned()
        .collect();

    child.connection_genes = child_connection_genes;
    child.node_genes = child_node_genes;

    child.node_order().and(Some(child))
}
