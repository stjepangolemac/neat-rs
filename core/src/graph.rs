trait Node {}

trait Edge {
    fn from() -> usize;
    fn to() -> usize;
}

trait Graph {
    type NodeType: Node;
    type EdgeType: Edge;

    fn nodes(&mut self) -> &mut [Self::NodeType];
    fn edges(&mut self) -> &mut [Self::EdgeType];

    fn add_node(&mut self, node: Self::NodeType);
    fn add_edge(&mut self, edge: Self::EdgeType);

    fn filter_modify_nodes<P: Fn(&Self::NodeType) -> bool, F: Fn(&mut Self::NodeType)>(
        &mut self,
        p: P,
        f: F,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct ExampleNode(usize);

    impl Node for ExampleNode {}

    #[derive(Debug)]
    struct ExampleEdge;

    impl Edge for ExampleEdge {
        fn from() -> usize {
            0
        }

        fn to() -> usize {
            0
        }
    }

    #[derive(Debug)]
    struct ExampleGraph {
        nodes: Vec<ExampleNode>,
        edges: Vec<ExampleEdge>,
    }

    impl ExampleGraph {
        pub fn new() -> Self {
            ExampleGraph {
                nodes: vec![],
                edges: vec![],
            }
        }
    }

    impl Graph for ExampleGraph {
        type NodeType = ExampleNode;
        type EdgeType = ExampleEdge;

        fn nodes(&mut self) -> &mut [ExampleNode] {
            &mut self.nodes[..]
        }

        fn edges(&mut self) -> &mut [ExampleEdge] {
            &mut self.edges[..]
        }

        fn add_node(&mut self, node: ExampleNode) {
            self.nodes.push(node);
        }

        fn add_edge(&mut self, edge: ExampleEdge) {
            self.edges.push(edge);
        }

        fn filter_modify_nodes<P: Fn(&Self::NodeType) -> bool, F: Fn(&mut Self::NodeType)>(
            &mut self,
            p: P,
            f: F,
        ) {
            self.nodes().iter_mut().filter(|n| p(n)).for_each(|n| f(n));
        }
    }

    #[test]
    fn init_graph() {
        ExampleGraph::new();
    }

    #[test]
    fn add_node() {
        let mut graph = ExampleGraph::new();

        graph.add_node(ExampleNode(1));
    }

    #[test]
    fn add_edge() {
        let mut graph = ExampleGraph::new();

        graph.add_edge(ExampleEdge);
    }

    #[test]
    fn filter_and_modify_nodes() {
        let mut graph = ExampleGraph::new();

        graph.add_node(ExampleNode(1));
        graph.add_node(ExampleNode(2));
        graph.add_node(ExampleNode(3));

        graph.filter_modify_nodes(|n| n.0 > 1, |n| n.0 = 10);

        let nodes = graph.nodes();

        assert_eq!(nodes[0].0, 1);
        assert_eq!(nodes[1].0, 10);
        assert_eq!(nodes[2].0, 10);
    }
}
