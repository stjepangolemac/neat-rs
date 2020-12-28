use neat_core::Network;
use std::fs::{read, write};
use std::path::Path;

pub fn to_bytes(network: &Network) -> Vec<u8> {
    bincode::serialize(network).unwrap()
}

pub fn from_bytes(bytes: &[u8]) -> Network {
    bincode::deserialize(bytes).unwrap()
}

pub fn to_file<S: AsRef<Path>>(path: S, network: &Network) {
    write(path, to_bytes(&network)).unwrap();
}

pub fn from_file<S: AsRef<Path>>(path: S) -> Network {
    from_bytes(&read(path).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::Genome;

    #[test]
    fn to_bytes_works() {
        let network: Network = (&Genome::new(3, 1)).into();

        to_bytes(&network);
    }

    #[test]
    fn from_bytes_works() {
        let mut network: Network = (&Genome::new(3, 1)).into();
        let output_before = network.forward_pass(vec![1., 2., 3.]);

        let bytes = to_bytes(&network);
        let mut imported_network = from_bytes(&bytes);

        let output_after = imported_network.forward_pass(vec![1., 2., 3.]);

        assert_eq!(output_before, output_after);
    }

    #[test]
    fn file_import_export_works() {
        let filename = "network.bin";

        let mut network: Network = (&Genome::new(3, 1)).into();
        let output_before = network.forward_pass(vec![1., 2., 3.]);

        to_file(filename, &network);
        let mut imported_network = from_file(filename);

        let output_after = imported_network.forward_pass(vec![1., 2., 3.]);

        assert_eq!(output_before, output_after);

        std::fs::remove_file(filename).unwrap();
    }
}
