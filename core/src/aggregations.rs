use rand::distributions::{Distribution, Standard};
use rand::Rng;

pub fn aggregate(kind: &Aggregation, components: &[f64]) -> f64 {
    use Aggregation::*;

    let func: fn(components: &[f64]) -> f64 = match kind {
        Product => product,
        Sum => sum,
        Max => max,
        Min => min,
        MaxAbs => maxabs,
        Median => median,
        Mean => mean,
    };

    func(components)
}

#[derive(Debug, Clone, PartialEq, Hash)]
#[cfg_attr(
    feature = "network-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Aggregation {
    Product,
    Sum,
    Max,
    Min,
    MaxAbs,
    Median,
    Mean,
}

impl Distribution<Aggregation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Aggregation {
        use Aggregation::*;

        match rng.gen_range(0, 7) {
            0 => Product,
            1 => Sum,
            2 => Max,
            3 => Min,
            4 => MaxAbs,
            5 => Median,
            _ => Mean,
        }
    }
}

fn product(components: &[f64]) -> f64 {
    components
        .iter()
        .fold(1., |result, current| result * current)
}

fn sum(components: &[f64]) -> f64 {
    components.iter().sum()
}

fn max(components: &[f64]) -> f64 {
    components.iter().fold(
        f64::MIN,
        |max, current| if *current > max { *current } else { max },
    )
}

fn min(components: &[f64]) -> f64 {
    components.iter().fold(
        f64::MAX,
        |min, current| if *current < min { *current } else { min },
    )
}

fn maxabs(components: &[f64]) -> f64 {
    let abs_components: Vec<f64> = components.iter().map(|component| component.abs()).collect();
    max(&abs_components)
}

fn median(components: &[f64]) -> f64 {
    use std::cmp::Ordering;

    if components.is_empty() {
        return 0.;
    }

    let mut sorted = components.to_vec();
    sorted.sort_by(|a, b| {
        if a < b {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });

    let length = sorted.len();
    let is_length_even = length % 2 == 0;
    let median_index = if is_length_even {
        length / 2 - 1
    } else {
        length / 2
    };

    *sorted.get(median_index).unwrap()
}

fn mean(components: &[f64]) -> f64 {
    let sum: f64 = components.iter().sum();
    sum / components.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_works() {
        let components = vec![1., 2., 3., 4.];

        assert!((product(&components) - 24.).abs() < f64::EPSILON);
    }

    #[test]
    fn sum_works() {
        let components = vec![1., 2., 3., 4.];

        assert!((sum(&components) - 10.).abs() < f64::EPSILON);
    }

    #[test]
    fn max_works() {
        let components = vec![1., 2., 3., 4.];

        assert!((max(&components) - 4.).abs() < f64::EPSILON);
    }

    #[test]
    fn min_works() {
        let components = vec![1., 2., 3., 4.];

        assert!((min(&components) - 1.).abs() < f64::EPSILON);
    }

    #[test]
    fn maxabs_works() {
        let components = vec![-5., -3., 1., 2., 3., 4.];

        assert!((maxabs(&components) - 5.).abs() < f64::EPSILON);
    }

    #[test]
    fn median_works() {
        let components = vec![3., -3., 4., -5., 1., 2.];

        assert!((median(&components) - 1.).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_works() {
        let components = vec![1., 2., 3., 4.];

        assert!((mean(&components) - 2.5).abs() < f64::EPSILON);
    }
}
