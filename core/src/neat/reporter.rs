use super::NEAT;

pub type Hook = fn(i: usize, &NEAT) -> ();

pub struct Reporter {
    hooks: Vec<(usize, Hook)>,
}

impl Reporter {
    pub fn new() -> Self {
        Reporter { hooks: vec![] }
    }

    pub fn register(&mut self, every: usize, hook: Hook) {
        self.hooks.push((every, hook));
    }

    pub fn report(&self, i: usize, system: &NEAT) {
        self.hooks
            .iter()
            .filter(|(every, _)| i % every == 0)
            .for_each(|(_, hook)| hook(i, system));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_every() {
        use crate::neat::NEAT;

        let mut reporter = Reporter::new();

        reporter.register(1, |i, _| {
            dbg!(i);
        });

        let system = NEAT::new(1, 1, |_| 0.);

        for i in 1..=10 {
            reporter.report(i, &system);
        }
    }

    #[test]
    fn print_every_3() {
        use crate::neat::NEAT;

        let mut reporter = Reporter::new();

        reporter.register(3, |i, _| {
            dbg!(i);
        });

        let system = NEAT::new(1, 1, |_| 0.);

        for i in 1..=10 {
            reporter.report(i, &system);
        }
    }

    #[test]
    fn access_system() {
        use crate::neat::NEAT;

        let mut reporter = Reporter::new();

        reporter.register(4, |i, system| {
            println!("At generation {} input count is {}", i, system.inputs);
        });

        let system = NEAT::new(1, 1, |_| 0.);

        for i in 1..=10 {
            reporter.report(i, &system);
        }
    }
}
