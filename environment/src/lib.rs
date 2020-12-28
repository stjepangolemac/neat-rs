pub trait Environment {
    type State;
    type Input;

    fn state(&self) -> Self::State;
    fn step(&mut self, input: Self::Input) -> Result<(), ()>;

    fn done(&self) -> bool;
    fn reset(&mut self);

    fn render(&self);

    fn fitness(&self) -> f64;
}
