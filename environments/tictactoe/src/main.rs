use rand::random;

use neat_core::{Configuration, Network, NEAT};
use neat_environment::Environment;

#[derive(Clone, Copy, Debug)]
enum Mark {
    X,
    O,
    Empty,
}

#[derive(Clone, Debug)]
enum Player {
    External,
    Internal,
}

type Field = [Mark; 9];

#[derive(Debug)]
struct TicTacToe {
    field: Field,
    first_player: Player,
    turn: Player,
}

impl TicTacToe {
    pub fn new() -> Self {
        let first_player: Player = if random::<f64>() < 0.5 {
            Player::External
        } else {
            Player::Internal
        };

        let mut ttt = TicTacToe {
            field: [Mark::Empty; 9],
            first_player: first_player.clone(),
            turn: first_player.clone(),
        };

        if let Player::Internal = first_player {
            ttt.step_internal();
        }

        ttt
    }

    fn step_internal(&mut self) {
        if self.game_over() || self.is_external_turn() {
            return;
        }

        let empty_indexes: Vec<usize> = self
            .field
            .iter()
            .enumerate()
            .filter(|(_, mark)| matches!(mark, Mark::Empty))
            .map(|(index, _)| index)
            .collect();

        let random_index = empty_indexes
            .get(random::<usize>() % empty_indexes.len())
            .unwrap();

        let mark_to_place = if matches!(self.first_player, Player::Internal) {
            Mark::X
        } else {
            Mark::O
        };

        *self.field.get_mut(*random_index).unwrap() = mark_to_place;
        self.turn = Player::External;
    }

    pub fn is_external_first(&self) -> bool {
        matches!(self.first_player, Player::External)
    }

    pub fn is_external_turn(&self) -> bool {
        matches!(self.turn, Player::External)
    }

    pub fn external_mark(&self) -> Mark {
        match self.first_player {
            Player::External => Mark::X,
            Player::Internal => Mark::O,
        }
    }

    fn game_over(&self) -> bool {
        let fields_full = self.field.iter().all(|mark| !matches!(mark, Mark::Empty));

        fields_full || self.did_external_win() || self.did_internal_win()
    }

    fn did_external_win(&self) -> bool {
        self.did_mark_win(self.external_mark())
    }

    fn did_internal_win(&self) -> bool {
        let internal_mark = {
            match self.first_player {
                Player::Internal => Mark::X,
                Player::External => Mark::O,
            }
        };

        self.did_mark_win(internal_mark)
    }

    fn is_draw(&self) -> bool {
        self.game_over() && !self.did_external_win() && !self.did_internal_win()
    }

    fn did_mark_win(&self, check_mark: Mark) -> bool {
        let winning_lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        let external_won = winning_lines.iter().any(|line| {
            line.iter()
                .map(|mark_index| self.field.get(*mark_index).unwrap())
                .all(|mark| match (mark, check_mark) {
                    (Mark::X, Mark::X) => true,
                    (Mark::O, Mark::O) => true,
                    _ => false,
                })
        });

        external_won
    }
}

impl Environment for TicTacToe {
    type State = Field;
    type Input = usize;

    fn state(&self) -> &Self::State {
        &self.field
    }

    fn step(&mut self, input: Self::Input) -> Result<(), ()> {
        if input >= 9 {
            panic!("Field index out of bounds");
        }

        if self.game_over() || !self.is_external_turn() {
            return Err(());
        }

        let mark_to_place = if matches!(self.first_player, Player::External) {
            Mark::X
        } else {
            Mark::O
        };
        let mark = self.field.get_mut(input).unwrap();

        if matches!(mark, Mark::Empty) {
            *mark = mark_to_place;
        } else {
            return Err(());
        }

        self.turn = Player::Internal;
        self.step_internal();

        Ok(())
    }

    fn done(&self) -> bool {
        self.game_over()
    }

    fn reset(&mut self) {
        *self = TicTacToe::new();
    }

    fn render(&self) {
        self.field.iter().enumerate().for_each(|(index, mark)| {
            let character: String = match mark {
                Mark::X => "X".to_owned(),
                Mark::O => "O".to_owned(),
                Mark::Empty => "_".to_owned(),
            };

            if index % 3 == 0 {
                print!("\n");
            }
            print!("{} ", character);
        });

        print!("\n\n");
    }

    fn fitness(&self) -> f64 {
        if self.did_external_win() {
            1.
        } else {
            0.
        }
    }
}

fn state_to_inputs(env: &TicTacToe) -> Vec<f64> {
    let player_mark = env.external_mark();

    env.state()
        .iter()
        .map(|mark| match (player_mark, *mark) {
            (Mark::X, Mark::X) => 1.,
            (Mark::O, Mark::O) => 1.,
            (Mark::X, Mark::O) => -1.,
            (Mark::O, Mark::X) => -1.,
            _ => 0.,
        })
        .collect()
}

fn move_from_outputs(outputs: &[f64]) -> usize {
    outputs
        .iter()
        .enumerate()
        .fold((0, -999.), |(max_index, max_output), (index, output)| {
            if output > &max_output {
                (index, *output)
            } else {
                (max_index, max_output)
            }
        })
        .0
}

fn play_network(network: &mut Network) {
    println!("Playing...");

    let mut env = TicTacToe::new();

    let player_mark = env.external_mark();
    println!("Player mark is {:?}", player_mark);

    loop {
        env.render();

        if env.game_over() {
            break;
        }

        let inputs = state_to_inputs(&env);
        let outputs: Vec<f64> = network.forward_pass(inputs.clone());
        let max_output_index: usize = move_from_outputs(&outputs);

        if env.step(max_output_index).is_err() {
            break;
        }
    }

    println!("Game over, last state");
    env.render();
}

fn main() {
    let mut system = NEAT::new(9, 9, |network| {
        let games = 100;
        let mut turns = 0;
        let mut games_won = 0;
        let mut games_draw = 0;

        let mut env = TicTacToe::new();

        for _ in 0..games {
            env.reset();

            loop {
                if env.game_over() {
                    break;
                }

                let inputs = state_to_inputs(&env);
                let outputs: Vec<f64> = network.forward_pass(inputs.clone());
                let max_output_index: usize = move_from_outputs(&outputs);

                if env.step(max_output_index).is_ok() {
                    turns += 1;
                } else {
                    break;
                }
            }

            games_won += if env.did_external_win() { 1 } else { 0 };
            games_draw += if env.is_draw() { 1 } else { 0 };
        }

        // games as f64 / (games_won as f64 + games_draw as f64) //+ turns as f64 * 0.01
        // turns as f64 / games as f64
        (games_won as f64 + games_draw as f64) / games as f64
    });

    system.set_configuration(Configuration {
        population_size: 50,
        max_generations: 500,
        node_cost: 0.001,
        connection_cost: 0.0005,
        compatibility_threshold: 3.,
        ..Default::default()
    });
    system.add_hook(1, |i, system| {
        let (_, _, fitness) = system.get_best();

        println!("Generation {}, best fitness is {}", i, fitness);
    });

    let (mut network, fitness) = system.start();

    println!(
        "Found network with {} nodes and {} connections, of fitness {}",
        network.nodes.len(),
        network.connections.len(),
        fitness
    );

    for _ in 0..5 {
        play_network(&mut network);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_run() {
        let mut env = TicTacToe::new();

        if env.is_external_first() {
            println!("I am X");
        } else {
            println!("I am O");
        }

        loop {
            if env.game_over() {
                break;
            }

            while env.step(random::<usize>() % 9).is_err() {}
        }

        println!("I WON: {}", env.did_external_win());
        env.render();
        env.reset();
    }
}
