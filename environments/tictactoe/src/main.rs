use rand::random;

trait Environment {
    type State;
    type Input;

    fn state(&self) -> &Self::State;
    fn step(&mut self, input: Self::Input) -> Result<(), ()>;

    fn done(&self) -> bool;
    fn reset(&mut self);

    fn render(&self);

    fn fitness(&self) -> f64;
}

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

    fn game_over(&self) -> bool {
        let fields_full = self.field.iter().all(|mark| !matches!(mark, Mark::Empty));

        fields_full || self.did_external_win() || self.did_internal_win()
    }

    fn did_external_win(&self) -> bool {
        let external_mark = {
            match self.first_player {
                Player::External => Mark::X,
                Player::Internal => Mark::O,
            }
        };

        self.did_mark_win(external_mark)
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

fn main() {
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
