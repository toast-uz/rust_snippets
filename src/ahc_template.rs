/*

AHC共通テンプレート

# 目的

- AtCoder Heuristic Contest (AHC) 系の問題において、
  自明解 → 貪欲 → 焼きなまし/ビームサーチ という発展的なアルゴリズム開発を、
  型安全かつ再利用可能な形で支援することを目的とする

# 特徴

- 問題依存処理と共通処理を分離することで再利用性を高める
  - 問題依存処理は前半に、共通処理は後半にまとめておく
- AHCの実装スタイルを整理し、段階的な改良を促す
　- 問題依存処理は、上から順に実装していく

*/

use std::time::Instant;
use itertools::Itertools;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rustc_hash::FxHashSet as HashSet;
use proconio::input;

// ----- 問題固有の実装 -----

const LIMIT: f64 = 0.0; // 制限時間（秒）
#[allow(dead_code)]
const DEBUG: bool = true;

// ソルバーの切り替え（実装したら変更する）
const SOLVER_INIT: SolverState = SolverState::None;  // 初期ソルバー
const SOLVER_MAIN: SolverState = SolverState::None;  // メインソルバー


#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }

// --------------------------------------------------------------------
// [1] 環境を定義して入力を記載する
// --------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct Env {
    // 貪欲法用の定義
    max_depth: usize,
    // ビームサーチ用の定義
    beam_width_init: usize,
    beam_width_max: usize,
    beam_width_min: usize,
    beam_width_min_rate: f64,
    beam_width_max_rate: f64,
    // 焼きなまし法用の定義
    sa_start_temp: f64,
    sa_end_temp: f64,
    sa_duration_temp: f64, // 温度の変化量
    sa_timer_resolution: usize,  // 焼きなまし法の解像度（温度を変化させるターン数）
    sa_patience: usize,   // 最適解を更新するまでの許容ターン数
}

impl Env {
    fn new() -> Self {
        let mut e = Self::default();
        // 入力を読み込む
        input! {
            n: usize,
        }
        e.init(n); e
    }
    // 入力をもとに初期化する（標準入力をともなわないテスト用に、関数をわけておく）
    fn init(&mut self, _n: usize) {
        self.init_greedy();     // 貪欲法の初期化
        self.init_beam();       // ビームサーチの初期化
        self.init_annealing();  // 焼きなまし法の初期化
    }
}

// --------------------------------------------------------------------
// [2] 自明解を作成する
// --------------------------------------------------------------------

// [2-1] 状態を定義する（オプショナル）
#[derive(Debug, Clone, Default)]
struct State {
    // 状態を記述する
    // 例: a: usize, b: usize
    score: isize,
}

impl State {
    fn new(_e: &Env) -> Self {
        Self::default()
    }
}

// [2-2] 操作を定義する（オプショナル）
#[derive(Debug, Clone, Default)]
struct Op {
    // 操作の内容を記述する
    // 例: a: usize, b: usize
    pre_id: usize, // 前の状態のID
    hash: u64, // ハッシュ値
    score: isize,
}

impl Op {
    fn new() -> Self {
        Self::default()
    }
}

// [2-3] ソルバーを定義する
#[derive(Debug, Clone, Default)]
struct Solver {
    // ソルバーの定義を記述する
    //ops: Vec<Op>, // 操作のリスト
    //state: State, // 現在の状態
    counter: usize,
    score: isize,
}

impl Solver {
    fn solve_trivial(&mut self, _e: &Env) {}                    // 自明解のアルゴリズムを記述する
    fn compute_score(&self, _e: &Env) -> isize { 0 }            // スコアを計算する
    fn make_output(&self, _e: &Env) -> String { String::new() } // 出力を作成する
}

// 以上を実装したら、SOLVER_INITをTrivialに変更する

// --------------------------------------------------------------------
// [3] 貪欲法を実装する
// --------------------------------------------------------------------

// [3-1] 貪欲法固有の環境定義
impl Env {
    fn init_greedy(&mut self) {
        self.max_depth = 100; // 深さの上限
    }
}

impl State {
    fn get_next_greedy_op(&self, _e: &Env) -> Op {
        // 貪欲法の次の操作を決定する
        Op::new()
    }
    fn apply_op(&mut self, _e: &Env, _op: &Op) {
        // 操作を適用する
    }
}

impl Solver {
    fn save_ops(&mut self, _e: &Env, _ops: Vec<Op>) {
        // 操作を登録する
    }
    fn save_state(&mut self, _e: &Env, _state: State) {
        // 状態を登録する
    }
}

// 以上を実装したら、SOLVER_INITをGreedyに変更する

// --------------------------------------------------------------------
// [4] 高度なアルゴリズムを実装する
// 文脈が強ければビームサーチを、文脈が弱ければ焼きなまし法を選択する
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// [4-A] ビームサーチを実装する
// --------------------------------------------------------------------

// [4-A-1] ビームサーチ固有の環境定義
impl Env {
    fn init_beam(&mut self) {
        self.beam_width_init = 1; // ビームの幅（初期値）
        self.beam_width_max = 1; // ビームの幅（最小値）
        self.beam_width_min = 1; // ビームの幅（最大値）
        self.beam_width_min_rate = 0.5; // ビーム幅の変化率（最小値）
        self.beam_width_max_rate = 2.0; // ビーム幅の変化率（最大値）
    }
}

// [4-A-2] Stateへのメソッドの追加
impl State {
    #[inline]
    // 前の状態のIDを取得する
    fn get_op(&self) -> Option<&Op> {
        None
    }
    //次の操作を列挙する
    fn enum_next_ops(&self, _e: &Env, _id: usize) -> Vec<Op> {
        vec![Op::new()]
    }
}

// 以上を実装したら、SOLVER_MAINをBeamに変更する
// [4-A-3] ビーム幅を1固定にして、ビームサーチを実行して結果を確認する
// [4-A-4] ビーム幅を調整する

// --------------------------------------------------------------------
// [4-B] 焼きなまし法を実装する
// --------------------------------------------------------------------

// [4-B-1] 焼きなまし固有の環境定義
impl Env {
    fn init_annealing(&mut self) {
        self.sa_start_temp = 1000.0; // 初期温度
        self.sa_end_temp = 1.0;   // 終了温度
        self.sa_timer_resolution = 100; // 焼きなまし法の解像度（温度を変化させるターン数）
        self.sa_patience = 1000; // 最適解を更新するまでの許容ターン数
        // 以下は変更しない
        self.sa_duration_temp = self.sa_end_temp - self.sa_start_temp; // 温度の変化量
    }
}

// [4-B-2] 遷移候補を定義
#[derive(Debug)]
#[allow(dead_code)]
enum Transition {
    // 遷移候補を定義する
    // 例: Swap(usize, usize),
    // 例: Insert(usize, usize),
    Swap(usize, usize),
}

// [4-B-3] StateにTransition関連のメソッドを追加
impl State {
    fn choose_transition(&self, _e: &Env, _rng: &mut impl rand::Rng) -> Transition {
        // 遷移候補を決定する
        Transition::Swap(0, 0)
    }
    fn compute_score(&self, _e: &Env) -> isize {
        // スコアを計算する
        0
    }
    fn compute_score_diff(&self, e: &Env, trans: &Transition) -> isize {
        // 遷移候補のスコア差分を計算する
        // 初期実装では、スコアフル計算をもとに差分を計算する
        let ord_score = self.score;
        let mut new_state = self.clone();
        new_state.apply_transition(e, trans);
        let new_score = new_state.compute_score(e);
        new_score - ord_score
    }
    fn apply_transition(&mut self, _e: &Env, _trans: &Transition) {
        // 遷移候補を適用する
    }
}

impl Solver {
    fn load_state(&mut self, _e: &Env) -> State {
        // 状態を読み込む
        State::new(_e)
    }
}

// 以上を実装したら、SOLVER_MAINをAnnealingに変更する
// [4-B-4] 焼きなまし法を実行して結果を確認する
// [4-B-5] 遷移候補を増やしてみる
// [4-B-6] 焼きなまし法の差分計算を実装する

// ------------------------------------------------------
// サポートライブラリ
// （ここから下を変更することもあるが、基本的には変更しない）
// ------------------------------------------------------

fn main() {
    let timer = Instant::now();
    let e = Env::new();
    let mut solver = Solver::new(&e);
    solver.solve(&e, &timer, LIMIT);
    println!("{}", solver.make_output(&e));
}

impl Env {
    // 経過時間とターン消化率を考慮してビーム幅を調整する
    fn get_next_beam_width(&self, beam_width: usize, depth: usize, timer: &Instant, start_time: f64, limit: f64) -> usize {
        let elapsed = timer.elapsed().as_secs_f64();
        let turn_rate = (depth as f64) / (self.max_depth as f64);
        let timer_rate = (elapsed - start_time) / (limit - start_time);
        let ratio = (turn_rate / timer_rate).clamp(self.beam_width_min_rate, self.beam_width_max_rate);
        ((beam_width as f64 * ratio) as usize).clamp(self.beam_width_max, self.beam_width_min)
    }
    // 焼きなまし法の温度を計算する、時間オーバーならErrを返す
    #[inline]
    fn get_temp(&self, timer: &Instant, start_time: f64, limit: f64) -> Result<f64, ()> {
        let elapsed = timer.elapsed().as_secs_f64();
        if elapsed > limit { return Err(()); }
        let temp = self.sa_start_temp + self.sa_duration_temp * (elapsed - start_time) / (limit - start_time);
        Ok(temp)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum SolverState { Trivial, Greedy, Beam, Annealing, None }

// ソルバーの基本実装
impl Solver {
    fn new(e: &Env) -> Self {
        let mut solver = Self::default();
        match SOLVER_INIT {
            SolverState::Trivial => solver.solve_trivial(e), // 自明解
            SolverState::Greedy  => solver.solve_greedy(e),  // 貪欲法
            SolverState::None => {}
            _ => panic!("Invalid SOLVER_INIT: {:?}", SOLVER_INIT),
        }
        solver.score = solver.compute_score(e);
        solver
    }
    fn solve(&mut self, e: &Env, timer: &Instant, limit: f64) {
        // メインソルバーの実行
        match SOLVER_MAIN {
            SolverState::Beam => self.solve_beam(e, timer, limit), // ビームサーチ
            SolverState::Annealing  => self.solve_anealing(e, timer, limit),  // 焼きなまし法
            SolverState::None => {}
            _ => panic!("Invalid SOLVER_MAIN: {:?}", SOLVER_MAIN),
        }
    }
}

// 貪欲法の実装
impl Solver {
    fn solve_greedy(&mut self, e: &Env) {
        let mut state = State::new(e);
        let mut ops = Vec::new();
        for _ in 0..e.max_depth {
            let op = state.get_next_greedy_op(e);
            state.apply_op(e, &op);
            ops.push(op);
        }
        self.save_ops(e, ops);
        self.save_state(e, state);
    }
}

// ビームサーチの実装
impl Solver {
    fn solve_beam(&mut self, e: &Env, timer: &Instant, limit: f64) {
        let mut beam_width = e.beam_width_init;
        let mut beam = vec![vec![State::new(e)]]; // 初期状態をビームに追加する
        let start_time = timer.elapsed().as_secs_f64();
        for depth in 0..e.max_depth {
            // ビームの状態列をもとに、次の操作を列挙する
            let mut cands = Vec::new();
            let mut cands_hash = HashSet::default();
            for (id, state) in beam[depth].iter().enumerate() {
                let ops = state.enum_next_ops(e, id);
                self.counter += ops.len();
                for op in ops {
                    if cands_hash.insert(op.hash) { cands.push(op); } // 多様性を高めるためhash値の重複を避ける
                }
            }
            // ビーム幅以内の候補を状態に適用して、次の状態列とする
            // select_nth_unstable_by_key は、O(n)であるため、高速である
            cands.select_nth_unstable_by_key(beam_width, |op| op.score);
            cands.truncate(beam_width);
            let new_beam = cands.iter().map(|op| {
                let mut state = beam[depth][op.pre_id].clone();
                state.apply_op(e, op);
                state
            }).collect_vec();
            beam.push(new_beam);
            // ビーム幅を制限時間とターン消化率を考慮して調整する
            beam_width = e.get_next_beam_width(beam_width, depth, timer, start_time, limit);
            dbg!("#{} counter={}, width={}", depth, self.counter, beam_width);
        }
        // ビームを復元する
        let mut depth = e.max_depth;
        let mut best_id = beam[depth].iter().enumerate()
            .max_by_key(|&(_, state)| state.get_op().unwrap().score).unwrap().0;
        let mut ops = Vec::new();
        while let Some(op) = beam[depth][best_id].get_op() {
            ops.push(op.clone());
            best_id = op.pre_id;
            depth -= 1;
        }
        self.save_ops(e, ops);
    }
}

// 焼きなまし法の実装
impl Solver {
    fn solve_anealing(&mut self, e: &Env, timer: &Instant, limit: f64) {
        let start_time = timer.elapsed().as_secs_f64();
        let mut temp = e.sa_start_temp; // 初期温度
        let mut state = self.load_state(e); // 初期状態を読み込む
        let mut best_state = state.clone();
        let mut best_counter = self.counter;
        let mut rng = SmallRng::seed_from_u64(1);
        loop {
            self.counter += 1;
            // 現在の温度を計算（少し重いので計算は sa_timer_resolution 間隔で行う）
            if self.counter % e.sa_timer_resolution == 0 {
                let Ok(new_temp) = e.get_temp(timer, start_time, limit) else { break; };
                temp = new_temp;
            }
            // PATIENCE回、ベスト更新されなかったら，現在のカウンターをベストにコピーして、ベストから再開する
            if self.counter > best_counter + e.sa_patience {
                best_counter = self.counter;
                state = best_state.clone();
                dbg!("counter:{} score:{} restart from the best", self.counter, self.score);
            }
            // 遷移候補を決めて、遷移した場合のコスト差分を計算する
            let trans = state.choose_transition(e, &mut rng);
            let score_diff = state.compute_score_diff(e, &trans);
            // スコアが高いほど良い場合
            // スコアが低いほど良い場合は-score_diff、score_diff <= 0 とする
            let prob = (score_diff as f64 / temp).exp();
            if score_diff >= 0 || prob > rng.gen() { // 確率probで遷移する
                state.apply_transition(e, &trans);
                state.score += score_diff;
                // スコアが高いほど良い場合
                // スコアが低いほど良い場合は self.score < best.score とする
                if best_state.score < state.score {
                    best_state = state.clone();
                    dbg!("counter:{} score:{} new best", self.counter, best_state.score);
                }
            }
        }
        // 現在のベストを最終結果として採用する
        self.save_state(e, best_state);
    }
}
