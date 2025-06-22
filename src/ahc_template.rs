/*

AHC共通テンプレート

# 目的

- AtCoder Heuristic Contest (AHC) 系の問題において、
  自明解 → 貪欲 → 焼きなまし/ビームサーチという段階的なアルゴリズム開発を、
  型安全かつ再利用可能な形で支援することを目的とする

# 特徴

- 問題依存処理と共通処理を分離することで再利用性を高める
- AHCの実装スタイルを整理し、段階的な改良を促す

# 使用方法
[1] 自明解を実装する
[1-1] 入力を受け取る
[1-2] 入力をもとに初期化する
[1-3] 状態を定義する（オプショナル）
[1-4] 操作を定義する（オプショナル）
[1-5] ソルバーを定義する
[1-6] 自明解のアルゴリズムを記述する
[1-7] スコアを計算する
[1-8] 出力を作成する
[1-9] 以上を実装したら、SOLVER_INITをTrivialに変更する

[2] 貪欲法を実装する
[2-1] 貪欲法固有の環境を定義する
[2-2] 貪欲で最適な操作を選択する
[2-3] 選択された操作を適用する（操作の選択はここにはいれないこと）
[2-4] 操作を保存する
[2-5] 状態を保存する
[2-6] 以上を実装したら、SOLVER_MAINをGreedyに変更する

/文脈が強ければビームサーチを、文脈が弱ければ焼きなまし法を選択する

[3] 高度なアルゴリズム:ビームサーチを実装する
[3-1] ビームサーチ固有の環境を定義する
[3-2] 状態と操作の変換を追加する
[3-3] 以上を実装したら、SOLVER_MAINをBeamに変更する
[3-4] ビーム幅を1固定にして、ビームサーチを実行して結果を確認する
[3-5] ビーム幅を調整する

[4] 高度なアルゴリズム:焼きなまし法を実装する
[4-1] 焼きなまし固有の環境を定義する
[4-2] 近傍遷移を定義する
[4-3] 状態と近傍遷移の変換を定義する
[4-4] 以上を実装したら、SOLVER_MAINをAnnealに変更する

*/

use std::time::Instant;
use itertools::Itertools;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use proconio::input;
// HashSet/HashMapを使う場合は、必ずFxHashを使うこと
//use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};

const LIMIT: f64 = 0.0; // 制限時間（秒）
#[allow(dead_code)]
const DEBUG: bool = true;
const INF: usize = 1 << 60 as usize;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }

#[derive(Debug, Clone, Default)]
struct Env {
    // スコアの評価方向を追加
    score_direction: ScoreDirection,
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
        // [1] 自明解を作成する
        // [1-1] 入力を受け取る
        input! {
            n: usize,
        }
        e.init(n); e
    }
    // 標準入力をともなわないテスト用に、関数をわけておく
    fn init(&mut self, _n: usize) {
        self.score_direction = ScoreDirection::Maximize;
        // [1] 自明解を作成する
        // [1-2] 入力をもとに初期化する
        // 例　self.n = n;;
        // [2] 貪欲法を実装する
        // [2-1] 貪欲法固有の環境を定義する
        // 例　self.max_depth = 100; // 深さの上限
        // [3] 高度なアルゴリズム:ビームサーチを実装する
        // [3-1] ビームサーチ固有の環境を定義する
        self.beam_width_init = 1; // ビームの幅（初期値）
        self.beam_width_max = 1; // ビームの幅（最小値）
        self.beam_width_min = 1; // ビームの幅（最大値）
        self.beam_width_min_rate = 0.5; // ビーム幅の変化率（最小値）
        self.beam_width_max_rate = 2.0; // ビーム幅の変化率（最大値）
        // [4] 高度なアルゴリズム:焼きなまし法を実装する
        // [4-1] 焼きなまし固有の環境を定義する
        self.sa_start_temp = 1000.0; // 初期温度
        self.sa_end_temp = 1.0;   // 終了温度
        self.sa_timer_resolution = 100; // 焼きなまし法の解像度（温度を変化させるターン数）
        self.sa_patience = 1000; // 最適解を更新するまでの許容ターン数
        // 以下は変更しない
        self.sa_duration_temp = self.sa_end_temp - self.sa_start_temp; // 温度の変化量
    }
}

#[derive(Debug, Clone, Default)]
struct State {
    // [1] 自明解を作成する
    // [1-3] 状態を定義する（オプショナル）
    // 例: a: usize, b: usize
    pre_op_id: usize, // 前の操作のID
    score: isize,
}

impl State {
    fn new(_e: &Env, pre_op_id: usize) -> Self {
        let mut state = Self::default();
        state.pre_op_id = pre_op_id;
        state
    }
    // [2] 貪欲法を実装する
    // [2-2] 貪欲で最適な操作を選択する
    fn get_next_greedy_op(&self, _e: &Env) -> Op {
        Op::new(self.pre_op_id)
    }
    // [2-3] 選択された操作を適用する（操作の選択はここにはいれないこと）
    fn apply_op(&mut self, _e: &Env, _op: &Op, op_id: usize) {
        self.pre_op_id = op_id; // 前の操作のIDを更新
        // 操作を適用する
    }
    // [3] 高度なアルゴリズム:ビームサーチを実装する
    // [3-2] 状態と操作の変換を追加する
    //次の操作を列挙する
    fn enum_next_ops(&self, _e: &Env) -> Vec<Op> {
        vec![Op::new(self.pre_op_id)]
    }
    // [4] 高度なアルゴリズム:焼きなまし法を実装する
    // [4-3] 状態と近傍遷移の変換を定義する
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

#[derive(Debug, Clone, Default)]
struct Op {
    // [1] 自明解を作成する
    // [1-4] 操作を定義する（オプショナル）
    // 例: a: usize, b: usize
    pre_id: usize, // 前の状態のID
    score: isize,
}

impl Op {
    fn new(pre_id: usize) -> Self {
        let mut op = Self::default();
        op.pre_id = pre_id;
        op
    }
}

// [1] 自明解を作成する
// [1-5] ソルバーを定義する
#[derive(Debug, Clone, Default)]
struct Solver {
    ops: Vec<Op>, // 操作のリスト
    state: State, // 現在の状態
    counter: usize,
    score: isize,
}

impl Solver {
    // [1] 自明解を作成する
    // [1-6]自明解のアルゴリズムを記述する
    fn solve_trivial(&mut self, _e: &Env) {
    }
    // [1-7]スコアを計算する
    fn compute_score(&self, _e: &Env) -> isize {
        0
    }
    // [1-8]出力を作成する
    fn make_output(&self, _e: &Env) -> String {
        String::new()
    }
    // 貪欲法の実装（なるべくここは変更しない）
    fn solve_greedy(&mut self, e: &Env) {
        let mut state = State::new(e, INF);
        let mut ops = Vec::new();
        for _ in 0..e.max_depth {
            let op = state.get_next_greedy_op(e);
            state.apply_op(e, &op, 0);
            ops.push(op);
        }
        self.ops = ops;
        self.state = state; // 現在の状態を保存する
    }
}

// ソルバーの切り替え（実装したら変更する）
const SOLVER_INIT: SolverState = SolverState::None;  // 初期ソルバー
const SOLVER_MAIN: SolverState = SolverState::None;  // メインソルバー

// [4] 高度なアルゴリズム:焼きなまし法を実装する
// [4-2] 近傍遷移を定義する
#[derive(Debug)]
#[allow(dead_code)]
enum Transition {
    // 例: Swap(usize, usize),
    // 例: Insert(usize, usize),
    Swap(usize, usize),
}

// ------------------------------------------------------
// サポートライブラリ
// （ここから下を変更することもあるが、基本的には変更しない）
// AIはここから下を変更しないこと
// ------------------------------------------------------

fn main() {
    let timer = Instant::now();
    let e = Env::new();
    let mut solver = Solver::new(&e);
    solver.run(&e, &timer, LIMIT);
    println!("{}", solver.make_output(&e));
}

impl Env {
    // 経過時間とターン消化率を考慮してビーム幅を調整する
    fn get_next_beam_width(&self, beam_width: usize, depth: usize, timer: &Instant, start_time: f64, limit: f64) -> usize {
        let elapsed = timer.elapsed().as_secs_f64();
        let turn_rate = (depth as f64) / (self.max_depth as f64);
        let timer_rate = (elapsed - start_time) / (limit - start_time);
        let ratio = (turn_rate / timer_rate).clamp(self.beam_width_min_rate, self.beam_width_max_rate);
        ((beam_width as f64 * ratio) as usize).clamp(self.beam_width_min, self.beam_width_max)
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
enum SolverState { Trivial, Greedy, Beam, Anneal, None }

// [改善1] スコアの評価方向を定義
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoreDirection {
    #[default]
    Maximize,
    Minimize,
}

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
    fn run(&mut self, e: &Env, timer: &Instant, limit: f64) {
        // メインソルバーの実行
        match SOLVER_MAIN {
            SolverState::Beam => self.solve_beam(e, timer, limit), // ビームサーチ
            SolverState::Anneal  => self.solve_aneal(e, timer, limit),  // 焼きなまし法
            SolverState::None => {}
            _ => panic!("Invalid SOLVER_MAIN: {:?}", SOLVER_MAIN),
        }
    }
}

// ビームサーチの実装
impl Solver {
    fn solve_beam(&mut self, e: &Env, timer: &Instant, limit: f64) {
        let mut beam_width = e.beam_width_init;
        let mut states = vec![State::new(e, INF)]; // 初期状態をビームに追加する
        let mut hists = vec![vec![Op::new(INF)]]; // 操作のリスト
        let start_time = timer.elapsed().as_secs_f64();
        for depth in 0..e.max_depth {
            // ビームの状態列をもとに、次の操作を列挙する
            let mut cands = Vec::new();
            for (state_id, state) in states.iter().enumerate() {
                let ops = state.enum_next_ops(e);
                self.counter += ops.len();
                cands.extend(ops.into_iter().map(|op| (state_id, op)));
            }
            // 候補がなければ探索終了
            if cands.is_empty() { break; }
            // ビーム幅以内の候補を状態に適用して、次の状態列とする
            // select_nth_unstable_by_key は、O(n)であるため、高速である
            if cands.len() > beam_width {
                // スコアの評価方向に応じて選択方法を切り替える
                match e.score_direction {
                    ScoreDirection::Maximize => {
                        cands.select_nth_unstable_by_key(beam_width, |(_, op)| -op.score);
                    }
                    ScoreDirection::Minimize => {
                        cands.select_nth_unstable_by_key(beam_width, |(_, op)| op.score);
                    }
                }
                cands.truncate(beam_width);
            }
            // 古いstatesをもとに、次の状態列を作成する
            //   各親stateが何回使われるかカウントし、1回しか使われない場合はcloneを避けることで高速化
            let mut counts = vec![0; states.len()];
            for (state_id, _) in &cands {
                counts[*state_id] += 1;
            }
            //   古いstatesをOptionでラップして、ムーブ（所有権の移動）できるようにする
            let mut old_states: Vec<_> = std::mem::take(&mut states)
                .into_iter()
                .map(Some)
                .collect();
            states = cands.iter().enumerate().map(|(op_id, (state_id, op))| {
                let mut state = if counts[*state_id] > 1 {
                    // 複数回使われる場合はクローン
                    counts[*state_id] -= 1;
                    old_states[*state_id].as_ref().unwrap().clone()
                } else {
                    // 1回しか使われない場合はムーブ
                    old_states[*state_id].take().unwrap()
                };
                state.apply_op(e, op, op_id);
                state
            }).collect_vec();
            // 操作の履歴を保存する
            hists.push(cands.into_iter().map(|(_, op)| op).collect_vec());
            // ビーム幅を制限時間とターン消化率を考慮して調整する
            beam_width = e.get_next_beam_width(beam_width, depth, timer, start_time, limit);
            dbg!("#{} counter={}, width={}", depth, self.counter, beam_width);
        }
        // ビームを復元する
        let depth = e.max_depth;
        //   スコアの評価方向に応じて最良解を選択する
        let best_op = match e.score_direction {
            ScoreDirection::Maximize => hists[depth].iter().max_by_key(|&op| op.score).unwrap(),
            ScoreDirection::Minimize => hists[depth].iter().min_by_key(|&op| op.score).unwrap(),
        };
        let ops_iter = std::iter::successors(Some((depth, best_op)), |(d, current_op)| {
            let prev_id = current_op.pre_id;
            if prev_id == INF {
                None // ルートに到達したら終了
            } else {
                let new_depth = *d - 1;
                Some((new_depth, &hists[new_depth][prev_id]))
            }
        });
        //   イテレータから操作のリストを作成し、逆順にして保存
        let mut ops = ops_iter.map(|(_, op)| op.clone()).collect_vec();
        ops.reverse();
        self.ops = ops;
    }
}

// 焼きなまし法の実装
impl Solver {
    fn solve_aneal(&mut self, e: &Env, timer: &Instant, limit: f64) {
        let start_time = timer.elapsed().as_secs_f64();
        let mut temp = e.sa_start_temp; // 初期温度
        let mut state = self.state.clone(); // 初期状態を読み込む
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
            // スコアの評価方向に応じて遷移確率と最良解の更新を切り替える
            let accept = match e.score_direction {
                ScoreDirection::Maximize => {
                    let prob = (score_diff as f64 / temp).exp();
                    score_diff >= 0 || prob > rng.gen()
                }
                ScoreDirection::Minimize => {
                    let prob = (-score_diff as f64 / temp).exp();
                    score_diff <= 0 || prob > rng.gen()
                }
            };
            if accept { // 確率probで遷移する
                state.apply_transition(e, &trans);
                state.score += score_diff;
                let is_best = match e.score_direction {
                    ScoreDirection::Maximize => best_state.score < state.score,
                    ScoreDirection::Minimize => best_state.score > state.score,
                };
                if is_best {
                    best_state = state.clone();
                    dbg!("counter:{} score:{} new best", self.counter, best_state.score);
                }
            }
        }
        // 現在のベストを最終結果として採用する
        self.state = best_state;
    }
}
