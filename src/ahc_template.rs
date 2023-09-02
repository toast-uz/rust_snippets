#![allow(non_snake_case)]

use proconio::input;
//use proconio::input_interactive;  // interactiveな問題用
//use itertools::{iproduct, Itertools};
//use rustc_hash::{FxHashSet, FxHashMap};
use std::time::Instant;
use xorshift_rand::*;

const LIMIT: f64 = 1.9;
const DEBUG_LEVEL: usize = 0;

macro_rules! dbg {( $level:expr, $( $x:expr ),* ) => ( if DEBUG_LEVEL >= $level {eprintln!($( $x ),* );})}

fn main() {
    let timer = Instant::now();
    let mut rng = xorshift_rng();
    let e = Env::new();
    let mut a = Agent::new(&e);
    a.optimize(&e, &mut rng, &timer, LIMIT);
    println!("{}", a.result());
    eprintln!("N:{} counter:{} score:{}", e.N, a.counter, a.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    N: usize,
}

impl Env {
    fn new() -> Self {
        input! { N: usize }
        Self { N }
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    score: isize,
    counter: usize,
}

impl Agent {
    const START_TEMP: f64 = 100.0;
    const END_TEMP: f64 = 10.0;
    const PATIENCE: usize = 100;

    fn new(e: &Env) -> Self {
        let mut a = Self::default();
        a.init(e); a
    }

    fn init(&mut self, e: &Env) {
        self.score = self.compute_score(e);
    }

    fn optimize(&mut self, e: &Env, rng: &mut XorshiftRng, timer: &Instant, limit: f64) {
        let start_time = timer.elapsed().as_secs_f64();
        let mut best = self.clone();
        let mut temp;
        loop {
            self.counter += 1;
            // PATIENCE回、ベスト更新されなかったら，現在のカウンターをベストにコピーして、ベストから再開する
            if self.counter > best.counter + Self::PATIENCE {
                best.counter = self.counter;
                *self = best.clone();
                dbg!(1, "counter:{} score:{} restart from the best", self.counter, self.score);
            }
            // 遷移候補を決めて、遷移した場合のコスト差分を計算する
            let neighbor = self.select_neighbor(e, rng);
            let score_diff = self.compute_score_diff(e, neighbor);
            // 現在の温度を計算して遷移確率を求める
            let time = timer.elapsed().as_secs_f64();
            if time >= limit { break; }
            temp = Self::START_TEMP + (Self::END_TEMP - Self::START_TEMP) * (time - start_time) / (limit - start_time);
            let prob = (score_diff as f64 / temp).exp();
            if prob > rng.gen() || neighbor.forced() { // 確率prob or 強制近傍か で遷移する
                self.transfer_neighbor(e, neighbor);
                self.score += score_diff;
                // ベストと比較してベストなら更新する
                if best.score < self.score {
                    best = self.clone();
                    dbg!(1, "counter:{} score:{} new best", best.counter, best.score);
                }
            }
        }
        // 現在のベストを最終結果として採用する
        best.counter = self.counter;
        *self = best;
    }

    // 近傍を選択する
    fn select_neighbor(&self, _e: &Env, rng: &mut XorshiftRng) -> Neighbor {
        let p = rng.gen();
        if p < 0.5 {
            Neighbor::None
        } else {
            let (mut a, mut b) = (0, 0);
            while a == b {
                a = rng.gen_range(0..100);
                b = rng.gen_range(0..100);
            }
            Neighbor::Swap(a, b)
        }
    }

    // 指定された近傍に遷移する
    fn transfer_neighbor(&mut self, _e: &Env, neighbor: Neighbor) {
        // 近傍遷移
        match neighbor {
            Neighbor::Swap(_a, _b) => (),
            Neighbor::None => (),
        }
    }

    // スコアの差分計算
    fn compute_score_diff(&self, e: &Env, _neighbor: Neighbor) -> isize {
        // 差分計算をしない場合
        let score_old = self.score;
        let score_new = self.compute_score(e);
        score_new - score_old
    }

    // スコアのフル計算
    fn compute_score(&self, _e: &Env) -> isize {
        0
    }

    // 結果出力
    fn result(&self) -> String {
        "".to_string()
    }
}

// 近傍識別
#[derive(Debug, Clone, Copy)]
enum Neighbor {
    Swap(usize, usize), // aとbを交換
    None,
}

impl Neighbor {
    // 近傍を元に戻す
    // kick系の非可逆なNeighborはNone（戻さない）とする
    #[allow(dead_code)]
    fn reversed(&self) -> Self {
        match *self {
            Self::Swap(a, b) => Self::Swap(b, a),
            Self::None => Self::None,
        }
    }
    // 強制で遷移する近傍かどうか
    // kick系の非可逆なNeighborはtrueとする
    fn forced(&self) -> bool {
        false
    }
}

// 実際に使う場合は、各モジュールの中身をコピーする
mod xorshift_rand;
mod kyopro_args;
mod kyopro_array;
mod union_find;
