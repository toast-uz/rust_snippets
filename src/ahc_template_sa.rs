/*

焼きなまし法のテンプレート

使い方
1) 最初の入力、ハイパーパラメータをもとに、イミュータブルなEnvを作成
2) 出力を作るAgentについて、initを初期解として作成
3) resultで出力を作成 → 初期解の提出
4) compute_scoreでスコアを計算 → ローカルテストツールのスコアと照合
5) 近傍を設計、select_neighborで近傍を選択、transfer_neighborで遷移
6) optimizeで最適化を行ってみて、スコアが改善するか確認
   （スコアが高い方が良いか、低い方が良いかで、実装を変更）
7) 制限時間を増やしてみて、さらにスコアが改善するか確認（→ 差分計算する必要性）
8) score_diffでスコアの差分計算を実装、最初はフル計算と照合
9) フル計算との照合をやめて、差分計算のみでカウンターとスコアが改善するか確認
10) 近傍を増やしてみて試行錯誤
11) ハイパーパラメータをチューニング

*/
use std::time::Instant;
use proconio::input;
//use itertools::{iproduct, Itertools};
//use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
//use rust_snippets::xorshift_rand::*;
//use rust_snippets::kyopro_args::*;
use xorshift_rand::*;
use kyopro_args::*;

const LIMIT: f64 = 0.0;
const DEBUG: bool = true;
const START_TEMP: f64 = 1e9;
const END_TEMP: f64 = 1e-9;
const PATIENCE: usize = 100;

macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }

fn main() {
    let timer = Instant::now();
    let mut rng = XorshiftRng::from_seed(1e18 as u64);
    let e = Env::new();
    let mut a = Agent::new(&e);
    a.optimize(&e, &mut rng, &timer, LIMIT);
    println!("{}", a.result());
    dbg!("N:{} counter:{}", e.n, a.counter);
    dbg!("Score = {}", a.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    n: usize,
    start_temp: f64,
    duration_temp: f64,
    patience: usize,
}

impl Env {
    fn new() -> Self {
        input! { n: usize }
        let mut e = Self::default();
        e.init(n); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, n: usize) {
        // 問題入力の設定
        self.n = n;
        // ハイパーパラメータの設定
        self.start_temp= os_env::get("start_temp").unwrap_or(START_TEMP);
        let end_temp= os_env::get("end_temp").unwrap_or(END_TEMP);
        self.duration_temp = end_temp - self.start_temp;
        self.patience= os_env::get("patience").unwrap_or(PATIENCE);
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    score: isize,
    counter: usize,
}

impl Agent {
    fn new(e: &Env) -> Self {
        let mut a = Self::default();
        a.init(e); a
    }

    fn init(&mut self, e: &Env) {
        self.score = self.compute_score(e);
    }

    fn optimize(&mut self, e: &Env, rng: &mut XorshiftRng, timer: &Instant, limit: f64) {
        let start_time = timer.elapsed().as_secs_f64();
        let mut time = start_time;
        let mut best = self.clone();
        let mut temp;
        while time < limit {
            self.counter += 1;
            // PATIENCE回、ベスト更新されなかったら，現在のカウンターをベストにコピーして、ベストから再開する
            if self.counter > best.counter + e.patience {
                best.counter = self.counter;
                *self = best.clone();
                dbg!("counter:{} score:{} restart from the best", self.counter, self.score);
            }
            // 遷移候補を決めて、遷移した場合のコスト差分を計算する
            let neighbor = self.select_neighbor(e, rng);
            let score_diff = self.compute_score_diff(e, neighbor);
            // 現在の温度を計算して遷移確率を求める
            time = timer.elapsed().as_secs_f64();
            temp = e.start_temp + e.duration_temp * (time - start_time) / (limit - start_time);
            let prob = (score_diff as f64 / temp).exp();
            // スコアが高いほど良い場合
            // スコアが低いほど良い場合はprob < rng.gen()とする
            if prob > rng.gen() || neighbor.forced() { // 確率prob or 強制近傍か で遷移する
                self.transfer_neighbor(e, neighbor);
                self.score += score_diff;
                // スコアが高いほど良い場合
                // スコアが低いほど良い場合は self.score < best.score とする
                if best.score < self.score {
                    best = self.clone();
                    dbg!("counter:{} score:{} new best", best.counter, best.score);
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
            let v = rng.gen_range_multiple(0..100, 2);
            Neighbor::Swap(v[0], v[1])
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
    // 以下のいずれかで実装する
    // 1) cloneしたnew_stateを遷移させて、フル計算する
    // 2) selfを遷移させて、フル計算し、その後、selfを逆遷移させる
    // 3) 差分計算をする
    // 3)の場合は、1)のコードを最初は残して、結果を照合する
    fn compute_score_diff(&mut self, e: &Env, neighbor: Neighbor) -> isize {
        // 1) 差分計算をしない場合の実装
        let score_old = self.score;
        let mut new_state = self.clone();
        new_state.transfer_neighbor(e, neighbor);
        let score_new = new_state.compute_score(e);
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
    // 近傍を逆遷移させるための近傍を返す
    // kick系の非可逆なNeighborはNone（戻さない）とする
    /*
    fn reversed(&self) -> Self {
        match *self {
            Self::Swap(a, b) => Self::Swap(b, a),
            Self::None => Self::None,
        }
    }
    */
    // 強制で遷移する近傍かどうか
    // kick系の非可逆なNeighborはtrueとする
    #[inline]
    fn forced(&self) -> bool {
        false
    }
}

mod xorshift_rand {
    #![allow(dead_code)]
    use std::time::SystemTime;
    use rustc_hash::FxHashSet as HashSet;

    pub fn xorshift_rng() -> XorshiftRng {
        let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
            .unwrap().as_secs() as u64;
        let mut rng = XorshiftRng::from_seed(seed);
        for _ in 0..100 { rng._xorshift(); }    // 初期値が偏らないようにウォーミングアップ
        rng
    }
    pub struct XorshiftRng { seed: u64, }

    impl XorshiftRng {
        pub fn from_seed(seed: u64) -> Self { Self { seed, } }
        fn _xorshift(&mut self) {
            self.seed ^= self.seed << 3;
            self.seed ^= self.seed >> 35;
            self.seed ^= self.seed << 14;
        }
        // [low, high) の範囲のusizeの乱数を求める
        pub fn gen_range<R: std::ops::RangeBounds<usize>>(&mut self, range: R) -> usize {
            let (start, end) = Self::unsafe_decode_range_(&range);
            self._xorshift();
            (start as u64 + self.seed % (end - start) as u64) as usize
        }
        // 重み付きで乱数を求める
        pub fn gen_range_weighted<R: std::ops::RangeBounds<usize>>(&mut self, range: R, weights: &[usize]) -> usize {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert_eq!(end - start, weights.len());
            let sum = weights.iter().sum::<usize>();
            let x = self.gen_range(0..sum);
            let mut acc = 0;
            for i in 0..weights.len() {
                acc += weights[i];
                if acc > x { return i; }
            }
            unreachable!()
        }
        // [low, high) の範囲から重複なくm個のusizeの乱数を求める
        pub fn gen_range_multiple<R: std::ops::RangeBounds<usize>>(&mut self, range: R, m: usize) -> Vec<usize> {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert!(m <= end - start);
            let many = m > (end - start) / 2; // mが半分より大きいか
            let n = if many { end - start - m } else { m };
            let mut res = HashSet::default();
            while res.len() < n {   // 半分より小さい方の数をランダムに選ぶ
                self._xorshift();
                let x = (start as u64 + self.seed % (end - start) as u64) as usize;
                res.insert(x);
            }
            (start..end).filter(|&x| many ^ res.contains(&x)).collect()
        }
        // rangeをもとに半開区間の範囲[start, end)を求める
        fn unsafe_decode_range_<R: std::ops::RangeBounds<usize>>(range: &R) -> (usize, usize) {
            let std::ops::Bound::Included(&start) = range.start_bound() else { panic!(); };
            let end = match range.end_bound() {
                std::ops::Bound::Included(&x) => x + 1,
                std::ops::Bound::Excluded(&x) => x,
                _ => panic!(),
            };
            (start, end)
        }
        // [0, 1] の範囲のf64の乱数を求める
        pub fn gen(&mut self) -> f64 {
            self._xorshift();
            self.seed as f64 / u64::MAX as f64
        }
        // u64の乱数を求める
        pub fn gen_u64(&mut self) -> u64 {
            self._xorshift();
            self.seed
        }
    }

    pub trait SliceXorshiftRandom<T> {
        fn choose(&self, rng: &mut XorshiftRng) -> T;
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T>;
        fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T;
        fn shuffle(&mut self, rng: &mut XorshiftRng);
    }

    impl<T: Clone> SliceXorshiftRandom<T> for [T] {
        fn choose(&self, rng: &mut XorshiftRng) -> T {
            let x = rng.gen_range(0..self.len());
            self[x].clone()
        }
        fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T {
            let x = rng.gen_range_weighted(0..self.len(), weights);
            self[x].clone()
        }
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T> {
            let selected = rng.gen_range_multiple(0..self.len(), m);
            selected.iter().map(|&i| self[i].clone()).collect()
        }
        fn shuffle(&mut self, rng: &mut XorshiftRng) {
            // Fisher-Yates shuffle
            for i in (1..self.len()).rev() {
                let x = rng.gen_range(0..=i);
                self.swap(i, x);
            }
        }
    }
}

mod kyopro_args {
    #![allow(dead_code)]

    use regex::Regex;
    use itertools::Itertools;

    pub struct Args {
        args_str: String,
    }

    impl Args {
        pub fn new() -> Self {
            let args: Vec<String> = std::env::args().collect();
            let args_str = args[1..].iter().join(" ");
            Self { args_str }
        }

        pub fn get<T: std::str::FromStr>(&self, arg_name: &str) -> Option<T> {
            let re_str = format!(r"-{}=([\d.-]+)", arg_name);
            let re = Regex::new(&re_str).unwrap();
            let Some(captures) = re.captures(&self.args_str) else { return None; };
            captures[1].parse().ok()
        }
    }

    pub mod os_env {
        const PREFIX: &str = "AHC_PARAMS_";

        pub fn get<T: std::str::FromStr>(name: &str) -> Option<T> {
            let name = format!("{}{}", PREFIX, name.to_uppercase());
            std::env::var(name).ok()?.parse().ok()
        }
    }
}