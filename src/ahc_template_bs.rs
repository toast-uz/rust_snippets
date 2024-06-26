/*

ビームサーチのテンプレート

使い方

*/

use std::time::Instant;
use proconio::input;
//use itertools::Itertools;
use heapmap::*;
//use rustc_hash::FxHashMap as HashMap;

const MAX_BEAM_WIDTH: usize = 30000;
const CHOKUDAI_WIDTH: usize = 1000;

//const LIMIT: f64 = 0.8; // 大量メモリの解放に時間がかかるので、余裕を持たせる
const DEBUG: bool = true;

#[allow(unused_macros)]
macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }
#[allow(unused_macros)]
macro_rules! dbg2 {( $( $x:expr ),* ) => ( if DEBUG {
    eprintln!($( $x ),* );
    println!("## {}", format!($( $x ),* ));
}) }

fn main() {
    //let timer = Instant::now();
    let e = Env::new();
    let mut a = Agent::new(&e);
    //a.chokudai_search(&e, &timer, LIMIT);
    a.beam_search(&e);
    println!("{}", a.result(&e));
    dbg!("counter = {}", a.counter);
    dbg!("Computed_score = {}", a.best_state.score);
}

#[derive(Debug, Clone, Default)]
struct Env {
    t: usize,   // ターン数
}

impl Env {
    fn new() -> Self {
        input! {
            t: usize,
        }
        let mut e = Self::default();
        e.init(t); e
    }

    // テストが作りやすいように、newとinitを分離
    fn init(&mut self, t: usize) {
        // 問題入力の設定
        self.t = t;
        // ハイパーパラメータの設定
    }
}

#[derive(Debug, Clone, Default)]
struct Agent {
    best_state: State,
    counter: usize,
}

impl Agent {
    fn new(_e: &Env) -> Self { Self::default() }

    #[allow(dead_code)]
    fn beam_search(&mut self, e: &Env) {
        // 初期状態を登録
        let mut todo = vec![State::default()];
        // ビームサーチ
        for t in 0..e.t {
            let mut next_todo = Vec::new();
            // 枝刈りのための準備
            let mut max_score = 0;
            // 現在の状態を順番に取り出す
            while let Some(state) = todo.pop() {
                // 次の状態を列挙
                for next_state in state.neighbors(e, t) {
                    // 枝刈り
                    if max_score < next_state.score {
                        max_score = next_state.score;
                    } else if next_state.score <= max_score - 3 {
                        continue;
                    }
                    // 処理候補に登録
                    self.counter += 1;
                    next_todo.push(next_state);
                }
            }
            // 処理候補をスコアが大きい順にソートして、ビーム幅分だけ残す
            next_todo.sort_by_key(|state| -state.score);
            next_todo.truncate(MAX_BEAM_WIDTH);
            todo = next_todo;   // reverseしない　->　popはスコアが小さい順に取り出す
        }
        self.best_state = todo[0].clone();
    }

    #[allow(dead_code)]
    fn chokudai_search(&mut self, e: &Env, timer: &Instant, limit: f64) {
        // 初期状態を登録
        let mut todo = vec![HeapMap::new(false); e.t + 1];
        todo[0].push((0, State::default()));
        // chokudaiサーチ
        'outer: loop { for t in 0..e.t { for _ in 0..CHOKUDAI_WIDTH {
            if timer.elapsed().as_secs_f64() > limit { break 'outer; }
            // 現在の状態を順番に取り出す
            let Some((_, state)) = todo[t].pop() else { continue; };
            // 次の状態を列挙
            for next_state in state.neighbors(e, t) {
                // 処理候補に登録
                self.counter += 1;
                todo[t + 1].push((next_state.score, next_state));
            }
        } } }
        self.best_state = todo[e.t].pop().unwrap().1;
    }

    // 結果出力
    fn result(&self, _e: &Env) -> String {
        "".to_string()
    }
}

#[derive(Debug, Clone, Default)]
struct State {
    score: isize,
}

impl State {
    fn new(score: isize) -> Self {
        Self { score }
    }

    fn neighbors(&self, _e: &Env, _t: usize) -> Vec<Self> {
        let mut res = Vec::new();
        // 次の状態を列挙してresちpushする
        res.push(Self::new(0));
        res
    }
}

mod heapmap {
    #![allow(dead_code)]

    use std::collections::BinaryHeap;
    use rustc_hash::FxHashMap as HashMap;
    use std::cmp::Reverse;

    #[derive(Debug, Clone)]
    pub struct HeapMap<K: Ord, V> {
        pq: PriorityQueue<K>,
        map: HashMap<usize, V>,
        counter: usize,
    }

    impl<K: Ord, V> HeapMap<K, V> {
        pub fn new(priority_min: bool) -> Self {
            Self { pq: PriorityQueue::new(priority_min), map: HashMap::default(), counter: 0, }
        }
        pub fn len(&self) -> usize { self.pq.len() }
        pub fn is_empty(&self) -> bool { self.pq.is_empty() }
        pub fn push(&mut self, (key, value): (K, V)) {
            self.pq.push((key, self.counter));
            self.map.insert(self.counter, value);
            self.counter += 1;  // unsafe
        }
        pub fn pop(&mut self) -> Option<(K, V)> {
            let (key, counter) = self.pq.pop()?;
            let value = self.map.remove(&counter)?;
            Some((key, value))
        }
    }

    #[derive(Debug, Clone)]
    enum PriorityQueue<K> {
        Max(BinaryHeap<(K, usize)>),
        Min(BinaryHeap<(Reverse<K>, usize)>),
    }

    impl<K: Ord> PriorityQueue<K> {
        fn new(priority_min: bool) -> Self {
            if priority_min { Self::Min(BinaryHeap::new()) } else { Self::Max(BinaryHeap::new()) }
        }
        fn len(&self) -> usize {
            match self {
                Self::Max(pq) => pq.len(),
                Self::Min(pq) => pq.len(),
            }
        }
        fn is_empty(&self) -> bool {
            match self {
                Self::Max(pq) => pq.is_empty(),
                Self::Min(pq) => pq.is_empty(),
            }
        }
        fn push(&mut self, (key, counter): (K, usize)) {
            match self {
                Self::Max(pq) => pq.push((key, counter)),
                Self::Min(pq) => pq.push((Reverse(key), counter)),
            }
        }
        fn pop(&mut self) -> Option<(K, usize)> {
            match self {
                Self::Max(pq) => pq.pop(),
                Self::Min(pq) => pq.pop().map(|(Reverse(key), counter)| (key, counter)),
            }
        }
    }
}
