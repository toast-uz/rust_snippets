#![allow(dead_code)]

use std::hash::Hash;
use itertools::{Itertools, iproduct};
use rustc_hash::FxHashMap as HashMap;
use ac_library::fenwicktree::FenwickTree;
use crate::xorshift_rand::*;

// PartialOrd は、比較結果がNoneにはならない前提とする
const ERR_PARTIALORD: &str = "PartialOrd cannot be None";
const INF: usize = 1 << 60;
const DEBUG: bool = false;

macro_rules! dbg {( $( $x:expr ),* ) => ( if DEBUG {eprintln!($( $x ),* );}) }

// 置換計算

/* Permutation
    Compatible with Python SymPy's Permutation.

    Example 1:
    `Self::new(&[1, 2, 0])` represents the permutation (0, 1, 2) -> (1, 2, 0).
    maps 0 -> 1, 1 -> 2, 2 -> 0.
    where the object at index 1 is moved to index 0, 
          the object at index 2 is moved to index 1,
          and the object at index 0 is moved to index 2.

    let perm = Permutation::new(&[1, 2, 0]);
    let x = vec!["A", "B", "C"];    // able to use any type with Clone trait
    assert_eq!(perm.apply(&x), vec!["B", "C", "A"]);

    Example 2:
    The power operator can be used to apply the permutation multiple times.
    (A negative power is used to apply the inverse permutation.)
    let perm = Permutation::new(&[1, 2, 0]);
    let x = vec!["A", "B", "C"];    // able to use any type with Clone trait
    assert_eq!(perm.pow(2).apply(&x), vec!["C", "A", "B"]);
    assert_eq!(perm.pow(-1).apply(&x), vec!["C", "A", "B"]);
*/

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    perm: Vec<usize>,
}

impl Permutation {
    pub fn len(&self) -> usize { self.perm.len() }
    pub fn new(perm: &[usize]) -> Self { Self { perm: perm.to_vec() } }
    pub fn inv(&self) -> Self {
        let mut res = vec![0; self.len()];
        for i in 0..self.len() { res[self.perm[i]] = i; }
        Self::new(&res)
    }
    pub fn pow(&self, power: isize) -> Self {
        let mut res = Self::new(&(0..self.len()).collect::<Vec<_>>());
        if power > 0 {
            for _ in 0..power { res *= self.clone(); }
        } else if power < 0 {
            let inv = self.inv();
            for _ in 0..(-power) { res *= inv.clone(); }
        }
        res
    }
    // apply the permutation to the vector x
    pub fn apply<T: Clone>(&self, x: &[T]) -> Vec<T> {
        let mut res = vec![x[0].clone(); self.len()];
        for i in 0..self.len() { res[i] = x[self.perm[i]].clone(); }
        res
    }
}

impl std::ops::Mul for Permutation {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(&(0..self.len())
            .map(|i| rhs.perm[self.perm[i]]).collect::<Vec<_>>())
    }
}
impl std::ops::MulAssign for Permutation {
    fn mul_assign(&mut self, rhs: Self) { *self = self.clone() * rhs; }
}


// Trait for measuring dissimilarity between two instances.

pub trait WrongMetric {
    fn wrong_metric(&self, other: &Self) -> usize;
}

impl<T: PartialEq> WrongMetric for [T] {
    fn wrong_metric(&self, other: &Self) -> usize {
        (0..self.len()).filter(|&i| self[i] != other[i]).count()
    }
}


// 引数順序
pub trait ArgPartialOrd<T> {
    fn argmax(&self) -> Option<usize>;
    fn argmin(&self) -> Option<usize>;
    fn argsort(&self) -> Vec<usize>;
}

impl<T: PartialOrd> ArgPartialOrd<T> for [T] {
    fn argmax(&self) -> Option<usize> { (0..self.len()).rev().max_by(|&i, &j|
        self[i].partial_cmp(&self[j]).expect(ERR_PARTIALORD)) } // 最初のインデックスが返るようにrevを使う
    fn argmin(&self) -> Option<usize> { (0..self.len()).min_by(|&i, &j|
        self[i].partial_cmp(&self[j]).expect(ERR_PARTIALORD)) }
    fn argsort(&self) -> Vec<usize> { (0..self.len()).sorted_by(|&i, &j|
        self[i].partial_cmp(&self[j]).expect(ERR_PARTIALORD)).collect() }
}

// 座標圧縮(initは圧縮後の最小値)
pub trait CoordinateCompress<T> {
    fn coordinate_compress(&self, init: usize) -> Vec<usize>;
    fn to_ord(&self, init: usize) -> Vec<usize>;    // 同じ値にも順序をつける
}

impl<T: Clone + PartialOrd> CoordinateCompress<T> for [T] {
    fn coordinate_compress(&self, init: usize) -> Vec<usize> {
        let mut xs: Vec<T> = self.to_vec();
        xs.sort_by(|x, y| x.partial_cmp(y).expect(ERR_PARTIALORD)); xs.dedup();
        self.iter().map(|x| xs.binary_search_by(|y|
            y.partial_cmp(x).expect(ERR_PARTIALORD)).unwrap() + init).collect()
    }
    fn to_ord(&self, init: usize) -> Vec<usize> {
        self.argsort().iter().enumerate()
            .sorted_by_key(|&(_, &p)| p).map(|(i, _)| init + i).collect()
    }
}

pub trait CoordinateCompressPlus<T> {
    fn to_ord_weighted(&self, init: usize) -> Vec<f64>;    // 同じ値は一度順序をつけたものを平均化する
}

impl<T: Clone + PartialOrd + Hash + Eq> CoordinateCompressPlus<T> for [T] {
    fn to_ord_weighted(&self, init: usize) -> Vec<f64> {
        let to_order_ids = self.to_order_ids();
        let mut res = self.to_ord(init).to_f64();
        for (_, v) in &to_order_ids {
            if v.is_empty() { continue; }
            let mut sum = 0.0;
            for &i in v { sum += res[i]; }
            let mean = sum / v.len() as f64;
            for &i in v { res[i] = mean; }
        }
        res
    }
}

// f64化
pub trait ToF64 {
    fn to_f64(&self) -> Vec<f64>;
}
impl ToF64 for [usize] {
    fn to_f64(&self) -> Vec<f64> { self.iter().map(|&x| x as f64).collect() }
}
impl ToF64 for [isize] {
    fn to_f64(&self) -> Vec<f64> { self.iter().map(|&x| x as f64).collect() }
}
impl ToF64 for [i32] {
    fn to_f64(&self) -> Vec<f64> { self.iter().map(|&x| x as f64).collect() }
}

// f64を前提とした正規化
pub trait Normalized {
    fn normalized(&self, l: f64, r: f64) -> Vec<f64>;
}

impl Normalized for [f64] {
    fn normalized(&self, l: f64, r: f64) -> Vec<f64> {
        if self.is_empty() { return Vec::new(); }
        let min = self.iter().min_by(|&x, &y| x.partial_cmp(y).expect(ERR_PARTIALORD)).cloned().unwrap();
        let max = self.iter().max_by(|&x, &y| x.partial_cmp(y).expect(ERR_PARTIALORD)).cloned().unwrap();
        if min == max { return vec![l; self.len()]; }
        self.iter().map(|&x| (x - min) / (max - min) * (r - l) + l).collect()
    }
}

pub trait ToDict<T> {
    fn to_count(&self) -> HashMap<T, usize>; // 辞書化 T -> count
    fn to_order_id(&self) -> HashMap<T, usize>; // 逆写像 T -> order_id (0-indexed) 1:1の場合のみ
    fn to_order_ids(&self) -> HashMap<T, Vec<usize>>; // 逆写像 T -> order_ids (0-indexed) 1:多の場合
}

impl<T: Clone + Eq + Hash> ToDict<T> for [T] {
    fn to_count(&self) -> HashMap<T, usize> {
        let mut res: HashMap<T, usize> = HashMap::default();
        for x in self { *res.entry(x.clone()).or_default() += 1; }
        res
    }
    fn to_order_id(&self) -> HashMap<T, usize> {
        (0..self.len()).map(|i| (self[i].clone(), i)).collect()
    }
    fn to_order_ids(&self) -> HashMap<T, Vec<usize>> {
        let mut res: HashMap<T, Vec<usize>> = HashMap::default();
        for (i, x) in self.iter().enumerate() { res.entry(x.clone()).or_default().push(i); }
        res
    }
}

// 2次元ベクトルの転置
pub trait Transpose<T> {
    fn transpose(&self) -> Vec<Vec<T>>;
}

impl<T: Clone> Transpose<T> for [Vec<T>] {
    fn transpose(&self) -> Vec<Vec<T>> {
        let n = self.len();
        let m = self[0].len();
        let mut res = vec![vec![]; m];
        for (i, j) in iproduct!(0..n, 0..m) {
            res[j].push(self[i][j].clone());
        }
        res
    }
}

// 転倒数
pub trait InversionNumber<T> {
    fn inversion_number(&self) -> usize;
}

impl<T: Clone + PartialOrd> InversionNumber<T> for [T] {
    fn inversion_number(&self) -> usize {
        let compressed = self.coordinate_compress(1);
        let Some(&max) = compressed.iter().max() else { return 0; };
        let mut bit = FenwickTree::new(max, 0usize);
        let mut res = 0;
        for &x in &compressed {
            res += bit.sum(x..max);
            bit.add(x - 1, 1);
            }
        res
    }
}

// TSP
pub trait TSP<T> {
    // 0 <= i, j <= n
    fn tsp_2opt(&mut self, i: usize, j: usize);
    fn kick(&mut self);
    fn tsp_2opt_diff_cost(&self, i: usize, j: usize,
        cost: &dyn Fn(&T, &T) -> isize) -> isize;
    fn tsp_cost(&self, cost: &dyn Fn(&T, &T) -> isize) -> isize;
    fn tsp_exact(&mut self, cost_f: &dyn Fn(&T, &T) -> isize);
    fn tsp_climb(&mut self, cost: &dyn Fn(&T, &T) -> isize,
        max_counter: usize, patience: usize, rng: &mut XorshiftRng);
}

impl<T: Clone + std::fmt::Debug> TSP<T> for Vec<T> {
    // 2-opt
    fn tsp_2opt(&mut self, i: usize, j: usize) {
        for k in (i + 1)..=((i + j) / 2) {
            self.swap(k, i + j - k + 1);
        }
    }
    // kick: double bridge
    fn kick(&mut self) {
        if self.len() < 5 { return; }
        let mut rng = xorshift_rng();
        let v = rng.gen_range_multiple(1..self.len(), 4);
        let mut res = Vec::new();
        for i in 0..v[0] { res.push(self[i].clone()); }
        for i in v[2]..v[3] { res.push(self[i].clone()); }
        for i in v[1]..v[2] { res.push(self[i].clone()); }
        for i in v[0]..v[1] { res.push(self[i].clone()); }
        for i in v[3]..self.len() { res.push(self[i].clone()); }
        *self = res;
    }
    // 差分コスト
    fn tsp_2opt_diff_cost(&self, i: usize, j: usize,
            cost_f: &dyn Fn(&T, &T) -> isize) -> isize {
        cost_f(&self[i], &self[j]) + cost_f(&self[i + 1], &self[j + 1])
        - cost_f(&self[i], &self[i + 1]) - cost_f(&self[j], &self[j + 1])
    }
    // フルコスト
    fn tsp_cost(&self, cost_f: &dyn Fn(&T, &T) -> isize) -> isize {
        let mut res = 0;
        for i in 0..(self.len() - 1) { res += cost_f(&self[i], &self[i + 1]); }
        res
    }
    // 厳密解（両端固定）
    fn tsp_exact(&mut self, cost_f: &dyn Fn(&T, &T) -> isize) {
        if self.len() <= 3 { return; }
        dbg!("START tsp exact solution, size:{} cost:{} {:?}", self.len(), self.tsp_cost(cost_f), self);
        let n = self.len() - 2;
        let mut dp = vec![vec![INF as isize; n]; 1 << n];
        let mut pre = vec![vec![INF; n]; 1 << n];  // dp復元用
        (0..n).for_each(|u| dp[1 << u][u] = cost_f(&self[0], &self[u + 1]));
        for bit in 0..(1 << n) {
            for u in 0..n {
                if (bit >> u) & 1 == 0 && bit > 0 { continue; }
                for v in 0..n {
                    if (bit >> v) & 1 == 1 { continue; }
                    let next_bit = bit | (1 << v);
                    let new_cost = dp[bit][u] + cost_f(&self[u + 1], &self[v + 1]);
                    if new_cost < dp[next_bit][v] {
                        dp[next_bit][v] = new_cost;
                        pre[next_bit][v] = u;
                    }
                }
            }
        }
        let mut res = vec![self[self.len() - 1].clone()];
        let mut bit = (1 << n) - 1;
        let mut v = (0..n).min_by_key(|&u|
            dp[bit][u] + cost_f(&self[u + 1], &self[self.len() - 1])).unwrap();
        while v < INF {
            res.push(self[v + 1].clone());
            let u = pre[bit][v];
            bit -= 1 << v;
            v = u;
        }
        res.push(self[0].clone());
        res.reverse();
        *self = res;
        dbg!("result size:{} cost:{} {:?}", self.len(), self.tsp_cost(cost_f), self);
    }
    // 山登り法+キック+差分コスト計算
    fn tsp_climb(&mut self, cost_f: &dyn Fn(&T, &T) -> isize,
            max_counter: usize, patience: usize, rng: &mut XorshiftRng) {
        if self.len() <= 3 { return; }  // gen_range_multipleが3以上を要求するため
        let mut best = (self.tsp_cost(cost_f), 0, self.clone());
        let mut cost = best.0;
        dbg!("START tsp climb, size:{}", self.len());
        for counter in 0..max_counter {
            if counter >= best.1 + patience {
                // ベストに戻してキックする
                *self = best.2.clone();
                best.1 = counter;
                self.kick();
                cost = self.tsp_cost(cost_f);
                dbg!("back to best and kick {} {}", counter, cost);
                continue;
            }
            let v = rng.gen_range_multiple(0..(self.len() - 1), 2);
            let (i, j) = (v[0], v[1]);
            let diff_cost = self.tsp_2opt_diff_cost(i, j, cost_f);
            if diff_cost >= 0 { continue; }
            self.tsp_2opt(i, j);
            cost += diff_cost;
            if cost < best.0 {
                best = (cost, counter, self.clone());
                dbg!("new_best {} {}", counter, cost);
            }
        }
    }
}

// Zobrist Hash

pub trait VariantCount {
    const VARIANT_COUNT: usize;
    fn variant_id(&self) -> usize;
}

impl VariantCount for bool {
    const VARIANT_COUNT: usize = 2;
    fn variant_id(&self) -> usize { *self as usize }
}

pub trait ZobristHash {
    fn new_zobrist_hash_seed(&self) -> Vec<u64>;
    fn zobrist_hash(&self, seed: &[u64]) -> u64;
    fn zobrist_hash_diff<T: VariantCount>(&self, i: usize, x: &T, seed: &[u64]) -> u64;
}

impl<T: VariantCount> ZobristHash for [T] {
    fn new_zobrist_hash_seed(&self) -> Vec<u64> {
        let mut rng = xorshift_rng();
        (0..(self.len() * T::VARIANT_COUNT))
            .map(|_| rng.gen_u64()).collect_vec()
    }
    fn zobrist_hash(&self, seed: &[u64]) -> u64 {
        let mut res = 0;
        for i in 0..self.len() {
            res ^= seed[i + self[i].variant_id() * self.len()];
        }
        res
    }
    fn zobrist_hash_diff<S: VariantCount>(&self, i: usize, x: &S, seed: &[u64]) -> u64 {
        let mut res = 0;
        res ^= seed[i + self[i].variant_id() * self.len()];
        res ^= seed[i + x.variant_id() * self.len()];
        res
    }
}

// 統計関数
pub trait Statistics<T> {
    fn mean(&self) -> T;
    fn var(&self) -> T;
}

impl Statistics<f64> for [f64] {
    fn mean(&self) -> f64 { self.iter().sum::<f64>() / self.len() as f64 }
    fn var(&self) -> f64 {
        let mean = self.mean();
        self.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.len() as f64
    }
}


///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test -r kyopro_array::tests::benchmark1    1.9 sec（hashでのidx取得10億回）
// * FxHasherを使わないと、約15 sec

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // 通常の符号付き整数
        let a = vec![-1, 3, 2, -4, 1_000_000_000];
        assert_eq!(a.argmax(), Some(4));
        assert_eq!(a.argmin(), Some(3));
        assert_eq!(a.argsort(), vec![3, 0, 2, 1, 4]);
        let hs = a.to_order_id();
        assert_eq!(hs[&-1], 0);
        assert_eq!(hs[&2], 2);
        assert_eq!(hs[&1_000_000_000], 4);
        assert_eq!(a.coordinate_compress(0), vec![1, 3, 2, 0, 4]);
        assert_eq!(a.to_ord(0), vec![1, 3, 2, 0, 4]);
        assert_eq!(a.to_ord_weighted(0), vec![1.0, 3.0, 2.0, 0.0, 4.0]);
        assert_eq!(a.to_ord_weighted(0).normalized(0.0, 1.0), vec![0.25, 0.75, 0.5, 0.0, 1.0]);
        assert_eq!(a.inversion_number(), 4);
        // 実数にも対応可能
        let a = vec![-0.5, 0.2, 0.9, 0.1, 1.0];
        assert_eq!(a.argmax(), Some(4));
        assert_eq!(a.argmin(), Some(0));
        assert_eq!(a.argsort(), vec![0, 3, 1, 2, 4]);
        assert_eq!(a.coordinate_compress(2), vec![2, 4, 5, 3, 6]);
        assert_eq!(a.to_ord(0), vec![0, 2, 3, 1, 4]);
        assert_eq!(a.inversion_number(), 2);
        // 同一の要素が存在
        let a = vec![1, 0, 2, 0, 2, 1, 0];
        assert_eq!(a.argmax(), Some(2));    // 最初に出た最大値のインデックスが返る
        assert_eq!(a.argmin(), Some(1));    // 最初に出た最小値のインデックスが返る
        assert_eq!(a.argsort(), vec![1, 3, 6, 0, 5, 2, 4]);    // 同じ数は最初が若番になる
        assert_eq!(a.coordinate_compress(0), vec![1, 0, 2, 0, 2, 1, 0]);
        assert_eq!(a.to_ord(0), vec![3, 0, 5, 1, 6, 4, 2]);
        assert_eq!(a.to_ord_weighted(0), vec![3.5, 1.0, 5.5, 1.0, 5.5, 3.5, 1.0]);
        assert_eq!(a.inversion_number(), 9);
        // 符号無し整数
        let a: Vec<usize> = vec![0, 3, 2, 4, 5];
        assert_eq!(a.inversion_number(), 1);
        // 転倒数のバリエーション
        let a = vec![0, 3, 2, 6, 5, 4];
        assert_eq!(a.inversion_number(), 4);
        let a = vec![5, 6, 4, 3, 0, 2];
        assert_eq!(a.inversion_number(), 13);
        // 転置
        let a = vec![vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(a.transpose(), vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    }

    #[test]
    fn benchmark1_to_order_id() {
        let a: Vec<usize> = (0..1000).collect();
        let hs = a.to_order_id();
        for _ in 0..500_000_000 {
            assert_eq!(hs[&0], 0);
            assert_eq!(hs[&999], 999);
        }
    }

    #[test]
    fn test_zobrist_hash() {
        let mut x = vec![false; 1000];
        let seed = x.new_zobrist_hash_seed();
        let mut rng = xorshift_rng();
        for _ in 0..1000 {
            for i in 0..x.len() {
                let b = match rng.gen_range(0..=1) {
                    0 => false,
                    1 => true,
                    _ => unreachable!(),
                };
                x[i] = b;
            }
            let hash = x.zobrist_hash(&seed);
            let i = rng.gen_range(0..x.len());
            let b = match rng.gen_range(0..=1) {
                0 => false,
                1 => true,
                _ => unreachable!(),
            };
            let hash_diff = x.zobrist_hash_diff(i, &b, &seed);
            let same = x[i] == b;
            x[i] = b;
            let hash2 = x.zobrist_hash(&seed);
            if same {
                assert_eq!(hash, hash2);
            } else {
                assert_ne!(hash, hash2);
            }
            assert_eq!(hash ^ hash_diff, hash2);
        }
    }

    #[test]
    fn test_permutation() {
        let perm = Permutation::new(&[1, 2, 3, 0]);
        let x = vec!["Apple", "Banana", "Cherry", "Durian"];
        assert_eq!(perm.apply(&x), vec!["Banana", "Cherry", "Durian", "Apple"]);
        assert_eq!(perm.pow(2).apply(&x), vec!["Cherry", "Durian", "Apple", "Banana"]);
        assert_eq!(perm.pow(-1).apply(&x), vec!["Durian", "Apple", "Banana", "Cherry"]);
    }
}
