#![allow(dead_code)]

use std::hash::Hash;
use itertools::{Itertools, iproduct};
use rustc_hash::FxHashMap as HashMap;
use ac_library::fenwicktree::FenwickTree;

// PartialOrd は、比較結果がNoneにはならない前提とする
const ERR_PARTIALORD: &str = "PartialOrd cannot be None";

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

fn main() {
}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test benchmark1 --bin kyopro_array --release    2.0 sec（hashでのidx取得10億回）
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
}
