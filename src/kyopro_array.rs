use std::hash::Hash;
use itertools::Itertools;
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
}

impl<T: Clone + PartialOrd> CoordinateCompress<T> for [T] {
    fn coordinate_compress(&self, init: usize) -> Vec<usize> {
        let mut xs: Vec<T> = self.to_vec();
        xs.sort_by(|x, y| x.partial_cmp(y).expect(ERR_PARTIALORD)); xs.dedup();
        self.iter().map(|x| xs.binary_search_by(|y|
            y.partial_cmp(x).expect(ERR_PARTIALORD)).unwrap() + init).collect()
    }
}

// T -> order_id (0-indexed)
pub trait ToOrderId<T> {
    fn to_order_id(&self) -> HashMap<T, usize>;
}

impl<T: Clone + Eq + Hash> ToOrderId<T> for [T] {
    fn to_order_id(&self) -> HashMap<T, usize> {
        (0..self.len()).map(|i| (self[i].clone(), i)).collect()
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

#[allow(dead_code)]
fn main() {
}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test benchmark1 --bin array --release    2.0 sec（hashでのidx取得10億回）
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
        assert_eq!(a.inversion_number(), 4);
        // 実数にも対応可能
        let a = vec![-0.5, 0.2, 0.9, 0.1, 1.0];
        assert_eq!(a.argmax(), Some(4));
        assert_eq!(a.argmin(), Some(0));
        assert_eq!(a.argsort(), vec![0, 3, 1, 2, 4]);
        assert_eq!(a.coordinate_compress(2), vec![2, 4, 5, 3, 6]);
        assert_eq!(a.inversion_number(), 2);
        // 同一の要素が存在
        let a = vec![0.1, 0.0, 0.2, -0.0, 0.2, 0.1, 0.0];
        assert_eq!(a.argmax(), Some(2));    // 最初に出た最大値のインデックスが返る
        assert_eq!(a.argmin(), Some(1));    // 最初に出た最小値のインデックスが返る（-0.0 == 0.0） * total_cmpでは無い
        assert_eq!(a.argsort(), vec![1, 3, 6, 0, 5, 2, 4]);    // 同じ数は最初が若番になる
        assert_eq!(a.coordinate_compress(0), vec![1, 0, 2, 0, 2, 1, 0]);
        assert_eq!(a.inversion_number(), 9);

        // 符号無し整数
        let a: Vec<usize> = vec![0, 3, 2, 4, 5];
        assert_eq!(a.inversion_number(), 1);
        // 転倒数のバリエーション
        let a = vec![0, 3, 2, 6, 5, 4];
        assert_eq!(a.inversion_number(), 4);
        let a = vec![5, 6, 4, 3, 0, 2];
        assert_eq!(a.inversion_number(), 13);
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
