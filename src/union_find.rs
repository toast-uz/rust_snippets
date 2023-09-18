/*
Union Find
経路圧縮を意識的に行わないことで、高速化・Undoを可能にする
通常操作は、root_mut、union、same_mut、size_mutを使う
変更がほぼ無いなら、root、same、sizeを使うと速い
結合操作はUndoしたいなら、undoable_unionとundoを使う

Refer https://github.com/rust-lang-ja/ac-library-rs
Refer https://note.nkmk.me/python-unite-find/
Refer https://nyaannyaan.github.io/library/data-structure/rollback-unite-find.hpp.html
*/

#![allow(dead_code)]
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};

#[derive(Debug, Clone, Default)]
pub struct UnionFind {
    parents: Vec<isize>,
    history: Vec<(usize, isize)>,   // with_capacityで確保すると遅くなる・・？
}

impl UnionFind {
    pub fn new(n: usize) -> Self { Self { parents: vec![-1; n], ..Default::default() } }
    #[inline]
    pub fn is_root(&self, x: usize) -> bool { self.parents[x] < 0 }
    #[inline]
    pub fn size_of_root(&self, x: usize) -> usize { -self.parents[x] as usize }
    // 木の根 再帰版  O(α(N)) 経路圧縮あり
    pub fn root(&mut self, x: usize) -> usize {
        if self.is_root(x) { x } else {
            self.parents[x] = self.root(self.parents[x] as usize) as isize;
            self.parents[x] as usize
        }
    }
    // 経路圧縮なし
    pub fn root_wo_compress(&self, x: usize) -> usize {
        if self.is_root(x) { x } else { self.root_wo_compress(self.parents[x] as usize) }
    }
    // まとめて経路圧縮する sameやsizeをたくさん操作する前にやっておきたい
    pub fn squeeze(&mut self) {
        self.clear_history();
        (0..self.parents.len()).for_each(|i| { self.root(i); });
    }
    // 木を結合する  O(α(N))
    pub fn unite(&mut self, x: usize, y: usize) {
        let x = self.root(x);
        let y = self.root(y);
        self.unite_roots_(x, y, false);
    }
    pub fn undoable_unite(&mut self, x: usize, y: usize) {
        self.unite_roots_(self.root_wo_compress(x), self.root_wo_compress(y), true);
    }
    fn unite_roots_(&mut self, mut x: usize, mut y: usize, history: bool) {  // 根同士の結合
        if history {
            self.history.push((x, self.parents[x]));
            self.history.push((y, self.parents[y]));
        }
        if x == y { return; }
        if self.parents[x] > self.parents[y] { std::mem::swap(&mut x, &mut y); }
        self.parents[x] += self.parents[y];
        self.parents[y] = x as isize;
    }
    pub fn undo(&mut self) {    // undoable_unionのアンドゥ、historyを2回分ロールバックする
        assert!(self.history.len() >= 2);
        if let Some((x, y)) = self.history.pop() { self.parents[x] = y; }
        if let Some((x, y)) = self.history.pop() { self.parents[x] = y; }
    }
    pub fn clear_history(&mut self) { self.history = Vec::new(); }
    // 同じ木に属するか  O(α(N))
    pub fn same(&mut self, x: usize, y: usize) -> bool { self.root(x) == self.root(y) }
    pub fn same_wo_compress(&self, x: usize, y: usize) -> bool { self.root_wo_compress(x) == self.root_wo_compress(y) }
    // 木のサイズ     O(α(N))
    pub fn size(&mut self, x: usize) -> usize {
        let y = self.root(x);
        self.size_of_root(y)
    }
    pub fn size_wo_compress(&self, x: usize) -> usize { self.size_of_root(self.root_wo_compress(x)) }

    // その他の参照関数
    // 木の根の列挙 O(N)
    pub fn roots(&self) -> HashSet<usize> {
        (0..self.parents.len()).filter(|&x| self.is_root(x)).collect()
    }
    // historyをもとに、木の根の減少差分を求める
    pub fn roots_diff(&self) -> HashSet<usize> {
        self.history.iter().map(|&(x, _)| x)
            .filter(|&x| !self.is_root(x)).collect()
    }
    // グループ数  O(N)
    pub fn group_count(&self) -> usize { self.roots().len() }
    pub fn group_count_diff(&self) -> usize { self.roots_diff().len() }
    // 木のサイズの列挙 O(N)
    pub fn sizes(&self) -> Vec<usize> {
        (0..self.parents.len()).filter(|&x| self.is_root(x))
            .map(|x| self.size_of_root(x)).collect()
    }
    // 典型スコア（2乗ノルム）
    fn norm2_(v: &[usize]) -> usize { v.iter().map(|&size| size.pow(2)).sum() }
    pub fn norm2(&self) -> usize { Self::norm2_(&self.sizes()) }
    // historyをもとに、サイズ差分（削除されたサイズ列, 追加されたサイズ列）を求める
    pub fn sizes_diff(&self) -> (Vec<usize>, Vec<usize>) {
        let hm: HashMap<usize, isize> = self.history.iter().rev().cloned().collect();
        let res1 = hm.iter().map(|(_, &size)| -size as usize).collect();
        let res2 = hm.keys()
            .filter(|&&x| self.is_root(x))
            .map(|&x| self.size_of_root(x)).collect();
        (res1, res2)
    }
    pub fn norm2_diff(&self) -> usize {
        let (remove, append) = self.sizes_diff();
        Self::norm2_(&append) - Self::norm2_(&remove)
    }
}

///////////////////////////////////////////////////////////
// AtCoder問題解答
// https://atcoder.jp/contests/atc001/tasks/unionfind_a
// 39ms

use proconio::input;
use proconio::fastout;

#[fastout]
fn main() {
    input! {
        n: usize, q: usize,
        s: [(usize, usize, usize); q],
    }
    let mut uf = UnionFind::new(n);
    for (p, a, b) in s {
        if p == 0 {
            uf.unite(a, b);
        } else {
            println!("{}", if uf.same_wo_compress(a, b) { "Yes" } else { "No" });
        }
    }
}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test -r union_find::tests::benchmark1    1.6 sec
// cargo test -r union_find::tests::benchmark2    2.7 sec（Undo機能未利用 10万回）
// cargo test -r union_find::tests::benchmark3    2.2 sec（Undo機能利用 1億回）
// cargo test -r union_find::tests::benchmark4    1.0 sec（スコアフル計算 1万回）
// cargo test -r union_find::tests::benchmark5    2.1 sec（スコア差分計算 1000万回）

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet as HashSet;
    use super::UnionFind;

    #[test]
    fn basic() {
        let n = 10;
        let mut uf = UnionFind::new(n);
        assert_eq!(uf.roots(), (0..n).collect::<HashSet<usize>>());
        assert_eq!(uf.norm2(), 10);
        uf.unite(1, 5);
        assert_eq!(uf.norm2(), 12);
        uf.unite(2, 3);
        assert_eq!(uf.norm2(), 14);
        uf.undoable_unite(7, 8);
        uf.undoable_unite(4, 8);
        uf.undoable_unite(0, 4);
        assert_eq!(uf.norm2(), 26);
        assert_eq!(uf.roots_diff(), HashSet::from_iter(vec![0, 4, 8]));
        assert_eq!(uf.sizes_diff(), (vec![1, 1, 1, 1], vec![4]));
        assert_eq!(uf.norm2_diff(), 12);
        let same = vec![
            vec![true, false, false, false, true, false, false, true, true, false],
            vec![false, true, false, false, false, true, false, false, false, false],
            vec![false, false, true, true, false, false, false, false, false, false],
            vec![false, false, true, true, false, false, false, false, false, false],
            vec![true, false, false, false, true, false, false, true, true, false],
            vec![false, true, false, false, false, true, false, false, false, false],
            vec![false, false, false, false, false, false, true, false, false, false],
            vec![true, false, false, false, true, false, false, true, true, false],
            vec![true, false, false, false, true, false, false, true, true, false],
            vec![false, false, false, false, false, false, false, false, false, true],
        ];
        for i in 0..n {
            for j in 0..n {
                assert_eq!(uf.same_wo_compress(i, j), same[i][j]);
            }
        }
        let size = vec![4, 2, 2, 2, 4, 2, 1, 4, 4, 1];
        for i in 0..n {
            assert_eq!(uf.size_wo_compress(i), size[i]);
        }
        assert_eq!(uf.group_count(), 5);
    }

    struct A {
        uf: UnionFind,
    }

    impl A {
        fn new() -> Self { Self { uf: UnionFind::new(5), } }
        fn init(&mut self) {
            self.uf.unite(1, 2);
            self.uf.unite(4, 2);
            self.uf.unite(0, 3);
        }
        fn compute_score(&self) -> usize { self.uf.norm2() }
    }

    #[test]
    fn class() {
        let mut a = A::new();
        a.init();
        assert_eq!(a.compute_score(), 13);
    }

    #[test]
    fn benchmark1() {
        let n = 100_000_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 1) {
            uf.unite(i, i + 1);
        }
        assert_eq!(uf.group_count(), 1);
    }

    #[test]
    fn benchmark2_undo_by_clone() {
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.unite(i, i + 1);
        }
        for _ in 0..100_000 {
            let mut uf1 = uf.clone();
            assert!(!uf1.same(n - 2, n - 1));
            uf1.unite(n - 2, n - 1);
            assert!(uf1.same(n - 2, n - 1));
        }
    }

    #[test]
    fn benchmark3_undoable() {
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.unite(i, i + 1);
        }
        uf.squeeze();
        for _ in 0..100_000_000 {
            assert!(!uf.same(n - 2, n - 1));
            uf.undoable_unite(n - 2, n - 1);
            assert!(uf.same(n - 2, n - 1));
            uf.undo();
        }
    }

    #[test]
    fn benchmark4_score() {
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.unite(i, i + 1);
        }
        assert_eq!(uf.norm2(), (n - 1).pow(2) + 1);
        for _ in 0..10_000 {
            let mut uf1 = uf.clone();
            uf1.unite(n - 2, n - 1);
            assert_eq!(uf1.norm2(), n.pow(2));
        }
    }

    #[test]
    fn benchmark5_score_diff() {
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.unite(i, i + 1);
        }
        uf.squeeze();
        let score = uf.norm2();
        assert_eq!(score, (n - 1).pow(2) + 1);
        for _ in 0..10_000_000 {
            uf.undoable_unite(n - 2, n - 1);
            assert_eq!(score + uf.norm2_diff(), n.pow(2));
            uf.undo();
        }
    }
}

