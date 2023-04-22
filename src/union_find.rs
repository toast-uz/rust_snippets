// Union Find
// Refer https://note.nkmk.me/python-union-find/
// Almost same with ac-library-rs dsu

#[derive(Debug, Clone, Default)]
struct UnionFind {
    parents: Vec<isize>,
    history: Vec<(usize, isize)>,
}

#[allow(dead_code)]
impl UnionFind {
    fn new(n: usize) -> Self { Self { parents: vec![-1; n], ..Default::default() } }
    // 木の根 再帰版  O(α(N))
    fn root(&mut self, x: usize) -> usize {
        if self.parents[x] < 0 {
            x
        } else {
            self.parents[x] = self.root(self.parents[x] as usize) as isize;
            self.parents[x] as usize
        }
    }
    // 木の根 経路圧縮しない
    fn root_without_compress(&mut self, x: usize) -> usize {
        if self.parents[x] < 0 {
            x
        } else {
            self.root(self.parents[x] as usize)
        }
    }
    // 木を結合する  O(α(N))
    fn union(&mut self, x: usize, y: usize) {
        let mut x = self.root(x);
        let mut y = self.root(y);
        if x == y { return; }
        if self.parents[x] > self.parents[y] { std::mem::swap(&mut x, &mut y); }
        self.parents[x] += self.parents[y];
        self.parents[y] = x as isize;
    }
    // 木を結合する（戻すことが可能、経路圧縮をしない）
    fn undoable_union(&mut self, x: usize, y: usize) {
        let mut x = self.root_without_compress(x);
        let mut y = self.root_without_compress(y);
        self.history.push((x, self.parents[x]));
        self.history.push((y, self.parents[y]));
        if x == y { return; }
        if self.parents[x] > self.parents[y] { std::mem::swap(&mut x, &mut y); }
        self.parents[x] += self.parents[y];
        self.parents[y] = x as isize;
    }
    // 結合を戻す
    fn undo(&mut self) {    // historyを2回分ロールバックする
        if let Some((x, y)) = self.history.pop() {
            self.parents[x] = y;
        }
        if let Some((x, y)) = self.history.pop() {
            self.parents[x] = y;
        }
    }
    // 木のサイズ     O(α(N))
    fn size(&mut self, x: usize) -> usize {
        let y = self.root(x);
        -self.parents[y] as usize
    }
    // 同じ木に属するか  O(α(N))
    fn same(&mut self, x: usize, y: usize) -> bool { self.root(x) == self.root(y) }
    // 木の根の列挙 O(N)
    fn roots(&mut self) -> Vec<usize> {
        (0..self.parents.len()).filter(|&i| self.parents[i] < 0).collect()
    }
    // グループ数  O(N)
    fn group_count(&mut self) -> usize { self.roots().len() }
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
            uf.union(a, b);
        } else {
            println!("{}", if uf.same(a, b) { "Yes" } else { "No" });
        }
    }
}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test --release benchmark1   1.5 sec
// cargo test --release benchmark2  35.1 sec
// cargo test --release benchmark3  16.3 sec

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let n = 10;
        let mut uf = UnionFind::new(n);
        assert_eq!(uf.roots(), (0..n).collect::<Vec<usize>>());
        uf.union(1, 5);
        uf.union(2, 3);
        uf.union(7, 8);
        uf.union(4, 8);
        uf.union(0, 4);
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
                assert_eq!(uf.same(i, j), same[i][j]);
            }
        }
        let size = vec![4, 2, 2, 2, 4, 2, 1, 4, 4, 1];
        for i in 0..n {
            assert_eq!(uf.size(i), size[i]);
        }
        assert_eq!(uf.group_count(), 5);
    }

    struct A {
        uf: UnionFind,
    }

    impl A {
        fn new() -> Self { Self { uf: UnionFind::new(5), } }
        fn init(&mut self) {
            self.uf.union(1, 2);
            self.uf.union(4, 2);
            self.uf.union(0, 3);
        }
        fn compute_score(&mut self) -> usize {
            self.uf.roots().iter().map(|&i| self.uf.size(i).pow(2)).sum()
        }
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
            uf.union(i, i + 1);
        }
        assert_eq!(uf.group_count(), 1);
    }

    #[test]
    fn benchmark2_undo() {
        let n = 1_000_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.union(i, i + 1);
        }
        for _ in 0..10_000 {
            let mut uf1 = uf.clone();
            assert_eq!(uf.group_count(), 2);
            uf1.union(n - 2, n - 1);
            assert_eq!(uf1.group_count(), 1);
        }
    }

    #[test]
    fn benchmark3_undo() {
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..(n - 2) {
            uf.union(i, i + 1);
        }
        for _ in 0..100_000 {
            assert_eq!(uf.group_count(), 2);
            uf.undoable_union(n - 2, n - 1);
            assert_eq!(uf.group_count(), 1);
            uf.undo();
        }
    }
}

