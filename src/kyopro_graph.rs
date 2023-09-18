// todo! Zobrist Hash
// todo! dfs

#![allow(dead_code)]

use std::collections::VecDeque;
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
use bitvec::prelude::*;

use crate::heapmap::*;

const INF: usize = 1e18 as usize;

// Map グラフの頂点集合

#[derive(Clone, Debug)]
pub struct Map<T> {
    pub n: usize,
    pub coordinate_limit: Coordinate,
    pub data: Vec<T>,   // 空の場合を許容する
}

impl<T: Clone> Map<T> {
    pub fn new(coordinate_limit: &Coordinate) -> Self {
        Self { n: coordinate_limit.norm(), coordinate_limit: coordinate_limit.clone(), data: Vec::new() }
    }
    pub fn new_with_fill(coordinate_limit: &Coordinate, fill: &T) -> Self {
        let n = coordinate_limit.norm();
        Self { n, coordinate_limit: coordinate_limit.clone(), data: vec![fill.clone(); n] }
    }
}

// トレイトではなく単独でMapを出力として使う時に、インデックス指定できるようにしておく
impl<T: Clone> std::ops::Index<&Coordinate> for Map<T> {
    type Output = T;
    #[inline]
    fn index(&self, c: &Coordinate) -> &Self::Output { &self.get(c).unwrap() }
}

impl<T: Clone> std::ops::IndexMut<&Coordinate> for Map<T> {
    #[inline]
    fn index_mut(&mut self, c: &Coordinate) -> &mut Self::Output {
        let res = self.get_mut(c).unwrap();
        res
    }
}


#[derive(Clone, Debug)]
pub struct BitMap {
    pub coordinate_limit: Coordinate,
    pub data: BitVec,
}

impl BitMap {
    pub fn new(coordinate_limit: &Coordinate) -> Self {
        let n = coordinate_limit.norm();
        let mut res = Self { coordinate_limit: coordinate_limit.clone(), data: BitVec::with_capacity(n) };
        unsafe { res.data.set_len(n); }
        todo!()
    }
}

// トレイトではなく単独でBitMapを出力として使う時に、インデックス指定できるようにしておく
impl std::ops::Index<&Coordinate> for BitMap {
    type Output = bool;
    #[inline]
    fn index(&self, c: &Coordinate) -> &Self::Output { &self.get(c).unwrap() }
}

pub trait MapOperation<T> {
    fn len(&self) -> usize;
    fn coordinate_limit(&self) -> Coordinate;
    fn get(&self, c: &Coordinate) -> Option<&T>;
    fn get_mut(&mut self, c: &Coordinate) -> Option<&mut T>;
    fn is_empty(&self, c: &Coordinate) -> bool where T: Cell {
        self.get(c).is_some_and(|t| t.is_empty())
    }
    fn is_obstacle(&self, c: &Coordinate) -> bool where T: Cell {
        self.get(c).is_some_and(|t| t.is_obstacle())
    }
    fn contains_key(&self, c: &Coordinate) -> bool {
        c.partial_cmp(&self.coordinate_limit())
            .is_some_and(|cmp| cmp == std::cmp::Ordering::Less)
    }
    fn set(&mut self, c: &Coordinate, x: T);
    fn p2c(&self, p: usize) -> Coordinate { self.coordinate_limit().p2c(p) }
    fn c2p(&self, c: &Coordinate) -> usize { self.coordinate_limit().c2p(c) }
}

impl<T: Clone> MapOperation<T> for Map<T> {
    fn len(&self) -> usize { self.n }
    fn get(&self, c: &Coordinate) -> Option<&T> { self.data.get(self.c2p(c)) }
    fn get_mut(&mut self, c: &Coordinate) -> Option<&mut T> {
        let p = self.c2p(c);
        self.data.get_mut(p)
    }
    fn set(&mut self, c: &Coordinate, x: T) {
        let p = self.c2p(c);
        self.data[p] = x;
    }
    fn coordinate_limit(&self) -> Coordinate { self.coordinate_limit.clone() }
}

impl MapOperation<bool> for BitMap {
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, c: &Coordinate) -> Option<&bool> {
        let p = self.c2p(c);
        if p >= self.len() { return None; }
        let Some(&res) = self.data.get(p).as_deref() else { return None; };
        if res { Some(&true) } else { Some(&false) }
    }
    fn get_mut(&mut self, _c: &Coordinate) -> Option<&mut bool> {
        unimplemented!("cannot get mut from BitMap")
    }
    fn set(&mut self, c: &Coordinate, x: bool) {
        let p = self.c2p(c);
        self.data.set(p, x);
    }
    fn coordinate_limit(&self) -> Coordinate { self.coordinate_limit.clone() }
}

impl<T: Clone> std::ops::Index<&Coordinate> for dyn MapOperation<T> {
    type Output = T;
    #[inline]
    fn index(&self, c: &Coordinate) -> &Self::Output { &self.get(c).unwrap() }
}

impl<T: Clone> std::ops::IndexMut<&Coordinate> for dyn MapOperation<T> {
    #[inline]
    fn index_mut(&mut self, c: &Coordinate) -> &mut Self::Output {
        let res = self.get_mut(c).unwrap();
        res
    }
}


// Cell グラフの頂点の種類

pub trait Cell {
    fn is_empty(&self) -> bool;
    fn is_obstacle(&self) -> bool;
}

impl Cell for bool {
    fn is_empty(&self) -> bool { !self }    // falseが空白
    fn is_obstacle(&self) -> bool { *self } // trueが障害物
}

#[derive(Clone, Copy, PartialEq, Eq, std::hash::Hash, Debug)]
pub enum DefaultCell {
    Empty,
    Obstacle,
    Other,
}

impl Cell for DefaultCell {
    fn is_empty(&self) -> bool { *self == Self::Empty }
    fn is_obstacle(&self) -> bool { *self == Self::Obstacle }
}

// Coordinate 座標

#[macro_export]
macro_rules! coord {
    ( $x: expr ) => { Coordinate::D1($x) };
    ( $i: expr, $j: expr ) => { Coordinate::D2 { i: $i, j: $j } };
    ( $x: expr, $y: expr, $z: expr ) => { Coordinate::D3 { x: $x, y: $y, z: $z } };
}

#[derive(Clone, PartialEq, Eq, std::hash::Hash, Debug)]
pub enum Coordinate {
    D1(usize),
    D2 { i: usize, j: usize },
    D3 { x: usize, y: usize, z: usize },
}

impl Coordinate {
    pub fn norm(&self) -> usize {
        match self {
            Self::D1(x) => *x,
            Self::D2 { i, j } => *i * *j,
            Self::D3 { x, y, z } => *x * *y * *z,
        }
    }
    pub fn invert(&self) -> Self {
        match self {
            Self::D1(x) => coord!(0usize.wrapping_sub(*x)),
            Self::D2 { i, j } => coord!(
                0usize.wrapping_sub(*i), 0usize.wrapping_sub(*j)
            ),
            Self::D3 { x, y, z } => coord!(
                0usize.wrapping_sub(*x), 0usize.wrapping_sub(*y), 0usize.wrapping_sub(*z)
            ),
        }
    }
    pub fn p2c(&self, p: usize) -> Self {
        match self {
            Self::D1(_) => coord!(p),
            Self::D2 { i: _, j: width } => coord!(p / *width, p % *width),
            Self::D3 { x: width, y: height, z: _ }
                => coord!(p % *width , (p / *width) % *height, p % (*width * *height)),
        }
    }
    pub fn c2p(&self, c: &Self) -> usize {
        match (self, c) {
            (Self::D1(_), Self::D1(x)) => *x,
            (Self::D2 { i: _, j: width }, Self::D2 { i, j }) => *i * *width + *j,
            (Self::D3 { x: width, y: height, z: _ }, Self::D3 {x, y, z}) => {
                *x + (*y + *z * *height) * *width
            },
            _ => panic!("cannot convert different dimension coordinates"),
         }
    }
}

// 全ての軸の順序が一致する場合に限り、比較可能とする
impl std::cmp::PartialOrd for Coordinate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::D1(x0), Self::D1(x1)) => Some(x0.cmp(x1)),
            (Self::D2 { i: i0, j: j0 }, Self::D2 { i: i1, j: j1 }) => {
                let cmp_i = i0.cmp(i1);
                let cmp_j = j0.cmp(j1);
                if cmp_i == cmp_j { Some(cmp_i) } else { None }
            },
            (Self::D3 { x: x0, y: y0, z: z0 }, Self::D3 { x: x1, y: y1, z: z1 }) => {
                let cmp_x = x0.cmp(x1);
                let cmp_y = y0.cmp(y1);
                let cmp_z = z0.cmp(z1);
                if cmp_x == cmp_y && cmp_y == cmp_z { Some(cmp_x) } else { None }
            },
            _ => panic!("cannot compare different dimension coordinates"),
        }
    }
}

// !0を負数として扱う
impl std::ops::Add for Coordinate {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::D1(x0), Self::D1(x1)) => coord!(x0.wrapping_add(x1)),
            (Self::D2 { i: i0, j: j0 }, Self::D2 { i: i1, j: j1 })
                => coord!(i0.wrapping_add(i1), j0.wrapping_add(j1)),
            (Self::D3 { x: x0, y: y0, z: z0 }, Self::D3 { x: x1, y: y1, z: z1 })
                => coord!(x0.wrapping_add(x1), y0.wrapping_add(y1), z0.wrapping_add(z1)),
            _ => panic!("cannot add different dimension coordinates"),
        }
    }
}

impl std::fmt::Display for Coordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::D1(x) => write!(f, "{}", x),
            Self::D2 { i, j } => write!(f, "{} {}", i, j),
            Self::D3 { x, y, z } => write!(f, "{} {} {}", x, y, z),
        }
    }
}


// Adjacency 隣接関係

pub enum Adjacency {
    D2dir4 { dir: [Coordinate; 4], name: [&'static str; 4] },
    D2dir8 { dir: [Coordinate; 8], name: [&'static str; 8] },
    D3dir6 { dir: [Coordinate; 6], name: [&'static str; 6] },
    UnDirected { adj: HashMap<usize, HashSet<(usize, isize)>> },
    Directed { adj: HashMap<usize, HashSet<(usize, isize)>> },
}

impl Adjacency {
    pub fn new_d2dir4() -> Self {
        Self::D2dir4 {
            dir: [coord!(!0, 0), coord!(0, 1), coord!(1, 0), coord!(0, !0)],
            name: ["U", "R", "D", "L"],
        }
    }
    pub fn new_d2dir8() -> Self {
        Self::D2dir8 {
            dir: [coord!(!0, 0), coord!(!0, 1), coord!(0, 1), coord!(1, 1),
                coord!(1, 0), coord!(1, !0), coord!(0, !0), coord!(!0, !0)],
            name: ["U", "UR", "R", "DR", "D", "DL", "L", "UL"],
        }
    }
    pub fn new_d3dir6() -> Self {
        Self::D3dir6 {
            dir: [coord!(1, 0, 0), coord!(!0, 0, 0), coord!(0, 1, 0),
                coord!(0, !0, 0), coord!(0, 0, 1), coord!(0, 0, !0)],
            name: ["F", "B", "R", "L", "U", "D"],
        }
    }
    pub fn new_undirected(edges: &[(usize, usize)]) -> Self {
        let mut adj: HashMap<usize, HashSet<(usize, isize)>> = HashMap::default();
        for &(u, v) in edges {
            adj.entry(u).or_default().insert((v, 1));
            adj.entry(v).or_default().insert((u, 1));
        }
        Self::UnDirected { adj }
    }
    pub fn new_undirected_with_cost(edges: &[(usize, usize, isize)]) -> Self {
        let mut adj: HashMap<usize, HashSet<(usize, isize)>> = HashMap::default();
        for &(u, v, c) in edges {
            adj.entry(u).or_default().insert((v, c));
            adj.entry(v).or_default().insert((u, c));
        }
        Self::UnDirected { adj }
    }
    pub fn new_directed(edges: &[(usize, usize)]) -> Self {
        let mut adj: HashMap<usize, HashSet<(usize, isize)>> = HashMap::default();
        for &(u, v) in edges {
            adj.entry(u).or_default().insert((v, 1));
        }
        Self::Directed { adj }
    }
    pub fn new_directed_with_cost(edges: &[(usize, usize, isize)]) -> Self {
        let mut adj: HashMap<usize, HashSet<(usize, isize)>> = HashMap::default();
        for &(u, v, c) in edges {
            adj.entry(u).or_default().insert((v, c));
        }
        Self::Directed { adj }
    }
    pub fn get(&self, c: &Coordinate) -> Vec<Coordinate> {
        match (self, c) {
            (Self::D2dir4 { dir, .. }, _) =>
                dir.iter().map(|d| c.clone() + d.clone()).collect(),
            (Self::UnDirected { adj }, Coordinate::D1(u)) => {
                let Some(vs) = adj.get(u) else { return Vec::new() };
                vs.iter().map(|&(v, _)| coord!(v)).collect()
            },
            _ => panic!("cannot get adjacent coordinates")
        }
    }
    pub fn get_with_cost(&self, c: &Coordinate) -> Vec<(Coordinate, isize)> {
        match (self, c) {
            (Self::D2dir4 { dir, .. }, _) =>
                dir.iter().map(|d| (c.clone() + d.clone(), 1)).collect(),
            (Self::UnDirected { adj }, Coordinate::D1(u)) => {
                let Some(vs) = adj.get(u) else { return Vec::new() };
                vs.iter().map(|&(v, c)| (coord!(v), c)).collect()
            },
            _ => panic!("cannot get adjacent coordinates")
        }
    }
}


// グラフ分析に関するユーティリティ関数群

// ダイクストラ法での(距離, dp復元用の1つ前の頂点)を求める
pub fn dijkstra_template<T: Cell + Clone>(start: &Coordinate, map: &(dyn MapOperation<T> + 'static), adj: &Adjacency)
        -> Map<(usize, Option<Coordinate>)> {
    let mut res =
        Map::new_with_fill(&map.coordinate_limit(), &(INF, None));
    res[start] = (0, None);
    let mut heapq = HeapMap::new(true);
    heapq.push((0, start.clone()));
    while let Some((d, pos)) = heapq.pop() {
        if d != res[&pos].0 { continue; }
        for (next, cost) in &adj.get_with_cost(&pos) {
            if !map.contains_key(next) { continue; }
            if map.is_obstacle(next) { continue; }
            let next_d = d + *cost as usize;
            if next_d < res[next].0 {
                heapq.push((next_d, next.clone()));
                res[next] = (next_d, Some(pos.clone()));
            }
        }
    }
    res
}

// dfsで連結サイスを求める（再帰版、遅いため基本的には非再帰の方を使う）
// startは必ず障害物でないことが保証されている
pub fn dfs_recursive_template<T: Cell + Clone>(start: &Coordinate, map: &(dyn MapOperation<T> + 'static), adj: &Adjacency)
        -> usize {
    pub fn dfs_recursive_sub<T: Cell + Clone>(pos: &Coordinate, map: &(dyn MapOperation<T> + 'static), adj: &Adjacency,
            seen: &mut Map<bool>) -> usize {
        seen.set(pos, true);
        let mut res = 1;
        for next in &adj.get(pos) {
            if !map.contains_key(next) { continue; }
            if map.is_obstacle(next) || seen[next] { continue; }
            res += dfs_recursive_sub(next, map, adj, seen);
        }
        res
    }
    let mut seen = Map::new_with_fill(&map.coordinate_limit(), &false);
    dfs_recursive_sub(start, map, adj, &mut seen)
}

// dfs非再帰版
// startは必ず障害物でないことが保証されている
pub fn dfs_template<T: Cell + Clone>(start: &Coordinate, map: &(dyn MapOperation<T> + 'static), adj: &Adjacency)
        -> usize {
    let mut res = 0;
    let mut seen = Map::new_with_fill(&map.coordinate_limit(), &false);
    let mut todo = vec![start.clone()];
    while let Some(pos) = todo.pop() {
        if seen[&pos] { continue; }
        seen.set(&pos, true);
        res += 1;
        for next in &adj.get(&pos) {
            if map.contains_key(next) && map.is_empty(next) { todo.push(next.clone()); }
        }
    }
    res
}

// bfsでの(距離, dp復元用の1つ前の頂点)を求める
pub fn bfs_template<T: Cell + Clone>(start: &Coordinate, map: &(dyn MapOperation<T> + 'static), adj: &Adjacency)
        -> Map<(usize, Option<Coordinate>)> {
    let mut res =
        Map::new_with_fill(&map.coordinate_limit(), &(INF, None));
    res[start] = (0, None);
    let mut todo: VecDeque<(usize, Coordinate, Option<Coordinate>)> = VecDeque::new();
    todo.push_front((0, start.clone(), None));
    let mut seen = Map::new_with_fill(&map.coordinate_limit(), &false);
    while let Some((dist, pos, pre)) = todo.pop_back() {
        if seen[&pos] { continue; }
        seen.set(&pos, true);
        res[&pos] = (dist, pre.clone());
        for next in &adj.get(&pos) {
            if !map.contains_key(next) { continue; }
            if map.is_obstacle(next) { continue; }
            todo.push_front((dist + 1, next.clone(), Some(pos.clone())));
        }
    }
    res
}


// Beginning of the other owner's code.
// lowlink by terry_u16
pub struct LowLink<'a, T: 'static> {
    map: &'a (dyn MapOperation<T> + 'static),
    used: Map<bool>,
    order: Map<usize>,
    low: Map<usize>,
    aps: HashSet<Coordinate>,
}
impl<'a, T: Cell + Clone + 'static> LowLink<'a, T> {
    pub fn calc_aps(start: &Coordinate, map: &'a (dyn MapOperation<T> + 'static),
            adj: &Adjacency) -> HashSet<Coordinate> {
        let used = Map::new_with_fill(&map.coordinate_limit(), &false);
        let order = Map::new_with_fill(&map.coordinate_limit(), &0);
        let low = Map::new_with_fill(&map.coordinate_limit(), &0);
        let aps = HashSet::default();
        let mut lowlink = Self { map, used, order, low, aps, };
        let k = 0;
        lowlink.dfs(start, k, None, adj);
        lowlink.aps
    }

    fn dfs(&mut self, c: &Coordinate, mut k: usize, parent: Option<&Coordinate>,
            adj: &Adjacency) -> usize {
        self.used.set(c, true);
        self.order[c] = k;
        k += 1;
        self.low[c] = self.order[c];
        let mut is_aps = false;
        let mut count = 0;

        for next in &adj.get(c) {
            // 空白でないなら、スキップする
            if !self.map.contains_key(next) { continue; }
            if self.map.is_obstacle(next) { continue; }

            if !self.used[next] {
                count += 1;
                k = self.dfs(next, k, Some(c), adj);
                self.low[c] = self.low[c].min(self.low[next]);
                if parent.is_some() && self.order[c] <= self.low[next] { is_aps = true; }
            } else if parent.is_none() || next != parent.unwrap() {
                self.low[c] = self.low[c].min(self.order[next]);
            }
        }
        if parent == None && count >= 2 { is_aps = true; }
        if is_aps { self.aps.insert(c.clone()); }
        k
    }
}
// End of the other owner's code.


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_matrix2d() {
        let coordinate_limit = coord!(2, 3);
        let mut map = Map::new_with_fill(&coordinate_limit, &DefaultCell::Empty);
        map[&coord!(0, 1)] = DefaultCell::Obstacle;
        let start = coord!(0, 0);
        let adj = Adjacency::new_d2dir4();
        let res = bfs_template(&start, &map, &adj);
        assert_eq!(res.data, vec![(0, None), (INF, None), (4, Some(coord!(1, 2))),
            (1, Some(coord!(0, 0))), (2, Some(coord!(1, 0))), (3, Some(coord!(1, 1)))]);
    }

    #[test]
    fn test_basic_grid() {
        let map: Map<bool> = Map::new(&coord!(7));
        let start = coord!(0);
        let edges =
            vec![(0, 1, 1), (0, 2, 2), (1, 4, 5), (2, 4, 5), (2, 6, 1), (4, 6, 2), (4, 5, 1)];
        let adj = Adjacency::new_undirected_with_cost(&edges);
        let res = bfs_template(&start, &map, &adj);
        assert_eq!(res.data, vec![(0, None), (1, Some(coord!(0))), (1, Some(coord!(0))),
            (INF, None), (2, Some(coord!(2))), (3, Some(coord!(4))), (2, Some(coord!(2)))]);
        let res = dijkstra_template(&start, &map, &adj);
        assert_eq!(res.data, vec![(0, None), (1, Some(coord!(0))), (2, Some(coord!(0))),
            (INF, None), (5, Some(coord!(6))), (6, Some(coord!(4))), (3, Some(coord!(2)))]);
    }

    #[test]
    fn test_lowlink() {
        let map: Map<bool> = Map::new(&coord!(7));
        let start = coord!(0);
        let edges =
            vec![(0, 1), (0, 2), (1, 4), (2, 4), (2, 6), (4, 6), (4, 5)];
        let adj = Adjacency::new_undirected(&edges);
        let res = LowLink::calc_aps(&start, &map, &adj);
        assert_eq!(res, HashSet::from_iter(vec![coord!(4)]));
    }

    #[test]
    fn test_dfs_benchmark() {
        let coordinate_limit = coord!(2000, 2000);  // 0.6s
        let mut map = Map::new_with_fill(&coordinate_limit, &false);
        map.set(&coord!(0, 1), true);
        let start = coord!(0, 0);
        let adj = Adjacency::new_d2dir4();
        let res = dfs_template(&start, &map, &adj);
        assert_eq!(res, coordinate_limit.norm() - 1);
    }

    #[test]
    fn test_dfs_recursive_benchmark() {
        // 環境変数 RUST_MIN_STACK を大きくしておく必要がある
        let coordinate_limit = coord!(2000, 2000);  // 1.7s
        let mut map = Map::new_with_fill(&coordinate_limit, &false);
        map.set(&coord!(0, 1), true);
        let start = coord!(0, 0);
        let adj = Adjacency::new_d2dir4();
        let res = dfs_recursive_template(&start, &map, &adj);
        assert_eq!(res, coordinate_limit.norm() - 1);
    }
}
