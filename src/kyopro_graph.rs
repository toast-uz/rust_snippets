// todo! differentiate bfs
// todo! differentiate dijkstra
// todo! 結節点判定（切らない・つながない） 3*3、3x3x3
// https://x.com/chokudai/status/1706124817915908481?s=20

#![allow(dead_code)]

use std::collections::VecDeque;
use itertools::{Itertools, iproduct};
use rustc_hash::{FxHashSet as HashSet, FxHashMap as HashMap};
use fixedbitset::FixedBitSet;

use crate::heapmap::*;
use crate::xorshift_rand::*;

const INF: usize = 1e18 as usize;

// Map グラフの頂点集合

#[derive(Clone, Debug, Default)]
pub struct Map<T> {
    pub n: usize,
    pub coordinate_limit: Coordinate,
    pub data: Vec<T>,   // 空の場合を許容する
}

impl<T: Clone> Map<T> {
    pub fn new(coordinate_limit: &Coordinate) -> Self {
        Self { n: coordinate_limit.size(), coordinate_limit: coordinate_limit.clone(), data: Vec::new() }
    }
    pub fn new_with_fill(coordinate_limit: &Coordinate, fill: &T) -> Self {
        let n = coordinate_limit.size();
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


#[derive(Clone, Debug, Default)]
pub struct BitMap {
    pub coordinate_limit: Coordinate,
    pub data: FixedBitSet,
}

impl BitMap {
    pub fn new(coordinate_limit: &Coordinate) -> Self {
        let n = coordinate_limit.size();
        Self { coordinate_limit: coordinate_limit.clone(), data: FixedBitSet::with_capacity(n) }
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
        if self.data.contains(p) { Some(&true) } else { Some(&false) }
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

// Zobrist Hash

pub trait ZobristHash {
    fn new_zobrist_hash_seed(&self) -> Map<Vec<u64>>;
    fn zobrist_hash(&self, seed: &Map<Vec<u64>>) -> u64;
    fn zobrist_hash_diff<T>(&self, c: &Coordinate, x: &T, seed: &Map<Vec<u64>>)
        -> u64 where T: Cell + VariantCount;
}

impl<T: Cell + VariantCount> ZobristHash for Map<T> {
    fn new_zobrist_hash_seed(&self) -> Map<Vec<u64>> {
        let mut rng = xorshift_rng();
        let mut res = Map::new(&self.coordinate_limit());
        res.data = (0..self.coordinate_limit().size())
            .map(|_| (0..T::VARIANT_COUNT).map(|_| rng.gen_u64()).collect_vec())
            .collect();
        res
    }
    fn zobrist_hash(&self, seed: &Map<Vec<u64>>) -> u64 {
        let mut res = 0;
        for p in 0..self.len() {
            res ^= seed[&seed.p2c(p)][self[&self.p2c(p)].variant_id()];
        }
        res
    }
    fn zobrist_hash_diff<S>(&self, c: &Coordinate, x: &S, seed: &Map<Vec<u64>>) -> u64
            where S: Cell + VariantCount {
        let mut res = 0;
        res ^= seed[&c][self[&c].variant_id()];
        res ^= seed[&c][x.variant_id()];
        res
    }
}

// Cell グラフの頂点の種類

pub trait Cell: Clone {
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

// VariantCount trait for enum

pub trait VariantCount {
    const VARIANT_COUNT: usize;
    fn variant_id(&self) -> usize;
}

impl VariantCount for bool {
    const VARIANT_COUNT: usize = 2;
    fn variant_id(&self) -> usize { *self as usize }
}

impl VariantCount for DefaultCell {
    const VARIANT_COUNT: usize = 3;
    fn variant_id(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Obstacle => 1,
            Self::Other => 2,
        }
    }
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
    pub fn size(&self) -> usize {   // 負数にも対応するため、isizeに変換してから計算する
        match self {
            Self::D1(x) => (*x as isize).abs() as usize,
            Self::D2 { i, j } => (*i as isize).abs() as usize * (*j as isize).abs() as usize,
            Self::D3 { x, y, z } => (*x as isize).abs() as usize * (*y as isize).abs() as usize * (*z as isize).abs() as usize,
        }
    }
    pub fn norm1(&self) -> usize {   // 負数にも対応するため、isizeに変換してから計算する
        match self {
            Self::D1(x) => (*x as isize).abs() as usize,
            Self::D2 { i, j } => (*i as isize).abs() as usize + (*j as isize).abs() as usize,
            Self::D3 { x, y, z } => (*x as isize).abs() as usize + (*y as isize).abs() as usize + (*z as isize).abs() as usize,
        }
    }
    pub fn norm2(&self) -> usize {   // 負数にも対応するため、isizeに変換してから計算する
        match self {
            Self::D1(x) => (*x as isize).pow(2) as usize,
            Self::D2 { i, j } => (*i as isize).pow(2) as usize + (*j as isize).pow(2) as usize,
            Self::D3 { x, y, z } => (*x as isize).pow(2) as usize + (*y as isize).pow(2) as usize + (*z as isize).pow(2) as usize,
        }
    }
    pub fn elms(&self) -> Vec<usize> {
        match self {
            Self::D1(x) => vec![*x],
            Self::D2 { i, j } => vec![*i, *j],
            Self::D3 { x, y, z } => vec![*x, *y, *z],
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
    pub fn iter(&self) -> CoordinateIter {
        let len = self.size();
        CoordinateIter { coordinate_limit: self.clone(), len, pos: 0 }
    }
}

impl Default for Coordinate {
    fn default() -> Self { Self::D1(0) }
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

impl std::ops::Sub for Coordinate {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs.invert()
    }
}

// selfをcoordinate_limitとして、selfの範囲の座標を生成するイテレータ
pub struct CoordinateIter {
    coordinate_limit: Coordinate,
    len: usize,
    pos: usize,
}

impl Iterator for CoordinateIter {
    type Item = Coordinate;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len { return None; }
        let res = self.pos;
        self.pos += 1;
        Some(self.coordinate_limit.p2c(res))
    }
}

impl IntoIterator for Coordinate {
    type Item = Coordinate;
    type IntoIter = CoordinateIter;
    fn into_iter(self) -> Self::IntoIter {
        let len = self.size();
        CoordinateIter { coordinate_limit: self, len, pos: 0 }
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

// 2次元マップの表示
pub fn repr_2d_map<T: Clone>(map: &Map<T>, f: &dyn Fn(&T) -> String, sep: &str) -> String {
    let mut res = Vec::new();
    let coordinate_limit = map.coordinate_limit.elms();
    let (height, width) = (coordinate_limit[0], coordinate_limit[1]);
    for i in 0..height {
        let mut row = Vec::new();
        for j in 0..width { row.push(f(&map[&coord!{i, j}])); }
        res.push(row.iter().join(sep));
    }
    res.iter().join("\n")
}


// Adjacency 隣接関係

type HashedAdjacency = HashMap<Coordinate, HashSet<(Coordinate, isize)>>;

#[derive(Clone, Debug)]
pub enum Adjacency {
    D2dir4 { dir: Vec<Coordinate>, name: Vec<&'static str> },
    D2dir8 { dir: Vec<Coordinate>, name: Vec<&'static str> },
    D3dir6 { dir: Vec<Coordinate>, name: Vec<&'static str> },
    D2dir4wall { dir: Vec<Coordinate>, name: Vec<&'static str>, h: Vec<Vec<bool>>, v: Vec<Vec<bool>> },
    UnDirected { adj: HashedAdjacency },
    Directed { adj: HashedAdjacency },
}

impl Adjacency {
    pub fn new_d2dir4() -> Self {
        Self::D2dir4 {
            dir: vec![coord!(!0, 0), coord!(0, 1), coord!(1, 0), coord!(0, !0)],
            name: vec!["U", "R", "D", "L"],
        }
    }
    pub fn new_d2dir8() -> Self {
        Self::D2dir8 {
            dir: vec![coord!(!0, 0), coord!(!0, 1), coord!(0, 1), coord!(1, 1),
                coord!(1, 0), coord!(1, !0), coord!(0, !0), coord!(!0, !0)],
            name: vec!["U", "UR", "R", "DR", "D", "DL", "L", "UL"],
        }
    }
    pub fn new_d3dir6() -> Self {
        Self::D3dir6 {
            dir: vec![coord!(1, 0, 0), coord!(!0, 0, 0), coord!(0, 1, 0),
                coord!(0, !0, 0), coord!(0, 0, 1), coord!(0, 0, !0)],
            name: vec!["F", "B", "R", "L", "U", "D"],
        }
    }
    pub fn new_d2dir4wall(h: &[Vec<char>], v: &[Vec<char>]) -> Self {
        Self::D2dir4wall {
            dir: vec![coord!(!0, 0), coord!(0, 1), coord!(1, 0), coord!(0, !0)],
            name: vec!["U", "R", "D", "L"],
            h: h.iter().map(|row| row.iter().map(|&c| c == '1').collect()).collect(),
            v: v.iter().map(|row| row.iter().map(|&c| c == '1').collect()).collect(),
        }
    }
    pub fn new_undirected(edges: &[(Coordinate, Coordinate)]) -> Self {
        let mut adj: HashedAdjacency = HashMap::default();
        for (u, v) in edges {
            adj.entry(u.clone()).or_default().insert((v.clone(), 1));
            adj.entry(v.clone()).or_default().insert((u.clone(), 1));
        }
        Self::UnDirected { adj }
    }
    pub fn new_undirected_with_cost(edges: &[(Coordinate, Coordinate, isize)]) -> Self {
        let mut adj: HashedAdjacency = HashMap::default();
        for (u, v, c) in edges {
            adj.entry(u.clone()).or_default().insert((v.clone(), *c));
            adj.entry(v.clone()).or_default().insert((u.clone(), *c));
        }
        Self::UnDirected { adj }
    }
    pub fn new_directed(edges: &[(Coordinate, Coordinate)]) -> Self {
        let mut adj: HashedAdjacency = HashMap::default();
        for (u, v) in edges {
            adj.entry(u.clone()).or_default().insert((v.clone(), 1));
        }
        Self::Directed { adj }
    }
    pub fn new_directed_with_cost(edges: &[(Coordinate, Coordinate, isize)]) -> Self {
        let mut adj: HashedAdjacency = HashMap::default();
        for (u, v, c) in edges {
            adj.entry(u.clone()).or_default().insert((v.clone(), *c));
        }
        Self::Directed { adj }
    }
    pub fn get(&self, c: &Coordinate) -> Vec<Coordinate> {
        match self {
            Self::D2dir4 { dir, .. }
            | Self::D2dir8 { dir, .. }
            | Self::D3dir6 { dir, .. } =>
                dir.iter().map(|d| c.clone() + d.clone()).collect(),
            Self::D2dir4wall { dir, name, h, v } => {
                let mut res = Vec::new();
                let Coordinate::D2{i, j} = *c else { panic!() };
                for (d, name) in dir.iter().zip(name.iter()) {
                    let next = c.clone() + d.clone();
                    match name {
                        &"U" => if i > 0 && !h[i - 1][j] { res.push(next); },
                        &"L" => if j > 0 && !v[i][j - 1] { res.push(next); },
                        &"D" => if i < h.len() && !h[i][j] { res.push(next); },
                        &"R" => if j < v[i].len() && !v[i][j] { res.push(next); },
                    _ => unreachable!(),
                    }
                }
                res
            },
            Self::UnDirected { adj }
            | Self::Directed { adj } => {
                let Some(vs) = adj.get(c) else { return Vec::new() };
                vs.iter().map(|(v, _)| v.clone()).collect()
            },
        }
    }
    pub fn get_with_cost(&self, c: &Coordinate) -> Vec<(Coordinate, isize)> {
        match self {
            Self::D2dir4 { dir, .. }
            | Self::D2dir8 { dir, .. }
            | Self::D3dir6 { dir, .. } =>
                dir.iter().map(|d| (c.clone() + d.clone(), 1)).collect(),
            Self::UnDirected { adj }
            | Self::Directed { adj } => {
                    let Some(vs) = adj.get(c) else { return Vec::new() };
                vs.iter().map(|(v, cost)| (v.clone(), *cost)).collect()
            },
            _ => unreachable!(),
        }
    }
    pub fn dir2name(&self, dir_: &Coordinate) -> String {
        match self {
            Self::D2dir4 { dir, name } |
                Self::D3dir6 { dir, name } |
                Self::D2dir8 { dir, name } |
                Self::D2dir4wall { dir, name, h: _, v: _ } => {
                    let i = dir.iter().position(|d| d == dir_).unwrap();
                    name[i].to_string()
            },
            _ => panic!("cannot convert direction to name"),
        }
    }
}

impl Default for Adjacency {
    fn default() -> Self { Self::new_d2dir4() }
}

// グラフ分析に関するユーティリティ関数群

type StaticMap<T> = dyn MapOperation<T> + 'static;

// ダイクストラ法での(距離, dp復元用の1つ前の頂点)を求める
pub fn dijkstra_template<T: Cell>(start: &Coordinate, map: &StaticMap<T>, adj: &Adjacency)
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

// 必須頂点のリストが与えられている時に、Steiner Treeを求める
// プリム法での近似解を返す
// 必須頂点をシャッフルすると結果が変わるため、何度も試すことで、最適解に近づきやすくなる
// 結果はAdjacencyのnew_with_costで使える形式で返す
pub fn steiner_tree<T: Cell>(terminals: &[Coordinate], map: &StaticMap<T>,
        adj: &Adjacency) -> Vec<(Coordinate, Coordinate, isize)> {
    if terminals.len() <= 1 { return Vec::new(); }
    let mut res = Vec::new();
    // 必須頂点間の距離と経路を求める
    let dists = terminals.iter().map(|start|
        dijkstra_template(start, map, adj)).collect_vec();
    // terminalをすべて使うまで繰り返す
    let mut used_terminal_ids = HashSet::from_iter(vec![0]);
    let mut dist_from_tree = vec![(INF, None); terminals.len()];
    let mut added_tree = HashSet::default();
    // ターミナルの1つを選び、木に含める
    added_tree.insert(terminals[0].clone());
    while used_terminal_ids.len() < terminals.len() {
        // 木から最小コストでつなげられるターミナルと接続点を選ぶ
        let terminal_ids = (0..terminals.len())
            .filter(|&i| !used_terminal_ids.contains(&i))
            .collect_vec();
        for (&terminal_id, pos) in iproduct!(&terminal_ids, &added_tree) {
            let dist = dists[terminal_id][pos].0;
            if dist < dist_from_tree[terminal_id].0 {
                dist_from_tree[terminal_id] = (dist, Some(pos.clone()));
            }
        }
        let Some((terminal_id, _, pos)) = dist_from_tree
            .iter().enumerate()
            .filter(|&(terminal_id, (_, pos))|
                !used_terminal_ids.contains(&terminal_id) && pos.is_some())
            .map(|(terminal_id, (d, pos))| (terminal_id, d, pos.clone().unwrap()))
            .min_by_key(|(_, &d, _)| d) else { break; };
        added_tree.clear();
        added_tree.insert(terminals[terminal_id].clone());
        used_terminal_ids.insert(terminal_id);
        // 接続点からターミナルまでの経路を求める
        let mut cur = pos;
        while let (d, Some(pre)) = &dists[terminal_id][&cur] {
            added_tree.insert(cur.clone());
            let d = *d as isize - dists[terminal_id][pre].0 as isize;
            res.push((cur.clone(), pre.clone(), d));
            cur = pre.clone();
        }
    }
    res
}

// dfsで連結サイスを求める（再帰版、遅いため基本的には非再帰の方を使う）
// startは必ず障害物でないことが保証されている
pub fn dfs_recursive_template<T: Cell>(start: &Coordinate, map: &StaticMap<T>, adj: &Adjacency)
        -> usize {
    pub fn dfs_recursive_sub<T: Cell>(pos: &Coordinate, map: &StaticMap<T>, adj: &Adjacency,
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
pub fn dfs_template<T: Cell>(start: &Coordinate, map: &StaticMap<T>, adj: &Adjacency)
        -> usize {
    let mut res = 0;
    let mut seen = BitMap::new(&map.coordinate_limit());
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
pub fn bfs_template<T: Cell>(start: &Coordinate, map: &StaticMap<T>, adj: &Adjacency)
        -> Map<(usize, Option<Coordinate>)> {
    let mut res =
        Map::new_with_fill(&map.coordinate_limit(), &(INF, None));
    res[start] = (0, None);
    let mut todo: VecDeque<(usize, Coordinate, Option<Coordinate>)> = VecDeque::new();
    todo.push_front((0, start.clone(), None));
    let mut seen = BitMap::new(&map.coordinate_limit());
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
    map: &'a StaticMap<T>,
    used: Map<bool>,
    order: Map<usize>,
    low: Map<usize>,
    aps: HashSet<Coordinate>,
}
impl<'a, T: Cell + 'static> LowLink<'a, T> {
    pub fn calc_aps(start: &Coordinate, map: &'a StaticMap<T>, adj: &Adjacency)
            -> HashSet<Coordinate> {
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


///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test -r kyopro_graph::test::dfs_benchmark    0.7 sec
// cargo test -r kyopro_graph::test::dfs_recursive_benchmark    1.4 sec

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_matrix2d() {
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
    fn basic_grid() {
        let map: Map<bool> = Map::new(&coord!(7));
        let start = coord!(0);
        let edges =
            vec![(0, 1, 1), (0, 2, 2), (1, 4, 5), (2, 4, 5), (2, 6, 1), (4, 6, 2), (4, 5, 1)];
        let edges = edges.iter().map(|&(u, v, c)| (coord!(u), coord!(v), c)).collect_vec();
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
        let edges = edges.iter().map(|&(u, v)| (coord!(u), coord!(v))).collect_vec();
        let adj = Adjacency::new_undirected(&edges);
        let res = LowLink::calc_aps(&start, &map, &adj);
        assert_eq!(res, HashSet::from_iter(vec![coord!(4)]));
    }

    #[test]
    fn dfs_benchmark() {
        let coordinate_limit = coord!(2000, 2000);  // 0.6s
        let mut map = BitMap::new(&coordinate_limit);
        map.set(&coord!(0, 1), true);
        let start = coord!(0, 0);
        let adj = Adjacency::new_d2dir4();
        let res = dfs_template(&start, &map, &adj);
        assert_eq!(res, coordinate_limit.size() - 1);
    }

    #[test]
    fn dfs_recursive_benchmark() {
        // 環境変数 RUST_MIN_STACK を大きくしておく必要がある
        let coordinate_limit = coord!(2000, 2000);  // 1.7s
        let mut map = BitMap::new(&coordinate_limit);
        map.set(&coord!(0, 1), true);
        let start = coord!(0, 0);
        let adj = Adjacency::new_d2dir4();
        let res = dfs_recursive_template(&start, &map, &adj);
        assert_eq!(res, coordinate_limit.size() - 1);
    }

    #[test]
    fn test_steiner_tree() {
        let coordinate_limit = coord!(5, 5);
        let map = Map::new_with_fill(&coordinate_limit, &false);
        let adj = Adjacency::new_d2dir4();
        let mut terminals = vec![
            coord!(0, 0), coord!(0, 2), coord!(2, 4), coord!(4, 1)];
        let res = steiner_tree(&terminals, &map, &adj);
        assert_eq!(res.len(), 9);
        let mut rng = xorshift_rng();
        for _ in 0..100 {
            terminals.shuffle(&mut rng);
            let res = steiner_tree(&terminals, &map, &adj);
            let res_len = res.iter().map(|(_, _, c)| c).sum::<isize>();
            if terminals[0] == coord!(4, 1) && terminals[1] == coord!(0, 0) {
                assert_eq!(res_len, 11, "{:?}", terminals);   // 最適解ではない
            } else {
                assert_eq!(res_len, 9, "{:?}", terminals);    // 最適解
            }
        }
    }

    #[test]
    fn test_zobrist_hash() {
        let coordinate_limit = coord!(5, 5);
        let mut map = Map::new_with_fill(&coordinate_limit, &DefaultCell::Empty);
        let seed = map.new_zobrist_hash_seed();
        let mut rng = xorshift_rng();
        for _ in 0..1000 {
            for p in 0..map.len() {
                let c = map.p2c(p);
                let x = match rng.gen_range(0..3) {
                    0 => DefaultCell::Empty,
                    1 => DefaultCell::Obstacle,
                    _ => DefaultCell::Other,
                };
                map[&c] = x;
            }
            let hash = map.zobrist_hash(&seed);
            let c = coord!(rng.gen_range(0..5), rng.gen_range(0..5));
            let x = match rng.gen_range(0..3) {
                0 => DefaultCell::Empty,
                1 => DefaultCell::Obstacle,
                _ => DefaultCell::Other,
            };
            let hash_diff = map.zobrist_hash_diff(&c, &x, &seed);
            let same = map[&c] == x;
            map[&c] = x;
            let hash2 = map.zobrist_hash(&seed);
            if same {
                assert_eq!(hash, hash2);
            } else {
                assert_ne!(hash, hash2);
            }
            assert_eq!(hash ^ hash_diff, hash2);
        }
    }
}
