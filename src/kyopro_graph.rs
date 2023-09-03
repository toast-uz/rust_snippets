use std::collections::{BinaryHeap, VecDeque};
use std::hash::Hash;
use std::cmp::Reverse;
use rustc_hash::FxHashSet as HashSet;

const INF: usize = 1e18 as usize;

#[derive(Debug, Clone, Default)]
struct Map<T> {
    n: usize,
    data: Vec<T>,
}

#[allow(dead_code)]
impl<T> Map<T>
    where T: Eq + Hash + Default + Clone + Cell {
    fn new_graph(n: usize) -> Self { Self { n, ..Default::default() } }
    fn new_grid<C: Coordinate>(coordinate_limit: &C) -> Self {
        let n = C::size(coordinate_limit);
        let mut data = vec![T::default(); n];
        for i in C::enum_outer_wall_ids(&coordinate_limit) { data[i] = T::wall(); }
        Self { n, data }
    }
    fn clone_with<S: Clone + Default>(&self, fill: &S) -> Map<S> {
        Map::<S> { n: self.n, data: vec![fill.clone(); self.n] }
    }
    #[inline]
    fn is_enterable(&self, i: usize) -> bool {
        let Some(t) = self.data.get(i) else { return true; };
        t.is_enterable()
    }
    #[inline]
    fn is_movable(&self, i: usize) -> bool {
        let Some(t) = self.data.get(i) else { return true; };
        t.is_movable()
    }
}

impl<T> std::ops::Index<usize> for Map<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &Self::Output { &self.data[i] }
}

impl<T> std::ops::IndexMut<usize> for Map<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut self.data[i] }
}

trait Cell {
    fn wall() -> Self;
    fn is_enterable(&self) -> bool;
    fn is_movable(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum DummyCell {
    #[default]
    None,
}

impl Cell for DummyCell {
    #[inline]
    fn wall() -> Self { Self::None }
    #[inline]
    fn is_enterable(&self) -> bool { true }
    #[inline]
    fn is_movable(&self) -> bool { true }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum BasicCell {
    #[default]
    Floor,
    Wall,
}

impl Cell for BasicCell {
    #[inline]
    fn wall() -> Self { Self::Wall }
    #[inline]
    fn is_enterable(&self) -> bool { *self != Self::Wall }
    #[inline]
    fn is_movable(&self) -> bool { *self != Self::Wall }
}

trait Coordinate<Rhs = Self> {
    type Output;
    fn size(coordinate_limit: &Rhs) -> usize;
    fn to_index(&self, coordinate_limit: &Rhs) -> usize;
    fn from_index(i: usize, coordinate_limit: &Rhs) -> Self::Output;
    fn enum_outer_wall_ids(coordinate_limit: &Rhs) -> HashSet<usize>;
}

#[derive(Debug, Clone, Default)]
pub struct Coordinate2D {
    pub x: usize,
    pub y: usize,
}

impl Coordinate for Coordinate2D {
    type Output = Self;
    #[inline]
    fn size(coordinate_limit: &Self) -> usize { (coordinate_limit.x + 2) * (coordinate_limit.y + 2) }
    #[inline]
    fn to_index(&self, coordinate_limit: &Self) -> usize {  // y, xの順
        self.y.wrapping_add(1) * (coordinate_limit.x + 2) + self.x.wrapping_add(1)
    }
    #[inline]
    fn from_index(i: usize, coordinate_limit: &Self) -> Self::Output {
        Self { x: (i % (coordinate_limit.x + 2)).wrapping_sub(1),
               y: (i / (coordinate_limit.x + 2)).wrapping_sub(1) }
    }
    fn enum_outer_wall_ids(coordinate_limit: &Self) -> HashSet<usize> {
        let mut res = HashSet::default();
        res.insert(Self { x: !0, y: !0 }.to_index(coordinate_limit));
        for x in 0..=coordinate_limit.x {
            res.insert(Self { x, y: !0 }.to_index(coordinate_limit));
            res.insert(Self { x, y: coordinate_limit.y }.to_index(coordinate_limit));
        }
        for y in 0..=coordinate_limit.y {
            res.insert(Self { x: !0, y }.to_index(coordinate_limit));
            res.insert(Self { x: coordinate_limit.x, y }.to_index(coordinate_limit));
        }
        res
    }
}

impl std::ops::Add for Coordinate2D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { x: self.x.wrapping_add(rhs.x), y: self.y.wrapping_add(rhs.y) }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Coordinate3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Coordinate for Coordinate3D {
    type Output = Self;
    #[inline]
    fn size(coordinate_limit: &Self) -> usize {
        (coordinate_limit.x + 2) * (coordinate_limit.y + 2) * (coordinate_limit.z + 2)
    }
    #[inline]
    fn to_index(&self, coordinate_limit: &Self) -> usize {  // z, y, xの順
        ((self.z.wrapping_add(1) * (coordinate_limit.y + 2))
            + self.y.wrapping_add(1)) * (coordinate_limit.x + 2) + self.x.wrapping_add(1)
    }
    #[inline]
    fn from_index(i: usize, coordinate_limit: &Self) -> Self::Output {
        Self {
            x: (i % (coordinate_limit.x + 2)).wrapping_sub(1),
            y: (i % (coordinate_limit.y + 2) / (coordinate_limit.x + 2)).wrapping_sub(1),
            z: (i / ((coordinate_limit.x + 2) * (coordinate_limit.y + 2))).wrapping_sub(1)
        }
    }
    fn enum_outer_wall_ids(coordinate_limit: &Self) -> HashSet<usize> {
        let mut res = HashSet::default();
        for x in 0..=coordinate_limit.x { for y in 0..=coordinate_limit.y {
            res.insert(Self { x, y, z: 0 }.to_index(coordinate_limit));
            res.insert(Self { x, y, z: coordinate_limit.z }.to_index(coordinate_limit));
        } }
        for y in 0..=coordinate_limit.y { for z in 0..=coordinate_limit.z {
            res.insert(Self { x: 0, y, z }.to_index(coordinate_limit));
            res.insert(Self { x: coordinate_limit.x, y, z }.to_index(coordinate_limit));
        } }
        for z in 0..=coordinate_limit.z { for x in 0..=coordinate_limit.x {
            res.insert(Self { x, y: 0, z }.to_index(coordinate_limit));
            res.insert(Self { x, y: coordinate_limit.y, z }.to_index(coordinate_limit));
        } }
        res
    }
}

impl std::ops::Add for Coordinate3D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { x: self.x.wrapping_add(rhs.x), y: self.y.wrapping_add(rhs.y), z: self.z.wrapping_add(rhs.z) }
    }
}

trait Adjacent {
    fn get(&self, i: usize) -> HashSet<usize>;
    fn get_with_cost(&self, i: usize) -> HashSet<(usize, isize)>;
}

struct Adjacent2D4dir {
    dir: Vec<usize>,
}

#[allow(dead_code)]
impl Adjacent2D4dir {
    fn new(coordinate_limit: &Coordinate2D) -> Self {
        Self { dir: vec![1, coordinate_limit.x + 2, !0, !0 - coordinate_limit.x - 1], }
    }
}

impl Adjacent for Adjacent2D4dir {
    fn get(&self, i: usize) -> HashSet<usize> {
        self.dir.iter().map(|&d| i.wrapping_add(d)).collect()
    }
    fn get_with_cost(&self, i: usize) -> HashSet<(usize, isize)> {
        self.dir.iter().map(|&d| (i.wrapping_add(d), 1)).collect()
    }
}

struct AdjacentUnDirected {
    adj: Vec<Vec<(usize, isize)>>,
}

#[allow(dead_code)]
impl AdjacentUnDirected {
    fn new(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj = vec![vec![]; n];
        for &(a, b) in edges {
            adj[a].push((b, 1));
            adj[b].push((a, 1));
        }
        Self { adj }
    }
    fn new_with_cost(n: usize, edges: &[(usize, usize, isize)]) -> Self {
        let mut adj = vec![vec![]; n];
        for &(a, b, c) in edges {
            adj[a].push((b, c));
            adj[b].push((a, c));
        }
        Self { adj }
    }
}

impl Adjacent for AdjacentUnDirected {
    fn get(&self, i: usize) -> HashSet<usize> {
        self.adj[i].iter().map(|&(j, _)| j).collect()
    }
    fn get_with_cost(&self, i: usize) -> HashSet<(usize, isize)> {
        self.adj[i].iter().map(|&(j, c)| (j, c)).collect()
    }
}

// ダイクストラ法での(距離, dp復元用の1つ前の頂点)を求める
#[allow(dead_code)]
fn dijkstra<T, A>(start: usize, map: &Map<T>, adj: &A) -> Map<(usize, Option<usize>)>
        where T: Eq + Hash + Default + Clone + Cell, A: Adjacent {
    let mut res = map.clone_with(&(INF, None));
    res[start] = (0, None);
    let mut heapq = BinaryHeap::new();
    heapq.push((Reverse(0), start));
    while let Some((Reverse(d), pos)) = heapq.pop() {
        if d != res[pos].0 { continue; }
        if !map.is_movable(pos) { continue; }
        for &(next, cost) in &adj.get_with_cost(pos) {
            if !map.is_enterable(next) { continue; }
            let next_d = d + cost as usize;
            if next_d < res[next].0 {
                heapq.push((Reverse(next_d), next));
                res[next] = (next_d, Some(pos));
            }
        }
    }
    res
}

// bfsでの(距離, dp復元用の1つ前の頂点)を求める
#[allow(dead_code)]
fn bfs<T, A>(start: usize, map: &Map<T>, adj: &A) -> Map<(usize, Option<usize>)>
        where T: Eq + Hash + Default + Clone + Cell, A: Adjacent {
    let mut res = map.clone_with(&(INF, None));
    res[start] = (0, None);
    let mut todo = VecDeque::new();
    todo.push_front((0, start, None));
    let mut seen = map.clone_with(&false);
    while let Some((dist, pos, pre)) = todo.pop_back() {
        if seen[pos] { continue; }
        seen[pos] = true;
        res[pos] = (dist, pre);
        if !map.is_movable(pos) { continue; }
        for &next in &adj.get(pos) {
            if !map.is_enterable(next) { continue; }
            todo.push_front((dist + 1, next, Some(pos)));
        }
    }
    res
}

fn main () {
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_basic_matrix2d() {
        let coordinate_limit = Coordinate2D { x: 3, y: 2 };
        let mut map = Map::<BasicCell>::new_grid(&coordinate_limit);
        map[Coordinate2D { x: 1, y: 0 }.to_index(&coordinate_limit)] = BasicCell::Wall;
        let start = Coordinate2D { x: 0, y: 0 }.to_index(&coordinate_limit);
        let adj = Adjacent2D4dir::new(&coordinate_limit);
        let res = bfs(start, &map, &adj);
        eprint!("{:?}\n", map.data);
        eprint!("{:?}\n", res.data);
        assert!(false);
    }

    #[test]
    fn test_basic_grid() {
        let map = Map::<DummyCell>::new_graph(6);
        let start = 0;
        let edges = vec![(0, 1, 1), (0, 2, 3), (1, 4, 4), (2, 4, 1), (4, 5, 1)];
        let adj = AdjacentUnDirected::new_with_cost(map.n, &edges);
        let res = bfs(start, &map, &adj);
        eprint!("{:?}\n", res.data);
        let res = dijkstra(start, &map, &adj);
        eprint!("{:?}\n", res.data);
        assert!(false);
    }
}