use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp;

const INF: usize = 1e18 as usize;

fn main () {
    const N: usize = 5;
    for n in 0..N {
        let mut counter = 0;
        for i in 0..(n + 2) {
            println!("i: {}", i);
            counter += 1;
            if counter > 10 { break; }
            println!("n - i: {}", n - i);
            let x = std::cmp::max(0, n - i);
            println!("x: {}", x);
        }
    }
}

// ダイクストラ法での(距離, dp復元用の1つ前の頂点)を求める
fn dijkstra(start: usize, n: usize, adj: &HashMap<usize, HashSet<(usize, isize)>>)
        -> Vec<(usize, Option<usize>)> {
    let mut res = vec![(INF, None); n];
    res[start] = (0, None);
    let mut heapq = BinaryHeap::new();
    heapq.push((cmp::Reverse(0), start));
    while let Some((cmp::Reverse(d), pos)) = heapq.pop() {
        if d != res[pos].0 { continue; }
        if let Some(pos_list) = adj.get(&pos) {
            for &(next, cost) in pos_list {
                let next_d = d + cost as usize;
                if next_d < res[next].0 {
                    heapq.push((cmp::Reverse(next_d), next));
                    res[next] = (next_d, Some(pos));
                }
            }
        }
    }
    res
}

// bfsでの(距離, dp復元用の1つ前の頂点)を求める
fn bfs(start: usize, n: usize, adj: &HashMap<usize, HashSet<usize>>)
        -> Vec<(usize, Option<usize>)> {
    let mut res = vec![(INF, None); n];
    res[start] = (0, None);
    let mut todo = VecDeque::new();
    todo.push_front((0, start, None));
    let mut seen = vec![false; n];
    while let Some((dist, pos, pre)) = todo.pop_back() {
        if seen[pos] { continue; }
        seen[pos] = true;
        res[pos] = (dist + 1, pre);
        if let Some(pos_list) = adj.get(&pos) {
            for &next in pos_list {
                todo.push_front((dist + 1, next, Some(pos)));
            }
        }
    }
    res
}
