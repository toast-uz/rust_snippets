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
