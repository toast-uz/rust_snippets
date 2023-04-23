/*
xorshiftのrand
Refer https://zenn.dev/karamawanu/articles/5769c3c27eaea3
Refer https://www.cepstrum.co.jp/hobby/xorshift/xorshift.html

Usage

let mut rng = Rand::new();
let random_range = rng.gen_range(0..10);    // [0, 10)のusizeの一様乱数
let random_range = rng.gen_range(0..=10);   // [0, 10]のusizeの一様乱数
let random_uniform = rng.gen();             // [0, 1]のf64の一様乱数
*/

use std::time::Instant;

struct Rand { seed: u64, }

#[allow(dead_code)]
impl Rand {
    fn new() -> Self { Self::new_from_seed(Instant::now().elapsed().as_nanos() as u64) }
    fn new_from_seed(seed: u64) -> Self { Self { seed, } }
    fn _xorshift(&mut self) {
        self.seed ^= self.seed << 3;
        self.seed ^= self.seed >> 35;
        self.seed ^= self.seed << 14;
    }
    // [low, high) の範囲のusizeの乱数を求める
    fn gen_range<R: std::ops::RangeBounds<usize>>(&mut self, range: R) -> usize {
        self._xorshift();
        let std::ops::Bound::Included(&start) = range.start_bound() else { panic!(); };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&x) => x + 1,
            std::ops::Bound::Excluded(&x) => x,
            _ => panic!(),
        };
        (start as u64 + self.seed % (end - start) as u64) as usize
    }
    // [0, 1] の範囲のf64の乱数を求める
    fn gen(&mut self) -> f64 {
        self._xorshift();
        self.seed as f64 / u64::MAX as f64
    }
}

fn main() {

}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test benchmark1 --bin rand --release   1.4 sec（1億回）世間一般のrandクレート
// cargo test benchmark2 --bin rand --release   2.3 sec（10億回）今回作成のxorshift


#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use super::*;

    #[test]
    fn basic() {
        let mut rng = Rand::new();
        for _ in 0..1000 {
            let random_range1 = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range1 && random_range1 < 10);
            let random_range2 = rng.gen_range(5..=10);   // [0, 10]のusizeの一様乱数
            assert!(5 <= random_range2 && random_range2 <= 10);
            let random_uniform = rng.gen();               // [0, 1]のf64の一様乱数
            assert!(0.0 <= random_uniform && random_uniform <= 1.0);
        }
    }

    #[test]
    fn benchmark1_public_rand() {
        let mut rng = rand::thread_rng();
        let mut total = 0;
        for _ in 0..100_000_000 {
            let random_range = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range && random_range < 10);
            total += random_range;
        }
        assert!(total >= 500_000_000usize);
    }

    #[test]
    fn benchmark2_my_rand() {
        let mut rng = Rand::new();
        let mut total = 0;
        for _ in 0..1_000_000_000 {
            let random_range = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range && random_range < 10);
            total += random_range;
        }
        assert!(total >= 5_000_000_000usize);
    }
}

