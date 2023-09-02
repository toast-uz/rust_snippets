/*
xorshiftのrand
Refer https://zenn.dev/karamawanu/articles/5769c3c27eaea3
Refer https://www.cepstrum.co.jp/hobby/xorshift/xorshift.html

Usage

let mut rng = xorshift_rand::xorshift_rng();
(let mut rng = xorshift_rand::XorshiftRng::from_seed(seed);)
let random_range = rng.gen_range(0..10);    // [0, 10)のusizeの一様乱数
let random_range = rng.gen_range(0..=10);   // [0, 10]のusizeの一様乱数
let random_uniform = rng.gen();             // [0, 1]のf64の一様乱数
*/

#[allow(dead_code)]
mod xorshift_rand {
    use std::time::Instant;
    use rustc_hash::FxHashSet;

    pub fn xorshift_rng() -> XorshiftRng {
        XorshiftRng::from_seed(Instant::now().elapsed().as_nanos() as u64)
    }
    pub struct XorshiftRng { seed: u64, }

    impl XorshiftRng {
        pub fn from_seed(seed: u64) -> Self { Self { seed, } }
        fn _xorshift(&mut self) {
            self.seed ^= self.seed << 3;
            self.seed ^= self.seed >> 35;
            self.seed ^= self.seed << 14;
        }
        // [low, high) の範囲のusizeの乱数を求める
        pub fn gen_range<R: std::ops::RangeBounds<usize>>(&mut self, range: R) -> usize {
            let (start, end) = Self::unsafe_decode_range_(&range);
            self._xorshift();
            (start as u64 + self.seed % (end - start) as u64) as usize
        }
        // [low, high) の範囲から重複なくm個のusizeの乱数を求める
        pub fn gen_range_multiple<R: std::ops::RangeBounds<usize>>(&mut self, range: R, m: usize) -> Vec<usize> {
            let (start, end) = Self::unsafe_decode_range_(&range);
            assert!(m <= end - start);
            let many = m > (end - start) / 2; // mが半分より大きいか
            let n = if many { end - start - m } else { m };
            let mut res = FxHashSet::default();
            while res.len() < n {   // 半分より小さい方の数をランダムに選ぶ
                self._xorshift();
                let x = (start as u64 + self.seed % (end - start) as u64) as usize;
                res.insert(x);
            }
            (start..end).filter(|&x| many ^ res.contains(&x)).collect()
        }
        // rangeをもとに半開区間の範囲[start, end)を求める
        fn unsafe_decode_range_<R: std::ops::RangeBounds<usize>>(range: &R) -> (usize, usize) {
            let std::ops::Bound::Included(&start) = range.start_bound() else { panic!(); };
            let end = match range.end_bound() {
                std::ops::Bound::Included(&x) => x + 1,
                std::ops::Bound::Excluded(&x) => x,
                _ => panic!(),
            };
            (start, end)
        }
        // [0, 1] の範囲のf64の乱数を求める
        pub fn gen(&mut self) -> f64 {
            self._xorshift();
            self.seed as f64 / u64::MAX as f64
        }
    }

    pub trait SliceXorshiftRandom<T> {
        fn choose(&self, rng: &mut XorshiftRng) -> T;
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T>;
        fn shuffle(&mut self, rng: &mut XorshiftRng);
    }

    impl<T: Clone> SliceXorshiftRandom<T> for [T] {
        fn choose(&self, rng: &mut XorshiftRng) -> T {
            let x = rng.gen_range(0..self.len());
            self[x].clone()
        }
        fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T> {
            let selected = rng.gen_range_multiple(0..self.len(), m);
            selected.iter().map(|&i| self[i].clone()).collect()
        }
        fn shuffle(&mut self, rng: &mut XorshiftRng) {
            for i in (1..self.len()).rev() {
                let x = rng.gen_range(0..=i);
                self.swap(i, x);
            }
        }
    }
}


// make binary date for https://github.com/dj-on-github/sp800_22_tests
// Refer https://tech.ateruimashin.com/2020/03/tools3/
/*
monobit_test                             0.4726615207621658 PASS
frequency_within_block_test              0.5703095540643596 PASS
runs_test                                0.8965676419009525 PASS
longest_run_ones_in_a_block_test         0.7592006906669653 PASS
binary_matrix_rank_test                  0.2049247626724776 PASS
dft_test                                 0.7896399427236418 PASS
non_overlapping_template_matching_test   0.9944184973839666 PASS
overlapping_template_matching_test       0.07805722693233187 PASS
maurers_universal_test                   0.6038796999164997 PASS
linear_complexity_test                   0.5031469546219416 PASS
serial_test                              0.8624576869293109 PASS
approximate_entropy_test                 0.8625398425658882 PASS
cumulative_sums_test                     0.2824631370718771 PASS
random_excursion_test                    0.21335520833619928 PASS
random_excursion_variant_test            0.07994960739002104 PASS
*/
use std::fs::File;
use std::io::Write;
use crate::xorshift_rand::XorshiftRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = XorshiftRng::from_seed(998244353);
    let mut file = File::create("./tools/sp800_22_tests/xorshift.bin")?;
    let buf: Vec<u8> = (0..(1024*1024)).map(|_| rng.gen_range(0..256) as u8).collect();
    file.write_all(&buf)?;
    file.flush()?;
    Ok(())
}

///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test benchmark1 --bin rand --release   1.0 sec（1億回）世間一般のrandクレート
// cargo test benchmark2 --bin rand --release   2.3 sec（10億回）今回作成のxorshift


#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use itertools::Itertools;
    use crate::xorshift_rand::XorshiftRng;

    use super::*;

    #[test]
    fn basic() {
        let mut rng = xorshift_rand::xorshift_rng();
        for _ in 0..1000 {
            let random_range1 = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range1 && random_range1 < 10);
            let random_range2 = rng.gen_range(5..=10);   // [0, 10]のusizeの一様乱数
            assert!(5 <= random_range2 && random_range2 <= 10);
            let random_range3 = rng.gen_range_multiple(0..10, 3); // 3個
            assert!(random_range3.iter().max().cloned().unwrap() < 10);
            assert_eq!(random_range3.len(), 3);
            assert!(random_range3.iter().all_unique());
            let random_range4 = rng.gen_range_multiple(0..10, 6); // 6個
            assert!(random_range4.iter().max().cloned().unwrap() < 10);
            assert_eq!(random_range4.len(), 6);
            assert!(random_range4.iter().all_unique());
            let random_range5 = rng.gen_range_multiple(0..10, 10); // 6個
            assert_eq!(random_range5, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            let random_uniform = rng.gen();               // [0, 1]のf64の一様乱数
            assert!(0.0 <= random_uniform && random_uniform <= 1.0);
        }
        basic_sub(&mut rng);
    }

    // サブルーチン呼び出し形式が通常のrandと同等かどうかを見る
    fn basic_sub(rng: &mut XorshiftRng) {
        let random_range = rng.gen_range(0..10);
        assert!(random_range < 10);
    }

    #[test]
    fn benchmark1_public_rand() {
        let mut rng = SmallRng::from_entropy();
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
        let mut rng = xorshift_rand::xorshift_rng();
        let mut total = 0;
        for _ in 0..1_000_000_000 {
            let random_range = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range && random_range < 10);
            total += random_range;
        }
        assert!(total >= 5_000_000_000usize);
    }
}

