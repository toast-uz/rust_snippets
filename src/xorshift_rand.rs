/*
xorshiftのrand
Refer https://zenn.dev/karamawanu/articles/5769c3c27eaea3
Refer https://www.cepstrum.co.jp/hobby/xorshift/xorshift.html

Usage

let mut rng = xorshift_rand::xorshift_rng();
(let mut rng = xorshift_rand::XorshiftRng::from_seed(seed);)
let random_range = rng.gen_range(0..10);    // [0, 10)のusizeの一様乱数
let random_range = rng.gen_range(0..=10);   // [0, 10]のusizeの一様乱数
let random_range = rng.gen_range_multiple(0..10, 3); // 3個
let random_uniform = rng.gen();             // [0, 1]のf64の一様乱数
*/
#![allow(dead_code)]
use std::time::SystemTime;
use rustc_hash::FxHashSet as HashSet;
use crate::kyopro_stats::*;

pub fn xorshift_rng() -> XorshiftRng {
    let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
        .unwrap().as_secs() as u64;
    let mut rng = XorshiftRng::from_seed(seed);
    for _ in 0..100 { rng._xorshift(); }    // 初期値が偏らないようにウォーミングアップ
    rng
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
        assert!(start < end);
        self._xorshift();
        (start as u64 + self.seed % (end - start) as u64) as usize
    }
    // 重み付きで乱数を求める
    pub fn gen_range_weighted<R: std::ops::RangeBounds<usize>>(&mut self, range: R, weights: &[usize]) -> usize {
        let (start, end) = Self::unsafe_decode_range_(&range);
        assert_eq!(end - start, weights.len());
        let sum = weights.iter().sum::<usize>();
        let x = self.gen_range(0..sum);
        let mut acc = 0;
        for i in 0..weights.len() {
            acc += weights[i];
            if acc > x { return i; }
        }
        unreachable!()
    }
    // [low, high) の範囲から重複なくm個のusizeの乱数を求める
    pub fn gen_range_multiple<R: std::ops::RangeBounds<usize>>(&mut self, range: R, m: usize) -> Vec<usize> {
        let (start, end) = Self::unsafe_decode_range_(&range);
        assert!(m <= end - start);
        let many = m > (end - start) / 2; // mが半分より大きいか
        let n = if many { end - start - m } else { m };
        let mut res = HashSet::default();
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
    // 正規分布に従う乱数を求める
    pub fn gen_gaussian(&mut self, mu: f64, sigma: f64) -> f64 {
        norm::ppf(self.gen(), mu, sigma)
    }
    // u64の乱数を求める
    pub fn gen_u64(&mut self) -> u64 {
        self._xorshift();
        self.seed
    }
}

pub trait SliceXorshiftRandom<T> {
    fn choose(&self, rng: &mut XorshiftRng) -> T;
    fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T>;
    fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T;
    fn shuffle(&mut self, rng: &mut XorshiftRng);
}

impl<T: Clone> SliceXorshiftRandom<T> for [T] {
    fn choose(&self, rng: &mut XorshiftRng) -> T {
        let x = rng.gen_range(0..self.len());
        self[x].clone()
    }
    fn choose_weighted(&self, rng: &mut XorshiftRng, weights: &[usize]) -> T {
        let x = rng.gen_range_weighted(0..self.len(), weights);
        self[x].clone()
    }
    fn choose_multiple(&self, rng: &mut XorshiftRng, m: usize) -> Vec<T> {
        let selected = rng.gen_range_multiple(0..self.len(), m);
        selected.iter().map(|&i| self[i].clone()).collect()
    }
    fn shuffle(&mut self, rng: &mut XorshiftRng) {
        // Fisher-Yates shuffle
        for i in (1..self.len()).rev() {
            let x = rng.gen_range(0..=i);
            self.swap(i, x);
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
// cargo test -r xorshift_rand::tests::benchmark1    1.0 sec（1億回）世間一般のrandクレート
// cargo test -r xorshift_rand::tests::benchmark2    2.2 sec（10億回）今回作成のxorshift


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut rng = XorshiftRng::from_seed(998244353);
        let x = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
        assert_eq!(x, 5);
        let x = rng.gen_range(5..=10);   // [0, 10]のusizeの一様乱数
        assert_eq!(x, 9);
        let x = rng.gen_range_multiple(0..10, 3); // 3個
        assert_eq!(x, vec![0, 4, 7]);
        let x = rng.gen_range_multiple(0..10, 6); // 6個
        assert_eq!(x, vec![2, 3, 4, 5, 7, 9]);
        let x = rng.gen_range_multiple(0..10, 10); // 6個
        assert_eq!(x, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let x = rng.gen();               // [0, 1]のf64の一様乱数
        assert!(0.70 < x && x < 0.71);
        let x = rng.gen();
        assert!(0.17 < x && x < 0.18);
        let x = rng.gen_range_weighted(0..10,
            &[0, 0, 0, 10, 1, 2, 1, 0, 1, 0]);
        assert_eq!(x, 5);
        basic_sub(&mut rng);
    }

    // サブルーチン呼び出し形式が通常のrandと同等かどうかを見る
    fn basic_sub(rng: &mut XorshiftRng) {
        let random_range = rng.gen_range(0..10);
        assert!(random_range < 10);
    }

    #[test]
    fn test_slice() {
        let mut rng = XorshiftRng::from_seed(998244353);
        let a = vec![1, 2, 3, 4, 5];
        let x = a.choose(&mut rng);
        assert_eq!(x, 1);
        let x = a.choose_multiple(&mut rng, 3);
        assert_eq!(x, vec![2, 4, 5]);
        let x = a.choose_weighted(&mut rng, &[0, 0, 0, 10, 1]);
        assert_eq!(x, 4);
        let mut a = vec![1, 2, 3, 4, 5];
        a.shuffle(&mut rng);
        assert_eq!(a, vec![4, 3, 2, 1, 5]);
    }

    #[test]
    fn benchmark1_public_rand() {
        use rand::prelude::*;
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
        let mut rng = xorshift_rng();
        let mut total = 0;
        for _ in 0..1_000_000_000 {
            let random_range = rng.gen_range(5..10);    // [0, 10)のusizeの一様乱数
            assert!(5 <= random_range && random_range < 10);
            total += random_range;
        }
        assert!(total >= 5_000_000_000usize);
    }
}

