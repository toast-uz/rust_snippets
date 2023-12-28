#![allow(dead_code)]

const RESOLUTION: usize = 100;
const UPPER_BOUND: f64 = 5.0;
const DISCRETIZATION_SIZE: usize = (UPPER_BOUND * RESOLUTION as f64) as usize;

// Standard Normal distribution 標準正規分布（上側確率）
// https://unit.aist.go.jp/mcml/rg-orgp/uncertainty_lecture/normsdist.html
// Beginning of the other owner's code.
// https://x.com/toomerhs/status/1704861185747931569?s=20
const SND: [f64; DISCRETIZATION_SIZE + 1] = {
    let mut res = [0f64; DISCRETIZATION_SIZE + 1];
    let mut i = 0;
    while i <= DISCRETIZATION_SIZE {
        let z = i as f64 / 100. * std::f64::consts::FRAC_1_SQRT_2;
        let mut n = 1;
        let mut a = 0f64;
        let mut b = std::f64::consts::FRAC_2_SQRT_PI;
        while n <= DISCRETIZATION_SIZE {
            a += z / (2. * n as f64 - 1.) * b;
            b *= -z * z / n as f64;
            n += 1;
        }
        res[i] = (1.0 - a) * 0.5;
        i += 1;
    }
    res
};
// End of the other owner's code.

pub fn normal_distribution(z: f64, mu: f64, sigma: f64) -> f64 {
    let x = z.normalized(mu, sigma);
    if x > 0.0 { SND[x.to_index()] } else { 1.0 - SND[x.to_index()] }
}

// 標準正規分布（上側確率）の期待値
const SND_MEAN: [f64; DISCRETIZATION_SIZE + 1] = {
    let mut res = [0f64; DISCRETIZATION_SIZE + 1];
    res[DISCRETIZATION_SIZE] = DISCRETIZATION_SIZE as f64 * SND[DISCRETIZATION_SIZE];
    let mut i = DISCRETIZATION_SIZE;
    while i > 0 {
        res[i - 1] = res[i] + (SND[i - 1] - SND[i]) * (i as f64 - 0.5);
        i -= 1;
    }
    let mut i = 0;
    while i <= DISCRETIZATION_SIZE {
        res[i] /= RESOLUTION as f64 * SND[i];
        i += 1;
    }
    res
};

// 標準正規分布の逆関数(pとして0.0と1.0を許容する)
fn standard_normal_distribution_inverse(p: f64) -> f64 {
    if p > 0.5 { return -standard_normal_distribution_inverse(1.0 - p); }
    assert!(0.0 <= p && p <= 0.5, "p must be in [0, 1]");
    match SND.binary_search_by(|&x| p.partial_cmp(&x).unwrap()) {
        Ok(i) => i as f64 / RESOLUTION as f64,
        Err(i) => {
            if i == 0 { 0.0 }
            else if i >= DISCRETIZATION_SIZE { UPPER_BOUND }
            else {
                let l = (i - 1) as f64 / RESOLUTION as f64;
                let r = i as f64 / RESOLUTION as f64;
                let l_p = SND[i - 1];
                let r_p = SND[i];
                l + (r - l) * (p - l_p) / (r_p - l_p)
            }
        }
    }
}

pub fn normal_distribution_inverse(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * standard_normal_distribution_inverse(p)
}

trait F64 {
    fn normalized(&self, mu: f64, sigma: f64) -> f64;
    fn to_index(&self) -> usize;
}

impl F64 for f64 {
    fn normalized(&self, mu: f64, sigma: f64) -> f64 {
        (self - mu) / sigma
    }
    fn to_index(&self) -> usize {
        ((self * RESOLUTION as f64).abs().round() as usize)
            .clamp(0, DISCRETIZATION_SIZE)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_normal_distribution() {
        let eps = 1e-6;
        for (x, mu, sigma, ans) in [
            (0.0, 0.0, 1.0, 0.5),
            (1.0, 0.0, 1.0, 0.158655),
            (2.0, 0.0, 1.0, 0.02275),
            (3.0, 0.0, 1.0, 0.00135),
            (4.0, 0.0, 1.0, 3.17e-05),
            (5.0, 0.0, 1.0, 2.87e-07),
            (-2.0, 0.0, 1.0, 0.97725),
            (2.9, 3.2, 2.5, 0.547758),
        ] {
            let res = super::normal_distribution(x, mu, sigma);
            assert!((res - ans).abs() < eps * sigma,
                "x:{x} mu:{mu} sigma:{sigma} res:{res} != ans:{ans}");
        }
    }
    #[test]
    fn test_inverse_normal_distribution() {
        let eps = 1e-3;
        for (p, mu, sigma, ans) in [
            (0.5, 0.0, 1.0, 0.0),
            (0.158655, 0.0, 1.0, 1.0),
            (0.02275, 0.0, 1.0, 2.0),
            (0.00135, 0.0, 1.0, 3.0),
            (3.17e-05, 0.0, 1.0, 4.0),
            (2.87e-07, 0.0, 1.0, 5.0),
            (0.97725, 0.0, 1.0, -2.0),
            (0.547758, 3.2, 2.5, 2.9),
        ] {
            let res = super::normal_distribution_inverse(p, mu, sigma);
            assert!((res - ans).abs() < eps * sigma,
                "p:{p} mu:{mu} sigma:{sigma} res:{res} != ans:{ans}");
        }
    }
}
