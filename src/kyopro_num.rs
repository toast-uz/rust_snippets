#![allow(dead_code)]

use std::ops::*;
use std::fmt::Display;

pub trait Num:
    Default + Copy + std::fmt::Debug + Display
    + PartialEq + PartialOrd
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(&self) -> Self { if *self >= Self::zero() { *self } else { Self::zero().checkked_sub(*self) } }
    fn gcd(&self, rhs: Self) -> Self;
    fn signum(&self) -> Self { if *self > Self::zero() { Self::one() } else if *self < Self::zero() { Self::zero().checkked_sub(Self::one()) } else { Self::zero() } }
    fn to_f64(&self) -> f64;
    fn to_isize(&self) -> isize;
    fn to_frac(&self) -> Frac<Self> { Frac::new(*self, Self::one()) }
    fn checkked_add(&self, rhs: Self) -> Self;
    fn checkked_sub(&self, rhs: Self) -> Self;
    fn checkked_mul(&self, rhs: Self) -> Self;
}

macro_rules! impl_num_int { ($($ty:ty),*) => {$(
    impl Num for $ty {
        fn zero() -> Self { 0 }
        fn one() -> Self { 1 }
        fn gcd(&self, rhs: Self) -> Self {
            assert!(rhs != 0);
            let mut a = self.abs();
            let mut b = rhs.abs();
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            a
        }
        fn to_f64(&self) -> f64 { *self as f64 }
        fn to_isize(&self) -> isize { *self as isize }
        fn checkked_add(&self, rhs: Self) -> Self { self.checked_add(rhs).unwrap_or_else(|| panic!("overflow by {} + {}", self, rhs)) }
        fn checkked_sub(&self, rhs: Self) -> Self { self.checked_sub(rhs).unwrap_or_else(|| panic!("overflow by {} - {}", self, rhs)) }
        fn checkked_mul(&self, rhs: Self) -> Self { self.checked_mul(rhs).unwrap_or_else(|| panic!("overflow by {} * {}", self, rhs)) }
    }
)*};}
macro_rules! impl_num_float { ($($ty:ty),*) => {$(
    impl Num for $ty {
        fn zero() -> Self { 0.0 }
        fn one() -> Self { 1.0 }
        fn gcd(&self, rhs: Self) -> Self {
            assert!(rhs != 0.0);
            rhs.abs()   // Fracの分母を1.0にするため
        }
        fn to_f64(&self) -> f64 { *self as f64 }
        fn to_isize(&self) -> isize { *self as isize }
        fn checkked_add(&self, rhs: Self) -> Self { *self + rhs }   // no check
        fn checkked_sub(&self, rhs: Self) -> Self { *self - rhs }   // no check
        fn checkked_mul(&self, rhs: Self) -> Self { *self * rhs }   // no check
    }
)*};}

impl_num_int!(isize, i32, i64, i128, usize, u32, u64, u128);
impl_num_float!(f32, f64);

#[derive(Debug, Clone, Copy, Default)]
pub struct Frac<T> {
    pub n: T,
    pub d: T,
}

impl<T: Num> Frac<T> {
    pub fn new(n: T, d: T) -> Self {
        assert!(d != T::zero());
        let g = n.gcd(d);
        Self { n: (n / g).abs() * n.signum() * d.signum(), d: (d / g).abs() }
    }
    pub fn calc(&self) -> T { self.n / self.d }
}

impl<T: Num> Add for Frac<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let g = self.d.gcd(rhs.d);
        let lcm = (self.d / g).checkked_mul(rhs.d);
        let n = self.n.checkked_mul(lcm / self.d).checkked_add(rhs.n.checkked_mul(lcm / rhs.d));
        Self::new(n, lcm)
    }
}
impl<T: Num> Sub for Frac<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let g = self.d.gcd(rhs.d);
        let lcm = (self.d / g).checkked_mul(rhs.d);
        let n = self.n.checkked_mul(lcm / self.d).checkked_sub(rhs.n.checkked_mul(lcm / rhs.d));
        Self::new(n, lcm)
    }
}
impl<T: Num> Mul<Frac<T>> for Frac<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let g1 = self.n.gcd(rhs.d);
        let g2 = rhs.n.gcd(self.d);
        Self::new((self.n / g1).checkked_mul(rhs.n / g2), (self.d / g2).checkked_mul(rhs.d / g1))
    }
}
impl<T: Num> Div<Frac<T>> for Frac<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let g1 = self.n.gcd(rhs.n);
        let g2 = self.d.gcd(rhs.d);
        Self::new((self.n / g1).checkked_mul(rhs.d / g2), (self.d / g2).checkked_mul(rhs.n / g1))
    }
}
impl<T: Num + Neg<Output=T>> Neg for Frac<T> {
    type Output = Self;
    fn neg(self) -> Self { Self::new(-self.n, self.d) }
}
impl<T: Num> AddAssign<Frac<T>> for Frac<T> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<T: Num> Add<T> for Frac<T> {
    type Output = Self;
    fn add(self, rhs: T) -> Self { self + rhs.to_frac() }
}
impl<T: Num> Sub<T> for Frac<T> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self { self - rhs.to_frac() }
}
impl<T: Num> Mul<T> for Frac<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self { self * rhs.to_frac() }
}
impl<T: Num> Div<T> for Frac<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self { self / rhs.to_frac() }
}
impl<T: Num> AddAssign<T> for Frac<T> {
    fn add_assign(&mut self, rhs: T) { *self += rhs.to_frac() }
}
impl<T: Num> PartialEq<Frac<T>> for Frac<T> {
    fn eq(&self, other: &Self) -> bool {
        self.n.checkked_mul(other.d) == self.d.checkked_mul(other.n)
    }
}
impl<T: Num> PartialEq<T> for Frac<T> {
    fn eq(&self, other: &T) -> bool { *self == other.to_frac() }
}
impl<T: Num> PartialOrd<Frac<T>> for Frac<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { // avoid overflow
        if self.d > T::zero() && other.d > T::zero() || self.d < T::zero() && other.d < T::zero() {
            self.n.checkked_mul(other.d).partial_cmp(&self.d.checkked_mul(other.n))
        } else {
            self.d.checkked_mul(other.n).partial_cmp(&self.n.checkked_mul(other.d))
        }
    }
}
impl<T: Num> PartialOrd<T> for Frac<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> { self.partial_cmp(&other.to_frac()) }
}
impl<T: Num> Display for Frac<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.n, self.d)
    }
}

impl<T: Num> Num for Frac<T> {
    fn zero() -> Self { Frac::new(T::zero(), T::one()) }
    fn one() -> Self { Frac::new(T::one(), T::one()) }
    fn gcd(&self, rhs: Self) -> Self { rhs.abs() }
    fn to_f64(&self) -> f64 { self.n.to_f64() / self.d.to_f64() }
    fn to_isize(&self) -> isize { self.n.to_isize() / self.d.to_isize() }
    fn checkked_add(&self, rhs: Self) -> Self { *self + rhs }   // checkeed
    fn checkked_sub(&self, rhs: Self) -> Self { *self - rhs }   // checkeed
    fn checkked_mul(&self, rhs: Self) -> Self { *self * rhs }   // checkeed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frac() {
        let o = Frac::zero();
        let a = Frac::new(1, 2);
        let b = Frac::new(3, 4);
        assert_eq!(o + o, o);
        assert_eq!(o * o, o);
        assert_eq!(a + b, Frac::new(5, 4));
        assert_eq!(a - b, Frac::new(-1, 4));
        assert_eq!(a * b, Frac::new(3, 8));
        assert_eq!(a / b, Frac::new(2, 3));
        assert_eq!(-a, Frac::new(-1, 2));
        assert_eq!(a + 1.to_frac(), Frac::new(3, 2));
        assert_eq!(a - 1.to_frac(), Frac::new(-1, 2));
        assert_eq!(a * 2.to_frac(), Frac::new(1, 1));
        assert_eq!(a / 2.to_frac(), Frac::new(1, 4));
        let a_ = Frac::new(-1, -2);
        let a__ = Frac { n: -1, d: -2 };
        assert_eq!(a.n, a_.n);
        assert_ne!(a.n, a__.n);
        assert_eq!(a, a_);
        assert_eq!(a, a__);
        assert!(a < b);
        assert!(a__ < b);
        assert!(o < a);
        assert!(o < a__);
        assert!(a <= a);
        assert!(a <= a__);
        assert!(a__ <= a);
        assert!(a >= a);
        assert!(a >= a__);
        assert!(a__ >= a);
        assert!(-a <= -a);
        assert!(-a >= -a__);
        assert!(-a__ <= -a);
        assert!(-a <= -a);
        assert!(-a <= -a__);
        assert!(-a__ <= -a);
        assert!(-b < -a);
        assert!(-b < -a__);
        assert!(-a < o);
        assert!(-a__ < o);
        assert_eq!(a.to_f64(), 0.5);
        assert_eq!(a.calc(), 0);
        let o = Frac::<f64>::zero();
        assert_eq!(o + 0.0, 0.0);
        assert_eq!(o * 0.0, 0.0);
        let o = Frac::zero();
        assert_eq!(a + o, a);
        assert_eq!(a * 0, 0);
        assert_eq!(a * 6, 3);
        let e = 1;
        assert_eq!(a * e, a);
        assert_eq!(a / e, a);
        assert_eq!(a * e * e, a);
        assert_eq!((a / a).to_isize(), e);
        assert_ne!(0.1 + 0.2, 0.3); // 誤差が出るため等しくならない
        assert_eq!(Frac::new(1, 10) + Frac::new(1, 5), Frac::new(3, 10));   // 正しく計算できる
    }
}
