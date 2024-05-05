#![allow(dead_code)]

use std::ops::*;
use std::fmt::Display;

pub trait NumRing:
    Default + Copy + std::fmt::Debug
    + PartialEq + PartialOrd
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn as_f64(&self) -> f64;
    fn as_isize(&self) -> isize;
}
macro_rules! impl_primnum_int { ($($ty:ty),*) => {$(
    impl NumRing for $ty {
        fn zero() -> Self { 0 }
        fn one() -> Self { 1 }
        fn as_f64(&self) -> f64 { *self as f64 }
        fn as_isize(&self) -> isize { *self as isize }
    }
)*};}
macro_rules! impl_primnum_float { ($($ty:ty),*) => {$(
    impl NumRing for $ty {
        fn zero() -> Self { 0.0 }
        fn one() -> Self { 1.0 }
        fn as_f64(&self) -> f64 { *self as f64 }
        fn as_isize(&self) -> isize { *self as isize }
    }
)*};}
impl_primnum_int!(isize, i32, i64);
impl_primnum_float!(f32, f64);

type FracPoint<T> = (Point<T>, T);

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Point<T> (pub T, pub T);

impl<T: NumRing> Point<T>{
    pub fn zero() -> Self { Self::default() }
    pub fn new((i, j): (T, T)) -> Self { Self (i, j) }
    pub fn abs2(&self) -> T { self.0 * self.0 + self.1 * self.1 }
    pub fn as_isize(&self) -> Point<isize> { Point::<isize>::new((self.0.as_isize(), self.1.as_isize())) }
    pub fn as_f64(&self) -> Point<f64> { Point::new((self.0.as_f64(), self.1.as_f64())) }
    pub fn abs(&self) -> f64 { self.abs2().as_f64().sqrt() }
    pub fn set_abs(&self, norm: f64) -> Point<f64> {
        let abs_org = self.abs();
        assert_ne!(abs_org, 0.0);
        self.as_f64() * norm / abs_org
    }
    pub fn normalized(&self) -> Point<f64> { self.set_abs(1.0) }
    // 範囲正規化したベクトルを返す
    pub fn clamp(&self, min_v: f64, max_v: f64) -> Point<f64> {
        assert!(min_v <= max_v && min_v >= 0.0);
        let abs = self.abs();
        if abs == 0.0 { return Point::zero(); }
        let norm = abs.clamp(min_v, max_v);
        self.as_f64().set_abs(norm)
    }
    pub fn cos(&self, rhs: Self) -> f64 {
        let abs1 = self.abs();
        let abs2 = rhs.abs();
        assert_ne!(abs1, 0.0);
        assert_ne!(abs2, 0.0);
        self.dot(rhs).as_f64() / (abs1 * abs2)
    }
    pub fn dist2(&self, rhs: Self) -> T { (*self - rhs).abs2() }
    pub fn dist(&self, rhs: Self) -> f64 { self.dist2(rhs).as_f64().sqrt() }
    pub fn dot(&self, rhs: Self) -> T { self.0 * rhs.0 + self.1 * rhs.1 }
    pub fn det(&self, rhs: Self) -> T { self.0 * rhs.1 - self.1 * rhs.0 }
    pub fn is_parallel(&self, rhs: Self) -> bool { self.det(rhs) == T::default() }
    pub fn perpendicular(&self) -> Self { Self (-self.1, self.0) }
    pub fn perpendicular_line(&self, line: &Line<T>) -> Line<T> {
        assert!(line.p != line.q);
        Line::new((*self, *self + line.to_vector().perpendicular()))
    }
    // 交点、Point<T> / det が答え
    pub fn perpendicular_line_foot_frac(&self, line: &Line<T>) -> FracPoint<T> {
        line.cross_point(&self.perpendicular_line(line)).unwrap()
    }
    pub fn perpendicular_segment_foot_frac(&self, seg: &Segment<T>) -> Option<FracPoint<T>> {
        seg.cross_point_with_line_frac(&self.perpendicular_line(&Line::from_segment(seg)))
    }
    // 交点、Point<T> が答え
    pub fn perpendicular_line_foot(&self, line: &Line<T>) -> Point<T> {
        let (r, det) = self.perpendicular_line_foot_frac(line);
        r / det
    }
    pub fn perpendicular_segment_foot(&self, seg: &Segment<T>) -> Option<Point<T>> {
        let Some((r, det)) = self.perpendicular_segment_foot_frac(seg) else { return None; };
        Some(r / det)
    }
    pub fn dist2_line(&self, line: &Line<T>) -> T {
        let foot = self.perpendicular_line_foot(line);
        self.dist2(foot)
    }
    pub fn dist2_segment(&self, seg: &Segment<T>) -> T {
        let dist2p = self.dist2(seg.p);
        let dist2q = self.dist2(seg.q);
        let mut res = if dist2p < dist2q { dist2p } else { dist2q };
        if let Some(foot) = self.perpendicular_segment_foot(seg) {
            let dist2foot = self.dist2(foot);
            if dist2foot < res { res = dist2foot; }
        }
        res
    }
    pub fn dist_line(&self, line: &Line<T>) -> f64 {
        self.dist2_line(line).as_f64().sqrt()
    }
    pub fn dist_segment(&self, seg: &Segment<T>) -> f64 {
        self.dist2_segment(seg).as_f64().sqrt()
    }
}

impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self (self.0 + rhs.0, self.1 + rhs.1) }
}
impl<T: Sub<Output = T>> Sub for Point<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self (self.0 - rhs.0, self.1 - rhs.1) }
}
impl<T: Copy + Mul<Output = T>> Mul<T> for Point<T> {   // スカラー倍
    type Output = Self;
    fn mul(self, scalar: T) -> Self { Self (self.0 * scalar, self.1 * scalar) }
}
impl<T: Copy + Div<Output = T>> Div<T> for Point<T> {   // スカラー分の1
    type Output = Self;
    fn div(self, scalar: T) -> Self { Self (self.0 / scalar, self.1 / scalar) }
}
impl<T: Copy + std::ops::Add<Output = T>> std::ops::AddAssign for Point<T> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<T: Copy + std::ops::Neg<Output = T>> std::ops::Neg for Point<T> {
    type Output = Self;
    fn neg(self) -> Self { Self ( -self.0, -self.1) }
}

impl<T: Display> std::fmt::Display for Point<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl Point<f64> {
    pub fn round(&self) -> Self { Self (self.0.round(), self.1.round()) }
    pub fn floor(&self) -> Self { Self (self.0.floor(), self.1.floor()) }
    pub fn ceil(&self) -> Self { Self (self.0.ceil(), self.1.ceil()) }
    pub fn trunc(&self) -> Self { Self (self.0.trunc(), self.1.trunc()) }   // floor_abs
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Segment<T> {
    pub p: Point<T>, pub q: Point<T>,
}

impl<T: NumRing> Segment<T> {
    pub fn new((p, q): (Point<T>, Point<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Point<T> { self.q - self.p }
    pub fn abs2(&self) -> T { self.to_vector().abs2() }
    pub fn as_f64(&self) -> Segment<f64> { Segment::new((self.p.as_f64(), self.q.as_f64())) }
    pub fn as_isize(&self) -> Segment<isize> { Segment::new((self.p.as_isize(), self.q.as_isize())) }
    pub fn contains(&self, p: Point<T>) -> bool {
        (p - self.p).is_parallel(self.q - self.p) && (p - self.p).dot(p - self.q) <= T::default()
    }
    pub fn cross(&self, rhs: &Segment<T>) -> bool {
        self.cross_with_line(&Line::from_segment(rhs)) && rhs.cross_with_line(&Line::from_segment(self))
    }
    pub fn cross_with_halfline(&self, hl: &HalfLine<T>) -> bool {
        let Some((r, det)) = self.cross_point_with_line_frac(&Line::from_halfline(hl)) else { return false; };
        let deg = (r - hl.p * det).dot(hl.to_vector());
        // オーバーフロー対策のため、degとdetをかけない
        deg >= T::default() && det >= T::default() || deg <= T::default() && det <= T::default()
    }
    pub fn cross_with_line(&self, line: &Line<T>) -> bool {
        line.to_vector().det(self.p - line.p) * line.to_vector().det(self.q - line.p) <= T::default()
    }
    // 交点、Point<T> / det が答え
    pub fn cross_point_frac(&self, rhs: &Segment<T>) -> Option<FracPoint<T>> {
        if !self.cross(rhs) { return None; }
        Line::from_segment(self).cross_point(&Line::from_segment(rhs))
    }
    pub fn cross_point_with_halfline_frac(&self, hl: &HalfLine<T>) -> Option<FracPoint<T>> {
        if !self.cross_with_halfline(hl) { return None; }
        Line::from_segment(self).cross_point(&Line::from_halfline(hl))
    }
    pub fn cross_point_with_line_frac(&self, line: &Line<T>) -> Option<FracPoint<T>> {
        if !self.cross_with_line(line) { return None; }
        Line::from_segment(self).cross_point(line)
    }
    // 線分の範囲を意識した交点
    pub fn cross_point(&self, rhs: &Segment<T>) -> Option<Point<T>> {
        let (r, det) = self.cross_point_frac(rhs)?;
        Some(r / det)
    }
    pub fn cross_point_with_halfline(&self, hl: &HalfLine<T>) -> Option<Point<T>> {
        let (r, det) = self.cross_point_with_halfline_frac(hl)?;
        Some(r / det)
    }
    pub fn cross_point_with_line(&self, line: &Line<T>) -> Option<Point<T>> {
        let (r, det) = self.cross_point_with_line_frac(line)?;
        Some(r / det)
    }
    // 線分と線分との距離
    pub fn dist2(&self, rhs: &Segment<T>) -> T {
        if self.cross(rhs) { return T::default(); }
        let candidate = [
            self.p.dist2_segment(rhs), self.q.dist2_segment(rhs),
            rhs.p.dist2_segment(self), rhs.q.dist2_segment(self)];
        candidate.iter().min_by(|&a, &b| a.partial_cmp(b).unwrap()).cloned().unwrap()
    }
    pub fn dist(&self, rhs: &Segment<T>) -> f64 {
        self.dist2(rhs).as_f64().sqrt()
    }
}

// 向きが反対でも同じとみなす
impl<T: NumRing> PartialEq for Segment<T> {
    fn eq(&self, rhs: &Self) -> bool {
        (self.p == rhs.p && self.q == rhs.q)
        || (self.p == rhs.q && self.q == rhs.p)
    }
}

// 半直線 p -> q
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfLine<T> {
    pub p: Point<T>, pub q: Point<T>,
}

impl<T: NumRing> HalfLine<T> {
    pub fn new((p, q): (Point<T>, Point<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Point<T> { self.q - self.p }
    pub fn from_segment(seg: &Segment<T>) -> Self { Self { p: seg.p, q: seg.q } }
    pub fn contains(&self, p: Point<T>) -> bool {
        Line::from_halfline(self).contains(p) && (p - self.p).dot(self.q - self.p) >= T::default()
    }
}

// 起点と向きが同じであれば同じとみなす
impl<T: NumRing> PartialEq for HalfLine<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.p == rhs.p && (self.q - self.p).is_parallel(rhs.q - rhs.p)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Line<T> {
    pub p: Point<T>, pub q: Point<T>,
}

impl<T: NumRing> Line<T> {
    pub fn new((p, q): (Point<T>, Point<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Point<T> { self.q - self.p }
    pub fn from_segment(seg: &Segment<T>) -> Self { Self { p: seg.p, q: seg.q } }
    pub fn from_halfline(hl: &HalfLine<T>) -> Self { Self { p: hl.p, q: hl.q } }
    pub fn contains(&self, p: Point<T>) -> bool { (p - self.p).is_parallel(self.q - self.p) }
    // 直線と直線との交点、Point<T> / det が答え
    pub fn cross_point(&self, rhs: &Line<T>) -> Option<FracPoint<T>> {
        let det = self.to_vector().det(rhs.to_vector());
        if det == T::default() { return None; } // 平行
        let r = self.p * det + self.to_vector() * rhs.to_vector().det(self.p - rhs.p);
        Some((r, det))
    }
    pub fn cross_point_f64(&self, rhs: &Line<T>) -> Option<Point<f64>> {
        let (r, det) = self.cross_point(rhs)?;
        Some(r.as_f64() / det.as_f64())
    }
}

// 同一直線上にあれば同じとみなす
impl<T: NumRing> PartialEq for Line<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.contains(rhs.p) && self.contains(rhs.q)
    }
}


///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test -r kyopro_linalg::tests::basic

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! eq_f64 { ($a:expr, $b:expr) => { ($a - $b).abs() < 1.0e-9 }; }
    macro_rules! assert_eq_f64 { ($a:expr, $b:expr) => { assert!(eq_f64!($a, $b), "left: {} != right: {}", $a, $b) }; }

    #[test]
    fn basic() {
        let o = Point::zero();
        let a = Point::new((1, 2));
        let b = Point::new((3, 3));
        let c = Point::new((-2, -1));
        let d = Point::new((2, 4));
        assert_eq!(a.abs2(), 5);
        assert_eq!(a * 2, Point::new((2, 4)));
        assert_eq!(a + b, Point::new((4, 5)));
        assert_eq!(a - b, Point::new((-2, -1)));
        let oa = Segment::new((o, a));
        let ob = Segment::new((o, b));
        let ac = Segment::new((a, c));
        let ad = Segment::new((a, d));
        let bc = Segment::new((b, c));
        assert_eq!(ac.abs2(), 18);
        let line_oa = Line::from_segment(&oa);
        let line_ob = Line::from_segment(&ob);
        let line_ac = Line::from_segment(&ac);
        // 交点はo
        let cross_oa_ob = line_oa.cross_point(&line_ob).unwrap();
        assert_eq!(cross_oa_ob.0, o);
        // 交点なし（2直線は平行）
        let cross_ob_ac = line_ob.cross_point(&line_ac);
        assert!(cross_ob_ac.is_none());
        // 交点は解析的に求められる
        let cross_loa_lbc = Line::from_segment(&oa.as_f64()).cross_point_f64(&Line::from_segment(&bc.as_f64())).unwrap();
        assert_eq!(cross_loa_lbc, Point::new((0.5, 1.0)));
        let cross_oa_lbc = oa.as_f64().cross_point_with_line(&Line::from_segment(&bc.as_f64())).unwrap();
        assert_eq!(cross_oa_lbc, Point::new((0.5, 1.0)));
        let cross_oa_bc = oa.as_f64().cross_point(&bc.as_f64()).unwrap();
        assert_eq!(cross_oa_bc, Point::new((0.5, 1.0)));
        let cross_ad_bc = ad.cross_point(&bc);
        assert!(cross_ad_bc.is_none());
        // 交点との距離
        let dist_a_ob = a.as_f64().dist_segment(&ob.as_f64());
        assert_eq_f64!(dist_a_ob, 1.0 / 2.0_f64.sqrt());
        let dist_b_ac = b.as_f64().dist_segment(&ac.as_f64());
        assert_eq_f64!(dist_b_ac, 5.0_f64.sqrt());
        let dist_ad_ob = ad.as_f64().dist(&ob.as_f64());
        assert_eq_f64!(dist_ad_ob, 1.0 / 2.0_f64.sqrt());
        // 線と点の包含判定
        assert!(ob.contains(o));
        assert!(ob.contains(b));
        assert!(!ob.contains(a));
        assert!(!ob.contains(b * 5));
        assert!(HalfLine::from_segment(&ob).contains(b * 5));
        assert!(!HalfLine::from_segment(&ob).contains(-b));
        assert!(Line::from_segment(&ob).contains(-b));
        assert!(!ob.as_f64().contains(b.as_f64() / 2.0 + a.as_f64() * 0.01));
        assert!(!ob.as_f64().contains(b.as_f64() / 2.0 - a.as_f64() * 0.01));
        assert!(!ob.as_f64().contains(b.as_f64() * -0.01));
        assert!(!ob.as_f64().contains(b.as_f64() * 1.01));
        // 線分・直線・半直線と線分の交差判定
        let oa_hl = HalfLine::from_segment(&oa);
        let ac_hl = HalfLine::from_segment(&ac);
        let ob_hl = HalfLine::from_segment(&ob);
        let oa_l = Line::from_segment(&oa);
        let ac_l = Line::from_segment(&ac);
        let ob_l = Line::from_segment(&ob);
        let ld = Point::new((-10, -10));
        let rd = Point::new((10, -10));
        let lu = Point::new((-10, 10));
        let ru = Point::new((10, 10));
        let l = Segment::new((ld, lu));
        let r = Segment::new((rd, ru));
        let u = Segment::new((lu, ru));
        let d = Segment::new((ld, rd));
        // 線分は全て交差しない
        assert!(!l.cross(&oa));
        assert!(!r.cross(&oa));
        assert!(!u.cross(&oa));
        assert!(!d.cross(&oa));
        assert!(!l.cross(&ac));
        assert!(!r.cross(&ac));
        assert!(!u.cross(&ac));
        assert!(!d.cross(&ac));
        // 半直線は1方向だけ交差する
        assert!(!l.cross_with_halfline(&oa_hl));
        assert!(!r.cross_with_halfline(&oa_hl));
        assert!(u.cross_with_halfline(&oa_hl));
        assert!(!d.cross_with_halfline(&oa_hl));
        assert!(l.cross_with_halfline(&ac_hl));
        assert!(!r.cross_with_halfline(&ac_hl));
        assert!(!u.cross_with_halfline(&ac_hl));
        assert!(!d.cross_with_halfline(&ac_hl));
        // 直線は2方向で交差する
        assert!(!l.cross_with_line(&oa_l));
        assert!(!r.cross_with_line(&oa_l));
        assert!(u.cross_with_line(&oa_l));
        assert!(d.cross_with_line(&oa_l));
        assert!(l.cross_with_line(&ac_l));
        assert!(!r.cross_with_line(&ac_l));
        assert!(u.cross_with_line(&ac_l));
        assert!(!d.cross_with_line(&ac_l));
        // 端に重なる場合は、両方向で交差する
        assert!(!l.cross_with_halfline(&ob_hl));
        assert!(r.cross_with_halfline(&ob_hl));
        assert!(u.cross_with_halfline(&ob_hl));
        assert!(!d.cross_with_halfline(&ob_hl));
        assert!(l.cross_with_line(&ob_l));
        assert!(r.cross_with_line(&ob_l));
        assert!(u.cross_with_line(&ob_l));
        assert!(d.cross_with_line(&ob_l));
    }
}
