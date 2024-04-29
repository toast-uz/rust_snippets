#![allow(dead_code)]
#![allow(unused_macros)]

pub const LINALG_EPS: f64 = 1e-10;
const DISPLAY_DIGIT: usize = 1;

#[macro_export]
macro_rules! eq_f64 { ($a:expr, $b:expr) => {
    if $a != 0.0 && $b != 0.0 { ($a / $b - 1.0).abs() < LINALG_EPS }
    else { ($a - $b).abs() < LINALG_EPS } }
}
#[macro_export]
macro_rules! eq_f64_zero { ($a:expr) => { eq_f64!($a, 0.0) };}
#[macro_export]
macro_rules! assert_eq_f64 { ($a:expr, $b:expr) => {
    assert!(eq_f64!($a, $b), "{:?} != {:?} as eps {LINALG_EPS}", $a, $b)
};}
#[macro_export]
macro_rules! assert_ne_f64 { ($a:expr, $b:expr) => {
    assert!(!eq_f64!($a, $b), "{:?} == {:?} as eps {LINALG_EPS}", $a, $b)
};}
#[macro_export]
macro_rules! eq_vector_f64 { ($a:expr, $b:expr) => {
    eq_f64!($a.x, $b.x) && eq_f64!($a.y, $b.y)
};}
#[macro_export]
macro_rules! parallel_vector_f64 { ($a:expr, $b:expr) => {
    !eq_f64_zero!($a.norm()) && !eq_f64_zero!($b.norm()) && eq_f64_zero!($a.det($b) / ($a.norm() * $b.norm()))
};}
#[macro_export]
macro_rules! eq_segment_f64 { ($a:expr, $b:expr) => {
    (eq_vector_f64!($a.p, $b.p) && eq_vector_f64!($a.q, $b.q))
        || (eq_vector_f64!($a.p, $b.q) && eq_vector_f64!($a.q, $b.p))
};}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vector<T> {
    pub x: T, pub y: T,
}

impl<T> Vector<T>
where T: Default + Copy + PartialOrd
    + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
    + std::ops::Mul<Output = T> + std::ops::Div<Output = T>
    + std::ops::Neg<Output = T>
{
    pub fn zero() -> Self { Self::default() }
    pub fn new((x, y): (T, T)) -> Self { Self { x, y } }
    pub fn norm2(&self) -> T { self.x * self.x + self.y * self.y }
    pub fn dist2(&self, rhs: Self) -> T { (*self - rhs).norm2() }
    pub fn dot(&self, rhs: Self) -> T { self.x * rhs.x + self.y * rhs.y }
    pub fn det(&self, rhs: Self) -> T { self.x * rhs.y - self.y * rhs.x }
    pub fn perpendicular(&self) -> Self { Self { x: -self.y, y: self.x } }
    pub fn perpendicular_line(&self, line: &Line<T>) -> Line<T> {
        Line::new((*self, *self + line.to_vector().perpendicular()))
    }
    pub fn mul(&self, k: T) -> Self { Self { x: self.x * k, y: self.y * k } }
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for Vector<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } }
}
impl<T: std::ops::Sub<Output = T>> std::ops::Sub for Vector<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } }
}
impl<T: Copy + std::ops::Add<Output = T>> std::ops::AddAssign for Vector<T> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<T: Copy + std::ops::Neg<Output = T>> std::ops::Neg for Vector<T> {
    type Output = Self;
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}

impl std::fmt::Display for Vector<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.DISPLAY_DIGIT$} {:.DISPLAY_DIGIT$}", self.x, self.y)
    }
}
impl std::fmt::Display for Vector<isize> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.x, self.y)
    }
}

impl Vector<f64> {
    pub fn up() -> Self { Self { x: 0.0, y: 1.0 } }
    pub fn down() -> Self { Self { x: 0.0, y: -1.0 } }
    pub fn left() -> Self { Self { x: -1.0, y: 0.0 } }
    pub fn right() -> Self { Self { x: 1.0, y: 0.0 } }
    // EPSを考慮して、範囲正規化したベクトルを返す
    pub fn clamp(&self, min_v: f64, max_v: f64) -> Self {
        assert!(min_v <= max_v && min_v >= 0.0);
        let norm = self.norm();
        let mul = if eq_f64_zero!(norm) { 1.0 }
        else if norm < min_v as f64 {
            (min_v * (1.0 + LINALG_EPS)).min(max_v) / norm
        }
        else if norm > max_v as f64 {
            (max_v / (norm * (1.0 + LINALG_EPS))).max(min_v / norm)
        }
        else { 1.0 };
        self.mul(mul)
    }
    pub fn norm(&self) -> f64 { self.norm2().sqrt() }
    pub fn set_norm(&self, norm: f64) -> Self {
        let norm_org = self.norm();
        assert!(!eq_f64_zero!(norm_org));
        Vector::new((self.x * norm / norm_org, self.y * norm / norm_org))
    }
    pub fn dist(&self, rhs: Self) -> f64 { (*self - rhs).norm() }
    pub fn cos(&self, rhs: Self) -> f64 { self.dot(rhs) / (self.norm() * rhs.norm()) }
    pub fn round(&self) -> Self { Self { x: self.x.round(), y: self.y.round() } }
    pub fn floor(&self) -> Self { Self { x: self.x.floor(), y: self.y.floor() } }
    pub fn ceil(&self) -> Self { Self { x: self.x.ceil(), y: self.y.ceil() } }
    pub fn floor_abs(&self) -> Self { Self { x: self.x.signum() * self.x.abs().floor(), y: self.y.signum() * self.y.abs().floor() } }
    pub fn ceil_abs(&self) -> Self { Self { x: self.x.signum() * self.x.abs().ceil(), y: self.y.signum() * self.y.abs().ceil() } }
    pub fn to_int(&self) -> Vector<isize> { Vector::<isize>::new((self.x.round() as isize, self.y.round() as isize)) }
    pub fn perpendicular_line_foot(&self, line: &Line<f64>) -> Vector<f64> {
        if line.p == line.q { return line.p; }
        line.cross_point(&self.perpendicular_line(line)).unwrap()
    }
    pub fn dist_line(&self, line: &Line<f64>) -> f64 {
        let foot = self.perpendicular_line_foot(line);
        self.dist(foot)
    }
    pub fn perpendicular_segment_foot(&self, seg: &Segment<f64>) -> Option<Vector<f64>> {
        let line = Line::from_segment(seg);
        let foot = self.perpendicular_line_foot(&line);
        if seg.contains(foot) { Some(foot) } else { None }
    }
    pub fn dist_segment(&self, segment: &Segment<f64>) -> f64 {
        let mut res = self.dist(segment.p).min(self.dist(segment.q));
        if let Some(foot) = self.perpendicular_segment_foot(segment) {
            res = res.min(self.dist(foot));
        }
        res
    }
}

impl PartialEq for Vector<isize> {
    fn eq(&self, other: &Self) -> bool { self.x == other.x && self.y == other.y }
}

impl PartialEq for Vector<f64> {
    fn eq(&self, other: &Self) -> bool { eq_vector_f64!(*self, *other) }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Segment<T> {
    pub p: Vector<T>, pub q: Vector<T>,
}

impl<T> Segment<T>
where T: Default + Copy + PartialOrd
    + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
    + std::ops::Mul<Output = T> + std::ops::Div<Output = T>
    + std::ops::Neg<Output = T>
{
    pub fn new((p, q): (Vector<T>, Vector<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Vector<T> { self.q - self.p }
    pub fn norm2(&self) -> T { self.to_vector().norm2() }
}

impl Segment<f64> {
    pub fn norm(&self) -> f64 { self.norm2().sqrt() }
    pub fn contains(&self, p: Vector<f64>) -> bool {
        Line::from_segment(self).contains(p) && (p - self.p).dot(p - self.q) < LINALG_EPS
    }
    // 線分の範囲を意識した交点
    pub fn cross_point(&self, rhs: &Segment<f64>) -> Option<Vector<f64>> {
        Line::from_segment(self).cross_point(&Line::from_segment(rhs))
            .filter(|&v| self.contains(v) && rhs.contains(v))
    }
    pub fn cross_point_with_halfline(&self, rhs: &HalfLine<f64>) -> Option<Vector<f64>> {
        Line::from_segment(self).cross_point(&Line::from_halfline(rhs))
            .filter(|&v| (v - self.p).dot(v - self.q) < LINALG_EPS && (v - rhs.p).dot(rhs.q - rhs.p) > -LINALG_EPS)
    }
    // 線分と線分との距離
    pub fn dist(&self, rhs: &Segment<f64>) -> f64 {
        if self.cross_point(rhs).is_some() { return 0.0; }
        self.p.dist_segment(rhs).min(self.q.dist_segment(rhs))
            .min(rhs.p.dist_segment(self).min(rhs.q.dist_segment(self)))
    }
}

impl PartialEq for Segment<f64> {
    fn eq(&self, rhs: &Self) -> bool { eq_segment_f64!(*self, *rhs) }
}

// 半直線 p -> q
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfLine<T> {
    pub p: Vector<T>, pub q: Vector<T>,
}

impl<T> HalfLine<T>
where T: Default + Copy + PartialOrd
    + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
    + std::ops::Mul<Output = T> + std::ops::Div<Output = T>
    + std::ops::Neg<Output = T>
{
    pub fn new((p, q): (Vector<T>, Vector<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Vector<T> { self.q - self.p }
    pub fn from_segment(seg: Segment<T>) -> Self { Self { p: seg.p, q: seg.q } }
}

impl HalfLine<f64> {
    pub fn contains(&self, p: Vector<f64>) -> bool {
        Line::from_halfline(self).contains(p) && (p - self.p).dot(self.q - self.p) > -LINALG_EPS
    }
}

impl PartialEq for HalfLine<f64> {
    fn eq(&self, rhs: &Self) -> bool {
        eq_vector_f64!(self.p, rhs.p) && parallel_vector_f64!(self.q - self.p, rhs.q - rhs.p)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Line<T> {
    pub p: Vector<T>, pub q: Vector<T>,
}

impl<T> Line<T>
where T: Default + Copy + PartialOrd
    + std::ops::Add<Output = T> + std::ops::Sub<Output = T>
    + std::ops::Mul<Output = T> + std::ops::Div<Output = T>
    + std::ops::Neg<Output = T>
{
    pub fn new((p, q): (Vector<T>, Vector<T>)) -> Self { Self { p, q } }
    pub fn to_vector(&self) -> Vector<T> { self.q - self.p }
    pub fn from_segment(seg: &Segment<T>) -> Self { Self { p: seg.p.clone(), q: seg.q.clone() } }
    pub fn from_halfline(hl: &HalfLine<T>) -> Self { Self { p: hl.p.clone(), q: hl.q.clone() } }
}

impl Line<f64> {
    pub fn contains(&self, p: Vector<f64>) -> bool {
        parallel_vector_f64!(self.p - p, self.q - p)
    }
    pub fn cross_point(&self, rhs: &Line<f64>) -> Option<Vector<f64>> {
        let d0 = self.q - self.p;
        let d1 = rhs.q - rhs.p;
        if parallel_vector_f64!(d0, d1) { return None; }
        let det = d0.det(d1);
        let x = -(d0.x * d1.x * (rhs.p.y - self.p.y) - d0.x * d1.y * rhs.p.x + d0.y * d1.x * self.p.x) / det;
        let y = (d0.y * d1.y * (rhs.p.x - self.p.x) - d0.y * d1.x * rhs.p.y + d0.x * d1.y * self.p.y) / det;
        Some(Vector::new((x, y)))
    }
}

impl PartialEq for Line<f64> {
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

    #[test]
    fn basic() {
        let o = Vector::<f64>::zero();
        let a = Vector::new((1, 2));
        let b = Vector::new((3, 3));
        let c = Vector::new((-2.0, -1.0));
        let d = Vector::new((2.0, 4.0));
        assert_eq!(a.norm2(), 5);
        assert_eq!(a.mul(2), Vector::new((2, 4)));
        assert_eq!(a + b, Vector::new((4, 5)));
        assert_eq!(a - b, Vector::new((-2, -1)));
        let a = Vector::new((1.0, 2.0));
        let b = Vector::new((3.0, 3.0));
        let oa = Segment::new((o, a));
        let ob = Segment::new((o, b));
        let ac = Segment::new((a, c));
        let ad = Segment::new((a, d));
        let bc = Segment::new((b, c));
        assert_eq_f64!(ac.norm2(), 18.0);
        let line_oa = Line::from_segment(&oa);
        let line_ob = Line::from_segment(&ob);
        let line_ac = Line::from_segment(&ac);
        // 交点はo
        let p_oa_ob = line_oa.cross_point(&line_ob).unwrap();
        assert_eq_f64!(p_oa_ob.x, 0.0);
        assert_eq_f64!(p_oa_ob.y, 0.0);
        // 交点なし（2直線は平行）
        let p_ob_ac = line_ob.cross_point(&line_ac);
        assert!(p_ob_ac.is_none());
        // 交点は解析的に求められる
        let p_oa_bc = oa.cross_point(&bc).unwrap();
        assert_eq_f64!(p_oa_bc.x, 0.5);
        assert_eq_f64!(p_oa_bc.y, 1.0);
        let p_ad_bc = ad.cross_point(&bc);
        assert!(p_ad_bc.is_none());
        let dist_a_ob = a.dist_segment(&ob);
        assert_eq_f64!(dist_a_ob, 1.0 / 2.0_f64.sqrt());
        let dist_b_ac = b.dist_segment(&ac);
        assert_eq_f64!(dist_b_ac, 5.0_f64.sqrt());
        let dist_ad_ob = ad.dist(&ob);
        assert_eq_f64!(dist_ad_ob, 1.0 / 2.0_f64.sqrt());
        assert!(ob.contains(o));
        assert!(ob.contains(b));
        assert!(!ob.contains(a));
        assert!(ob.contains(b.mul(0.5)));
        assert!(!ob.contains(b.mul(0.5) + a.mul(0.01)));
        assert!(!ob.contains(b.mul(0.5) - a.mul(0.01)));
        assert!(!ob.contains(b.mul(-0.01)));
        assert!(!ob.contains(b.mul(1.01)));
    }
}
