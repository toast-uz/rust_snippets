#![allow(dead_code)]

use std::ops::*;
use std::fmt::Display;
use crate::kyopro_num::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct P<T> (pub T, pub T);

impl<T: Num> P<T> {
    pub fn zero() -> Self { Self::default() }
    pub fn new((i, j): (T, T)) -> Self { Self (i, j) }
    pub fn abs2(&self) -> T { self.dot(*self) }
    pub fn abs_manhattan(&self) -> T { self.0.abs().checkked_add(self.1.abs()) }
    pub fn to_isize(&self) -> P<isize> { P::<isize>::new((self.0.to_isize(), self.1.to_isize())) }
    pub fn to_f64(&self) -> P<f64> { P::new((self.0.to_f64(), self.1.to_f64())) }
    pub fn to_frac(&self) -> P<Frac<T>> { P::new((self.0.to_frac(), self.1.to_frac())) }
    pub fn abs(&self) -> f64 { self.abs2().to_f64().sqrt() }
    pub fn set_abs(&self, norm: f64) -> P<f64> {
        let abs_org = self.abs();
        assert_ne!(abs_org, 0.0);
        self.to_f64() * norm / abs_org
    }
    pub fn normalized(&self) -> P<f64> { self.set_abs(1.0) }
    // 範囲正規化したベクトルを返す
    pub fn clamp(&self, min_v: f64, max_v: f64) -> P<f64> {
        assert!(min_v <= max_v && min_v >= 0.0);
        let abs = self.abs();
        if abs == 0.0 { return P::zero(); }
        let norm = abs.clamp(min_v, max_v);
        self.to_f64().set_abs(norm)
    }
    pub fn cos(&self, rhs: Self) -> f64 {
        let abs1 = self.abs();
        let abs2 = rhs.abs();
        assert_ne!(abs1, 0.0);
        assert_ne!(abs2, 0.0);
        self.dot(rhs).to_f64() / (abs1 * abs2)
    }
    pub fn dist2(&self, rhs: Self) -> T { (*self - rhs).abs2() }
    pub fn dist_manhattan(&self, rhs: Self) -> T { (*self - rhs).abs_manhattan() }
    pub fn dist(&self, rhs: Self) -> f64 { self.dist2(rhs).to_f64().sqrt() }
    pub fn dot(&self, rhs: Self) -> T { self.0.checkked_mul(rhs.0).checkked_add(self.1.checkked_mul(rhs.1)) }
    pub fn det(&self, rhs: Self) -> T { self.0.checkked_mul(rhs.1).checkked_sub(rhs.0.checkked_mul(self.1)) }
    pub fn is_parallel(&self, rhs: Self) -> bool { self.det(rhs) == T::default() }
}

impl<T: Num + Neg<Output=T>> P<T> {
    pub fn perpendicular(&self) -> Self { Self (-self.1, self.0) }
    pub fn perpendicular_line(&self, line: &Line<T>) -> Line<T> {
        assert!(line.p != line.q);
        Line::new((*self, *self + line.to_vector().perpendicular()))
    }
    // 交点、Point<T> / det が答え
    pub fn perpendicular_line_foot_frac(&self, line: &Line<T>) -> P<Frac<T>> {
        line.cross_point_frac(&self.perpendicular_line(line)).unwrap()
    }
    pub fn perpendicular_segment_foot_frac(&self, seg: &Segment<T>) -> Option<P<Frac<T>>> {
        seg.cross_point_with_line_frac(&self.perpendicular_line(&Line::from_segment(seg)))
    }
    // 交点、Point<T> が答え
    pub fn perpendicular_line_foot(&self, line: &Line<T>) -> P<T> {
        self.perpendicular_line_foot_frac(line).calc()
    }
    pub fn perpendicular_segment_foot(&self, seg: &Segment<T>) -> Option<P<T>> {
        self.perpendicular_segment_foot_frac(seg).and_then(|r| Some(r.calc()))
    }
    pub fn dist2_line_frac(&self, line: &Line<T>) -> Frac<T> {
        self.to_frac().dist2(self.perpendicular_line_foot_frac(line))
    }
    pub fn dist2_segment_frac(&self, seg: &Segment<T>) -> Frac<T> {
        let dist2p = self.dist2(seg.p);
        let dist2q = self.dist2(seg.q);
        let mut res = if dist2p < dist2q { dist2p } else { dist2q }.to_frac();
        if let Some(foot) = self.perpendicular_segment_foot_frac(seg) {
            let dist2foot = self.to_frac().dist2(foot);
            if dist2foot < res { res = dist2foot; }
        }
        res
    }
    pub fn dist_line(&self, line: &Line<T>) -> f64 {
        self.dist2_line_frac(line).to_f64().sqrt()
    }
    pub fn dist_segment(&self, seg: &Segment<T>) -> f64 {
        self.dist2_segment_frac(seg).to_f64().sqrt()
    }
}

impl<T: Num> P<Frac<T>> {
    fn calc(&self) -> P<T> { P::new((self.0.calc(), self.1.calc())) }
}

impl<T: Num> Add for P<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self (self.0.checkked_add(rhs.0), self.1.checkked_add(rhs.1)) }
}
impl<T: Num> Sub for P<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self (self.0.checkked_sub(rhs.0), self.1.checkked_sub(rhs.1)) }
}
impl<T: Num> Mul<T> for P<T> {   // スカラー倍
    type Output = Self;
    fn mul(self, scalar: T) -> Self { Self (self.0.checkked_mul(scalar), self.1.checkked_mul(scalar)) }
}
impl<T: Num> Div<T> for P<T> {   // スカラー分の1
    type Output = Self;
    fn div(self, scalar: T) -> Self { Self (self.0 / scalar, self.1 / scalar) }
}
impl<T: Num> AddAssign for P<T> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<T: Num + Neg<Output=T>> Neg for P<T> {
    type Output = Self;
    fn neg(self) -> Self { Self ( -self.0, -self.1) }
}

impl<T: Num> Display for P<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl P<f64> {
    pub fn round(&self) -> Self { Self (self.0.round(), self.1.round()) }
    pub fn floor(&self) -> Self { Self (self.0.floor(), self.1.floor()) }
    pub fn ceil(&self) -> Self { Self (self.0.ceil(), self.1.ceil()) }
    pub fn trunc(&self) -> Self { Self (self.0.trunc(), self.1.trunc()) }   // floor_abs
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Segment<T> {
    pub p: P<T>, pub q: P<T>,
}

impl<T: Num> Segment<T> {
    pub fn to_f64(&self) -> Segment<f64> { Segment::new((self.p.to_f64(), self.q.to_f64())) }
    pub fn to_isize(&self) -> Segment<isize> { Segment::new((self.p.to_isize(), self.q.to_isize())) }
    pub fn to_frac(&self) -> Segment<Frac<T>> { Segment::new((self.p.to_frac(), self.q.to_frac())) }
    pub fn contains(&self, p: P<T>) -> bool {
        (p - self.p).is_parallel(self.q - self.p) && (p - self.p).dot(p - self.q) <= T::default()
    }
    pub fn cross(&self, rhs: &Segment<T>) -> bool {
        self.cross_with_line(&Line::from_segment(rhs)) && rhs.cross_with_line(&Line::from_segment(self))
    }
    pub fn cross_with_halfline(&self, hl: &HalfLine<T>) -> bool {
        if !self.cross_with_line(&Line::from_halfline(hl)) { return false; }
        let det1 = (hl.p - self.p).det(self.q - self.p);
        let det2 = (hl.p - self.p).det(hl.q - self.p);
        let det3 = (hl.p - self.q).det(self.p - self.q);
        let det4 = (hl.p - self.q).det(hl.q - self.q);
        det1 > T::zero() && det2 > T::zero() || det1 < T::zero() && det2 < T::zero()
            || det3 > T::zero() && det4 > T::zero() || det3 < T::zero() && det4 < T::zero()
        // overfloww for isize
        //let Some(r) = self.cross_point_with_line_frac(&Line::from_halfline(hl)) else { return false; };
        //(r - hl.p.to_frac()).dot(hl.to_vector().to_frac()) >= Frac::<T>::zero()
    }
    pub fn cross_with_line(&self, line: &Line<T>) -> bool {
        let det1 = line.to_vector().det(self.p - line.p);
        let det2 = line.to_vector().det(self.q - line.p);
        det1 >= T::zero() && det2 <= T::zero() || det1 <= T::zero() && det2 >= T::zero()
    }
    // 交点、Point<T> / det が答え
    pub fn cross_point_frac(&self, rhs: &Segment<T>) -> Option<P<Frac<T>>> {
        if !self.cross(rhs) { return None; }
        self.to_line().cross_point_frac(&rhs.to_line())
    }
    pub fn cross_point_with_halfline_frac(&self, hl: &HalfLine<T>) -> Option<P<Frac<T>>> {
        if !self.cross_with_halfline(hl) { return None; }
        self.to_line().cross_point_frac(&hl.to_line())
    }
    pub fn cross_point_with_line_frac(&self, line: &Line<T>) -> Option<P<Frac<T>>> {
        if !self.cross_with_line(line) { return None; }
        self.to_line().cross_point_frac(line)
    }
    // 線分の範囲を意識した交点
    pub fn cross_point(&self, rhs: &Segment<T>) -> Option<P<T>> {
        self.cross_point_frac(rhs).and_then(|r| Some(r.calc()))
    }
    pub fn cross_point_with_halfline(&self, hl: &HalfLine<T>) -> Option<P<T>> {
        self.cross_point_with_halfline_frac(hl).and_then(|r| Some(r.calc()))
    }
    pub fn cross_point_with_line(&self, line: &Line<T>) -> Option<P<T>> {
        self.cross_point_with_line_frac(line).and_then(|r| Some(r.calc()))
    }
}

impl<T: Num + Neg<Output=T>> Segment<T> {
    // 線分と線分との距離
    pub fn dist2_frac(&self, rhs: &Segment<T>) -> Frac<T> {
        if self.cross(rhs) { return Frac::zero(); }
        let candidate = [
            self.p.dist2_segment_frac(rhs), self.q.dist2_segment_frac(rhs),
            rhs.p.dist2_segment_frac(self), rhs.q.dist2_segment_frac(self)];
        candidate.iter().min_by(|&a, &b| a.partial_cmp(b).unwrap()).cloned().unwrap()
    }
    pub fn dist(&self, rhs: &Segment<T>) -> f64 {
        self.dist2_frac(rhs).to_f64().sqrt()
    }
}

// 向きが反対でも同じとみなす
impl<T: Num> PartialEq for Segment<T> {
    fn eq(&self, rhs: &Self) -> bool {
        (self.p == rhs.p && self.q == rhs.q)
        || (self.p == rhs.q && self.q == rhs.p)
    }
}

// 半直線 p -> q
#[derive(Debug, Clone, Copy, Default)]
pub struct HalfLine<T> {
    pub p: P<T>, pub q: P<T>,
}

impl<T: Num> HalfLine<T> {
    pub fn to_f64(&self) -> HalfLine<f64> { HalfLine::new((self.p.to_f64(), self.q.to_f64())) }
    pub fn to_isize(&self) -> HalfLine<isize> { HalfLine::new((self.p.to_isize(), self.q.to_isize())) }
    pub fn to_frac(&self) -> HalfLine<Frac<T>> { HalfLine::new((self.p.to_frac(), self.q.to_frac())) }
    pub fn contains(&self, p: P<T>) -> bool {
        Line::from_halfline(self).contains(p) && (p - self.p).dot(self.q - self.p) >= T::default()
    }
}

// 起点と向きが同じであれば同じとみなす
impl<T: Num> PartialEq for HalfLine<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.p == rhs.p && (self.q - self.p).is_parallel(rhs.q - rhs.p)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Line<T> {
    pub p: P<T>, pub q: P<T>,
}

impl<T: Num> Line<T> {
    pub fn to_f64(&self) -> Line<f64> { Line::new((self.p.to_f64(), self.q.to_f64())) }
    pub fn to_isize(&self) -> Line<isize> { Line::new((self.p.to_isize(), self.q.to_isize())) }
    pub fn to_frac(&self) -> Line<Frac<T>> { Line::new((self.p.to_frac(), self.q.to_frac())) }
    pub fn contains(&self, p: P<T>) -> bool { (p - self.p).is_parallel(self.q - self.p) }
    // 直線と直線との交点、Point<T> / det が答え
    pub fn cross_point_frac(&self, rhs: &Line<T>) -> Option<P<Frac<T>>> {
        let det = self.to_vector().det(rhs.to_vector());
        if det == T::zero() { return None; } // 平行
        let r = self.p * det + self.to_vector() * rhs.to_vector().det(self.p - rhs.p);
        Some(r.to_frac() / det.to_frac())
    }
    pub fn cross_point(&self, rhs: &Line<T>) -> Option<P<T>> {
        let r = self.cross_point_frac(rhs)?;
        Some(r.calc())
    }
}

// 同一直線上にあれば同じとみなす
impl<T: Num> PartialEq for Line<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.contains(rhs.p) && self.contains(rhs.q)
    }
}

macro_rules! impl_lines { ($($ty:ty),*) => {$(
    impl<T: Num> $ty {
        pub fn new((p, q): (P<T>, P<T>)) -> Self { Self { p, q } }
        pub fn abs2(&self) -> T { self.to_vector().abs2() }
        pub fn from_segment(seg: &Segment<T>) -> Self { Self { p: seg.p, q: seg.q } }
        pub fn from_halfline(hl: &HalfLine<T>) -> Self { Self { p: hl.p, q: hl.q } }
        pub fn from_line(line: &Line<T>) -> Self { Self { p: line.p, q: line.q } }
        pub fn to_vector(&self) -> P<T> { self.q - self.p }
        pub fn to_segment(&self) -> Segment<T> { Segment { p: self.p, q: self.q } }
        pub fn to_halfline(&self) -> HalfLine<T> { HalfLine { p: self.p, q: self.q } }
        pub fn to_line(&self) -> Line<T> { Line { p: self.p, q: self.q } }
    }
)*};}

impl_lines!(Segment<T>, HalfLine<T>, Line<T>);


///////////////////////////////////////////////////////////
// テストとベンチマーク
// cargo test -r kyopro_geometry::tests::basic

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_eq_f64 {
        ($a:expr, $b:expr) => { assert!(($a - $b).abs() < 1e-9); };
    }

    #[test]
    fn basic() {
        let o = P::zero();
        let a = P::new((1, 2));
        let b = P::new((3, 3));
        let c = P::new((-2, -1));
        let d = P::new((2, 4));
        assert_eq!(a.abs2(), 5);
        assert_eq!(a * 2, P::new((2, 4)));
        assert_eq!(a + b, P::new((4, 5)));
        assert_eq!(a - b, P::new((-2, -1)));
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
        assert_eq!(cross_oa_ob, o);
        // 交点なし（2直線は平行）
        let cross_ob_ac = line_ob.cross_point(&line_ac);
        assert!(cross_ob_ac.is_none());
        // 交点は解析的に求められる
        let cross_loa_lbc = Line::from_segment(&oa).cross_point_frac(&Line::from_segment(&bc)).unwrap();
        assert_eq!(cross_loa_lbc, P::new((Frac::new(1, 2), Frac::one())));
        let cross_oa_lbc = oa.to_f64().cross_point_with_line(&Line::from_segment(&bc.to_f64())).unwrap();
        assert_eq!(cross_oa_lbc, P::new((0.5, 1.0)));
        let cross_oa_bc = oa.to_f64().cross_point(&bc.to_f64()).unwrap();
        assert_eq!(cross_oa_bc, P::new((0.5, 1.0)));
        let cross_ad_bc = ad.cross_point(&bc);
        assert!(cross_ad_bc.is_none());
        // 交点との距離
        let dist_a_ob = a.to_f64().dist_segment(&ob.to_f64());
        assert_eq_f64!(dist_a_ob, 1.0 / 2.0_f64.sqrt());
        let dist_b_ac = b.to_f64().dist_segment(&ac.to_f64());
        assert_eq_f64!(dist_b_ac, 5.0_f64.sqrt());
        let dist_ad_ob = ad.to_f64().dist(&ob.to_f64());
        assert_eq_f64!(dist_ad_ob, 1.0 / 2.0_f64.sqrt());
        // 線と点の包含判定
        assert!(ob.contains(o));
        assert!(ob.contains(b));
        assert!(!ob.contains(a));
        assert!(!ob.contains(b * 5));
        assert!(ob.to_halfline().contains(b * 5));
        assert!(!ob.to_halfline().contains(-b));
        assert!(ob.to_line().contains(-b));
        assert!(!ob.to_f64().contains(b.to_f64() / 2.0 + a.to_f64() * 0.01));
        assert!(!ob.to_f64().contains(b.to_f64() / 2.0 - a.to_f64() * 0.01));
        assert!(!ob.to_f64().contains(b.to_f64() * -0.01));
        assert!(!ob.to_f64().contains(b.to_f64() * 1.01));

        // 線分・直線・半直線と線分の交差判定
        let oa_hl = oa.to_halfline();
        let ac_hl = ac.to_halfline();
        let ob_hl = ob.to_halfline();
        let oa_l = oa.to_line();
        let ac_l = ac.to_line();
        let ob_l = ob.to_line();
        let ld = P::new((-10, -10));
        let rd = P::new((10, -10));
        let lu = P::new((-10, 10));
        let ru = P::new((10, 10));
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
        assert_eq!(u.cross_point_with_halfline_frac(&oa_hl), Some(P::new((5, 10)).to_frac()));
        assert!(!d.cross_with_halfline(&oa_hl));
        assert_eq!(l.cross_point_with_halfline_frac(&ac_hl), Some(P::new((-10, -9)).to_frac()));
        assert!(!r.cross_with_halfline(&ac_hl));
        assert!(!u.cross_with_halfline(&ac_hl));
        assert!(!d.cross_with_halfline(&ac_hl));
        // 直線は2方向で交差する
        assert!(!l.cross_with_line(&oa_l));
        assert!(!r.cross_with_line(&oa_l));
        assert_eq!(u.cross_point_with_line_frac(&oa_l), Some(P::new((5, 10)).to_frac()));
        assert_eq!(u.to_f64().cross_point_with_line(&oa_l.to_f64()), Some(P::new((5, 10)).to_f64()));
        assert_eq!(d.cross_point_with_line_frac(&oa_l), Some(-P::new((5, 10)).to_frac()));
        assert_eq!(l.cross_point_with_line_frac(&ac_l), Some(P::new((-10, -9)).to_frac()));
        assert!(!r.cross_with_line(&ac_l));
        assert_eq!(u.cross_point_with_line_frac(&ac_l), Some(P::new((9, 10)).to_frac()));
        assert!(!d.cross_with_line(&ac_l));
        // 端に重なる場合は、両方向で交差する
        assert!(!l.cross_with_halfline(&ob_hl));
        assert_eq!(r.cross_point_with_halfline_frac(&ob_hl), Some(ru.to_frac()));
        assert_eq!(u.cross_point_with_halfline_frac(&ob_hl), Some(ru.to_frac()));
        assert!(!d.cross_with_halfline(&ob_hl));
        assert_eq!(l.cross_point_with_line_frac(&ob_l), Some(ld.to_frac()));
        assert_eq!(r.cross_point_with_line_frac(&ob_l), Some(ru.to_frac()));
        assert_eq!(u.cross_point_with_line_frac(&ob_l), Some(ru.to_frac()));
        assert_eq!(d.cross_point_with_line_frac(&ob_l), Some(ld.to_frac()));

        // オーバーフローする場合（isize）アルゴリズムを修正してオーバーフローを回避
        const MAX_LENGTH: isize = 1e5 as isize;
        let a = P::new((-MAX_LENGTH, MAX_LENGTH));
        let b = P::new((MAX_LENGTH - 1, 0));
        let p = P(-1, MAX_LENGTH - 1);
        let ab = Segment::new((a, b));
        let op_l = Line::new((o, p));
        let op_hl = HalfLine::new((o, p));
        let op_seg = Segment::new((o, p));
        assert!(ab.cross_with_line(&op_l)); // success
        assert!(ab.cross(&op_seg)); // success
        assert!(ab.cross_with_halfline(&op_hl));    // overflow for naive algorithm
        // オーバーフローしない場合（f64）
        const MAX_LENGTH2: f64 = 1e5;
        let a = P::new((-MAX_LENGTH2, MAX_LENGTH2));
        let b = P::new((MAX_LENGTH2 - 1.0, 0.0));
        let o = P::zero();
        let p = P(-1.0, MAX_LENGTH2 - 1.0);
        let ab = Segment::new((a, b));
        let op_l = Line::new((o, p));
        let op_hl = HalfLine::new((o, p));
        let op_seg = Segment::new((o, p));
        assert!(ab.cross_with_line(&op_l)); // success
        assert!(ab.cross(&op_seg)); // success
        assert!(ab.cross_with_halfline(&op_hl));    // no overflow
        // オーバーフローしない場合（i128）
        const MAX_LENGTH3: i128 = 1e5 as i128;
        let a = P::new((-MAX_LENGTH3, MAX_LENGTH3));
        let b = P::new((MAX_LENGTH3 - 1, 0));
        let o = P::zero();
        let p = P(-1, MAX_LENGTH3 - 1);
        let ab = Segment::new((a, b));
        let op_l = Line::new((o, p));
        let op_hl = HalfLine::new((o, p));
        let op_seg = Segment::new((o, p));
        assert!(ab.cross_with_line(&op_l)); // success
        assert!(ab.cross(&op_seg)); // success
        assert!(ab.cross_with_halfline(&op_hl));    // no overflow
    }
}
