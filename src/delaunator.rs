#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
//! Implements the Delaunay triangulation algorithm.
//!
//! This module was ported from the original [Delaunator](https://github.com/mapbox/delaunator), by Mapbox. If a triangulation is possible a given set of points in the 2D space, it returns a [`Triangulation`] structure. This structure contains three main components: [`triangles`], [`halfedges`] and [`hull`]:
//! ```ignore
//! let coords = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
//! let (delaunay, _) = delaunator::triangulate_from_tuple(&coords).unwrap();
//! ```
//! - `triangles`: A `Vec<usize>` that contains the indices for each vertex of a triangle in the original array. All triangles are directed counter-clockwise. To get the coordinates of all triangles, use:
//! ```ignore
//! let t = 0;
//! loop {
//!     println!("[{:?}, {:?}, {:?}]",
//!         coords[delaunay.triangles[t]],
//!         coords[delaunay.triangles[t+1]],
//!         coords[delaunay.triangles[t+2]],
//!     );
//!     t += 3;
//! }
//! ```
//! - `halfedges`:  `Vec<usize>` array of triangle half-edge indices that allows you to traverse the triangulation. i-th half-edge in the array corresponds to vertex `triangles[i]` the half-edge is coming from. `halfedges[i]` is the index of a twin half-edge in an adjacent triangle (or `INVALID_INDEX` for outer half-edges on the convex hull). The flat array-based data structures might be counterintuitive, but they're one of the key reasons this library is fast.
//! - `hull`: A `Vec<usize>` array of indices that reference points on the convex hull of the input data, counter-clockwise.
//!
//! The last two components, `inedges` and `outedges`, are for voronator internal use only.
//!
//! # Example
//!
//! ```
//! extern crate voronator;
//!
//! use voronator::delaunator::{Point, triangulate_from_tuple};
//!
//! fn main() {
//!     let points = vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
//!
//!     let (t, _) = triangulate_from_tuple::<Point>(&points)
//!         .expect("No triangulation exists for this input.");
//!
//!     for i in 0..t.len() {
//!         let i0 = t.triangles[3*i];
//!         let i1 = t.triangles[3*i + 1];
//!         let i2 = t.triangles[3*i + 2];
//!
//!         let p = vec![points[i0], points[i1], points[i2]];
//!
//!         println!("triangle {}: {:?}", i, p);
//!     }
//! }
//! ```
//!
//! [`Triangulation`]: ./struct.Triangulation.html
//! [`triangles`]: ./struct.Triangulation.html#structfield.triangles
//! [`halfedges`]: ./struct.Triangulation.html#structfield.halfedges
//! [`hull`]: ./struct.Triangulation.html#structfield.hull

use num::{Float, NumCast, Zero};
use rayon::prelude::*;
use std::{f64, fmt, usize};

/// Defines a comparison epsilon used for floating-point comparisons
pub const EPSILON: f64 = f64::EPSILON * 2.0;

/// Defines an invalid index in the Triangulation vectors
pub const INVALID_INDEX: usize = usize::max_value();

/// Trait for a coordinate (point) used to generate a Voronoi diagram. The default included struct `Point` is
/// included below as an example.
///
/// ```no_run
/// use num::Float;
/// use voronator::delaunator::Coord;
/// use std::fmt::Debug;
///
/// #[derive(Clone, PartialEq, Debug)]
/// /// Represents a point in the 2D space.
/// pub struct Point<F: Float + Send + Sync + Debug = f64> {
///    /// X coordinate of the point
///    pub x: F,
///    /// Y coordinate of the point
///    pub y: F,
/// }
///
/// impl<F: Float + Send + Sync + Debug> Coord for Point<F> {
///    type F = F;
///
///    // Inline these methods as otherwise we incur a heavy performance penalty
///    #[inline(always)]
///    fn from_xy(x: F, y: F) -> Self {
///        Point{x, y}
///    }
///    #[inline(always)]
///    fn x(&self) -> F {
///       self.x
///    }
///    #[inline(always)]
///    fn y(&self) -> F {
///        self.y
///    }
/// }
/// ```
///
use std::fmt::Debug;

pub trait Coord: Sync + Send + Clone + Debug {
    /// Floating point type for the coordinates
    type F: Float + Sync + Send + Debug;

    /// Create a coordinate from (x, y) positions
    fn from_xy(x: Self::F, y: Self::F) -> Self;
    /// Return x coordinate
    fn x(&self) -> Self::F;
    /// Return y coordinate
    fn y(&self) -> Self::F;

    /// Return the magnitude of the 2D vector represented by (x, y)
    #[inline]
    fn magnitude2(&self) -> Self::F {
        self.x() * self.x() + self.y() * self.y()
    }
}

#[inline]
fn vector<C: Coord>(p: &C, q: &C) -> C {
    C::from_xy(q.x() - p.x(), q.y() - p.y())
}

#[inline]
fn determinant<C: Coord>(p: &C, q: &C) -> C::F {
    p.x() * q.y() - p.y() * q.x()
}

#[inline]
fn dist2<C: Coord>(p: &C, q: &C) -> C::F {
    let d = vector(p, q);

    d.x() * d.x() + d.y() * d.y()
}

#[inline]
// https://floating-point-gui.de/errors/comparison/
fn nearly_equal<F: Float>(a: F, b: F, epsilon: F) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();

    if a == b {
        true
    } else if a == F::zero() || b == F::zero() || (abs_a + abs_b < F::min_positive_value()) {
        diff < (epsilon * F::min_positive_value())
    } else {
        diff / (abs_a + abs_b).min(F::max_value()) < epsilon
    }
}

/// Test whether two coordinates describe the same point in space
#[inline]
fn equals<C: Coord>(p: &C, q: &C) -> bool {
    nearly_equal(p.x(), q.x(), C::F::epsilon()) && nearly_equal(p.y(), q.y(), C::F::epsilon())
    //(p.x() - q.x()).abs() <= (C::F::min_positive_value() * num::cast(2.0).unwrap())
    //    && (p.y() - q.y()).abs() <= (C::F::min_positive_value() * num::cast(2.0).unwrap())
}

#[inline]
fn equals_with_span<C: Coord>(p: &C, q: &C, span: C::F) -> bool {
    let dist = dist2(p, q) / span;
    nearly_equal(dist, C::F::zero(), C::F::epsilon())
    //dist < num::cast(1e-20).unwrap() // dunno about this
}

#[derive(Clone, PartialEq, Debug)]
/// Represents a point in the 2D space.
pub struct Point<F: Float + Send + Sync + Debug = f64> {
    /// X coordinate of the point
    pub x: F,
    /// Y coordinate of the point
    pub y: F,
}

impl<F: Float + Send + Sync + Debug> Coord for Point<F> {
    type F = F;

    // Inline these methods as otherwise we incur a heavy performance penalty
    #[inline(always)]
    fn from_xy(x: F, y: F) -> Self {
        Point { x, y }
    }
    #[inline(always)]
    fn x(&self) -> F {
        self.x
    }
    #[inline(always)]
    fn y(&self) -> F {
        self.y
    }
}

/*impl<F: Float + Send + Sync> fmt::Debug for Point<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.x(), self.y())
    }
}*/

impl From<(f64, f64)> for Point<f64> {
    #[inline]
    fn from((x, y): (f64, f64)) -> Self {
        Point { x, y }
    }
}

#[inline]
fn in_circle<C: Coord>(p: &C, a: &C, b: &C, c: &C) -> bool {
    let zero = C::F::zero();

    let d = vector(p, a);
    let e = vector(p, b);
    let f = vector(p, c);

    let ap = d.x() * d.x() + d.y() * d.y();
    let bp = e.x() * e.x() + e.y() * e.y();
    let cp = f.x() * f.x() + f.y() * f.y();

    #[rustfmt::skip]
    let res = d.x() * (e.y() * cp  - bp  * f.y()) -
                   d.y() * (e.x() * cp  - bp  * f.x()) +
                   ap  * (e.x() * f.y() - e.y() * f.x()) ;

    res < zero
}

#[rustfmt::skip]
#[inline]
fn circumradius<C: Coord>(a: &C, b: &C, c: &C) -> C::F {
    let zero = C::F::zero();
    let point5: C::F = num::cast(0.5).unwrap();

    let d = vector(a, b);
    let e = vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = determinant(&d, &e);

    let x = (e.y() * bl - d.y() * cl) * (point5 / det);
    let y = (d.x() * cl - e.x() * bl) * (point5 / det);

    if (bl != zero) &&
       (cl != zero) &&
       (det != zero) {
        x * x + y * y
    } else {
        C::F::max_value()
    }
}

/// Calculates the circumcenter of a triangle, given it's three vertices
///
/// # Arguments
///
/// * `a` - The first vertex of the triangle
/// * `b` - The second vertex of the triangle
/// * `c` - The third vertex of the triangle
#[rustfmt::skip]
pub fn circumcenter<C: Coord>(a: &C, b: &C, c: &C) -> Option<C> {
    let zero = C::F::zero();
    let point5: C::F = num::cast(0.5).unwrap();

    let d = vector(a, b);
    let e = vector(a, c);

    let bl = d.magnitude2();
    let cl = e.magnitude2();
    let det = determinant(&d, &e);

    let x = (e.y() * bl - d.y() * cl) * (point5 / det);
    let y = (d.x() * cl - e.x() * bl) * (point5 / det);

    if (bl != zero) &&
       (cl != zero) &&
       (det != zero) {
        Some(C::from_xy(
            a.x() + x,
            a.y() + y)
        )
    } else {
        None
    }
}

fn counter_clockwise<C: Coord>(p0: &C, p1: &C, p2: &C) -> bool {
    let zero = C::F::zero();
    let large_number = num::cast(1e14).unwrap();

    let v0 = vector(p0, p1);
    let v1 = vector(p0, p2);
    let det = determinant(&v0, &v1);
    let dist = v0.magnitude2() + v1.magnitude2();

    if det == zero {
        return false;
    }

    let reldet = (dist / det).abs();

    if reldet > large_number {
        return false;
    }

    det > zero
}

/// Returs the next halfedge for a given halfedge
///
/// # Arguments
///
/// * `i` - The current halfedge index
#[inline]
pub fn next_halfedge(i: usize) -> usize {
    if i % 3 == 2 {
        i - 2
    } else {
        i + 1
    }
}

/// Returs the previous halfedge for a given halfedge
///
/// # Arguments
///
/// * `i` - The current halfedge index
#[inline]
pub fn prev_halfedge(i: usize) -> usize {
    if i % 3 == 0 {
        i + 2
    } else {
        i - 1
    }
}

/// Returns a vec containing indices for the 3 edges of a triangle t
///
/// # Arguments
///
/// * `t` - The triangle index
#[inline]
pub fn edges_of_triangle(t: usize) -> [usize; 3] {
    [3 * t, 3 * t + 1, 3 * t + 2]
}

/// Returns the triangle associated with the given edge
///
/// # Arguments
///
/// * `e` - The edge index
#[inline]
pub fn triangle_of_edge(e: usize) -> usize {
    ((e as f64) / 3.).floor() as usize
}

/// Returns a vec containing the indices of the corners of the given triangle
///
/// # Arguments
///
/// * `t` - The triangle index
/// * `delaunay` - A reference to a fully constructed Triangulation
#[inline]
pub fn points_of_triangle(t: usize, delaunay: &Triangulation) -> [usize; 3] {
    let mut triangle: [usize; 3] = [0; 3];
    let edges = edges_of_triangle(t);
    triangle[0] = delaunay.triangles[edges[0]];
    triangle[1] = delaunay.triangles[edges[1]];
    triangle[2] = delaunay.triangles[edges[2]];
    //edges.iter().map(|&e| delaunay.triangles[e]).collect()

    triangle
}

/// Returns a vec containing the indices for the adjacent triangles of the given triangle
///
/// # Arguments
///
/// * `t` - The triangle index
/// * `delaunay` - A reference to a fully constructed Triangulation
pub fn triangles_adjacent_to_triangle(t: usize, delaunay: &Triangulation) -> Vec<usize> {
    let mut adjacent_triangles: Vec<usize> = vec![];
    for &e in edges_of_triangle(t).iter() {
        let opposite = delaunay.halfedges[e];
        if opposite != INVALID_INDEX {
            adjacent_triangles.push(triangle_of_edge(opposite));
        }
    }
    adjacent_triangles
}

/// Returns a vec containing all edges around a point
///
/// # Arguments
///
/// * `start` - The start point index
/// * `delaunay` - A reference to a fully constructed Triangulation
#[inline]
pub fn edges_around_point(start: usize, delaunay: &Triangulation) -> Vec<usize> {
    let mut result: Vec<usize> = vec![];

    // If the starting index is invalid we can't continue
    if start == INVALID_INDEX {
        return result;
    }

    let mut incoming = start;
    loop {
        result.push(incoming);
        let outgoing = next_halfedge(incoming);
        incoming = delaunay.halfedges[outgoing];
        if incoming == INVALID_INDEX || incoming == start {
            break;
        }
    }
    result
}

/// Represents a Delaunay triangulation for a given set of points. See example in [`delaunator`] for usage details.
///
/// [`delaunator`]: ./index.html#example

pub struct Triangulation {
    /// Contains the indices for each vertex of a triangle in the original array. All triangles are directed counter-clockwise.
    pub triangles: Vec<usize>,
    /// A `Vec<usize>` of triangle half-edge indices that allows you to traverse the triangulation. i-th half-edge in the array corresponds to vertex `triangles[i]` the half-edge is coming from. `halfedges[i]` is the index of a twin half-edge in an adjacent triangle (or `INVALID_INDEX` for outer half-edges on the convex hull).
    pub halfedges: Vec<usize>,
    /// A `Vec<usize>` array of indices that reference points on the convex hull of the input data, counter-clockwise.
    pub hull: Vec<usize>,
    /// A `Vec<usize>` that contains indices for halfedges of points in the hull that points inwards to the diagram. Only for [`voronator`] internal use.
    ///
    /// [`voronator`]: ../index.html
    pub inedges: Vec<usize>,
    /// A `Vec<usize>` that contains indices for halfedges of points in the hull that points outwards to the diagram. Only for [`voronator`] internal use.
    ///
    /// [`voronator`]: ../index.html
    pub outedges: Vec<usize>,

    edge_stack: Vec<usize>,
}

impl Triangulation {
    fn new(n: usize) -> Self {
        let max_triangles = 2 * n - 5;
        Self {
            triangles: Vec::with_capacity(max_triangles * 3),
            halfedges: Vec::with_capacity(max_triangles * 3),
            hull: Vec::new(),
            inedges: vec![INVALID_INDEX; n],
            outedges: vec![INVALID_INDEX; n],
            edge_stack: Vec::new(),
        }
    }

    /// Returns the number of triangles calculated in the triangulation. Same as `triangles.len() / 3`.
    #[inline]
    pub fn len(&self) -> usize {
        self.triangles.len() / 3
    }

    fn legalize<C: Coord>(&mut self, p: usize, points: &[C], hull: &mut Hull<C>) -> usize {
        /* if the pair of triangles doesn't satisfy the Delaunay condition
         * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
         * then do the same check/flip recursively for the new pair of triangles
         *
         *           pl                    pl
         *          /||\                  /  \
         *       al/ || \bl            al/    \a
         *        /  ||  \              /      \
         *       /  a||b  \    flip    /___ar___\
         *     p0\   ||   /p1   =>   p0\---bl---/p1
         *        \  ||  /              \      /
         *       ar\ || /br             b\    /br
         *          \||/                  \  /
         *           pr                    pr
         */
        let mut i: usize = 0;
        let mut ar;
        let mut a = p;

        self.edge_stack.clear();

        loop {
            let b = self.halfedges[a];
            ar = prev_halfedge(a);

            if b == INVALID_INDEX {
                if i > 0 {
                    i -= 1;
                    a = self.edge_stack[i];
                    continue;
                } else {
                    break;
                }
            }

            let al = next_halfedge(a);
            let bl = prev_halfedge(b);

            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];

            let illegal = in_circle(&points[p1], &points[p0], &points[pr], &points[pl]);
            if illegal {
                self.triangles[a] = p1;
                self.triangles[b] = p0;

                let hbl = self.halfedges[bl];

                // Edge swapped on the other side of the hull (rare).
                // Fix the halfedge reference
                if hbl == INVALID_INDEX {
                    let mut e = hull.start;
                    loop {
                        if hull.tri[e] == bl {
                            hull.tri[e] = a;
                            break;
                        }

                        e = hull.prev[e];

                        if e == hull.start {
                            break;
                        }
                    }
                }

                self.link(a, hbl);
                self.link(b, self.halfedges[ar]);
                self.link(ar, bl);

                let br = next_halfedge(b);

                if i < self.edge_stack.len() {
                    self.edge_stack[i] = br;
                } else {
                    self.edge_stack.push(br);
                }

                i += 1;
            } else if i > 0 {
                i -= 1;
                a = self.edge_stack[i];
                continue;
            } else {
                break;
            }
        }

        ar
    }

    fn link(&mut self, a: usize, b: usize) {
        let s: usize = self.halfedges.len();

        match a.cmp(&s) {
            std::cmp::Ordering::Equal => self.halfedges.push(b),
            std::cmp::Ordering::Less => self.halfedges[a] = b,
            _ => panic!("Cannot link edge"),
        }

        /*if a == s {
            self.halfedges.push(b);
        } else if a < s {
            self.halfedges[a] = b;
        } else {
            // todo: fix hard error, make it recoverable or graceful
            panic!("Cannot link edge")
        }*/

        if b != INVALID_INDEX {
            let s2: usize = self.halfedges.len();

            match b.cmp(&s2) {
                std::cmp::Ordering::Equal => self.halfedges.push(a),
                std::cmp::Ordering::Less => self.halfedges[b] = a,
                _ => panic!("Cannot link edge"),
            }
            /*if b == s2 {
                self.halfedges.push(a);
            } else if b < s2 {
                self.halfedges[b] = a;
            } else {
                // todo: fix hard error, make it recoverable or graceful
                panic!("Cannot link edge")
            }*/
        }
    }

    fn add_triangle(
        &mut self,
        i0: usize,
        i1: usize,
        i2: usize,
        a: usize,
        b: usize,
        c: usize,
    ) -> usize {
        let t: usize = self.triangles.len();

        // eprintln!("adding triangle [{}, {}, {}]", i0, i1, i2);

        self.triangles.push(i0);
        self.triangles.push(i1);
        self.triangles.push(i2);

        self.link(t, a);
        self.link(t + 1, b);
        self.link(t + 2, c);

        t
    }
}

//@see https://stackoverflow.com/questions/33333363/built-in-mod-vs-custom-mod-function-improve-the-performance-of-modulus-op/33333636#33333636
#[inline]
fn fast_mod(i: usize, c: usize) -> usize {
    if i >= c {
        i % c
    } else {
        i
    }
}

// monotonically increases with real angle,
// but doesn't need expensive trigonometry
#[inline]
fn pseudo_angle<C: Coord>(d: &C) -> C::F {
    let zero: C::F = C::F::zero();
    let quarter: C::F = num::cast(0.25).unwrap();

    let p = d.x() / (d.x().abs() + d.y().abs());

    if d.y() > zero {
        let value: C::F = num::cast(3.0).unwrap();

        (value - p) * quarter
    } else {
        let value: C::F = num::cast(1.0).unwrap();

        (value + p) * quarter
    }
}

struct Hull<C: Coord> {
    prev: Vec<usize>,
    next: Vec<usize>,
    tri: Vec<usize>,
    hash: Vec<usize>,
    start: usize,
    center: C,
}

impl<C: Coord> Hull<C> {
    fn new(n: usize, center: C, i0: usize, i1: usize, i2: usize, points: &[C]) -> Self {
        // initialize a hash table for storing edges of the advancing convex hull
        let hash_len = (n as f64).sqrt().ceil() as usize;

        let mut hull = Self {
            prev: vec![0; n],
            next: vec![0; n],
            tri: vec![0; n],
            hash: vec![INVALID_INDEX; hash_len],
            start: i0,
            center,
        };

        hull.next[i0] = i1;
        hull.prev[i2] = i1;
        hull.next[i1] = i2;
        hull.prev[i0] = i2;
        hull.next[i2] = i0;
        hull.prev[i1] = i0;

        hull.tri[i0] = 0;
        hull.tri[i1] = 1;
        hull.tri[i2] = 2;

        // todo here

        hull.hash_edge(&points[i0], i0);
        hull.hash_edge(&points[i1], i1);
        hull.hash_edge(&points[i2], i2);

        hull
    }

    #[inline]
    fn hash_key(&self, p: &C) -> usize {
        let d = vector(&self.center, p);

        let angle: C::F = pseudo_angle(&d);
        let len: C::F = num::cast(self.hash.len()).unwrap();

        let to_mod: usize = num::cast((angle * len).floor()).unwrap();

        fast_mod(to_mod, self.hash.len())
    }

    #[inline]
    fn hash_edge(&mut self, p: &C, i: usize) {
        let key = self.hash_key(p);
        self.hash[key] = i;
    }

    fn find_visible_edge(&self, p: &C, span: C::F, points: &[C]) -> (usize, bool) {
        // find a visible edge on the convex hull using edge hash
        let mut start = 0;
        let key = self.hash_key(p);

        for j in 0..self.hash.len() {
            let index = fast_mod(key + j, self.hash.len());
            start = self.hash[index];
            if start != INVALID_INDEX && start != self.next[start] {
                break;
            }
        }

        // Make sure what we found is on the hull.
        // todo: return something that represents failure to fail gracefully instead
        if self.prev[start] == start || self.prev[start] == INVALID_INDEX {
            panic!("not in the hull");
        }

        start = self.prev[start];
        let mut e = start;
        let mut q: usize;

        //eprintln!("starting advancing...");

        // Advance until we find a place in the hull where our current point
        // can be added.
        loop {
            q = self.next[e];
            // eprintln!("p: {:?}, e: {:?}, q: {:?}", p, &points[e], &points[q]);
            if equals_with_span(p, &points[e], span) || equals_with_span(p, &points[q], span) {
                eprintln!("p: {:?}, e: {:?}, q: {:?}", p, &points[e], &points[q]);
                e = INVALID_INDEX;
                break;
            }
            if counter_clockwise(p, &points[e], &points[q]) {
                break;
            }
            e = q;
            if e == start {
                eprintln!("p: {:?}, e: {:?}, q: {:?}", p, &points[e], &points[q]);
                e = INVALID_INDEX;
                break;
            }
        }
        //eprintln!("returning from find_visible_edge...");
        (e, e == start)
    }
}

fn calculate_bbox_center<C: Coord>(points: &[C]) -> (C, C::F) {
    let mut max = Point {
        x: C::F::min_value(),
        y: C::F::min_value(),
    };
    let mut min = Point {
        x: C::F::max_value(),
        y: C::F::max_value(),
    };

    for point in points {
        min.x = min.x.min(point.x());
        min.y = min.y.min(point.y());
        max.x = max.x.max(point.x());
        max.y = max.y.max(point.y());
    }

    let half = num::cast(0.5).unwrap();

    let width = max.x - min.x;
    let height = max.y - min.y;
    let span = width * width + height * height;

    (
        C::from_xy((min.x + max.x) * half, (min.y + max.y) * half),
        span,
    )
}

fn find_closest_point<C: Coord>(points: &[C], p: &C) -> usize {
    let zero = C::F::zero();
    let mut min_dist = C::F::max_value();
    let mut k = INVALID_INDEX;

    for (i, q) in points.iter().enumerate() {
        if i != k {
            let d = dist2(p, q);

            if d < min_dist && d > zero {
                k = i;
                min_dist = d;
            }
        }
    }

    k
}

fn find_seed_triangle<C: Coord>(center: &C, points: &[C]) -> Option<(usize, usize, usize)> {
    let i0 = find_closest_point(points, center);
    let p0 = &points[i0];

    let i1 = find_closest_point(points, p0);
    let p1 = &points[i1];

    // find the third point which forms the smallest circumcircle
    // with the first two
    let mut min_radius = C::F::max_value();
    let mut i2 = INVALID_INDEX;
    for (i, p) in points.iter().enumerate() {
        if i != i0 && i != i1 {
            let r = circumradius(p0, p1, p);

            if r < min_radius {
                i2 = i;
                min_radius = r;
            }
        }
    }

    if min_radius == C::F::max_value() {
        None
    } else {
        match counter_clockwise(p0, p1, &points[i2]) {
            true => Some((i0, i2, i1)),
            false => Some((i0, i1, i2)),
        }
    }
}

fn to_points<C: Coord>(coords: &[C::F]) -> Vec<C> {
    coords
        .chunks(2)
        .map(|tuple| C::from_xy(tuple[0], tuple[1]))
        .collect()
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D
/// points.
///
/// Points are passed a flat array of `f64` of size `2n`, where n is the
/// number of points and for each point `i`, `{x = 2i, y = 2i + 1}` and
/// converted internally to `delaunator::Point`. It returns both the triangulation
/// and the vector of `delaunator::Point` to be used, if desired.
///
/// # Arguments
///
/// * `coords` - A vector of `f64` of size `2n`, where for each point `i`, `x = 2i`
/// and y = `2i + 1`.
pub fn triangulate_from_arr<C: Coord>(coords: &[C::F]) -> Option<(Triangulation, Vec<C>)> {
    let n = coords.len();

    if n % 2 != 0 {
        return None;
    }

    let points = to_points(coords);
    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D
/// points.
///
/// Points are passed as a tuple, `(f64, f64)`, and converted internally
/// to `delaunator::Point`. It returns both the triangulation and the vector of
/// Points to be used, if desired.
///
/// # Arguments
///
/// * `coords` - A vector of tuples, where each tuple is a `(f64, f64)`
pub fn triangulate_from_tuple<C: Coord>(
    coords: &[(C::F, C::F)],
) -> Option<(Triangulation, Vec<C>)> {
    let points: Vec<C> = coords.iter().map(|p| C::from_xy(p.0, p.1)).collect();

    let triangulation = triangulate(&points)?;

    Some((triangulation, points))
}

/// Calculates the Delaunay triangulation, if it exists, for a given set of 2D points
///
/// # Arguments
///
/// * `points` - The set of points
pub fn triangulate<C: Coord>(points: &[C]) -> Option<Triangulation> {
    if points.len() < 3 {
        return None;
    }

    // eprintln!("triangulating {} points...", points.len());

    //eprintln!("calculating bbox and seeds...");
    let (center_bbox, span) = calculate_bbox_center(points);
    let (i0, i1, i2) = find_seed_triangle(&center_bbox, points)?;

    let p0 = &points[i0];
    let p1 = &points[i1];
    let p2 = &points[i2];

    let center = circumcenter(p0, p1, p2)?;

    //eprintln!("calculating dists...");

    // Calculate the distances from the center once to avoid having to
    // calculate for each compare.
    let mut dists: Vec<(usize, C::F)> = points
        .par_iter()
        .enumerate()
        .map(|(i, _)| (i, dist2(&points[i], &center)))
        .collect();

    // sort the points by distance from the seed triangle circumcenter
    // Using sort_by and not sort_unstable_by reduces errors when swapping to f32 points. Not entirely sure why, maybe because the input is originally ordered by x-axis and this helps?
    dists.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

    //eprintln!("creating hull...");
    let mut hull = Hull::new(points.len(), center, i0, i1, i2, points);

    //eprintln!("calculating triangulation...");
    let mut triangulation = Triangulation::new(points.len());
    triangulation.add_triangle(i0, i1, i2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);

    let mut pp = &C::from_xy(C::F::nan(), C::F::nan());

    //eprintln!("iterating points...");
    // go through points based on distance from the center.
    for &(i, _dist) in dists.iter() {
        let p = &points[i];

        // This probably doesn't work as intended - if there are multiple points with the same distance from the center, but at opposite ends of the 'circle' around
        // the central point,
        // skip near-duplicate points
        /*if equals(p, pp) {
            continue;
        }*/

        // skip seed triangle points
        if i == i0 || i == i1 || i == i2 {
            continue;
        }

        // We don't need to clone here, can just keep track of the last point
        pp = p; //.clone();

        //eprintln!("finding visible edge...");
        let (mut e, backwards) = hull.find_visible_edge(p, span, points);
        if e == INVALID_INDEX {
            continue;
        }

        // add the first triangle from the point
        // eprintln!("first triangle of iteration");
        let mut t = triangulation.add_triangle(
            e,
            i,
            hull.next[e],
            INVALID_INDEX,
            INVALID_INDEX,
            hull.tri[e],
        );

        hull.tri[i] = triangulation.legalize(t + 2, points, &mut hull);
        hull.tri[e] = t;

        //eprintln!("walking forward in hull...");
        // walk forward through the hull, adding more triangles and
        // flipping recursively
        let mut next = hull.next[e];
        loop {
            let q = hull.next[next];
            if !counter_clockwise(p, &points[next], &points[q]) {
                break;
            }
            t = triangulation.add_triangle(next, i, q, hull.tri[i], INVALID_INDEX, hull.tri[next]);

            hull.tri[i] = triangulation.legalize(t + 2, points, &mut hull);
            hull.next[next] = next;
            next = q;
        }

        //eprintln!("walking backwards in hull...");
        // walk backward from the other side, adding more triangles
        // and flipping
        if backwards {
            loop {
                let q = hull.prev[e];
                if !counter_clockwise(p, &points[q], &points[e]) {
                    break;
                }
                t = triangulation.add_triangle(q, i, e, INVALID_INDEX, hull.tri[e], hull.tri[q]);
                triangulation.legalize(t + 2, points, &mut hull);
                hull.tri[q] = t;
                hull.next[e] = e;
                e = q;
            }
        }

        // update the hull indices
        hull.prev[i] = e;
        hull.next[e] = i;
        hull.prev[next] = i;
        hull.next[i] = next;
        hull.start = e;

        hull.hash_edge(p, i);
        hull.hash_edge(&points[e], e);
    }

    for e in 0..triangulation.triangles.len() {
        let endpoint = triangulation.triangles[next_halfedge(e)];
        if triangulation.halfedges[e] == INVALID_INDEX
            || triangulation.inedges[endpoint] == INVALID_INDEX
        {
            triangulation.inedges[endpoint] = e;
        }
    }

    let mut vert0: usize;
    let mut vert1 = hull.start;
    loop {
        vert0 = vert1;
        vert1 = hull.next[vert1];
        triangulation.inedges[vert1] = hull.tri[vert0];
        triangulation.outedges[vert0] = hull.tri[vert1];
        if vert1 == hull.start {
            break;
        }
    }

    //eprintln!("copying hull...");
    let mut e = hull.start;
    loop {
        triangulation.hull.push(e);
        e = hull.next[e];
        if e == hull.start {
            break;
        }
    }

    //eprintln!("done");

    Some(triangulation)
}
