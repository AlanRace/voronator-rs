//! Provides functions for handling polygons.
//!
//! Polygons are stored as a Vec<Point>
//!
//! # Example
//!
//! ```no_run
//! extern crate voronator;
//!
//! use voronator::delaunator::Point;
//! use voronator::polygon::Polygon;
//!
//! fn main() {
//!     let points = vec![Point{x: 0., y: 0.}, Point{x: 1., y: 0.}, Point{x: 1., y: 1.}, Point{x: 0., y: 1.}];
//!     let polygon = Polygon::from_points(points);
//! }
//!

use crate::delaunator::Coord;
use num::Zero;

/// Represents a polygon.
pub struct Polygon<C>
where
    //F: Float + Sync + Send,
    C: Coord,
{
    pub(crate) points: Vec<C>,
    is_clipped: bool,
}

impl<C> Default for Polygon<C>
where
    //F: Float + Sync + Send,
    C: Coord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> Polygon<C>
where
    //F: Float + Sync + Send,
    C: Coord,
{
    /// Create an empty polygon with no points.
    pub fn new() -> Self {
        Polygon {
            points: Vec::new(),
            is_clipped: false,
        }
    }

    /// Create a polygon consisting of the points supplied.
    pub fn from_points(points: Vec<C>) -> Self {
        Polygon {
            points,
            is_clipped: false,
        }
    }

    /// Return a slice of points representing the polygon.
    pub fn points(&self) -> &[C] {
        &self.points
    }

    /// Return whether the polygon has been clipped.
    pub fn is_clipped(&self) -> bool {
        self.is_clipped
    }

    /// Return area of polygon.
    pub fn area(&self) -> Option<C::F> {
        match self.points.len() {
            0 => None,
            1 => Some(C::F::zero()),
            _ => {
                let mut area_sum_double: C::F = C::F::zero();

                for i in 0..self.points.len() - 1 {
                    area_sum_double = area_sum_double
                        + triangle_area2(&self.points[0], &self.points[i], &self.points[i + 1]);
                }

                Some(area_sum_double / num::cast(2.0).unwrap())
            }
        }
    }

    /// Return the centroid of polygon.
    pub fn centroid(&self) -> Option<C> {
        match self.points.len() {
            0 => None,
            1 => Some(C::from_xy(self.points[0].x(), self.points[0].y())),
            _ => {
                let mut x: C::F = C::F::zero();
                let mut y: C::F = C::F::zero();
                let mut area_sum_double: C::F = C::F::zero();

                for i in 0..self.points.len() - 1 {
                    let triangle_cent3 =
                        triangle_centroid3(&self.points[0], &self.points[i], &self.points[i + 1]);
                    let area_squared =
                        triangle_area2(&self.points[0], &self.points[i], &self.points[i + 1]);

                    x = x + (area_squared * triangle_cent3.x());
                    y = y + (area_squared * triangle_cent3.y());

                    area_sum_double = area_sum_double + area_squared
                }

                Some(C::from_xy(
                    x / num::cast(3.0).unwrap() / area_sum_double,
                    y / num::cast(3.0).unwrap() / area_sum_double,
                ))
            }
        }
    }
}

fn triangle_centroid3<C: Coord>(p1: &C, p2: &C, p3: &C) -> C {
    C::from_xy(p1.x() + p2.x() + p3.x(), p1.y() + p2.y() + p3.y())
}

fn triangle_area2<C: Coord>(p1: &C, p2: &C, p3: &C) -> C::F {
    (p2.x() - p1.x()) * (p3.y() - p1.y()) - (p3.x() - p1.x()) * (p2.y() - p1.y())
}

fn inside<C: Coord>(p: &C, p1: &C, p2: &C) -> bool {
    (p2.y() - p1.y()) * p.x() + (p1.x() - p2.x()) * p.y() + (p2.x() * p1.y() - p1.x() * p2.y())
        < C::F::zero()
}

fn intersection<C: Coord>(cp1: &C, cp2: &C, s: &C, e: &C) -> C {
    let dc = C::from_xy(cp1.x() - cp2.x(), cp1.y() - cp2.y());
    let dp = C::from_xy(s.x() - e.x(), s.y() - e.y());

    let n1 = cp1.x() * cp2.y() - cp1.y() * cp2.x();
    let n2 = s.x() * e.y() - s.y() * e.x();

    let one: C::F = num::cast(1.0).unwrap();

    let n3 = one / (dc.x() * dp.y() - dc.y() * dp.x());

    C::from_xy(
        (n1 * dp.x() - n2 * dc.x()) * n3,
        (n1 * dp.y() - n2 * dc.y()) * n3,
    )
}

/// Sutherland-Hodgman clipping modified from https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#C.2B.2B
pub fn sutherland_hodgman<C>(subject: &Polygon<C>, clip: &Polygon<C>) -> Polygon<C>
where
    //    F: Float + Sync + Send,
    C: Coord,
{
    let mut output_polygon = Polygon::new();
    let mut input_polygon = Polygon::new();

    let mut clipped = false;
    output_polygon.points.clone_from(&subject.points);

    let mut new_polygon_size = subject.points.len();

    for j in 0..clip.points.len() {
        // copy new polygon to input polygon & set counter to 0
        input_polygon.points.clear();
        input_polygon.points.clone_from(&output_polygon.points);

        let mut counter = 0;
        output_polygon.points.clear();

        // get clipping polygon edge
        let cp1 = &clip.points[j];
        let cp2 = &clip.points[(j + 1) % clip.points.len()];

        for i in 0..new_polygon_size {
            // get subject polygon edge
            let s = &input_polygon.points[i];
            let e = &input_polygon.points[(i + 1) % new_polygon_size];

            // Case 1: Both vertices are inside:
            // Only the second vertex is added to the output list
            if inside(s, cp1, cp2) && inside(e, cp1, cp2) {
                output_polygon.points.push(e.clone());
                counter += 1;

            // Case 2: First vertex is outside while second one is inside:
            // Both the point of intersection of the edge with the clip boundary
            // and the second vertex are added to the output list
            } else if !inside(s, cp1, cp2) && inside(e, cp1, cp2) {
                output_polygon.points.push(intersection(cp1, cp2, s, e));
                output_polygon.points.push(e.clone());

                clipped = true;
                counter += 1;
                counter += 1;

            // Case 3: First vertex is inside while second one is outside:
            // Only the point of intersection of the edge with the clip boundary
            // is added to the output list
            } else if inside(s, cp1, cp2) && !inside(e, cp1, cp2) {
                output_polygon.points.push(intersection(cp1, cp2, s, e));
                clipped = true;
                counter += 1;

                // Case 4: Both vertices are outside
                //} else if !inside(s, cp1, cp2) && !inside(e, cp1, cp2) {
                // No vertices are added to the output list
            }
        }
        // set new polygon size
        new_polygon_size = counter;
    }

    //println!("Clipped? {}", clipped);

    output_polygon.is_clipped = clipped;

    output_polygon
}
