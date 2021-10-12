#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
//! Constructs a Voronoi diagram given a set of points.
//!
//! This module was adapted from [d3-delaunay](https://github.com/d3/d3-delaunay) and from
//! [Red Blog Games](https://www.redblobgames.com/x/2022-voronoi-maps-tutorial/) Voronoi maps tutorial.
//! It implements the Delaunay triangulation dual extraction, which is the Voronoi diagram.
//! It also implements a centroidal tesselation based on the Voronoi diagram, but using centroids
//! instead of circumcenters for the vertices of the cell polygons.
//!
//! Apart from the triangle center they are using, the Voronoi and Centroidal diagrams differ
//! in how they handle the hull cells. The Voronoi diagram implements a clipping algorithm that
//! clips the diagram into a bounding box, thus extracting neat polygons around the hull. The
//! Centroid diagram, in the other hand, doesn't. The outer cells can be missing or be distorted,
//! as triangles calculated by the Delaunay triangulation can be too thin in the hull, causing
//! centroid calculation to be "bad".
//!
//! If you have a robust solution for this particular problem, please let me know either by
//! creating an issue or through a pull-request, and I will make sure to add your solution with
//! the proper credits.
//!
//! # Example
//!
//! ## Voronoi Diagram
//! ```rust
//! extern crate voronator;
//! extern crate rand;
//!
//! use voronator::VoronoiDiagram;
//! use rand::prelude::*;
//! use rand::distributions::Uniform;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let range1 = Uniform::new(0., 100.);
//!     let range2 = Uniform::new(0., 100.);
//!     let mut points: Vec<(f64, f64)> = (0..10)
//!         .map(|_| (rng.sample(&range1), rng.sample(&range2)))
//!         .collect();
//!
//!     let diagram = VoronoiDiagram::from_tuple(&(0., 0.), &(100., 100.), &points).unwrap();
//!     
//!     for cell in diagram.cells() {
//!         let p: Vec<(f32, f32)> = cell.points().into_iter()
//!             .map(|x| (x.x as f32, x.y as f32))
//!             .collect();
//!         
//!         println!("{:?}", p);
//!     }
//! }
//! ```
//!
//! ## Centroidal Tesselation Diagram
//! ```rust
//! extern crate voronator;
//! extern crate rand;
//!
//! use voronator::CentroidDiagram;
//! use rand::prelude::*;
//! use rand::distributions::Uniform;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let range1 = Uniform::new(0., 100.);
//!     let range2 = Uniform::new(0., 100.);
//!     let mut points: Vec<(f64, f64)> = (0..10)
//!         .map(|_| (rng.sample(&range1), rng.sample(&range2)))
//!         .collect();
//!
//!     let diagram = CentroidDiagram::from_tuple(&points).unwrap();
//!     
//!     for cell in diagram.cells {
//!         let p: Vec<(f32, f32)> = cell.into_iter()
//!             .map(|x| (x.x as f32, x.y as f32))
//!             .collect();
//!         
//!         println!("{:?}", p);
//!     }
//! }
//! ```

pub mod polygon;
pub mod delaunator;

use std::{f64, usize};

use crate::delaunator::*;
use crate::polygon::*;

/// Represents a centroidal tesselation diagram.
pub struct CentroidDiagram {
    /// Contains the input data
    pub sites: Vec<Point>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    ///
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the centroid of each triangle
    pub centers: Vec<Point>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    pub cells: Vec<Vec<Point>>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,
}

impl CentroidDiagram {
    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `delaunator::Point`.
    pub fn new(points: &[Point]) -> Option<Self> {
        let delaunay = triangulate(points)?;
        let centers = calculate_centroids(points, &delaunay);
        let cells = CentroidDiagram::calculate_polygons(points, &centers, &delaunay);
        let neighbors = calculate_neighbors(points, &delaunay);
        Some(CentroidDiagram {
            sites: points.to_vec(),
            delaunay,
            centers,
            cells,
            neighbors,
        })
    }

    /// Creates a centroidal tesselation, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<Point> = coords.iter().map(|p| Point::from(*p)).collect();
        CentroidDiagram::new(&points)
    }

    fn calculate_polygons(
        points: &[Point],
        centers: &[Point],
        delaunay: &Triangulation,
    ) -> Vec<Vec<Point>> {
        let mut polygons: Vec<Vec<Point>> = vec![];

        for t in 0..points.len() {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<Point> = triangles.into_iter().map(|t| centers[t].clone()).collect();

            polygons.push(polygon);
        }

        polygons
    }
}

fn helper_points(polygon: &Polygon) -> Vec<Point> {
    let mut points = vec![];

    let mut min = Point{x: f64::MAX, y: f64::MAX};
    let mut max = Point{x: f64::MIN, y: f64::MIN};

    for point in polygon.points() {
        if point.x < min.x {
            min.x = point.x;
        }
        if point.x > max.x {
            max.x = point.x;
        }
        if point.y < min.y {
            min.y = point.y;
        }
        if point.y > max.y {
            max.y = point.y;
        }
    }

    let width = max.x - min.x;
    let height = max.y - min.y;

    points.push(Point{x: min.x - width, y: min.y + height / 2.0});
    points.push(Point{x: max.x + width, y: min.y + height / 2.0});
    points.push(Point{x: min.x + width / 2.0, y: min.y - height});
    points.push(Point{x: min.x + width / 2.0, y: max.y + height});

    points
}

/// Represents a Voronoi diagram.
pub struct VoronoiDiagram {
    /// Contains the input data
    pub sites: Vec<Point>,
    /// A [`Triangulation`] struct that contains the Delaunay triangulation information.
    ///
    /// [`Triangulation`]: ./delaunator/struct.Triangulation.html
    pub delaunay: Triangulation,
    /// Stores the circumcenter of each triangle
    pub centers: Vec<Point>,
    /// Stores the coordinates of each vertex of a cell, in counter-clockwise order
    cells: Vec<Polygon>,
    /// Stores the neighbor of each cell
    pub neighbors: Vec<Vec<usize>>,

    num_helper_points: usize,
}

impl VoronoiDiagram {
    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    ///
    /// Points are represented here as a [`delaunator::Point`].
    /// [`delaunator::Point`]: ./delaunator/struct.Point.html
    pub fn new(min: &Point, max: &Point, points: &[Point]) -> Option<Self> {
        // Create a polygon defining the region to clip to (rectangle from min point to max point)
        let clip_points = vec![Point{x: min.x, y: min.y}, Point{x:max.x, y: min.y}, Point{x: max.x, y:max.y}, Point{x: min.x, y:max.y}];
        let clip_polygon = polygon::Polygon::from_points(clip_points);

        VoronoiDiagram::with_bounding_polygon(points.to_vec(), &clip_polygon)
    }

    /// Creates a Voronoi diagram, if it exists, for a given set of points bounded by the supplied polygon.
    ///
    /// Points are represented here as a [`delaunator::Point`].
    /// [`delaunator::Point`]: ./delaunator/struct.Point.html
    pub fn with_bounding_polygon(mut points: Vec<Point>, clip_polygon: &Polygon) -> Option<Self> {
        // Add in the 
        let mut helper_points = helper_points(&clip_polygon);
        let num_helper_points = helper_points.len();
        points.append(&mut helper_points);

        VoronoiDiagram::with_helper_points(points, clip_polygon, num_helper_points)
    }

    fn with_helper_points(points: Vec<Point>, clip_polygon: &Polygon, num_helper_points: usize) -> Option<Self> {
        let delaunay = triangulate(&points)?;
        let centers = calculate_circumcenters(&points, &delaunay);
        let cells =
            VoronoiDiagram::calculate_polygons(&points, &centers, &delaunay, &clip_polygon);
        let neighbors = calculate_neighbors(&points, &delaunay);

        Some(VoronoiDiagram {
            sites: points,
            delaunay,
            centers,
            cells,
            neighbors,
            num_helper_points,
        })
    }

    /// Creates a Voronoi diagram, if it exists, for a given set of points.
    ///
    /// Points are represented here as a `(f64, f64)` tuple.
    pub fn from_tuple(min: &(f64, f64), max: &(f64, f64), coords: &[(f64, f64)]) -> Option<Self> {
        let points: Vec<Point> = coords.iter().map(|p| Point::from(*p)).collect();
        
        let clip_points = vec![Point{x: min.0, y: min.1}, Point{x:max.0, y: min.1}, Point{x: max.0, y:max.1}, Point{x: min.0, y:max.1}];
        let clip_polygon = polygon::Polygon::from_points(clip_points);

        VoronoiDiagram::with_bounding_polygon(points, &clip_polygon)
    }

    /// Returns slice containing the valid cells in the Voronoi diagram.
    ///
    /// Cells are represented as a `Polygon`.
    pub fn cells(&self) -> &[Polygon] {
        &self.cells[..self.cells.len()-self.num_helper_points]
    }

    fn calculate_polygons(
        points: &[Point],
        centers: &[Point],
        delaunay: &Triangulation,
        clip_polygon: &Polygon,
    ) -> Vec<Polygon> {
        let mut polygons: Vec<Polygon> = vec![];

        for t in 0..points.len() {
            let incoming = delaunay.inedges[t];
            let edges = edges_around_point(incoming, delaunay);
            let triangles: Vec<usize> = edges.into_iter().map(triangle_of_edge).collect();
            let polygon: Vec<Point> = triangles.into_iter().map(|t| centers[t].clone()).collect();

            let polygon = polygon::Polygon::from_points(polygon);
            let polygon = polygon::sutherland_hodgman(&polygon, &clip_polygon);

            polygons.push(polygon);
        }

        polygons
    }
}

fn calculate_centroids(points: &[Point], delaunay: &Triangulation) -> Vec<Point> {
    let num_triangles = delaunay.len();
    let mut centroids = Vec::with_capacity(num_triangles);
    for t in 0..num_triangles {
        let mut sum = Point { x: 0., y: 0. };
        for i in 0..3 {
            let s = 3 * t + i; // triangle coord index
            let p = &points[delaunay.triangles[s]];
            sum.x += p.x;
            sum.y += p.y;
        }
        centroids.push(Point {
            x: sum.x / 3.,
            y: sum.y / 3.,
        });
    }
    centroids
}

fn calculate_circumcenters(points: &[Point], delaunay: &Triangulation) -> Vec<Point> {
    let num_triangles = delaunay.len();
    let mut circumcenters = vec![Point { x: 0., y: 0. }; num_triangles];
    for t in 0..num_triangles {
        let v: Vec<Point> = points_of_triangle(t, delaunay)
            .into_iter()
            .map(|p| points[p].clone())
            .collect();
        if let Some(c) = circumcenter(&v[0], &v[1], &v[2]) {
            circumcenters[t] = c;
        }
    }
    circumcenters
}

fn calculate_neighbors(points: &[Point], delaunay: &Triangulation) -> Vec<Vec<usize>> {
    let num_points = points.len();
    let mut neighbors: Vec<Vec<usize>> = vec![vec![]; num_points];

    for t in 0..num_points {
        let e0 = delaunay.inedges[t];
        if e0 == INVALID_INDEX {
            continue;
        }
        let mut e = e0;
        loop {
            neighbors[t].push(delaunay.triangles[e]);
            e = next_halfedge(e);
            if delaunay.triangles[e] != t {
                break;
            }
            e = delaunay.halfedges[e];
            if e == INVALID_INDEX {
                neighbors[t].push(delaunay.triangles[delaunay.outedges[t]]);
                break;
            }
            if e == e0 {
                break;
            }
        }
    }

    neighbors
}
