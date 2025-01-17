extern crate voronator;

use voronator::delaunator::Point;
use voronator::VoronoiDiagram;

use svg::node;
use svg::node::element::Path;
use svg::Document;
use svg::Node;

fn main() {
    let points = [(2520.0, 856.0), (794.0, 66.0), (974.0, 446.0)];
    let voronoi =
        VoronoiDiagram::<Point>::from_tuple(&(0.0, 0.0), &(2560.0, 2560.0), &points).unwrap();

    let mut document = Document::new().set("viewBox", (0, 0, 2560, 2560));
    let colours = ["blue", "green", "red"];

    for (i, polygon) in voronoi.cells().iter().enumerate() {
        let cell_vertices = polygon.points();

        let mut is_start = true;
        let mut d: Vec<node::element::path::Command> = vec![];
        for point in cell_vertices.iter() {
            if is_start {
                d.push(node::element::path::Command::Move(
                    node::element::path::Position::Absolute,
                    (point.x, point.y).into(),
                ));
                is_start = false;
            } else {
                d.push(node::element::path::Command::Line(
                    node::element::path::Position::Absolute,
                    (point.x, point.y).into(),
                ));
            }
        }
        d.push(node::element::path::Command::Close);
        let data = node::element::path::Data::from(d);

        let path = Path::new()
            .set("fill", colours[i])
            .set("stroke", "black")
            .set("stroke-width", 1)
            .set("d", data);

        document.append(path);
    }

    for point in points {
        document.append(
            node::element::Rectangle::new()
                .set("x", point.0 - 15.0)
                .set("y", point.1 - 15.0)
                .set("width", 30)
                .set("height", 30),
        );
    }

    svg::save("example.svg", &document).unwrap();
}
