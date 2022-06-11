use std::env;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fs::File, path::Path};

use anyhow::{bail, Error};
use flatgeobuf::{ColumnType, FgbWriter, GeometryType};
use geo_types::{Geometry, MultiLineString};
use geozero::{ColumnValue, PropertyProcessor};
use itertools::Itertools;
use png::Decoder;
use rasterize::{
    ActiveEdgeRasterizer, BBox, Color, ColorU8, Image, ImageMut, Layer, LinColor, Line, LineCap,
    LineJoin, Paint, Scene, Segment, StrokeStyle, SubPath, Transform,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use time::format_description::well_known;
use time::OffsetDateTime;

mod import {
    use std::fs::File;
    use std::path::Path;
    use std::str::FromStr;

    use anyhow::Error;
    use csv::{Reader, StringRecord};
    use itertools::izip;
    use time::{format_description::well_known, OffsetDateTime};
    use zip::ZipArchive;

    use crate::Point;

    pub fn read_archive(path: &Path) -> Result<Vec<Point>, Error> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        let mut latitudes = Vec::new();
        let mut record = StringRecord::new();
        {
            let reader = archive.by_name("raw_location_latitude.csv")?;
            let mut reader = Reader::from_reader(reader);
            while reader.read_record(&mut record)? {
                latitudes.push(get_array_first_value(&record[2])?);
            }
        }
        let mut times = Vec::with_capacity(latitudes.len());
        let mut longitudes = Vec::with_capacity(latitudes.len());
        {
            let reader = archive.by_name("raw_location_longitude.csv")?;
            let mut reader = Reader::from_reader(reader);
            while reader.read_record(&mut record)? {
                let time = OffsetDateTime::parse(&record[0], &well_known::Rfc3339)?;
                times.push(time);
                longitudes.push(get_array_first_value(&record[2])?);
            }
        }
        let mut radiuses = Vec::with_capacity(latitudes.len());
        {
            let reader = archive.by_name("raw_location_horizontal-radius.csv")?;
            let mut reader = Reader::from_reader(reader);
            while reader.read_record(&mut record)? {
                radiuses.push(get_array_first_value(&record[2])?);
            }
        }
        let points = izip!(times, latitudes, longitudes, radiuses)
            .map(|(time, lat, lon, radius)| {
                let geom = geo_types::Point::new(lon, lat);
                Point { time, geom, radius }
            })
            .collect();
        Ok(points)
    }

    fn get_array_first_value<T>(s: &str) -> Result<T, T::Err>
    where
        T: FromStr,
        T::Err: std::fmt::Debug,
    {
        let p = s.find(|c| c == ']' || c == ',').expect("expected array");
        s[1..p].parse()
    }
}

mod osrm {
    use std::fmt::Write;

    use anyhow::{anyhow, Error};
    use geo_types::{LineString, MultiLineString};
    use reqwest::blocking::Client;
    use serde::Deserialize;

    use crate::Point;

    #[derive(Deserialize, Debug)]
    pub struct MatchResponse {
        pub matchings: Vec<Matching>,
    }

    #[derive(Deserialize, Debug)]
    pub struct Matching {
        pub geometry: String,
    }

    pub struct OsrmClient {
        client: Client,
        base_url: String,
    }

    impl OsrmClient {
        pub fn new(base_url: String) -> Self {
            let client = Client::new();
            Self { client, base_url }
        }

        pub fn match_map(
            &self,
            profile: &str,
            points: &[Point],
        ) -> Result<MultiLineString<f64>, Error> {
            let mut q = format!("{}match/v1/{profile}/polyline6(", self.base_url);
            let mut timestamps = String::new();
            let mut radiuses = String::from("&radiuses=");
            let coordinates = LineString::from_iter(points.iter().map(|p| p.geom));
            for point in points {
                write!(timestamps, "{};", point.time.unix_timestamp())?;
                write!(radiuses, "{};", point.radius)?;
            }
            timestamps.pop();
            radiuses.pop();
            let coordinates =
                polyline::encode_coordinates(coordinates, 6).map_err(|e| anyhow!(e))?;
            let pe = percent_encoding::percent_encode(
                coordinates.as_bytes(),
                percent_encoding::NON_ALPHANUMERIC,
            );
            write!(
                q,
                "{})?geometries=polyline6&tidy=true&steps=false&timestamps=",
                pe
            )?;
            q.push_str(&timestamps);
            q.push_str(&radiuses);
            let res = self.client.get(&q).send()?.json::<MatchResponse>()?;
            let mls = MultiLineString(
                res.matchings
                    .into_iter()
                    .map(|m| polyline::decode_polyline(&m.geometry, 6))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow!(e))?,
            );
            Ok(mls)
        }
    }
}

mod geo {
    pub fn epsg_4326_to_3857(x: f64, y: f64) -> (f64, f64) {
        const WGS84_EQUATORIAL_RADIUS: f64 = 6_378_137.0;
        const MAX_LATITUDE: f64 = 85.06;

        let x = x.to_radians();
        let y = if y > MAX_LATITUDE {
            std::f64::consts::PI
        } else if y < -MAX_LATITUDE {
            -std::f64::consts::PI
        } else {
            let y = y.to_radians() / 2.0 + std::f64::consts::FRAC_PI_4;
            y.tan().ln()
        };
        (x * WGS84_EQUATORIAL_RADIUS, y * WGS84_EQUATORIAL_RADIUS)
    }
}

pub struct Point {
    pub time: OffsetDateTime,
    pub geom: geo_types::Point<f64>,
    pub radius: f32,
}

fn export_fgb(name: &str, file: &Path, points: &[Point]) -> anyhow::Result<()> {
    let mut fgb = FgbWriter::create(name, GeometryType::Point, |_, _| {})?;
    fgb.set_crs(4326, |_fbb, _crs| {});
    fgb.add_column("time", ColumnType::DateTime, |_, col| {
        col.nullable = false;
    });
    fgb.add_column("radius", ColumnType::Float, |_, col| {
        col.nullable = false;
    });
    for point in points {
        let time = point.time.format(&well_known::Rfc3339)?;
        fgb.add_feature_geom(Geometry::Point(point.geom), |feat| {
            feat.property(0, "time", &ColumnValue::DateTime(&time))
                .unwrap();
            feat.property(1, "radius", &ColumnValue::Float(point.radius))
                .unwrap();
        })?;
    }
    let out_file = File::create(file)?;
    fgb.write(&mut BufWriter::new(out_file))?;
    Ok(())
}

fn reproject_mls(mls: &mut MultiLineString<f64>) {
    mls.iter_mut().for_each(|ls| {
        ls.0.iter_mut().for_each(|p| {
            let (x, y) = geo::epsg_4326_to_3857(p.x, p.y);
            p.x = x;
            p.y = y;
        });
    });
}

fn mls_to_path(mls: MultiLineString<f64>) -> rasterize::Path {
    let subpaths = mls
        .into_iter()
        .filter_map(|ls| {
            SubPath::new(
                ls.into_iter()
                    .map(|p| p.x_y())
                    .tuple_windows()
                    .map(|(p1, p2)| Segment::Line(Line::new(p1, p2)))
                    .collect(),
                false,
            )
        })
        .collect();
    rasterize::Path::new(subpaths)
}

struct RasterizeOptions {
    rasterizer: ActiveEdgeRasterizer,
    bbox: BBox,
    transform: Transform,
    stroke_color: Arc<LinColor>,
    stroke_style: StrokeStyle,
}

fn rasterize_route(
    options: &RasterizeOptions,
    output: PathBuf,
    mut mls: MultiLineString<f64>,
) -> anyhow::Result<Option<PathBuf>> {
    reproject_mls(&mut mls);
    let scene = Scene::stroke(
        Arc::new(mls_to_path(mls)),
        Arc::clone(&options.stroke_color) as Arc<dyn Paint>,
        options.stroke_style,
    );
    let layer = scene.render(
        &options.rasterizer,
        options.transform,
        Some(options.bbox),
        None,
    );
    if layer.width() > 0 && layer.height() > 0 {
        let out = File::create(&output)?;
        layer.write_png(out)?;
        Ok(Some(output))
    } else {
        Ok(None)
    }
}

fn accumulate_images(bbox: BBox, images: &[PathBuf]) -> anyhow::Result<()> {
    let mut layer = Layer::new(bbox, Some(LinColor::new(1.0, 1.0, 1.0, 1.0)));
    let mut buf = vec![0; layer.width() * layer.height() * 4];
    for img in images {
        {
            let decoder = Decoder::new(File::open(&img)?);
            let mut reader = decoder.read_info()?;
            let info = reader.next_frame(&mut buf)?;
            layer
                .iter_mut()
                .zip((&buf[..info.buffer_size()]).chunks(4))
                .for_each(|(pa, p)| {
                    let p = &p[..4];
                    *pa = pa.blend_over(&ColorU8::new(p[0], p[1], p[2], p[3]).into());
                });
        }
        let file = File::create(img)?;
        layer.write_png(file)?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        bail!(
            "Usage: <{}> <archive> <x0> <y0> <x1> <y1> <inv_scale>",
            args[0]
        );
    }
    let archive = Path::new(&args[1]);
    let mut points = import::read_archive(archive)?;
    export_fgb("location", &archive.with_extension("fgb"), &points)?;
    points.sort_unstable_by_key(|p| p.time);

    let inv_scale = args[6].parse::<f64>()?;
    let p1 = (args[2].parse()?, args[3].parse()?);
    let p2 = (args[4].parse()?, args[5].parse()?);
    let transform = Transform::new_scale(1.0 / inv_scale, -1.0 / inv_scale);
    let p1 = transform.apply(p1.into());
    let p2 = transform.apply(p2.into());
    let bbox = BBox::new(p1, p2);

    let stroke_color = Arc::new(LinColor::from(ColorU8::new(255, 96, 17, 255)));
    let stroke_style = StrokeStyle {
        width: 20.0,
        line_join: LineJoin::Round,
        line_cap: LineCap::Butt,
    };
    let rasterizer = ActiveEdgeRasterizer::default();
    let rasterize_options = RasterizeOptions {
        rasterizer,
        bbox,
        transform,
        stroke_color,
        stroke_style,
    };

    let points = points
        .into_iter()
        .filter(|p| p.radius < 50.0)
        .group_by(|p| p.time.date())
        .into_iter()
        .map(|(d, p)| (d, p.collect::<Vec<_>>()))
        .collect::<Vec<_>>();

    let osrm_client = osrm::OsrmClient::new("http://127.0.0.1:5000/".to_string());
    let routes = points
        .into_par_iter()
        .filter_map(|(date, points)| {
            osrm_client
                .match_map("foot", &points)
                .ok()
                .map(|m| (date, m))
        })
        .collect::<Vec<_>>();

    let mut images = routes
        .into_par_iter()
        .filter_map(|(date, mls)| {
            let raster_name = format!("{date}.png");
            rasterize_route(&rasterize_options, raster_name.into(), mls).transpose()
        })
        .collect::<Result<Vec<_>, Error>>()?;

    images.sort_unstable();
    accumulate_images(bbox, &images)?;

    Ok(())
}
