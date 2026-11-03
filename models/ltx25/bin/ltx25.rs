//! Native LTX 2.5 command line.
//!
//! `ltx25 infer PROFILE REQUEST OUTPUT` runs one request against a prepared
//! profile and writes latents, decoded frames/audio and `output.mp4`.
//! `LTX25_WORLD_SIZE` is 1, or 2 to run DiT blocks 24-47 on GPU 1. `LTX25_REPEAT=N` runs the
//! request N times in one process (warm after the first) and reports each request's seconds.
//! `ltx25 media-smoke OUTPUT.mp4` encodes a tiny clip; it needs no GPU or weights.
use serde_json::json;
use std::path::Path;

const USAGE: &str = "usage: ltx25 infer PROFILE_JSON REQUEST_JSON OUTPUT_DIR | ltx25 media-smoke OUTPUT_MP4";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| {
        match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
            ["infer", profile, request, output] => {
                let world = match std::env::var("LTX25_WORLD_SIZE").as_deref() {
                    Err(_) | Ok("1") => 1,
                    Ok("2") => 2,
                    Ok(other) => return Err(format!("LTX25_WORLD_SIZE must be 1 or 2, not {other:?}").into()),
                };
                let repeat = match std::env::var("LTX25_REPEAT") {
                    Err(_) => 1,
                    Ok(value) => value.parse::<usize>().ok().filter(|&n| n >= 1)
                        .ok_or("LTX25_REPEAT must be a positive request count")?,
                };
                let (profile, request, output) = (Path::new(profile), Path::new(request), Path::new(output));
                if repeat == 1 { ltx25_native::request::infer_on(world, profile, request, output) }
                else { ltx25_native::request::infer_repeat(world, profile, request, output, repeat) }
            }
            ["media-smoke", output] => {
                let (frames, width, height) = (3, 32, 16);
                let mut rgb = vec![0.; frames * width * height * 3];
                for frame in 0..frames {
                    for pixel in 0..width * height { rgb[(frame * width * height + pixel) * 3 + frame] = 1.; }
                }
                let audio = vec![0.; 12000];
                ltx25_native::media::encode_host(Path::new(output), &rgb, frames, width, height, 24, &audio, 6000, 48000)
                    .map(|report| json!({"passed": true, "rung": "native-media-smoke", "output": report}))
            }
            _ => Err(USAGE.into()),
        }
    })();
    match result {
        Ok(report) => {
            println!("{}", serde_json::to_string(&report).unwrap());
            if report["passed"] != true { std::process::exit(1); }
        }
        Err(error) => {
            eprintln!("{}", json!({"passed": false, "error": error.to_string()}));
            std::process::exit(2);
        }
    }
}
