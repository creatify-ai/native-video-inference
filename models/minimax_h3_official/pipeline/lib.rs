//! Native input generation for the H3 FL2VA pipeline.
//! Model outputs are supplied by the real native components, never gold fixtures.
use std::{fs, path::Path};
use sha2::{Digest, Sha256};
use serde_json::{json, Value};

pub mod audit;
#[path="../support/startup_trace.rs"]
pub mod startup_trace;
pub mod ref2va;
pub mod long_video;
pub mod resample;
pub mod noise;
#[path="../support/limits.rs"]
pub mod limits;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Geometry follows the pinned reference layout helpers, including the
/// 17*n+5 pixel-frame and 5*n+2 latent-frame grids. It contains no model output.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub width: usize, pub height: usize, pub num_frames: usize, pub keyframe_count: usize,
    pub latent_width: usize, pub latent_height: usize,
    pub num_latent_frames: usize, pub num_audio_latents: usize,
}
pub const MIN_RENDER_FRAMES:usize=5;
pub mod delivery;
impl Shape {
    pub fn new(width: usize, height: usize, requested_frames: usize) -> Result<Self> {
        if width == 0 || height == 0 || width % 32 != 0 || height % 32 != 0 {
            return Err("FL2VA width and height must be positive multiples of 32".into());
        }
        if requested_frames == 0 || requested_frames > 345 {
            return Err("FL2VA requested frame count must be between 1 and 345".into());
        }
        let num_frames = requested_frames + (5 + 17 - requested_frames % 17) % 17;
        if num_frames < MIN_RENDER_FRAMES || num_frames > 345 {
            return Err("aligned video frame count must be on the 5..345 native grid".into());
        }
        // Match Python round(num_frames / 24 * 40), including operation order.
        let num_audio_latents = (num_frames as f64 / 24.0 * 40.0).round_ties_even() as usize;
        width.checked_mul(height).and_then(|n| n.checked_mul(num_frames))
            .and_then(|n| n.checked_mul(3)).ok_or("video shape overflows address space")?;
        Ok(Self { width, height, num_frames, keyframe_count: 2, latent_width: width / 16,
            latent_height: height / 16, num_latent_frames: (num_frames - 5) / 17 * 5 + 2,
            num_audio_latents })
    }
    pub fn with_keyframes(mut self, count: usize) -> Result<Self> {
        if count>2 {return Err("expected zero, one or two keyframes".into());}
        self.keyframe_count=count; Ok(self)
    }
    pub fn long(width:usize,height:usize,frames:usize,count:usize)->Result<Self> {
        let ceiling=match std::env::var("H3_FL2VA_LONG_MAX_S") {
            Ok(value)=>value.parse::<f64>()?,Err(std::env::VarError::NotPresent)=>60.0,
            Err(error)=>return Err(error.into()),
        };
        let plan=long_video::Plan::new(frames,ceiling,true)?;
        let mut shape=Self::new(width,height,124)?.with_keyframes(count)?;
        width.checked_mul(height).and_then(|n|n.checked_mul(frames)).and_then(|n|n.checked_mul(3))
            .ok_or("long-video shape overflows address space")?;
        shape.num_frames=frames;shape.num_latent_frames=plan.num_latent_frames;
        shape.num_audio_latents=plan.num_audio_latents;Ok(shape)
    }
    pub fn case01() -> Self { Self::new(768, 768, 345).unwrap() }
    pub fn from_request(request: &Value) -> Result<Self> {
        if request.get("fps").is_some_and(|fps| fps != 24) {
            return Err("FL2VA uses the original fixed 24 fps".into());
        }
        let dimension = |key: &str| -> Result<usize> {
            usize::try_from(request[key].as_u64().ok_or_else(|| format!("missing positive integer {key}"))?)
                .map_err(Into::into)
        };
        let (width, height) = if request.get("width").is_none() && request.get("height").is_none()
            && request["case"] == "case-01" { (768, 768) }
            else { (dimension("width")?, dimension("height")?) };
        let first=request.get("first_png").is_some_and(|v| !v.is_null());
        let last=request.get("last_png").is_some_and(|v| !v.is_null());
        if last && !first {return Err("last keyframe requires first keyframe".into());}
        let count=usize::from(first)+usize::from(last);
        if request.get("keyframe_count").is_some_and(|v| v.as_u64()!=Some(count as u64)) {return Err("keyframe count disagrees with supplied images".into());}
        if request["long_video"]==true {
            if request["workflow"].as_str().is_some_and(|w|w!="fl2va") || request.get("refinement").is_some() {
                return Err("long-video requires the official FL2VA partition".into());
            }
            Self::long(width,height,dimension("num_frames")?,count)
        }else{
            let shape=Self::new(width, height, dimension("num_frames")?)?.with_keyframes(count)?;
            delivery::ShortDelivery::from_request(request,shape.num_frames)?;
            Ok(shape)
        }
    }
    pub fn rows_per_frame(self) -> usize { self.latent_height / 2 * (self.latent_width / 2) }
    pub fn condition_rows(self) -> usize { self.keyframe_count * self.rows_per_frame() }
    pub fn video_rows(self) -> usize { (self.num_latent_frames + self.keyframe_count) * self.rows_per_frame() }
    pub fn audio_rows(self) -> usize { 2 * self.num_audio_latents }
    pub fn posterior_elements(self) -> usize { 24 * self.latent_height * self.latent_width }
    pub fn audio_samples(self) -> usize { self.num_audio_latents * 800 }
    pub fn json(self) -> Value {
        json!({"width":self.width,"height":self.height,"num_frames":self.num_frames,"fps":24,"keyframe_count":self.keyframe_count,
            "latent_width":self.latent_width,"latent_height":self.latent_height,
            "num_latent_frames":self.num_latent_frames,"num_audio_latents":self.num_audio_latents,
            "condition_video_rows":self.condition_rows(),"video_rows":self.video_rows(),
            "audio_rows":self.audio_rows(),"audio_samples_per_channel":self.audio_samples()})
    }
}

pub use native_core_infra::rng::CpuGenerator;

pub fn sha256(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }
pub fn f32_bytes(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
pub fn i64_bytes(values: &[i64]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
pub fn f64_bytes(values: &[f64]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes=fs::read(path)?;
    if bytes.len()%4!=0 { return Err("malformed FP32 file".into()); }
    Ok(bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect())
}

/// [1,C,T,H,W] -> [T*(H/2)*(W/2), C*4], the original [1,2,2] patch.
pub fn patchify(values: &[f32], channels: usize, frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
    if height%2 != 0 || width%2 != 0 || values.len()!=channels*frames*height*width { return Err("invalid video shape".into()); }
    let mut out = Vec::with_capacity(values.len());
    for t in 0..frames { for h in (0..height).step_by(2) { for w in (0..width).step_by(2) {
        for c in 0..channels { for dh in 0..2 { for dw in 0..2 {
            out.push(values[((c*frames+t)*height+h+dh)*width+w+dw]);
        } } }
    } } }
    Ok(out)
}

pub fn unpatchify_video(values: &[f32], channels: usize, frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
    if height%2 != 0 || width%2 != 0 || values.len()!=channels*frames*height*width { return Err("invalid packed video shape".into()); }
    let mut out=vec![0.0;values.len()]; let mut i=0;
    for t in 0..frames { for h in (0..height).step_by(2) { for w in (0..width).step_by(2) {
        for c in 0..channels { for dh in 0..2 { for dw in 0..2 {
            out[((c*frames+t)*height+h+dh)*width+w+dw]=values[i]; i+=1;
        } } }
    } } }
    Ok(out)
}

pub fn unpack_audio(values: &[f32], channels: usize, frames: usize, width: usize) -> Result<Vec<f32>> {
    if values.len()!=channels*frames*width { return Err("invalid packed audio shape".into()); }
    let mut out=vec![0.0;values.len()];
    for c in 0..channels { for t in 0..frames { for d in 0..width {
        out[(c*width+d)*frames+t]=values[(c*frames+t)*width+d];
    } } }
    Ok(out)
}

/// NumPy's pairwise sum, including its sequential branch below eight elements.
/// <=128 covers the native duration range.
fn numpy_pairwise_sum(values: &[f64]) -> f64 {
    assert!(values.len()<=128);
    if values.len()<8 {return values.iter().fold(-0.0,|sum,&value|sum+value);}
    let mut r=[0.0;8]; r.copy_from_slice(&values[..8]);
    let end=values.len()/8*8;
    for i in (8..end).step_by(8) { for j in 0..8 { r[j]+=values[i+j]; } }
    let mut result=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
    for value in &values[end..] { result+=value; }
    result
}

pub struct Layout { pub position_ids: Vec<f64>, pub token_tags: Vec<i64>, pub video_indices: Vec<i64>, pub audio_indices: Vec<i64>, pub text_indices: Vec<i64> }
pub fn case01_layout(text_tags: &[i64]) -> Result<Layout> {
    fl2va_layout(text_tags, Shape::case01())
}

pub fn fl2va_layout(text_tags: &[i64], shape: Shape) -> Result<Layout> {
    fl2va_layout_with_delivery(text_tags,shape,None)
}

/// A short FL clip's final image guides its delivered end, not the padded tail.
pub fn fl2va_layout_with_delivery(text_tags:&[i64],shape:Shape,delivered:Option<usize>)->Result<Layout> {
    if delivered.is_some_and(|n|n==0||n>120||n>shape.num_frames){return Err("invalid short layout delivery".into());}
    if text_tags.is_empty() || text_tags.iter().any(|&t| t!=0 && t!=1) {
        return Err("conditioner token tags must be a nonempty sequence of text/video tags".into());
    }
    let ntext=text_tags.len(); let rows=shape.rows_per_frame();
    let gh=shape.latent_height/2; let gw=shape.latent_width/2;
    let audio_start=ntext+shape.keyframe_count*rows; let video_start=audio_start+shape.audio_rows();
    let len=video_start+shape.num_latent_frames*rows;
    let mut positions=vec![0.0;len*3]; let mut tags=vec![0;len];
    for i in 0..ntext { positions[3*i]=i as f64; tags[i]=text_tags[i]; }
    let sqrt_area=((shape.latent_height*shape.latent_width) as f64).sqrt();
    let axis=|dim:usize| {
        let ratio=dim as f64/sqrt_area; let left=(1.0-ratio)/2.0;
        let right=left+ratio; let step=(right-left)/(dim/2) as f64;
        (0..dim/2).map(|i| (left+i as f64*step)*32.0).collect::<Vec<f64>>()
    };
    let height_axis=axis(shape.latent_height); let width_axis=axis(shape.latent_width);
    let spans:Vec<f64>=(0..shape.num_latent_frames).map(|i| (5.0/3.0)*if i%5==0 {1.0} else {4.0}).collect();
    let last=ntext as f64+numpy_pairwise_sum(&spans)-5.0/3.0;
    let last=match delivered.filter(|&n|n<shape.num_frames){
        Some(n)=>ntext as f64+(n-1) as f64*(5.0/3.0),None=>last,
    };
    for k in 0..shape.keyframe_count { for h in 0..gh { for w in 0..gw {
        let row=ntext+k*rows+h*gw+w;
        positions[row*3]=if k==0 {ntext as f64} else {last};
        positions[row*3+1]=height_axis[h]; positions[row*3+2]=width_axis[w];
    } } }
    for c in 0..2 { for t in 0..shape.num_audio_latents {
        let row=audio_start+c*shape.num_audio_latents+t; positions[row*3]=(ntext+t) as f64;
        positions[row*3+2]=width_axis[if c==0 {0} else {gw-1}]; tags[row]=2;
    } }
    let mut elapsed=0.0;
    for (t,span) in spans.iter().enumerate() {
        for h in 0..gh { for w in 0..gw {
            let row=video_start+t*rows+h*gw+w;
            positions[row*3]=ntext as f64+elapsed;
            positions[row*3+1]=height_axis[h]; positions[row*3+2]=width_axis[w];
        } }
        elapsed+=span;
    }
    Ok(Layout { position_ids:positions, token_tags:tags,
        video_indices:(ntext..audio_start).chain(video_start..len).map(|i|i as i64).collect(),
        audio_indices:(audio_start..video_start).map(|i|i as i64).collect(),
        text_indices:(0..ntext).map(|i|i as i64).collect() })
}

/// Compatibility entry point for the original fixed case.
pub fn canvas_rgb(path: &Path) -> Result<Vec<u8>> {
    canvas_rgb_sized(path, 768, 768)
}

/// Decode canvas-sized keyframes; rejects a resize until its exact path is available.
pub fn canvas_rgb_sized(path: &Path, width: usize, height: usize) -> Result<Vec<u8>> {
    let mut decoder=png::Decoder::new(std::io::BufReader::new(fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader=decoder.read_info()?;
    let mut bytes=vec![0;reader.output_buffer_size()]; let info=reader.next_frame(&mut bytes)?;
    if info.width as usize!=width || info.height as usize!=height || info.bit_depth!=png::BitDepth::Eight {
        return Err(format!("expected canvas-sized {width}x{height} eight-bit PNG; received {}x{} {:?}",info.width,info.height,info.bit_depth).into());
    }
    bytes.truncate(info.buffer_size());
    match info.color_type {
        png::ColorType::Rgb=>Ok(bytes),
        png::ColorType::Rgba=>Ok(bytes.chunks_exact(4).flat_map(|p|p[..3].iter().copied()).collect()),
        png::ColorType::Grayscale=>Ok(bytes.iter().flat_map(|&p| [p,p,p]).collect()),
        png::ColorType::GrayscaleAlpha=>Ok(bytes.chunks_exact(2).flat_map(|p| [p[0],p[0],p[0]]).collect()),
        _=>Err("unsupported expanded PNG color type".into()),
    }
}

#[cfg(test)]
mod request_modes {
    use super::*;
    #[test]
    fn zero_one_two_keyframes_preserve_target_geometry() {
        let legacy=Shape::new(768,1376,124).unwrap();
        for count in 0..=2 {
            let shape=legacy.with_keyframes(count).unwrap();
            let tags=[1,1,1];let layout=fl2va_layout(&tags,shape).unwrap();
            assert_eq!(shape.condition_rows(),count*24*43);
            assert_eq!(layout.video_indices.len(),(shape.num_latent_frames+count)*24*43);
            assert_eq!(layout.audio_indices.len(),shape.audio_rows());
            let video_first=3+shape.condition_rows()+shape.audio_rows();
            assert_eq!(layout.position_ids[video_first*3],3.0);
            if count>0 {assert_eq!(layout.position_ids[3*3],3.0);}
            assert_eq!(layout.token_tags.len(),tags.len()+shape.video_rows()+shape.audio_rows());
        }
        let old=fl2va_layout(&[1,0,1],legacy).unwrap();
        let explicit=fl2va_layout(&[1,0,1],legacy.with_keyframes(2).unwrap()).unwrap();
        assert_eq!(old.position_ids,explicit.position_ids);
        assert_eq!(old.video_indices,explicit.video_indices);
        assert!(legacy.with_keyframes(3).is_err());
    }

    #[test]
    fn short_last_keyframe_uses_delivered_time_without_moving_other_tokens() {
        for frames in [1,12,24,91,119] {
            let shape=Shape::new(32,32,frames).unwrap();
            let original=fl2va_layout(&[1,0,1],shape).unwrap();
            let short=fl2va_layout_with_delivery(&[1,0,1],shape,Some(frames)).unwrap();
            let rows=shape.rows_per_frame();
            for row in 0..short.token_tags.len(){for axis in 0..3{
                let index=row*3+axis;
                if (3+rows..3+2*rows).contains(&row)&&axis==0 {
                    assert_eq!(short.position_ids[index],3.0+(frames-1) as f64*(5.0/3.0));
                }else{assert_eq!(short.position_ids[index],original.position_ids[index]);}
            }}
            assert_eq!(short.token_tags,original.token_tags);
            assert_eq!(short.video_indices,original.video_indices);
            assert_eq!(short.audio_indices,original.audio_indices);
            for count in 0..=1 {
                let shape=shape.with_keyframes(count).unwrap();
                assert_eq!(fl2va_layout_with_delivery(&[1],shape,Some(frames)).unwrap().position_ids,
                    fl2va_layout(&[1],shape).unwrap().position_ids);
            }
        }
        let shape=Shape::new(32,32,39).unwrap();
        assert_eq!(fl2va_layout_with_delivery(&[1],shape,Some(39)).unwrap().position_ids,
            fl2va_layout(&[1],shape).unwrap().position_ids);
        for frames in [0,40,121]{assert!(fl2va_layout_with_delivery(&[1],shape,Some(frames)).is_err());}
    }
    #[test]
    fn request_count_is_bound_to_real_files() {
        let mut request=json!({"width":768,"height":768,"num_frames":124,"keyframe_count":0});
        assert_eq!(Shape::from_request(&request).unwrap().keyframe_count,0);
        request["last_png"]=json!("last.png");assert!(Shape::from_request(&request).is_err());
        request["first_png"]=json!("first.png");assert!(Shape::from_request(&request).is_err());
        request["keyframe_count"]=json!(2);assert_eq!(Shape::from_request(&request).unwrap().keyframe_count,2);
        request.as_object_mut().unwrap().remove("last_png");request["keyframe_count"]=json!(1);
        assert_eq!(Shape::from_request(&request).unwrap().keyframe_count,1);
    }

}

pub mod recipe;

pub mod cli;
pub mod media;
