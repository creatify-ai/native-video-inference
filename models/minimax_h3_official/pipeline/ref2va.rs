//! Ordered Ref2VA positions adapted from MiniMax/HuggingFace before_denoise.py.
//! Copyright 2026 MiniMax and HuggingFace Teams, Apache-2.0; see licenses/.
//! Geometry must describe the actual encoded references, not the target canvas.
use crate::{Layout, Result, Shape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all="lowercase")]
pub enum Kind { Image, Video, Audio }

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceGeometry {
    pub kind: Kind,
    #[serde(default)] pub latent_frames: usize,
    #[serde(default)] pub latent_height: usize,
    #[serde(default)] pub latent_width: usize,
    #[serde(default)] pub audio_rows: usize,
}
impl ReferenceGeometry {
    fn video_rows(&self)->Result<usize> {
        if self.audio_rows%2!=0 {return Err("reference audio must have two equal channel-major spans".into());}
        match self.kind {
            Kind::Audio => {
                if self.audio_rows==0 || self.latent_frames!=0 || self.latent_height!=0 || self.latent_width!=0 {
                    return Err("audio reference requires audio rows and no visual geometry".into());
                }
                Ok(0)
            },
            Kind::Image | Kind::Video => {
                if self.latent_frames==0 || self.latent_height<2 || self.latent_width<2 ||
                   self.latent_height%2!=0 || self.latent_width%2!=0 ||
                   (self.kind==Kind::Image && (self.latent_frames!=1 || self.audio_rows!=0)) {
                    return Err("invalid image/video reference latent geometry".into());
                }
                self.latent_frames.checked_mul(self.latent_height/2).and_then(|n|n.checked_mul(self.latent_width/2))
                    .ok_or_else(||"reference video size overflow".into())
            },
        }
    }
}

pub struct ReferenceLayout {
    pub packed: Layout,
    pub condition_video_rows: usize,
    pub condition_audio_rows: usize,
}

fn axes(height:usize,width:usize)->(Vec<f64>,Vec<f64>) {
    let area=(height as f64*width as f64).sqrt();
    let axis=|dim:usize|{
        let ratio=dim as f64/area;let left=(1.0-ratio)/2.0;
        let step=((left+ratio)-left)/(dim/2) as f64;
        (0..dim/2).map(|i|(left+i as f64*step)*32.0).collect()
    };
    (axis(height),axis(width))
}
fn span(frame:usize)->f64 {(5.0/3.0)*if frame%5==0 {1.0}else{4.0}}
// Python 3.12 builtin sum uses compensated summation. The per-frame torch
// cumsum below remains sequential; only the reference block's span uses sum.
fn reference_span(frames:usize)->f64 {
    let mut hi:f64=0.0;let mut lo:f64=0.0;
    for frame in 0..frames {
        let value=span(frame);let total=hi+value;
        lo+=if hi.abs()>=value.abs(){(hi-total)+value}else{(value-total)+hi};
        hi=total;
    }
    hi+lo
}
fn audio(positions:&mut [f64],start:usize,rows:usize,origin:f64,width:&[f64]) {
    let count=rows/2;
    for channel in 0..2 {for t in 0..count {
        let row=start+channel*count+t;
        positions[row*3]=origin+t as f64;
        positions[row*3+2]=if channel==0 {width[0]}else{width[width.len()-1]};
    }}
}
fn video(positions:&mut [f64],start:usize,frames:usize,origin:f64,height:&[f64],width:&[f64])->f64 {
    let mut elapsed=0.0;
    for t in 0..frames {
        for (h,&y) in height.iter().enumerate() {for (w,&x) in width.iter().enumerate() {
            let row=start+(t*height.len()+h)*width.len()+w;
            positions[row*3]=origin+elapsed;
            positions[row*3+1]=y;positions[row*3+2]=x;
        }}
        elapsed+=span(t);
    }
    elapsed
}

pub fn layout(text_tags:&[i64],target:Shape,references:&[ReferenceGeometry])->Result<ReferenceLayout> {
    if text_tags.is_empty() || text_tags.iter().any(|&x|x!=0&&x!=1) {
        return Err("invalid Ref2VA text token tags".into());
    }
    if references.is_empty() || references.len()>12 || references.iter().all(|r|r.kind==Kind::Audio) {
        return Err("Ref2VA needs 1..12 ordered references including an image or video".into());
    }
    for (kind,limit) in [(Kind::Image,9),(Kind::Video,3),(Kind::Audio,3)] {
        if references.iter().filter(|r|r.kind==kind).count()>limit {return Err("reference modality count exceeds endpoint contract".into());}
    }
    let mut cv=0usize;let mut ca=0usize;
    for r in references {
        cv=cv.checked_add(r.video_rows()?).ok_or("video condition count overflow")?;
        ca=ca.checked_add(r.audio_rows).ok_or("audio condition count overflow")?;
    }
    let tv=target.num_latent_frames.checked_mul(target.rows_per_frame()).ok_or("target video size overflow")?;
    let ta=target.audio_rows();
    let total=crate::limits::packed_rows(&[text_tags.len(),cv,ca,tv,ta])?;
    let mut p=Layout{position_ids:vec![0.0;total*3],token_tags:vec![0;total],
        video_indices:Vec::with_capacity(cv+tv),audio_indices:Vec::with_capacity(ca+ta),
        text_indices:(0..text_tags.len() as i64).collect()};
    for (i,&tag) in text_tags.iter().enumerate(){p.position_ids[3*i]=i as f64;p.token_tags[i]=tag;}
    let (target_h,target_w)=axes(target.latent_height,target.latent_width);
    let mut cursor=text_tags.len();let mut origin=cursor as f64;
    for r in references {
        if r.kind==Kind::Audio {
            audio(&mut p.position_ids,cursor,r.audio_rows,origin,&target_w);
            p.audio_indices.extend((cursor..cursor+r.audio_rows).map(|i|i as i64));
            cursor+=r.audio_rows;origin+=(r.audio_rows/2) as f64;
        }else{
            let (h,w)=axes(r.latent_height,r.latent_width);
            if r.audio_rows>0 {
                audio(&mut p.position_ids,cursor,r.audio_rows,origin,&w);
                p.audio_indices.extend((cursor..cursor+r.audio_rows).map(|i|i as i64));
                cursor+=r.audio_rows;
            }
            let rows=r.video_rows()?;
            video(&mut p.position_ids,cursor,r.latent_frames,origin,&h,&w);
            p.video_indices.extend((cursor..cursor+rows).map(|i|i as i64));cursor+=rows;
            origin+=if r.kind==Kind::Image {1.0}else{reference_span(r.latent_frames).max((r.audio_rows/2) as f64)};
        }
    }
    audio(&mut p.position_ids,cursor,ta,origin,&target_w);
    p.audio_indices.extend((cursor..cursor+ta).map(|i|i as i64));cursor+=ta;
    video(&mut p.position_ids,cursor,target.num_latent_frames,origin,&target_h,&target_w);
    p.video_indices.extend((cursor..cursor+tv).map(|i|i as i64));
    for &i in &p.audio_indices {p.token_tags[i as usize]=2;}
    Ok(ReferenceLayout{packed:p,condition_video_rows:cv,condition_audio_rows:ca})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image()->ReferenceGeometry {ReferenceGeometry{kind:Kind::Image,latent_frames:1,latent_height:2,latent_width:2,audio_rows:0}}
    #[test]
    fn image_advances_one_rotary_slot_and_soundtrack_precedes_video() {
        let target=Shape::new(32,32,124).unwrap();
        let mut clip=image();clip.kind=Kind::Video;clip.audio_rows=6;
        let l=layout(&[1,0],target,&[image(),clip]).unwrap();
        assert_eq!((l.condition_video_rows,l.condition_audio_rows),(2,6));
        assert_eq!(&l.packed.video_indices[..2],&[2,9]);
        assert_eq!(&l.packed.audio_indices[..6],&[3,4,5,6,7,8]);
        assert_eq!(l.packed.position_ids[2*3],2.0);
        assert_eq!(l.packed.position_ids[3*3],3.0);
        assert_eq!(l.packed.position_ids[9*3],3.0);
        assert_eq!(l.packed.position_ids[10*3],6.0);
    }
    #[test]
    fn rejects_malformed_and_unbounded_reference_geometry() {
        let t=Shape::new(32,32,124).unwrap();let mut r=image();
        r.audio_rows=2;assert!(layout(&[1],t,&[r.clone()]).is_err());
        r.audio_rows=0;r.latent_frames=usize::MAX;assert!(layout(&[1],t,&[r.clone()]).is_err());
        r.kind=Kind::Video;r.latent_height=usize::MAX-1;assert!(layout(&[1],t,&[r]).is_err());
    }
}
