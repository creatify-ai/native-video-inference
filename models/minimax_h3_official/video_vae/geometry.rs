//! Literal pinned VAE tile allocation and temporal tail arithmetic.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Axis {pub starts:Vec<i32>,pub overlaps:Vec<i32>}
pub fn axis(length:usize)->Result<Axis,String>{
    if length<32||length%16!=0||length>i32::MAX as usize{return Err("VAE dimensions require latent-aligned tiles of at least 32 pixels".into());}
    if length<=256{return Ok(Axis{starts:vec![0],overlaps:vec![]});}
    let mut n=length.div_ceil(256);
    while 256*n-64*(n-1)<length{n+=1;}
    let mut overlaps=vec![64i32;n-1];
    let remaining=256*n-64*(n-1)-length;
    for i in 0..remaining/16{overlaps[i%(n-1)]+=16;}
    let mut starts=vec![0];
    for overlap in &overlaps{starts.push(starts.last().unwrap()+256-overlap);}
    if *starts.last().unwrap() as usize+256!=length{return Err("tile union does not cover requested axis".into());}
    Ok(Axis{starts,overlaps})
}
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Temporal {pub latent_frames:usize,pub pad_tokens:usize,pub chunks:usize,pub frames:usize}
pub fn temporal(latent_frames:usize)->Result<Temporal,String>{
    if latent_frames<2{return Err("tiled video decoder needs at least two latent frames".into());}
    // The first decoder tile consumes seven latents. Two latents are the
    // five-frame native shape: repeat the last latent through one full tile,
    // then trim its padded tail. Without this case the chunk count is zero.
    let pad_tokens=if latent_frames==2{5}else{(5-(latent_frames+3)%5)%5};
    let chunks=(latent_frames+3+pad_tokens)/5-1;
    let trim:usize=(0..pad_tokens).map(|k|if (latent_frames+k)%5==0{1}else{4}).sum();
    Ok(Temporal{latent_frames,pad_tokens,chunks,frames:chunks*17+5-trim})
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn original_tiles(){
        assert_eq!(axis(768).unwrap(),Axis{starts:vec![0,160,336,512],overlaps:vec![96,80,80]});
        assert_eq!(axis(1376).unwrap(),Axis{starts:vec![0,176,352,544,736,928,1120],overlaps:vec![80,80,64,64,64,64]});
        for n in (256..=2048).step_by(16){let a=axis(n).unwrap();assert_eq!(*a.starts.last().unwrap()+256,n as i32);assert!(a.overlaps.iter().all(|&x|x>=64&&x<256&&x%16==0));}
        assert!(axis(1375).is_err());assert!(axis(255).is_err());assert_eq!(axis(32).unwrap(),Axis{starts:vec![0],overlaps:vec![]});
    }
    #[test]fn original_temporal_tails(){
        for (tokens,frames) in [(102,345),(37,124),(67,226),(57,192),(97,328),(87,294)]{let t=temporal(tokens).unwrap();assert_eq!(t.frames,frames);assert_eq!(t.pad_tokens,0);}
        assert_eq!((3..=7).map(|t|temporal(t).unwrap().frames).collect::<Vec<_>>(),vec![9,13,17,18,22]);
        assert!(temporal(0).is_err());assert!(temporal(1).is_err());
    }
    #[test]fn five_frame_video_decodes_one_padded_tile(){
        assert_eq!(temporal(2).unwrap(),Temporal{latent_frames:2,pad_tokens:5,chunks:1,frames:5});
    }
}
