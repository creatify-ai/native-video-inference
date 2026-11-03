//! Original Ref2VA modality labels, 2 fps video sampling and Qwen presentation.
//! Inputs are normalized uint8 RGB, with videos at 24 fps. Audio contributes labels.
use crate::{pixels::Pixels,prepare::{array3,dump,i64bytes},Result};
use serde_json::{json,Value};
use std::{fs,path::{Path,PathBuf}};
use tokenizers::Tokenizer;

struct Visual {pixels:crate::pixels::Visual,video:bool}
impl Visual {
    fn blocks(&self)->usize {self.pixels.blocks()}
    fn grid(&self)->(usize,usize){self.pixels.grid()}
}
// Literal Qwen3VLVideoProcessor.smart_resize, including rounded temporal budget.
fn video_size(n:usize,h:usize,w:usize,min:usize,max:usize)->Result<(usize,usize)> {
    if n<2 || h==0 || w==0{return Err("invalid sampled video geometry".into());}
    let (mut h,mut w)=(h as f64,w as f64);
    if h<32.0 || w<32.0 {let scale=(32.0/h).max(32.0/w);h=(h*scale).trunc();w=(w*scale).trunc();}
    if h.max(w)/h.min(w)>200.0{return Err("video aspect ratio".into());}
    let (mut rh,mut rw)=((h/32.0).round_ties_even()*32.0,(w/32.0).round_ties_even()*32.0);
    let t=(n as f64/2.0).round_ties_even()*2.0;
    if t*rh*rw>max as f64 {
        let beta=(n as f64*h*w/max as f64).sqrt();rh=((h/beta/32.0).floor()*32.0).max(32.0);rw=((w/beta/32.0).floor()*32.0).max(32.0);
    } else if t*rh*rw<min as f64 {
        let beta=(min as f64/(n as f64*h*w)).sqrt();rh=(h*beta/32.0).ceil()*32.0;rw=(w*beta/32.0).ceil()*32.0;
    }
    if !rh.is_finite() || !rw.is_finite() || rh*rw>16777216.0{return Err("video processor size budget".into());}
    Ok((rw as usize,rh as usize))
}
fn processor(config:&Value)->Result<(usize,usize,[f32;3],[f32;3])> {
    if config["patch_size"]!=16 || config["merge_size"]!=2 || config["temporal_patch_size"]!=2 || config["resample"]!=3 ||
       config["do_resize"]!=true || config["do_rescale"]!=true || config["do_normalize"]!=true{return Err("unsupported Qwen processor".into());}
    let min=config["size"]["shortest_edge"].as_u64().ok_or("processor min pixels")? as usize;
    let max=config["size"]["longest_edge"].as_u64().ok_or("processor max pixels")? as usize;
    let scale=config["rescale_factor"].as_f64().ok_or("rescale factor")?;
    if min==0 || max<min || !scale.is_finite() || scale<=0.0{return Err("invalid processor configuration".into());}
    let mut mean=array3(config,"image_mean")?;let mut std=array3(config,"image_std")?;
    for c in 0..3 {mean[c]*=(1.0/scale) as f32;std[c]*=(1.0/scale) as f32;}
    Ok((min,max,mean,std))
}
/// Tokens, positions and geometry; image and video pixel rows are left to the caller.
pub fn plan(model:&Path,prompt:&Path,descriptor:&Path,output:&Path)->Result<(Value,Pixels,Pixels)> {
    let descriptor:Value=serde_json::from_slice(&fs::read(descriptor)?)?;
    if descriptor["schema"]!="h3-ref2va-normalized-rgb-v1"{return Err("normalized reference schema".into());}
    let refs=descriptor["references"].as_array().ok_or("references")?;
    if refs.is_empty() || refs.len()>12{return Err("reference count".into());}
    let config:Value=serde_json::from_slice(&fs::read(model.join("processor/processor_config.json"))?)?;
    let image_config=processor(&config["image_processor"])?;let video_config=processor(&config["video_processor"])?;
    let mut visuals=vec![];let mut mapping=vec![];let mut counts=[0usize;3];let mut rows=0usize;
    for reference in refs {
        let kind=reference["type"].as_str().ok_or("reference type")?;
        let slot=match kind {"image"=>0,"video"=>1,"audio"=>2,_=>return Err("unknown reference type".into())};counts[slot]+=1;
        if slot==0 && reference["has_audio"]==true{return Err("image reference cannot carry audio".into());}
        if slot==1 && reference.get("fps").is_some_and(|v|v.as_f64()!=Some(24.0)){return Err("normalized video must be at 24 fps".into());}
        if slot==2 {mapping.push(None);continue;}
        let w=reference["width"].as_u64().ok_or("width")? as usize;let h=reference["height"].as_u64().ok_or("height")? as usize;
        let bytes=crate::dynamic_request::geometry(w,h)?;
        let frames=if slot==0 {1}else{reference["frames"].as_u64().ok_or("video frame count")? as usize};
        if frames==0 || frames>345{return Err("normalized video frame count".into());}
        let path=PathBuf::from(reference["file"].as_str().ok_or("RGB path")?);
        if !path.is_absolute() || fs::metadata(&path)?.len()!=bytes.checked_mul(frames).ok_or("RGB bytes overflow")? as u64{return Err("normalized RGB bytes/path".into());}
        let sampled:Vec<_>=if slot==0 {vec![0]}else{(0..frames).step_by(12).collect()};
        if slot==1 && sampled.len()<2{return Err("video reference must provide at least two samples at 2 fps".into());}
        let (rw,rh)=if slot==0 {crate::resize::smart(w,h,32,image_config.0,image_config.1)?}else{video_size(sampled.len(),h,w,video_config.0,video_config.1)?};
        let visual=Visual{pixels:crate::pixels::Visual{path,w,h,rw,rh,sampled},video:slot==1};
        rows=rows.checked_add(visual.blocks()*rw/16*(rh/16)).ok_or("vision rows overflow")?;
        mapping.push(Some(visuals.len()));visuals.push(visual);
    }
    if counts[0]>9 || counts[1]>3 || counts[2]>3 || counts[0]+counts[1]==0{return Err("reference modality limits".into());}
    if fs::metadata(prompt)?.len()>crate::dynamic_request::budget("H3_QWEN_PROMPT_LIMIT_BYTES",16usize<<20)? as u64{return Err("prompt budget".into());}
    let prompt=fs::read_to_string(prompt)?;
    let tokenizer=crate::prepare::cached_tokenizer(&model.join("tokenizer/tokenizer.json"))?;
    let model_config:Value=serde_json::from_slice(&fs::read(model.join("model_config/config.json"))?)?;
    let token=|name:&str|->Result<i64>{Ok(tokenizer.token_to_id(name).ok_or("missing special token")? as i64)};
    let (start,end,image_pad,video_pad)=(token("<|vision_start|>")?,token("<|vision_end|>")?,token("<|image_pad|>")?,token("<|video_pad|>")?);
    if model_config["image_token_id"].as_i64()!=Some(image_pad) || model_config["video_token_id"].as_i64()!=Some(video_pad){return Err("model/tokenizer modality ids".into());}
    let (mut ids,mut tags,mut ordered_grids)=(vec![],vec![],vec![]);let mut numbers=[0usize;3];
    let text=|s:String,ids:&mut Vec<i64>,tags:&mut Vec<i64>|->Result<()>{let e=tokenizer.encode(s,false)?;ids.extend(e.get_ids().iter().map(|&n|n as i64));tags.extend(std::iter::repeat_n(1,e.len()));Ok(())};
    for (reference,visual) in refs.iter().zip(&mapping) {
        if reference["type"]=="audio" || reference["has_audio"]==true {numbers[2]+=1;text(format!("<Audio {}>: ",numbers[2]),&mut ids,&mut tags)?;}
        if let Some(index)=visual {
            let v=&visuals[*index];let (gh,gw)=v.grid();let slot=usize::from(v.video);numbers[slot]+=1;
            text(format!("<{} {}>: ",if v.video{"Video"}else{"Picture"},numbers[slot]),&mut ids,&mut tags)?;
            for block in 0..v.blocks() {
                if v.video {
                    let first=block*2;let last=(first+1).min(v.pixels.sampled.len()-1);
                    let ticks=((first+last) as f64/4.0*10.0).round_ties_even() as usize;
                    text(format!("<{}.{:01} seconds>",ticks/10,ticks%10),&mut ids,&mut tags)?;
                }
                ids.push(start);ids.extend(std::iter::repeat_n(if v.video{video_pad}else{image_pad},gh*gw/4));ids.push(end);
                tags.extend(std::iter::repeat_n(0,gh*gw/4+2));ordered_grids.push((gh,gw));
            }
        }
    }
    text(prompt,&mut ids,&mut tags)?;
    crate::dynamic_request::token_limits(ids.len(),model_config["text_config"]["max_position_embeddings"].as_u64().ok_or("max positions")? as usize)?;
    let scratch=rows.checked_mul(53986).and_then(|v|ids.len().checked_mul(188960).and_then(|t|v.checked_add(t))).ok_or("scratch overflow")?;
    if scratch>crate::dynamic_request::budget("H3_QWEN_SCRATCH_LIMIT_BYTES",16usize<<30)?{return Err("Qwen scratch budget".into());}
    let mm:Vec<i64>=ids.iter().map(|&n|if n==image_pad{1}else if n==video_pad{2}else{0}).collect();
    let positions=crate::ref2va::positions(&mm,&ordered_grids)?;
    let (mut image_grid,mut video_grid,mut grids,mut video_grids)=(vec![],vec![],vec![],vec![]);
    let mut images=Pixels{visuals:vec![],mean:image_config.2,std:image_config.3};
    let mut videos=Pixels{visuals:vec![],mean:video_config.2,std:video_config.3};
    for v in visuals {
        let (h,w)=v.grid();
        if v.video {video_grid.extend([v.blocks() as i64,h as i64,w as i64]);video_grids.extend(std::iter::repeat_n((h,w),v.blocks()));videos.visuals.push(v.pixels);}
        else {image_grid.extend([1,h as i64,w as i64]);grids.push((h,w));images.visuals.push(v.pixels);}
    }
    if images.total_rows()+videos.total_rows()!=rows{return Err("reference vision rows".into());}
    fs::create_dir_all(output)?;let mut tensors=serde_json::Map::new();
    for (name,values,shape) in [("input_ids.raw",&ids,vec![1,ids.len()]),("text_token_tags.raw",&tags,vec![tags.len()]),
        ("mm_token_type_ids.raw",&mm,vec![1,ids.len()]),("position_ids.raw",&positions,vec![3,1,ids.len()]),
        ("image_grid_thw.raw",&image_grid,vec![counts[0],3]),("video_grid_thw.raw",&video_grid,vec![counts[1],3])] {
        tensors.insert(name.into(),dump(output,name,&i64bytes(values),"int64",&shape)?);
    }
    let report=json!({"passed":true,"schema":"h3-ref2va-conditioner-prepared-v1","token_count":ids.len(),"vision_rows":rows,
        "image_tokens":images.total_rows()/4,"video_tokens":videos.total_rows()/4,"grids":grids,"video_grids":video_grids,
        "scratch_estimated_bytes":scratch,"tensors":tensors,"gold_activations_read":false,
        "input_scope":"normalized image/video RGB and soundtrack labels; media normalization external"});
    Ok((report,images,videos))
}
