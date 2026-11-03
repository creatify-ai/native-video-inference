//! Ref2VA Qwen preparation from independently normalized RGB references.
//! The descriptor supplies pixels and geometry, never tokens or activations.
use crate::{pixels::{Pixels,Visual},prepare::{array3,dump,i64bytes,hash},Result};
use serde_json::{json,Value};
use std::{fs,path::Path};
use tokenizers::Tokenizer;

pub fn run(model:&Path,prompt:&Path,descriptor:&Path,output:&Path)->Result<Value> {
    let (mut report,images,videos)=plan(model,prompt,descriptor,output)?;
    write(output,&mut report,&images,&videos)?;
    Ok(report)
}

/// The whole pixel tensors and the preparation record, as one process reads them.
pub fn write(output:&Path,report:&mut Value,images:&Pixels,videos:&Pixels)->Result<()> {
    crate::prepare::write_pixels(output,report,"pixel_values.raw",images)?;
    // The mixed-media entry always writes both tensors, possibly empty.
    if report.get("video_grids").is_some() {crate::prepare::write_pixels(output,report,"pixel_values_videos.raw",videos)?;}
    fs::write(output.join("prepared.json"),serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

/// Tokens, positions and geometry; the image and video pixel rows are left to the caller.
pub fn plan(model:&Path,prompt:&Path,descriptor:&Path,output:&Path)->Result<(Value,Pixels,Pixels)> {
    let input:Value=serde_json::from_slice(&fs::read(descriptor)?)?;
    if input["references"].as_array().is_some_and(|r|r.iter().any(|r|r["type"]!="image")) {
        return crate::ref2va_mixed::plan(model,prompt,descriptor,output);
    }
    let descriptor:Value=serde_json::from_slice(&fs::read(descriptor)?)?;
    if descriptor["schema"]!="h3-ref2va-normalized-rgb-v1" {return Err("invalid normalized reference descriptor".into());}
    let refs=descriptor["references"].as_array().ok_or("references")?;
    if refs.is_empty() || refs.len()>9 || refs.iter().any(|r|r["type"]!="image") {
        return Err("this conditioner entry requires 1..9 normalized image references; video/audio ingestion is not yet accepted".into());
    }
    let processor:Value=serde_json::from_slice(&fs::read(model.join("processor/processor_config.json"))?)?;
    let config=&processor["image_processor"];
    if config["patch_size"]!=16 || config["merge_size"]!=2 || config["temporal_patch_size"]!=2 ||
       config["do_resize"]!=true || config["resample"]!=3 || config["do_normalize"]!=true || config["do_rescale"]!=true {
        return Err("unsupported Qwen image processor".into());
    }
    let minimum=config["size"]["shortest_edge"].as_u64().or_else(||config["min_pixels"].as_u64()).ok_or("min pixels")? as usize;
    let maximum=config["size"]["longest_edge"].as_u64().or_else(||config["max_pixels"].as_u64()).ok_or("max pixels")? as usize;
    let scale=config["rescale_factor"].as_f64().ok_or("rescale factor")?;
    if !scale.is_finite() || scale<=0.0{return Err("invalid rescale factor".into());}
    let mut mean=array3(config,"image_mean")?;let mut std=array3(config,"image_std")?;
    for c in 0..3 {mean[c]*=(1.0/scale) as f32;std[c]*=(1.0/scale) as f32;}
    if std.iter().any(|s|!s.is_finite() || *s<=0.0){return Err("invalid standard deviation".into());}
    let model_config:Value=serde_json::from_slice(&fs::read(model.join("model_config/config.json"))?)?;
    let tokenizer=crate::prepare::cached_tokenizer(&model.join("tokenizer/tokenizer.json"))?;
    let start=tokenizer.token_to_id("<|vision_start|>").ok_or("vision start")? as i64;
    let end=tokenizer.token_to_id("<|vision_end|>").ok_or("vision end")? as i64;
    let pad=tokenizer.token_to_id("<|image_pad|>").ok_or("image pad")? as i64;
    if Some(pad)!=model_config["image_token_id"].as_i64(){return Err("image token identity".into());}
    if fs::metadata(prompt)?.len()>crate::dynamic_request::budget("H3_QWEN_PROMPT_LIMIT_BYTES",16usize<<20)? as u64{return Err("prompt byte budget".into());}
    let prompt=fs::read_to_string(prompt)?;
    let mut grids=vec![];let mut inputs=vec![];let mut total_rows=0usize;
    for r in refs {
        let w=r["width"].as_u64().ok_or("reference width")? as usize;
        let h=r["height"].as_u64().ok_or("reference height")? as usize;
        let bytes=crate::dynamic_request::geometry(w,h)?;
        let path=Path::new(r["file"].as_str().ok_or("reference RGB file")?);
        if !path.is_absolute() || fs::metadata(path)?.len()!=bytes as u64{return Err("reference RGB geometry/bytes".into());}
        let (rw,rh)=crate::resize::smart(w,h,32,minimum,maximum)?;
        total_rows=total_rows.checked_add(rw/16*(rh/16)).ok_or("vision rows overflow")?;
        grids.push((rh/16,rw/16));inputs.push((path,w,h,rw,rh));
    }
    let mut ids=vec![];let mut tags=vec![];
    for (i,&(gh,gw)) in grids.iter().enumerate() {
        let label=tokenizer.encode(format!("<Picture {}>: ",i+1),false)?;
        ids.extend(label.get_ids().iter().map(|&id|id as i64));tags.extend(std::iter::repeat_n(1i64,label.len()));
        ids.push(start);ids.extend(std::iter::repeat_n(pad,gh*gw/4));ids.push(end);
        tags.extend(std::iter::repeat_n(0i64,gh*gw/4+2));
    }
    let encoded=tokenizer.encode(prompt.as_str(),false)?;
    ids.extend(encoded.get_ids().iter().map(|&id|id as i64));tags.extend(std::iter::repeat_n(1i64,encoded.len()));
    crate::dynamic_request::token_limits(ids.len(),model_config["text_config"]["max_position_embeddings"].as_u64().ok_or("max positions")? as usize)?;
    let scratch=total_rows.checked_mul(53986).and_then(|v|ids.len().checked_mul(188960).and_then(|t|v.checked_add(t))).ok_or("scratch overflow")?;
    if scratch>crate::dynamic_request::budget("H3_QWEN_SCRATCH_LIMIT_BYTES",16usize<<30)?{return Err("Qwen scratch byte budget".into());}
    let mm:Vec<i64>=ids.iter().map(|&id|i64::from(id==pad)).collect();
    let positions=positions(&mm,&grids)?;
    let mut canvas_inputs=vec![];
    for (path,w,h,_,_) in &inputs {canvas_inputs.push(json!({"shape":[h,w,3],"sha256":hash(&fs::read(path)?)}));}
    let images=Pixels{visuals:inputs.iter().map(|&(path,w,h,rw,rh)|Visual{path:path.to_path_buf(),w,h,rw,rh,sampled:vec![0]}).collect(),mean,std};
    if images.total_rows()!=total_rows{return Err("reference vision rows".into());}
    fs::create_dir_all(output)?;
    let grid:Vec<i64>=grids.iter().flat_map(|&(h,w)|[1,h as i64,w as i64]).collect();
    let mut tensors=serde_json::Map::new();
    for (name,values,shape) in [("input_ids.raw",&ids,vec![1,ids.len()]),("text_token_tags.raw",&tags,vec![ids.len()]),
        ("mm_token_type_ids.raw",&mm,vec![1,ids.len()]),("position_ids.raw",&positions,vec![3,1,ids.len()]),
        ("image_grid_thw.raw",&grid,vec![grids.len(),3])] {
        tensors.insert(name.into(),dump(output,name,&i64bytes(values),"int64",&shape)?);
    }
    let report=json!({"passed":true,"schema":"h3-ref2va-conditioner-prepared-v1","token_count":ids.len(),
        "vision_rows":total_rows,"image_tokens":total_rows/4,"grids":grids,"canvas_inputs":canvas_inputs,
        "scratch_estimated_bytes":scratch,"tensors":tensors,"gold_activations_read":false,
        "input_scope":"independently normalized image RGB; no reference normalization in this entry"});
    Ok((report,images,Pixels{visuals:vec![],mean,std}))
}

pub(crate) fn positions(mm:&[i64],grids:&[(usize,usize)])->Result<Vec<i64>> {
    let mut out=vec![0;mm.len()*3];let (mut cursor,mut current,mut image)=(0usize,0i64,0usize);
    while cursor<mm.len() {
        let end=(cursor+1..mm.len()).find(|&i|mm[i]!=mm[cursor]).unwrap_or(mm.len());
        if mm[cursor]==0 {
            for i in cursor..end {for axis in 0..3 {out[axis*mm.len()+i]=current+(i-cursor) as i64;}}
            current+=(end-cursor) as i64;
        } else {
            let &(h,w)=grids.get(image).ok_or("excess vision run")?;
            if end-cursor!=h*w/4{return Err("vision span geometry".into());}
            for i in cursor..end {out[i]=current;out[mm.len()+i]=current+((i-cursor)/(w/2)) as i64;out[2*mm.len()+i]=current+((i-cursor)%(w/2)) as i64;}
            current+=(h.max(w)/2) as i64;image+=1;
        }
        cursor=end;
    }
    if image!=grids.len(){return Err("missing vision runs".into());}Ok(out)
}
