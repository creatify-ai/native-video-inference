//! Generic CPU admission: model geometry, indexing and explicit resource budgets.
use crate::{prepare::hash, Result};
use serde_json::Value;
use std::{fs,path::{Path,PathBuf}};
pub const MODEL_MAX_TEXT_ROWS:usize=262144;
pub fn budget(name:&str,default:usize)->Result<usize>{let n=std::env::var(name).map_or(Ok(default),|s|s.parse::<usize>())?;if n==0{return Err(format!("{name} must be positive").into());}Ok(n)}
pub fn scratch_bytes(gh:usize,gw:usize,nt:usize)->Result<usize>{
    gh.checked_mul(gw).and_then(|n|n.checked_mul(2)).and_then(|n|n.checked_mul(53986))
        .and_then(|v|nt.checked_mul(188960).and_then(|t|v.checked_add(t))).ok_or_else(||"scratch size overflow".into())
}
pub fn resource_limits(gh:usize,gw:usize,nt:usize)->Result<usize>{
    let bytes=scratch_bytes(gh,gw,nt)?;
    if bytes>budget("H3_QWEN_SCRATCH_LIMIT_BYTES",16usize<<30)?{return Err("Qwen native scratch estimate exceeds configured H3_QWEN_SCRATCH_LIMIT_BYTES".into());}
    Ok(bytes)
}

/// Idle peers take part in admission but execute no forward or weight load.
/// Charging them the leader's scratch can reject an otherwise fitting request.
pub fn device_memory_required(active:bool,scratch:usize,weights:usize,reserve:usize)->Result<usize>{
    if !active{return Ok(0);}
    scratch.checked_add(weights).and_then(|n|n.checked_add(reserve)).ok_or_else(||"memory estimate overflow".into())
}

pub struct Request {
    pub model:PathBuf,pub prompt:PathBuf,pub canvases:Vec<PathBuf>,pub output:PathBuf,
    pub width:usize,pub height:usize,
}
pub fn geometry(width:usize,height:usize)->Result<usize> {
    let bytes=width.checked_mul(height).and_then(|n|n.checked_mul(3)).ok_or("canvas byte overflow")?;
    if bytes>budget("H3_QWEN_CANVAS_LIMIT_BYTES",256usize<<20)?{return Err("canvas exceeds configured H3_QWEN_CANVAS_LIMIT_BYTES".into());}
    crate::resize::smart(width,height,32,65536,16777216)?;
    Ok(bytes)
}
impl Request {
    pub fn parse(args:&[String])->Result<Self> {
        if ![6,8].contains(&args.len()) || args[0]!="infer" {
            return Err("expected infer MODEL PROMPT FIRST_RGB LAST_RGB OUTPUT [WIDTH HEIGHT]".into());
        }
        let (width,height)=if args.len()==8{(args[6].parse()?,args[7].parse()?)}else{(768,768)};
        if args[3]=="-" && args[4]!="-" {return Err("last keyframe requires first".into());}
        let bytes=if args[3]=="-" {0}else{geometry(width,height)?};
        let request=Self{model:fs::canonicalize(&args[1])?,prompt:args[2].clone().into(),
            canvases:args[3..5].iter().filter(|s|s.as_str()!="-").map(PathBuf::from).collect(),output:args[5].clone().into(),width,height};
        if !request.output.is_absolute() || request.output.exists() {
            return Err("resident output must be a fresh absolute directory".into());
        }
        for path in &request.canvases {
            let info=fs::metadata(path)?;
            if !info.is_file() || info.len()!=bytes as u64 {return Err("canvas RGB byte count mismatch".into());}
        }
        // Reject invalid UTF-8 before processor work or any GPU operation.
        if fs::metadata(&request.prompt)?.len()>budget("H3_QWEN_PROMPT_LIMIT_BYTES",16usize<<20)? as u64{return Err("prompt exceeds configured byte budget".into());}
        fs::read_to_string(&request.prompt)?;
        Ok(request)
    }
}

/// Size, mtime, ctime, device and inode of the manifest and every hashed config.
/// Any write or replacement changes ctime/inode, so an unchanged fingerprint
/// reuses the digests verified when it was recorded.
fn config_fingerprint(root:&Path,names:&[String])->Result<Vec<(u64,i64,i64,i64,i64,u64,u64)>> {
    use std::os::unix::fs::MetadataExt;
    std::iter::once("model-manifest.json".to_string()).chain(names.iter().cloned()).map(|name|->Result<(u64,i64,i64,i64,i64,u64,u64)>{
        let m=fs::metadata(root.join(&name))?;
        Ok((m.len(),m.mtime(),m.mtime_nsec(),m.ctime(),m.ctime_nsec(),m.dev(),m.ino()))
    }).collect()
}
type VerifiedModel=(Vec<(u64,i64,i64,i64,i64,u64,u64)>,Vec<String>,Value,String);
static VERIFIED:std::sync::Mutex<Option<(std::path::PathBuf,VerifiedModel)>>=std::sync::Mutex::new(None);

pub fn checked_model(root:&Path)->Result<(Value,String)> {
    if let Some((path,(print,names,index,sha)))=VERIFIED.lock().map_err(|_|"model cache poisoned")?.as_ref() {
        if path==root && config_fingerprint(root,names).ok().as_ref()==Some(print) {return Ok((index.clone(),sha.clone()));}
    }
    let (index,sha)=verify_model(root)?;
    let names:Vec<String>=index["config_sha256"].as_object().ok_or("config hashes")?.keys().cloned().collect();
    // Fingerprint after hashing: a write during verification makes the next request re-hash.
    let print=config_fingerprint(root,&names)?;
    *VERIFIED.lock().map_err(|_|"model cache poisoned")?=Some((root.to_path_buf(),(print,names,index.clone(),sha.clone())));
    Ok((index,sha))
}

fn verify_model(root:&Path)->Result<(Value,String)> {
    let raw=fs::read(root.join("model-manifest.json"))?;
    let index:Value=serde_json::from_slice(&raw)?;
    if index["schema"]!="h3-conditioner-static-model-v1" || index["selected_hidden_state_index"]!=50 ||
        index["activation_files_in_index"]!=0 || index["request_files_in_index"]!=0 {
        return Err("invalid static conditioner model index".into());
    }
    for (name,digest) in index["config_sha256"].as_object().ok_or("config hashes")? {
        if Path::new(name).is_absolute() || name.split('/').any(|x|x=="..") {return Err("invalid config path".into());}
        if hash(&fs::read(root.join(name))?)!=digest.as_str().ok_or("config digest")? {
            return Err(format!("resident conditioner config changed: {name}").into());
        }
    }
    Ok((index,hash(&raw)))
}

pub fn token_limits(rows:usize,max_positions:usize)->Result<()> {
    if rows==0 || rows>MODEL_MAX_TEXT_ROWS || rows>max_positions || rows>i32::MAX as usize || rows.checked_mul(25600*2).is_none() {
        return Err("conditioner token sequence exceeds model or native indexing limit".into());
    }
    Ok(())
}

pub fn validate_prepared(request:&Request,prepared:&Path,prep:&Value,index:&Value,pixels:&crate::pixels::Pixels)->Result<usize> {
    let rows=prep["token_count"].as_u64().ok_or("prepared token count")? as usize;
    let config:Value=serde_json::from_slice(&fs::read(request.model.join("model_config/config.json"))?)?;
    let max_positions=config["text_config"]["max_position_embeddings"].as_u64().ok_or("text max positions")? as usize;
    token_limits(rows,max_positions)?;
    let (w,h)=if request.canvases.is_empty() {(32,32)}else{crate::resize::smart(request.width,request.height,32,65536,16777216)?};
    let gh=h/16;let gw=w/16;let image_tokens=request.canvases.len()*gh*gw/4;
    resource_limits(gh,gw,rows)?;
    if prep["image_tokens"].as_u64()!=Some(image_tokens as u64) || prep["grid_height"]!=gh || prep["grid_width"]!=gw {
        return Err("prepared vision layout mismatch".into());
    }
    let ids=fs::read(prepared.join("input_ids.raw"))?;
    let positions=fs::read(prepared.join("position_ids.raw"))?;
    if ids.len()!=rows*8 || positions.len()!=rows*3*8 {return Err("prepared token/position byte count".into());}
    let vocab=index["weights"]["language_model.embed_tokens.weight"]["shape"][0].as_u64().ok_or("embedding vocabulary")?;
    let image_id=config["image_token_id"].as_i64().ok_or("image token")?;
    let mut count=0;
    for raw in ids.chunks_exact(8) {
        let id=i64::from_le_bytes(raw.try_into().unwrap());
        if id<0 || id as u64>=vocab {return Err("token ID outside embedding vocabulary".into());}
        if id==image_id {count+=1;}
    }
    if count!=image_tokens {return Err("prepared image token count".into());}
    for raw in positions.chunks_exact(8) {
        let pos=i64::from_le_bytes(raw.try_into().unwrap());
        if pos<0 || pos as usize>=max_positions {return Err("position outside model limit".into());}
    }
    let expected=[("text_token_tags.raw",rows*8),("mm_token_type_ids.raw",rows*8),
        ("image_grid_thw.raw",request.canvases.len()*3*8)];
    for (name,bytes) in expected {
        if fs::metadata(prepared.join(name))?.len()!=bytes as u64 {return Err(format!("prepared tensor byte count: {name}").into());}
    }
    if pixels.total_rows()!=request.canvases.len()*gw*gh {return Err("prepared pixel row count".into());}
    Ok(rows)
}

#[cfg(test)] mod tests {
 use super::*;
 #[test] fn generic_geometries(){for (w,h) in [(768,768),(768,1344),(1344,768),(1440,1440),(769,1345),(4160,4160)]{assert_eq!(geometry(w,h).unwrap(),w*h*3);}}
 #[test] fn invalid_dimensions(){for (w,h) in [(0,768),(768,0),(1,201),(usize::MAX,768)]{assert!(geometry(w,h).is_err());}}
 #[test] fn model_not_fixture_nt(){for n in [1172,2456,8192,9000,16384,262144]{assert!(token_limits(n,262144).is_ok());}for n in [0,262145,usize::MAX]{assert!(token_limits(n,262144).is_err());}}
 #[test] fn resource_bounds(){assert!(resource_limits(48,48,9000).is_ok());assert!(resource_limits(256,256,262144).is_err());assert!(scratch_bytes(usize::MAX,4,1).is_err());}
 #[test] fn idle_rank_does_not_reject_the_leaders_memory_budget(){
   let free=[3584,0,512,64,0,0,2048,1024];
   // A short request runs only on rank 0. Busy idle GPUs must not veto it.
   assert!(free.iter().enumerate().all(|(rank,&bytes)|device_memory_required(rank==0,512,1024,2048).unwrap()<=bytes));
   // Keep the complete scratch/weight/vendor requirement on active ranks.
   assert!(device_memory_required(true,512,1024,2048).unwrap()>3583);
   assert!(device_memory_required(true,usize::MAX,1,0).is_err());
 }
}
