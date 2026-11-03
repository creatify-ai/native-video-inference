//! Qwen3-VL input presentation. Reads configuration, tokenizer data and pixels;
//! no gold token IDs, positions, pixel tensors or encoder activations are read.
use crate::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::{Arc, Mutex, OnceLock}};
use tokenizers::{AddedToken, Tokenizer};

/// Parsing Qwen's 11 MB tokenizer.json takes ~0.3 s, about 50x the cost of
/// encoding a prompt, so the resident conditioner keeps one parsed tokenizer
/// per file. The cache key includes the file's size and modification time, so
/// a replaced file is parsed again; the tokenizer itself is never mutated.
pub(crate) fn cached_tokenizer(path: &Path) -> Result<Arc<Tokenizer>> {
    type Stamp = (u64, Option<std::time::SystemTime>);
    type Key = (PathBuf, Stamp, Stamp);
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<Tokenizer>>>> = OnceLock::new();
    let metadata = fs::metadata(path)?;
    let config_path=path.with_file_name("tokenizer_config.json");
    let config_metadata=fs::metadata(&config_path)?;
    let key = (fs::canonicalize(path)?, (metadata.len(), metadata.modified().ok()),
               (config_metadata.len(), config_metadata.modified().ok()));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(tokenizer) = cache.lock().map_err(|_| "tokenizer cache poisoned")?.get(&key) { return Ok(tokenizer.clone()); }
    let mut tokenizer=Tokenizer::from_file(path)?;
    let config:Value=serde_json::from_slice(&fs::read(config_path)?)?;
    configure_special_tokens(&mut tokenizer,&config)?;
    let tokenizer = Arc::new(tokenizer);
    let mut entries = cache.lock().map_err(|_| "tokenizer cache poisoned")?;
    entries.retain(|k, _| k.0 != key.0);
    entries.insert(key, tokenizer.clone());
    Ok(tokenizer)
}

/// AutoTokenizer also registers specials declared only in tokenizer_config.json.
/// The official H3 bundle declares dialogue/lyrics markers outside tokenizer.json;
/// preserve their configured order so the appended IDs address the right weights.
fn configure_special_tokens(tokenizer:&mut Tokenizer,config:&Value)->Result<()> {
    if let Some(entries)=config.get("additional_special_tokens") {
        let entries=entries.as_array().ok_or("additional_special_tokens must be an array")?;
        let tokens=entries.iter().map(|entry|->Result<AddedToken>{
            if let Some(text)=entry.as_str(){return Ok(AddedToken::from(text,true));}
            let mut token:AddedToken=serde_json::from_value(entry.clone())?;
            token.special=true;Ok(token)
        }).collect::<Result<Vec<_>>>()?;
        tokenizer.add_special_tokens(&tokens);
    }
    Ok(())
}

#[cfg(test)] mod tokenizer_configuration_tests {
    use super::*;
    #[test] fn configuration_only_specials_keep_declared_order_and_whitespace() {
        use tokenizers::{models::wordlevel::WordLevel,pre_tokenizers::whitespace::Whitespace};
        let model=WordLevel::builder().vocab([("[UNK]".into(),0),("hello".into(),1)]
            .into_iter().collect()).unk_token("[UNK]".into()).build().unwrap();
        let mut tokenizer=Tokenizer::new(model);tokenizer.with_pre_tokenizer(Some(Whitespace));
        let config=json!({"additional_special_tokens":["<d>","</d>","<|lyrics_start|>"]});
        configure_special_tokens(&mut tokenizer,&config).unwrap();
        assert_eq!(tokenizer.token_to_id("<d>"),Some(2));
        assert_eq!(tokenizer.token_to_id("</d>"),Some(3));
        assert_eq!(tokenizer.encode("hello <d>hello</d>",false).unwrap().get_ids(),&[1,2,1,3]);
        configure_special_tokens(&mut tokenizer,&config).unwrap();
        assert_eq!(tokenizer.get_vocab_size(true),5);
    }
}

pub fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

pub(crate) fn dump(output: &Path, name: &str, bytes: &[u8], dtype: &str, shape: &[usize]) -> Result<Value> {
    fs::write(output.join(name), bytes)?;
    Ok(json!({"file":name,"dtype":dtype,"shape":shape,"bytes":bytes.len(),"sha256":crate::debug_audit::sha(||hash(bytes))}))
}
pub(crate) fn i64bytes(values: &[i64]) -> Vec<u8> { values.iter().flat_map(|x| x.to_le_bytes()).collect() }
/// Little-endian FP32 bytes, without a copy on little-endian hosts.
pub(crate) fn f32_le_bytes(values: &[f32]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    { std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }) }
    #[cfg(not(target_endian = "little"))]
    { std::borrow::Cow::Owned(values.iter().flat_map(|x| x.to_le_bytes()).collect()) }
}

pub(crate) fn array3(config: &Value, name: &str) -> Result<[f32; 3]> {
    let values = config[name].as_array().ok_or_else(|| format!("missing {name}"))?;
    if values.len() != 3 { return Err(format!("invalid {name}").into()); }
    Ok([values[0].as_f64().ok_or("number")? as f32,
        values[1].as_f64().ok_or("number")? as f32,
        values[2].as_f64().ok_or("number")? as f32])
}

pub fn run_with_size(export: &Path, prompt_path: &Path, canvases: &[&Path], output: &Path,
                     width: usize, height: usize) -> Result<Value> {
    let (mut report, pixels) = plan_with_size(export, prompt_path, canvases, output, width, height)?;
    write_pixels(output, &mut report, "pixel_values.raw", &pixels)?;
    if crate::debug_audit::enabled(){fs::write(output.join("prepared.json"), serde_json::to_vec_pretty(&report)?)?;}
    Ok(report)
}

/// The whole pixel tensor as the single-process forward reads it.
pub(crate) fn write_pixels(output: &Path, report: &mut Value, name: &str, pixels: &crate::pixels::Pixels) -> Result<()> {
    let rows = pixels.total_rows();
    let entry = dump(output, name, &f32_le_bytes(&pixels.rows(0..rows)?), "float32", &[rows, 1536])?;
    report["tensors"].as_object_mut().ok_or("prepared tensors")?.insert(name.into(), entry);
    Ok(())
}

/// Everything except the pixel rows, which the caller produces whole or by range.
pub fn plan_with_size(export: &Path, prompt_path: &Path, canvases: &[&Path], output: &Path,
                      width: usize, height: usize) -> Result<(Value, crate::pixels::Pixels)> {
    let processor: Value = serde_json::from_slice(&fs::read(export.join("processor/processor_config.json"))?)?;
    let config = &processor["image_processor"];
    let model: Value = serde_json::from_slice(&fs::read(export.join("model_config/config.json"))?)?;
    let tokenizer_path = export.join("tokenizer/tokenizer.json");
    let tokenizer = cached_tokenizer(&tokenizer_path)?;
    if fs::metadata(prompt_path)?.len()>crate::dynamic_request::budget("H3_QWEN_PROMPT_LIMIT_BYTES",16usize<<20)? as u64 {return Err("prompt exceeds configured byte budget".into());}
    let prompt = fs::read_to_string(prompt_path)?;
    if !canvases.is_empty() {crate::dynamic_request::geometry(width,height)?;}
    // The shared canvas remains unchanged; Qwen resizes its own vision input.
    let patch = config["patch_size"].as_u64().ok_or("patch_size")? as usize;
    let merge = config["merge_size"].as_u64().ok_or("merge_size")? as usize;
    let temporal = config["temporal_patch_size"].as_u64().ok_or("temporal_patch_size")? as usize;
    if canvases.len() > 2 || patch != 16 || merge != 2 || temporal != 2 {
        return Err("unsupported Qwen processor layout".into());
    }
    let factor = patch * merge;
    let min_pixels = config["size"]["shortest_edge"].as_u64()
        .or_else(|| config["min_pixels"].as_u64()).ok_or("processor min pixels")? as usize;
    let max_pixels = config["size"]["longest_edge"].as_u64()
        .or_else(|| config["max_pixels"].as_u64()).ok_or("processor max pixels")? as usize;
    let area = width.checked_mul(height).ok_or("canvas area overflow")?;
    if config["resample"]!=3 || config["do_resize"]!=true{return Err("unsupported Qwen resample configuration".into());}
    let (rw,rh)=if canvases.is_empty() {(factor,factor)}else{crate::resize::smart(width,height,factor,min_pixels,max_pixels)?};
    if config["do_normalize"] != true || config["do_rescale"] != true {
        return Err("unsupported normalization configuration".into());
    }
    let scale = config["rescale_factor"].as_f64().ok_or("rescale factor")?;
    let mut mean = array3(&config, "image_mean")?;
    let mut std = array3(&config, "image_std")?;
    // Transformers TorchvisionBackend fuses the rescale factor into FP32 mean
    // and std, then torchvision performs two separate FP32 sub/div operations.
    let reciprocal = (1.0f64 / scale) as f32;
    for channel in 0..3 { mean[channel] *= reciprocal; std[channel] *= reciprocal; }
    let gh = rh / patch;
    let gw = rw / patch;
    let patch_values = 3 * temporal * patch * patch;
    let vision_start = tokenizer.token_to_id("<|vision_start|>").ok_or("vision start ID")?;
    let vision_end = tokenizer.token_to_id("<|vision_end|>").ok_or("vision end ID")?;
    let image_pad = tokenizer.token_to_id("<|image_pad|>").ok_or("image pad ID")?;
    if image_pad as u64 != model["image_token_id"].as_u64().ok_or("model image token")? {
        return Err("tokenizer/model image ID mismatch".into());
    }
    let mut ids = vec![];
    let mut tags = vec![];
    let mut grid = vec![];
    let text_tag = 1i64;
    let video_tag = 0i64;
    for index in 0..canvases.len() {
        let label = tokenizer.encode(format!("<Picture {}>: ", index + 1), false)?;
        ids.extend(label.get_ids().iter().map(|&x| x as i64));
        tags.extend(std::iter::repeat_n(text_tag, label.len()));
        ids.push(vision_start as i64);
        ids.extend(std::iter::repeat_n(image_pad as i64, gh * gw / (merge * merge)));
        ids.push(vision_end as i64);
        tags.extend(std::iter::repeat_n(video_tag, 2 + gh * gw / (merge * merge)));
        grid.extend([1i64, gh as i64, gw as i64]);
    }
    let encoded = tokenizer.encode(prompt.as_str(), false)?;
    ids.extend(encoded.get_ids().iter().map(|&x| x as i64));
    tags.extend(std::iter::repeat_n(text_tag, encoded.len()));
    let mm_ids: Vec<i64> = ids.iter().map(|&x| if x == image_pad as i64 { 1 } else { 0 }).collect();
    let mut positions = vec![0i64; ids.len() * 3];
    let mut cursor = 0usize;
    let mut current_pos = 0i64;
    let mut image = 0usize;
    while cursor < ids.len() {
        let modality = mm_ids[cursor];
        let end = (cursor + 1..ids.len()).find(|&i| mm_ids[i] != modality).unwrap_or(ids.len());
        if modality == 0 {
            for i in cursor..end { for axis in 0..3 {
                positions[axis * ids.len() + i] = current_pos + (i - cursor) as i64;
            } }
            current_pos += (end - cursor) as i64;
        } else {
            if image >= canvases.len() || end - cursor != gh * gw / (merge * merge) { return Err("invalid image token span".into()); }
            for i in cursor..end {
                let offset = i - cursor;
                positions[i] = current_pos;
                positions[ids.len() + i] = current_pos + (offset / (gw / merge)) as i64;
                positions[2 * ids.len() + i] = current_pos + (offset % (gw / merge)) as i64;
            }
            current_pos += (gh.max(gw) / merge) as i64;
            image += 1;
        }
        cursor = end;
    }
    if image != canvases.len() { return Err("missing image token span".into()); }
    crate::dynamic_request::token_limits(ids.len(),model["text_config"]["max_position_embeddings"].as_u64().ok_or("model max positions")? as usize)?;
    let scratch_estimate=crate::dynamic_request::resource_limits(gh,gw,ids.len())?;
    let mut inputs = vec![];
    for canvas in canvases {
        if fs::metadata(canvas)?.len() != (area * 3) as u64 { return Err("canvas RGB byte count mismatch".into()); }
        let sha = if crate::debug_audit::enabled() { json!(hash(&fs::read(canvas)?)) } else { Value::Null };
        inputs.push(json!({"sha256":sha,"shape":[height,width,3],"dtype":"uint8"}));
    }
    if patch_values != 1536 { return Err("unsupported Qwen patch size".into()); }
    let pixels = crate::pixels::Pixels{visuals:canvases.iter().map(|canvas|crate::pixels::Visual{path:canvas.to_path_buf(),
        w:width,h:height,rw,rh,sampled:vec![0]}).collect(),mean,std};
    fs::create_dir_all(output)?;
    let mut tensors = serde_json::Map::new();
    for (name, values, shape) in [
        ("input_ids.raw", &ids, vec![1, ids.len()]),
        ("mm_token_type_ids.raw", &mm_ids, vec![1, ids.len()]),
        ("text_token_tags.raw", &tags, vec![ids.len()]),
        ("image_grid_thw.raw", &grid, vec![canvases.len(), 3]),
        ("position_ids.raw", &positions, vec![3, 1, ids.len()]),
    ] { tensors.insert(name.to_string(), dump(output, name, &i64bytes(values), "int64", &shape)?); }
    let report = json!({"schema":"h3-conditioner-prepared-v1","canvas_inputs":inputs,
        "canvas_width":width,"canvas_height":height,"keyframe_count":canvases.len(),"grid_height":gh,"grid_width":gw,
        "processor_resize":if (rw,rh)==(width,height){"identity"}else{"native_pinned_torch_uint8_bicubic_antialias"},
        "processor_width":rw,"processor_height":rh,"scratch_estimated_bytes":scratch_estimate,
        "processor_pixel_limits":[min_pixels,max_pixels],
        "prompt_sha256":crate::debug_audit::sha(||hash(prompt.as_bytes())),"tokenizer_sha256":if crate::debug_audit::enabled(){json!(hash(&fs::read(tokenizer_path)?))}else{Value::Null},
        "token_count":ids.len(),"image_tokens":gh*gw/(merge*merge)*canvases.len(),
        "tensors":tensors,"gold_activations_read":false,"passed":true});
    Ok((report, pixels))
}
