use crate::*;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};
include!("long_assemble.rs");

fn save(root:&Path,name:&str,bytes:&[u8])->Result<Value> {
    fs::write(root.join(name),bytes)?;
    Ok(json!({"file":name,"bytes":bytes.len(),"sha256":audit::hash(bytes)}))
}
/// BF16 request tensor that is byte-for-byte an existing component output: link
/// it instead of reading and rewriting it (copy if a link is impossible).
fn linked_bf16_tensor(root:&Path,name:&str,source:&Path,shape:&[usize])->Result<Value> {
    let bytes=shape.iter().product::<usize>()*2;
    if fs::metadata(source)?.len()!=bytes as u64{return Err("request tensor shape mismatch".into());}
    let target=root.join(format!("{name}.raw"));
    if fs::hard_link(source,&target).is_err(){fs::copy(source,&target)?;}
    let digest=if audit::enabled(){audit::hash(&fs::read(&target)?)}else{Value::Null};
    let mut stride=vec![1;shape.len()];
    for i in (0..shape.len()-1).rev(){stride[i]=stride[i+1]*shape[i+1];}
    Ok(json!({"file":format!("{name}.raw"),"bytes":bytes,"sha256":digest,"dtype":"torch.bfloat16",
        "shape":shape,"stride":stride,"contiguous":true,"storage_offset":0}))
}
/// The precomputed FP32 hidden_states file (`assemble-noise`), its keyframe rows written in place over
/// the zero prefix, moved into the request: the same bytes `tensor` would write, without rewriting
/// the noise rows.
fn in_place_hidden_states(root:&Path,source:&Path,prefix:&[f32],prefix_bytes:u64,shape:&[usize])->Result<Value> {
    use std::os::unix::fs::FileExt;
    let bytes=shape.iter().product::<usize>()*4;
    let file=fs::OpenOptions::new().write(true).open(source)?;
    if file.metadata()?.len()!=bytes as u64 || (prefix.len()*4) as u64!=prefix_bytes{return Err("precomputed hidden_states shape mismatch".into());}
    file.write_all_at(&f32_bytes(prefix),0)?;drop(file);
    let target=root.join("hidden_states.raw");
    if fs::rename(source,&target).is_err(){fs::copy(source,&target)?;}
    let digest=if audit::enabled(){audit::hash(&fs::read(&target)?)}else{Value::Null};
    let mut stride=vec![1;shape.len()];
    for i in (0..shape.len()-1).rev(){stride[i]=stride[i+1]*shape[i+1];}
    Ok(json!({"file":"hidden_states.raw","bytes":bytes,"sha256":digest,"dtype":"torch.float32",
        "shape":shape,"stride":stride,"contiguous":true,"storage_offset":0}))
}
fn tensor(root:&Path,name:&str,bytes:&[u8],dtype:&str,shape:&[usize])->Result<Value> {
    let width=match dtype {"torch.bfloat16"=>2,"torch.float32"=>4,"torch.int64"|"torch.float64"=>8,_=>return Err("unsupported tensor type".into())};
    if shape.iter().product::<usize>()*width!=bytes.len(){return Err("request tensor shape mismatch".into());}
    let mut stride=vec![1;shape.len()];
    for i in (0..shape.len()-1).rev(){stride[i]=stride[i+1]*shape[i+1];}
    let mut value=save(root,&format!("{name}.raw"),bytes)?;
    value["dtype"]=json!(dtype);value["shape"]=json!(shape);value["stride"]=json!(stride);
    value["contiguous"]=json!(true);value["storage_offset"]=json!(0);
    Ok(value)
}
fn shape_args(args:&[String], start:usize, include_frames:bool)->Result<Shape> {
    if args.len()==start {return Ok(Shape::case01());}
    let count=if include_frames {3}else{2};
    if args.len()!=start+count && args.len()!=start+count+1 {return Err("expected optional WIDTH HEIGHT [NUM_FRAMES]".into());}
    Shape::new(args[start].parse()?,args[start+1].parse()?,
        if include_frames {args[start+2].parse()?} else {345})?.with_keyframes(
        if args.len()==start+count+1 {args[start+count].parse()?} else {2})
}
fn read_tags(path:&str)->Result<Vec<i64>> {
    let bytes=fs::read(path)?;
    if bytes.is_empty() || bytes.len()%8!=0 {return Err("expected nonempty I64 conditioner tags".into());}
    Ok(bytes.chunks_exact(8).map(|x|i64::from_le_bytes(x.try_into().unwrap())).collect())
}
fn reference_structure(root:&Path,shape:Shape,references:&[ref2va::ReferenceGeometry],
    result:ref2va::ReferenceLayout,seed:u64,nfe:usize)->Result<Value> {
    let layout=result.packed;
    let (nv,na,nt)=(layout.video_indices.len(),layout.audio_indices.len(),layout.text_indices.len());
    let total=nv+na+nt;limits::packed_rows(&[nv,na,nt])?;
    fs::create_dir(root)?;
    let zeros=|name:&str,dtype:&str,dims:&[usize],width:usize|->Result<Value>{
        let bytes=dims.iter().try_fold(width,|n,d|n.checked_mul(*d)).ok_or("structure tensor size overflow")?;
        fs::File::create(root.join(format!("{name}.raw")))?.set_len(bytes as u64)?;
        let mut stride=vec![1;dims.len()];for i in (0..dims.len()-1).rev(){stride[i]=stride[i+1]*dims[i+1];}
        Ok(json!({"file":format!("{name}.raw"),"bytes":bytes,"sha256":Value::Null,"dtype":dtype,"shape":dims,
            "stride":stride,"contiguous":true,"storage_offset":0}))
    };
    let inputs=json!({
        "hidden_states":zeros("hidden_states","torch.float32",&[1,nv,96],4)?,
        "audio_hidden_states":zeros("audio_hidden_states","torch.float32",&[1,na,32],4)?,
        "encoder_hidden_states":zeros("encoder_hidden_states","torch.bfloat16",&[1,nt,5120],2)?,
        "position_ids":tensor(root,"position_ids",&f64_bytes(&layout.position_ids),"torch.float64",&[total,3])?,
        "token_tags":tensor(root,"token_tags",&i64_bytes(&layout.token_tags),"torch.int64",&[total])?,
        "video_indices":tensor(root,"video_indices",&i64_bytes(&layout.video_indices),"torch.int64",&[nv])?,
        "audio_indices":tensor(root,"audio_indices",&i64_bytes(&layout.audio_indices),"torch.int64",&[na])?,
        "text_indices":tensor(root,"text_indices",&i64_bytes(&layout.text_indices),"torch.int64",&[nt])?});
    let manifest=json!({"schema":"h3-native-request-v1","status":"complete","producer":"h3-native-pipeline",
        "seed":seed,"request_uint32_draws":0,"inputs":inputs,"workflow":"ref2va","nfe":nfe,
        "keyframe_noise_aug":0.999,"condition_video_rows":result.condition_video_rows,
        "condition_audio_rows":result.condition_audio_rows,"references":references,
        "shape":shape.json(),"gold_activations_used":false,"structure_only":true});
    fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&manifest)?)?;Ok(manifest)
}
fn layout_delivery(args:&mut Vec<String>)->Result<Option<usize>> {
    if args.len()>=2&&args[args.len()-2]=="--delivery-frames" {
        let frames=args.pop().unwrap().parse()?;args.pop();Ok(Some(frames))
    }else{Ok(None)}
}
/// A caller-supplied conditioned stream, admitted by its registered workflow.
/// Ordinary H3 calls do not supply an extension.
#[derive(Clone,Copy)]
pub struct AssemblyExtension {
    pub workflow: &'static str,
    pub audio: fn(&[f32],usize)->Result<Vec<f32>>,
    pub keyframes: usize,
}
pub fn run(args:Vec<String>, extension:Option<AssemblyExtension>)->Result<Value> {
    if args.len()<3 {return Err("usage: h3-native-pipeline generate CONFIG REQUEST OUTPUT | noise SEED OUTPUT COUNT... | canvas FIRST LAST OUTPUT [WIDTH HEIGHT] | layout TAGS OUTPUT [WIDTH HEIGHT NUM_FRAMES] | unpack VIDEO AUDIO OUTPUT [WIDTH HEIGHT NUM_FRAMES] | assemble CONDITIONING TAGS CONDITION0 CONDITION1 SEED OUTPUT [WIDTH HEIGHT NUM_FRAMES]".into());}
    let report=match args[1].as_str() {
        "audio-wav"=>{
            if args.len()!=6{return Err("audio-wav PLANAR_FP32 OUTPUT_WAV CHANNELS SAMPLE_RATE".into());}
            let raw=fs::read(&args[2])?;let channels:u16=args[4].parse()?;let rate:u32=args[5].parse()?;
            let wav=media::wav_bytes(&raw,channels,rate)?;fs::write(&args[3],&wav)?;
            json!({"mode":"audio-wav","passed":true,"channels":channels,"sample_rate":rate,"bytes":wav.len()})
        },
        "long-video-plan"=>{
            if args.len()!=4{return Err("long-video-plan INPUT_JSON OUTPUT_JSON".into());}
            let cases:Vec<Value>=serde_json::from_slice(&fs::read(&args[2])?)?;
            let plans:Vec<Value>=cases.iter().map(long_video::describe).collect::<Result<_>>()?;
            fs::write(&args[3],serde_json::to_vec(&plans)?)?;
            json!({"mode":"long-video-plan","passed":true,"plans":plans.len(),"inference_enabled":false})
        },
        "resample-audio"=>{
            if args.len()!=7{return Err("resample-audio INPUT OUTPUT SOURCE_RATE CHANNELS TARGET_FRAMES".into());}
            let raw=fs::read(&args[2])?;
            if raw.len()%4!=0{return Err("decoded audio must be FP32".into());}
            let input:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
            let output=resample::normalize(&input,args[5].parse()?,args[4].parse()?,args[6].parse()?)?;
            fs::write(&args[3],f32_bytes(&output))?;
            json!({"passed":true,"channels":2,"sample_rate":32000,"samples":output.len()/2})
        },
        "noise"=>{
            let seed:u64=args[2].parse()?; let root=PathBuf::from(args.get(3).ok_or("missing output")?); fs::create_dir_all(&root)?;
            let mut rng=CpuGenerator::new(seed);let mut outputs=Vec::new();
            for (i,arg) in args.iter().skip(4).enumerate() {
                let count:usize=arg.parse()?;let values=rng.normal(count)?;
                outputs.push(save(&root,&format!("noise{i}.raw"),&f32_bytes(&values))?);
            }
            json!({"mode":"noise","seed":seed,"uint32_draws":rng.draws,"outputs":outputs})
        },
        "canvas"=>{
            let shape=shape_args(&args,5,false)?;
            let root=Path::new(args.get(4).ok_or("missing output")?);fs::create_dir_all(root)?;
            let first=canvas_rgb_sized(Path::new(&args[2]),shape.width,shape.height)?;
            let last=if shape.keyframe_count==2 {
                Some(save(root,"last.rgb",&canvas_rgb_sized(Path::new(&args[3]),shape.width,shape.height)?)?)
            }else{None};
            if shape.keyframe_count==0 {return Err("text-only requests have no canvas stage".into());}
            json!({"mode":"canvas","shape":[shape.keyframe_count,shape.height,shape.width,3],"first":save(root,"first.rgb",&first)?,"last":last})
        },
        "layout-ref2va"=>{
            if args.len()!=5 {return Err("usage: layout-ref2va TAGS DESCRIPTOR OUTPUT".into());}
            let descriptor:Value=serde_json::from_slice(&fs::read(&args[3])?)?;
            let shape=Shape::from_request(&descriptor["target"])?;
            let references:Vec<ref2va::ReferenceGeometry>=serde_json::from_value(descriptor["references"].clone())?;
            let tags=read_tags(&args[2])?;
            let result=ref2va::layout(&tags,shape,&references)?;let layout=result.packed;
            let root=Path::new(&args[4]);fs::create_dir(root)?;
            json!({"mode":"layout-ref2va","condition_video_rows":result.condition_video_rows,
                "condition_audio_rows":result.condition_audio_rows,
                "position_ids":save(root,"position_ids.raw",&f64_bytes(&layout.position_ids))?,
                "token_tags":save(root,"token_tags.raw",&i64_bytes(&layout.token_tags))?,
                "video_indices":save(root,"video_indices.raw",&i64_bytes(&layout.video_indices))?,
                "audio_indices":save(root,"audio_indices.raw",&i64_bytes(&layout.audio_indices))?,
                "text_indices":save(root,"text_indices.raw",&i64_bytes(&layout.text_indices))?})
        },
        "assemble-ref2va"|"assemble-ref2va-structure"=>{
            if args.len()!=7{return Err("assemble-ref2va CONDITIONING TAGS DESCRIPTOR SEED OUTPUT".into());}
            let descriptor:Value=serde_json::from_slice(&fs::read(&args[4])?)?;
            let shape=Shape::from_request(&descriptor["target"])?;
            let entries=descriptor["references"].as_array().ok_or("reference entries")?;
            let references:Vec<ref2va::ReferenceGeometry>=entries.iter().map(|r|serde_json::from_value(r["geometry"].clone())).collect::<std::result::Result<_,_>>()?;
            let tags=read_tags(&args[3])?;
            let result=ref2va::layout(&tags,shape,&references)?;
            if args[1]=="assemble-ref2va-structure" {
                return finish_report(reference_structure(Path::new(&args[6]),shape,&references,result,args[5].parse()?,recipe::Recipe::from_request(&descriptor["target"])?.nfe)?);
            }
            let layout=result.packed;
            let conditioning=fs::read(&args[2])?;
            if conditioning.len()!=tags.len()*5120*2 || conditioning.chunks_exact(2).any(|b|u16::from_le_bytes(b.try_into().unwrap())&0x7f80==0x7f80) {
                return Err("conditioning dimensions/nonfinite values".into());
            }
            let seed:u64=args[5].parse()?;let mut generator=CpuGenerator::new(seed);
            let visual_conditions=references.iter().filter(|r|r.kind!=ref2va::Kind::Audio).count();
            let mut condition_draws=0usize;
            let mut video=vec![];let mut audio=vec![];
            for (entry,geometry) in entries.iter().zip(&references) {
                if geometry.kind!=ref2va::Kind::Audio {
                    let path=Path::new(entry["video_file"].as_str().ok_or("native visual latent file")?);
                    let condition=read_f32(path)?;
                    let count=24*geometry.latent_frames*geometry.latent_height*geometry.latent_width;
                    if condition.len()!=count || condition.iter().any(|x|!x.is_finite()){return Err("reference visual latent dimensions/nonfinite values".into());}
                    let (noise,draws)=noise::visual_condition(seed,shape.num_latent_frames,visual_conditions,
                        geometry.latent_frames,geometry.latent_height,geometry.latent_width)?;
                    condition_draws+=draws;let t=0.999f32;let sigma=1.0f32-t;
                    let values:Vec<_>=condition.iter().zip(noise).map(|(&v,n)|t*v+sigma*n).collect();
                    video.extend(patchify(&values,24,geometry.latent_frames,geometry.latent_height,geometry.latent_width)?);
                }
                if geometry.audio_rows>0 {
                    let values=read_f32(Path::new(entry["audio_file"].as_str().ok_or("native audio latent file")?))?;
                    if values.len()!=geometry.audio_rows*32 || values.iter().any(|x|!x.is_finite()){return Err("reference audio latent dimensions/nonfinite values".into());}
                    audio.extend(values);
                }
            }
            video.extend(patchify(&generator.normal(24*shape.num_latent_frames*shape.latent_height*shape.latent_width)?,24,shape.num_latent_frames,shape.latent_height,shape.latent_width)?);
            audio.extend(noise::audio(seed,shape.audio_rows()*32)?);
            let (nv,na,nt)=(video.len()/96,audio.len()/32,tags.len());let total=nv+na+nt;
            let root=Path::new(&args[6]);fs::create_dir(root)?;
            let inputs=json!({
                "hidden_states":tensor(root,"hidden_states",&f32_bytes(&video),"torch.float32",&[1,nv,96])?,
                "audio_hidden_states":tensor(root,"audio_hidden_states",&f32_bytes(&audio),"torch.float32",&[1,na,32])?,
                "encoder_hidden_states":tensor(root,"encoder_hidden_states",&conditioning,"torch.bfloat16",&[1,nt,5120])?,
                "position_ids":tensor(root,"position_ids",&f64_bytes(&layout.position_ids),"torch.float64",&[total,3])?,
                "token_tags":tensor(root,"token_tags",&i64_bytes(&layout.token_tags),"torch.int64",&[total])?,
                "video_indices":tensor(root,"video_indices",&i64_bytes(&layout.video_indices),"torch.int64",&[nv])?,
                "audio_indices":tensor(root,"audio_indices",&i64_bytes(&layout.audio_indices),"torch.int64",&[na])?,
                "text_indices":tensor(root,"text_indices",&i64_bytes(&layout.text_indices),"torch.int64",&[nt])?});
            let report=json!({"schema":"h3-native-request-v1","status":"complete","producer":"h3-native-pipeline",
                "workflow":"ref2va","nfe":recipe::Recipe::from_request(&descriptor["target"])?.nfe,"references":references,"inputs":inputs,"shape":shape.json(),
                "seed":seed,"request_uint32_draws":generator.draws+condition_draws+shape.audio_rows()*32,"keyframe_noise_aug":0.999,
                "condition_video_rows":result.condition_video_rows,"condition_audio_rows":result.condition_audio_rows,
                "gold_activations_used":false,"component_descriptor":descriptor});
            fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&report)?)?;report
        },
        "layout"=>{
            let shape=shape_args(&args,4,true)?;
            let root=Path::new(args.get(3).ok_or("missing output")?);fs::create_dir_all(root)?;
            let tags=read_tags(&args[2])?;
            let layout=fl2va_layout(&tags,shape)?;
            json!({"mode":"layout","shape":shape.json(),"position_ids":save(root,"position_ids.raw",&f64_bytes(&layout.position_ids))?,
              "token_tags":save(root,"token_tags.raw",&i64_bytes(&layout.token_tags))?,
              "video_indices":save(root,"video_indices.raw",&i64_bytes(&layout.video_indices))?,
              "audio_indices":save(root,"audio_indices.raw",&i64_bytes(&layout.audio_indices))?,
              "text_indices":save(root,"text_indices.raw",&i64_bytes(&layout.text_indices))?})
        },
        "unpack-ref2va"=>{
            if args.len()!=6{return Err("unpack-ref2va VIDEO AUDIO DESCRIPTOR OUTPUT".into());}
            let descriptor:Value=serde_json::from_slice(&fs::read(&args[4])?)?;
            let shape=Shape::from_request(&descriptor["target"])?;
            let cv=usize::try_from(descriptor["condition_video_rows"].as_u64().ok_or("condition video rows")?)?;
            let ca=usize::try_from(descriptor["condition_audio_rows"].as_u64().ok_or("condition audio rows")?)?;
            let packed_video=read_f32(Path::new(&args[2]))?;let packed_audio=read_f32(Path::new(&args[3]))?;
            let target_video=shape.num_latent_frames.checked_mul(shape.rows_per_frame()).ok_or("target rows overflow")?;
            if cv.checked_add(target_video).and_then(|n|n.checked_mul(96))!=Some(packed_video.len()) ||
               ca.checked_add(shape.audio_rows()).and_then(|n|n.checked_mul(32))!=Some(packed_audio.len()) || ca%2!=0 {
                return Err("Ref2VA packed streams differ from target geometry and conditioning counts".into());
            }
            let video=unpatchify_video(&packed_video[cv*96..],24,shape.num_latent_frames,shape.latent_height,shape.latent_width)?;
            let audio=unpack_audio(&packed_audio[ca*32..],2,shape.num_audio_latents,32)?;
            let root=Path::new(&args[5]);fs::create_dir(root)?;
            json!({"mode":"unpack-ref2va","shape":shape.json(),"condition_video_rows":cv,"condition_audio_rows":ca,
                "video":save(root,"video.raw",&f32_bytes(&video))?,"audio":save(root,"audio.raw",&f32_bytes(&audio))?})
        },
        "unpack"|"unpack-long"=>{
            let shape=if args[1]=="unpack-long"{long_shape_args(&args,5)?}else{shape_args(&args,5,true)?};
            let root=Path::new(args.get(4).ok_or("missing output")?);fs::create_dir_all(root)?;
            let packed_video=read_f32(Path::new(&args[2]))?;let packed_audio=read_f32(Path::new(&args[3]))?;
            if packed_video.len()!=shape.video_rows()*96{return Err("packed video shape differs from request".into());}
            let video=unpatchify_video(&packed_video[shape.condition_rows()*96..],24,
                shape.num_latent_frames,shape.latent_height,shape.latent_width)?;
            let audio=unpack_audio(&packed_audio,2,shape.num_audio_latents,32)?;
            json!({"mode":"unpack","shape":shape.json(),"video":save(root,"video.raw",&f32_bytes(&video))?,"audio":save(root,"audio.raw",&f32_bytes(&audio))?})
        },
        "assemble-noise"=>{
            // The exact draws `assemble` makes for an FL2VA request, in the same order,
            // computed before the conditioner and keyframe latents exist.
            let shape=shape_args(&args,4,true)?;
            let seed:u64=args[2].parse()?;let root=Path::new(&args[3]);fs::create_dir_all(root)?;
            let mut generator=CpuGenerator::new(seed);
            let mut condition_draws=0usize;
            let mut files=vec![];
            for index in 0..shape.keyframe_count {
                let name=format!("keyframe{index}.f32");
                let (values,draws)=noise::visual_condition(seed,shape.num_latent_frames,shape.keyframe_count,1,
                    shape.latent_height,shape.latent_width)?;
                condition_draws+=draws;
                fs::write(root.join(&name),f32_bytes(&values))?;files.push(name);
            }
            // The request's whole hidden_states tensor: the keyframe rows stay a zero prefix that
            // `assemble` overwrites in place once the keyframe latents exist.
            let prefix=(shape.keyframe_count*shape.posterior_elements()*4) as u64;
            {
                use std::io::{Seek,SeekFrom,Write};
                let mut file=fs::File::create(root.join("hidden_states.f32"))?;
                file.set_len(prefix)?;file.seek(SeekFrom::Start(prefix))?;
                file.write_all(&f32_bytes(&patchify(&generator.normal(shape.posterior_elements()*shape.num_latent_frames)?,
                    24,shape.num_latent_frames,shape.latent_height,shape.latent_width)?))?;
            }
            fs::write(root.join("audio.f32"),f32_bytes(&noise::audio(seed,shape.audio_rows()*32)?))?;
            let manifest=json!({"schema":"h3-public-assemble-noise-v1","seed":seed,"shape":shape.json(),"draws":generator.draws+condition_draws+shape.audio_rows()*32,
                "keyframes":files,"hidden_states":"hidden_states.f32","keyframe_prefix_bytes":prefix,"audio":"audio.f32"});
            fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&manifest)?)?;
            manifest
        },
        "assemble-structure"=>{
            // Same layout inputs as `assemble` from the conditioner's prepared token
            // tags; latent/conditioning tensors are zero placeholders of the exact
            // size. Used only to prebuild the DiT layout; never generated from.
            // The same --nfe binding as assemble: the prebuilt layout must select the request's model.
            let mut shape_arguments=args.clone();
            let nfe=if shape_arguments.len()>=2 && shape_arguments[shape_arguments.len()-2]=="--nfe" {
                let value:usize=shape_arguments.pop().unwrap().parse()?;shape_arguments.pop();value
            }else{recipe::DEFAULT_NFE};
            let workflow=extension.map_or("fl2va",|e|e.workflow);
            recipe::Recipe::from_request(&json!({"workflow":workflow,"nfe":nfe}))?;
            let delivered=layout_delivery(&mut shape_arguments)?;
            let shape=shape_args(&shape_arguments,8,true)?;
            let root=Path::new(&args[7]);fs::create_dir_all(root)?;
            let tags=read_tags(&args[3])?;let seed:u64=args[6].parse()?;
            let layout=fl2va_layout_with_delivery(&tags,shape,delivered)?;
            let nt=tags.len();let nv=shape.video_rows();let na=shape.audio_rows();let total=nt+nv+na;
            let zeros=|name:&str,dtype:&str,dims:&[usize],width:usize|->Result<Value>{
                let bytes=dims.iter().product::<usize>()*width;
                fs::File::create(root.join(format!("{name}.raw")))?.set_len(bytes as u64)?;
                let mut stride=vec![1;dims.len()];for i in (0..dims.len()-1).rev(){stride[i]=stride[i+1]*dims[i+1];}
                Ok(json!({"file":format!("{name}.raw"),"bytes":bytes,"sha256":Value::Null,"dtype":dtype,"shape":dims,
                    "stride":stride,"contiguous":true,"storage_offset":0}))
            };
            let inputs=json!({
                "hidden_states":zeros("hidden_states","torch.float32",&[1,nv,96],4)?,
                "audio_hidden_states":zeros("audio_hidden_states","torch.float32",&[1,na,32],4)?,
                "encoder_hidden_states":zeros("encoder_hidden_states","torch.bfloat16",&[1,nt,5120],2)?,
                "position_ids":tensor(root,"position_ids",&f64_bytes(&layout.position_ids),"torch.float64",&[total,3])?,
                "token_tags":tensor(root,"token_tags",&i64_bytes(&layout.token_tags),"torch.int64",&[total])?,
                "video_indices":tensor(root,"video_indices",&i64_bytes(&layout.video_indices),"torch.int64",&[nv])?,
                "audio_indices":tensor(root,"audio_indices",&i64_bytes(&layout.audio_indices),"torch.int64",&[na])?,
                "text_indices":tensor(root,"text_indices",&i64_bytes(&layout.text_indices),"torch.int64",&[nt])?});
            let result=json!({"schema":"h3-native-request-v1","status":"complete","producer":"h3-native-pipeline",
                "seed":seed,"request_uint32_draws":0,"inputs":inputs,"workflow":workflow,"nfe":nfe,
                "keyframe_noise_aug":0.999,"condition_video_rows":shape.condition_rows(),
                "condition_audio_rows":if extension.is_some(){na}else{0},
                "shape":shape.json(),"component_input_files":args[2..6],"gold_activations_used":false,"structure_only":true});
            fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&result)?)?;
            result
        },
        "assemble"|"assemble-long"=>{
            // Inputs are outputs from real native components. No gold/model directory is accepted.
            let mut shape_arguments=args.clone();
            let nfe=if shape_arguments.len()>=2 && shape_arguments[shape_arguments.len()-2]=="--nfe" {
                let value:usize=shape_arguments.pop().unwrap().parse()?;shape_arguments.pop();value
            }else{recipe::DEFAULT_NFE};
            let workflow=extension.map_or("fl2va",|e|e.workflow);
            recipe::Recipe::from_request(&json!({"workflow":workflow,"nfe":nfe}))?;
            let delivered=layout_delivery(&mut shape_arguments)?;
            if delivered.is_some()&&(args[1]!="assemble"||extension.is_some()){return Err("short layout delivery is FL-only".into());}
            let shape=if args[1]=="assemble-long"{long_shape_args(&shape_arguments,8)?}else{shape_args(&shape_arguments,8,true)?};
            let root=Path::new(&args[7]);fs::create_dir_all(root)?;
            let tags=read_tags(&args[3])?;
            let seed:u64=args[6].parse()?;let mut generator=CpuGenerator::new(seed);let mut video=Vec::new();
            let mut condition_draws=0usize;
            // Plain FL2VA may reuse draws precomputed by `assemble-noise` for this exact
            // seed and shape (same generator, same order); anything else draws here.
            let precomputed=if args[1]=="assemble" && extension.is_none() {
                root.parent().map(|p|p.join("assemble-noise")).filter(|d|d.join("manifest.json").is_file()).and_then(|dir|{
                    let m:Value=serde_json::from_slice(&fs::read(dir.join("manifest.json")).ok()?).ok()?;
                    (m["schema"]=="h3-public-assemble-noise-v1" && m["seed"]==json!(seed) && m["shape"]==shape.json()).then_some((dir,m))
                })
            }else{None};
            let mut keyframe_noise=Vec::new();
            if let Some((dir,m))=&precomputed {
                for name in m["keyframes"].as_array().ok_or("noise keyframes")? {keyframe_noise.push(read_f32(&dir.join(name.as_str().ok_or("noise file")?))?);}
                if keyframe_noise.len()!=shape.keyframe_count||keyframe_noise.iter().any(|n|n.len()!=shape.posterior_elements()){return Err("precomputed noise shape mismatch".into());}
            }
            for (index,name) in args[4..4+shape.keyframe_count].iter().enumerate() {
                let condition=read_f32(Path::new(name))?;
                if condition.len()!=shape.posterior_elements() || condition.iter().any(|v|!v.is_finite()){return Err("invalid native keyframe latent".into());}
                let noise=if precomputed.is_some(){std::mem::take(&mut keyframe_noise[index])}else{
                    let (values,draws)=noise::visual_condition(seed,shape.num_latent_frames,shape.keyframe_count,1,
                        shape.latent_height,shape.latent_width)?;condition_draws+=draws;values};
                let t=0.999f32;let sigma=1.0f32-t;
                let noised:Vec<f32>=condition.iter().zip(noise).map(|(&v,n)|t*v+sigma*n).collect();
                video.extend(patchify(&noised,24,1,shape.latent_height,shape.latent_width)?);
            }
            // With a precompute the noise rows are already in its hidden_states file; `video` keeps the keyframe rows.
            if precomputed.is_none() {
                video.extend(patchify(&generator.normal(shape.posterior_elements()*shape.num_latent_frames)?,
                    24,shape.num_latent_frames,shape.latent_height,shape.latent_width)?);
            }
            let audio=if let Some(extension)=extension {
                if shape.keyframe_count!=extension.keyframes{return Err("conditioned workflow keyframe count".into());}
                (extension.audio)(&read_f32(Path::new(&args[5]))?,shape.num_audio_latents)?
            }else if let Some((dir,m))=&precomputed {
                let audio=read_f32(&dir.join("audio.f32"))?;
                if audio.len()!=shape.audio_rows()*32{return Err("precomputed noise shape mismatch".into());}
                generator.draws=m["draws"].as_u64().ok_or("noise draws")? as usize;
                audio
            }else{
                let values=noise::audio(seed,shape.audio_rows()*32)?;
                generator.draws+=condition_draws+values.len();values
            };
            if args[1]=="assemble-long" {
                assemble_long(root,shape,&fs::read(&args[2])?,&tags,&video,&audio,seed,generator.draws,nfe)?
            }else{
            let layout=fl2va_layout_with_delivery(&tags,shape,delivered)?;
            let nt=tags.len();let nv=shape.video_rows();let na=shape.audio_rows();let total=nt+nv+na;
            let inputs=json!({
                "hidden_states":if let Some((dir,m))=&precomputed {
                    in_place_hidden_states(root,&dir.join(m["hidden_states"].as_str().ok_or("precomputed hidden_states")?),&video,
                        m["keyframe_prefix_bytes"].as_u64().ok_or("precomputed keyframe prefix")?,&[1,nv,96])?
                }else{tensor(root,"hidden_states",&f32_bytes(&video),"torch.float32",&[1,nv,96])?},
                "audio_hidden_states":tensor(root,"audio_hidden_states",&f32_bytes(&audio),"torch.float32",&[1,na,32])?,
                "encoder_hidden_states":linked_bf16_tensor(root,"encoder_hidden_states",Path::new(&args[2]),&[1,nt,5120])?,
                "position_ids":tensor(root,"position_ids",&f64_bytes(&layout.position_ids),"torch.float64",&[total,3])?,
                "token_tags":tensor(root,"token_tags",&i64_bytes(&layout.token_tags),"torch.int64",&[total])?,
                "video_indices":tensor(root,"video_indices",&i64_bytes(&layout.video_indices),"torch.int64",&[nv])?,
                "audio_indices":tensor(root,"audio_indices",&i64_bytes(&layout.audio_indices),"torch.int64",&[na])?,
                "text_indices":tensor(root,"text_indices",&i64_bytes(&layout.text_indices),"torch.int64",&[nt])?});
            let result=json!({"schema":"h3-native-request-v1","status":"complete","producer":"h3-native-pipeline",
                "seed":seed,"request_uint32_draws":generator.draws,"inputs":inputs,
                "workflow":workflow,"nfe":nfe,
                "keyframe_noise_aug":0.999,"condition_video_rows":shape.condition_rows(),"condition_audio_rows":if extension.is_some(){na}else{0},
                "shape":shape.json(),
                "component_input_files":args[2..6],"gold_activations_used":false});
            fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&result)?)?;
            result
            }
        },
        _=>return Err("unsupported command".into()),
    };
    finish_report(report)
}
pub fn finish_report(mut report:Value)->Result<Value> {
    report["debug_audit"]=json!(audit::enabled());
    if audit::enabled() {
    let maps=fs::read_to_string("/proc/self/maps")?;
    let lower=maps.to_lowercase();
    if ["libpython","libtorch","libaten","libc10"].iter().any(|name|lower.contains(name)) {
        return Err("forbidden native process dependency".into());
    }
    report["clean_native_dependencies"]=json!(true);
    report["process_maps"]=json!(maps);
    } else {
        report["clean_native_dependencies"]=Value::Null;
        report["process_maps"]=Value::Null;
    }
    Ok(report)
}

#[cfg(all(test,target_os="linux",feature="avx-noise"))]
mod reference_structure_tests {
    use super::*;
    #[test]
    fn prebuilt_reference_layout_matches_real_assembly_and_never_reads_component_values()->Result<()> {
        audit::begin_request()?;
        let root=std::env::temp_dir().join(format!("h3-reference-structure-{}",std::process::id()));
        fs::create_dir(&root)?;
        struct Cleanup(PathBuf);impl Drop for Cleanup{fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}
        let _cleanup=Cleanup(root.clone());
        let shape=Shape::new(32,32,5)?.with_keyframes(0)?;
        let tags=root.join("tags.raw");fs::write(&tags,i64_bytes(&[1,0,0,1]))?;
        let descriptor=root.join("geometry.json");
        let mut geometry=json!({"target":shape.json(),"references":[{"geometry":{"kind":"image",
            "latent_frames":1,"latent_height":2,"latent_width":2},"video_file":"missing-component"}]});
        fs::write(&descriptor,serde_json::to_vec(&geometry)?)?;
        let structure=root.join("structure");let actual=root.join("actual");let conditioning=root.join("conditioning.raw");
        let args=|mode:&str,out:&Path|vec!["pipeline".into(),mode.into(),conditioning.to_string_lossy().into_owned(),
            tags.to_string_lossy().into_owned(),descriptor.to_string_lossy().into_owned(),"0".into(),out.to_string_lossy().into_owned()];
        let prebuilt=run(args("assemble-ref2va-structure",&structure),None)?;
        assert_eq!(prebuilt["structure_only"],true);assert_eq!(prebuilt["request_uint32_draws"],0);
        // The component files do not exist while structure is prepared.
        assert!(!conditioning.exists());
        let visual=root.join("visual.raw");fs::write(&visual,f32_bytes(&vec![0.25;96]))?;
        geometry["references"][0]["video_file"]=json!(visual);
        fs::write(&descriptor,serde_json::to_vec(&geometry)?)?;
        fs::write(&conditioning,vec![0x3fu8;4*5120*2])?;
        let generated=run(args("assemble-ref2va",&actual),None)?;
        for key in ["workflow","nfe","shape","references","condition_video_rows","condition_audio_rows","keyframe_noise_aug"] {
            assert_eq!(prebuilt[key],generated[key],"{key}");
        }
        for name in ["position_ids","token_tags","video_indices","audio_indices","text_indices"] {
            assert_eq!(fs::read(structure.join(format!("{name}.raw")))?,fs::read(actual.join(format!("{name}.raw")))?,"{name}");
        }
        for name in ["hidden_states","audio_hidden_states","encoder_hidden_states"] {
            assert_eq!(prebuilt["inputs"][name]["shape"],generated["inputs"][name]["shape"]);
            assert!(fs::read(structure.join(format!("{name}.raw")))?.iter().all(|&b|b==0));
            assert!(fs::read(actual.join(format!("{name}.raw")))?.iter().any(|&b|b!=0));
        }
        Ok(())
    }
}
