// Included by the native pipeline CLI. Initial noise is sampled once, in the
// original keyframes -> full video -> full stereo audio order before slicing.
fn long_shape_args(args:&[String],start:usize)->Result<Shape> {
    if args.len()!=start+4{return Err("long geometry requires WIDTH HEIGHT FRAMES KEYFRAMES".into());}
    Shape::long(args[start].parse()?,args[start+1].parse()?,args[start+2].parse()?,args[start+3].parse()?)
}

fn assemble_long(root:&Path,shape:Shape,conditioning:&[u8],tags:&[i64],video:&[f32],audio:&[f32],
                 seed:u64,draws:usize,nfe:usize)->Result<Value> {
    let plan=long_video::Plan::new(shape.num_frames,shape.num_frames as f64/24.0,true)?;
    let rows=shape.rows_per_frame();let first=shape.keyframe_count>0;let last=shape.keyframe_count==2;
    if conditioning.len()!=tags.len()*5120*2 || video.len()!=shape.video_rows()*96 || audio.len()!=shape.audio_rows()*32 {
        return Err("long-video initial streams differ from geometry".into());
    }
    let mut windows=Vec::new();
    for (index,window) in plan.windows.iter().enumerate() {
        let anchors=window.anchors(first,last,plan.windows.len()>1);
        let count=usize::from(!anchors["first"].is_null())+usize::from(!anchors["last"].is_null());
        let local=Shape::new(shape.width,shape.height,window.pixel_frames)?.with_keyframes(count)?;
        // Under FL's accepted contracts every window retains the same real
        // input keyframes in text_for(): carried anchors add no new image.
        let layout=fl2va_layout(tags,local)?;
        crate::limits::packed_rows(&[local.video_rows(),local.audio_rows(),tags.len()])?;
        let mut v=Vec::with_capacity(local.video_rows()*96);
        if !anchors["first"].is_null(){v.extend_from_slice(&video[..rows*96]);}
        if !anchors["last"].is_null(){
            let offset=if anchors["last"]=="real"{(shape.keyframe_count-1)*rows*96}else{0};
            v.extend_from_slice(&video[offset..offset+rows*96]);
        }
        let start=(shape.condition_rows()+window.latent_frames[0]*rows)*96;
        let end=(shape.condition_rows()+window.latent_frames[1]*rows)*96;
        v.extend_from_slice(&video[start..end]);
        let mut a=Vec::with_capacity(local.audio_rows()*32);
        for channel in 0..2 {
            let start=(channel*shape.num_audio_latents+window.audio_latents[0])*32;
            let end=(channel*shape.num_audio_latents+window.audio_latents[1])*32;
            a.extend_from_slice(&audio[start..end]);
        }
        let directory=format!("windows/{index:03}");let output=root.join(&directory);fs::create_dir_all(&output)?;
        let(nv,na,nt)=(v.len()/96,a.len()/32,tags.len());let total=nv+na+nt;
        let inputs=json!({
            "hidden_states":tensor(&output,"hidden_states",&f32_bytes(&v),"torch.float32",&[1,nv,96])?,
            "audio_hidden_states":tensor(&output,"audio_hidden_states",&f32_bytes(&a),"torch.float32",&[1,na,32])?,
            "encoder_hidden_states":tensor(&output,"encoder_hidden_states",conditioning,"torch.bfloat16",&[1,nt,5120])?,
            "position_ids":tensor(&output,"position_ids",&f64_bytes(&layout.position_ids),"torch.float64",&[total,3])?,
            "token_tags":tensor(&output,"token_tags",&i64_bytes(&layout.token_tags),"torch.int64",&[total])?,
            "video_indices":tensor(&output,"video_indices",&i64_bytes(&layout.video_indices),"torch.int64",&[nv])?,
            "audio_indices":tensor(&output,"audio_indices",&i64_bytes(&layout.audio_indices),"torch.int64",&[na])?,
            "text_indices":tensor(&output,"text_indices",&i64_bytes(&layout.text_indices),"torch.int64",&[nt])?});
        let manifest=json!({"schema":"h3-native-request-v1","status":"complete","producer":"h3-native-pipeline",
            "workflow":"fl2va","nfe":nfe,"seed":seed,"inputs":inputs,"shape":local.json(),"keyframe_noise_aug":0.999,
            "condition_video_rows":local.condition_rows(),"condition_audio_rows":0,"gold_activations_used":false});
        fs::write(output.join("manifest.json"),serde_json::to_vec_pretty(&manifest)?)?;
        let next=plan.windows.get(index+1);
        windows.push(json!({"request":directory,"geometry":window,"anchors":anchors,
            "video_weights":long_video::fuse_weights(local.num_latent_frames,window.overlap_latent_frames,
                next.map_or(0,|w|w.overlap_latent_frames))?,
            "audio_weights":long_video::fuse_weights(local.num_audio_latents,window.overlap_audio_latents,
                next.map_or(0,|w|w.overlap_audio_latents))?}));
    }
    let result=json!({"schema":"h3-native-long-request-v1","status":"complete","producer":"h3-native-pipeline",
        "workflow":"fl2va","nfe":nfe,"gold_activations_used":false,"shape":shape.json(),"plan":plan,
        "seed":seed,"request_uint32_draws":draws,"keyframe_noise_aug":0.999,"windows":windows,
        "inputs":{"hidden_states":tensor(root,"hidden_states",&f32_bytes(video),"torch.float32",&[1,shape.video_rows(),96])?,
                  "audio_hidden_states":tensor(root,"audio_hidden_states",&f32_bytes(audio),"torch.float32",&[1,shape.audio_rows(),32])?}});
    fs::write(root.join("manifest.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
