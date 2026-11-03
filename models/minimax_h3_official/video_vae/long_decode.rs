// Long decode: 20 groups with one neighbouring context group,
// temporal slicing before VAE, then discard context pixels before streaming.
fn decode_long_body(args:&[String],ctx:&mut Context,started:Instant,model_load_seconds:f64)->Result<Value>{
    if args.len()!=8{return Err("decode-long MODEL LATENTS OUTPUT WIDTH HEIGHT FRAMES".into());}
    let(w,h,frames):(usize,usize,usize)=(args[5].parse()?,args[6].parse()?,args[7].parse()?);
    Spatial::new(w,h)?;
    let ceiling=std::env::var("H3_FL2VA_LONG_MAX_S").unwrap_or_else(|_|"60".into()).parse::<f64>()?;
    if !ceiling.is_finite() || frames<124 || frames%17!=5 || frames as f64/24.0>ceiling{
        return Err("long decode frame count outside the supported duration".into());
    }
    let groups=(frames-5)/17;
    if groups<24{let mut normal=args.to_vec();normal[1]="decode".into();return decode_slice_body(&normal,ctx,started,model_load_seconds,None);}
    let request=ctx.request.clone().ok_or("long decode requires a streaming request context")?;
    let out=Path::new(&args[4]);fs::create_dir_all(out)?;
    let latent_frames=groups.checked_mul(5).and_then(|n|n.checked_add(2)).ok_or("long latent overflow")?;
    let plane=(w/16).checked_mul(h/16).and_then(|n|n.checked_mul(4)).ok_or("long spatial overflow")?;
    let expected=24usize.checked_mul(latent_frames).and_then(|n|n.checked_mul(plane)).ok_or("long input overflow")?;
    let bytes=crate::startup_trace::read(&args[3])?;
    if bytes.len()!=expected || !finite_f32(&bytes){return Err("long decoder input shape/nonfinite mismatch".into());}
    let mut stream=FrameStream::new(&request,w,h,frames,out)?;
    let mut reports=Vec::new();let mut written=0;
    for first in (0..groups).step_by(20){
        let last=(first+20).min(groups);let lo=first.saturating_sub(1);let hi=(last+1).min(groups);
        let from=5*lo;let until=5*hi+2;let local_frames=17*(hi-lo)+5;
        let head=17*(first-lo);let kept=17*(last-first)+if last==groups{5}else{0};
        let directory=out.join(format!("slice-{first:03}"));fs::create_dir(&directory)?;
        let file=directory.join("latents.raw");let mut input=Vec::with_capacity(24*(until-from)*plane);
        for channel in 0..24{input.extend_from_slice(&bytes[(channel*latent_frames+from)*plane..(channel*latent_frames+until)*plane]);}
        fs::write(&file,&input)?;drop(input);
        let mut local=args.to_vec();local[1]="decode".into();local[3]=file.to_string_lossy().into_owned();
        local[4]=directory.to_string_lossy().into_owned();local[7]=local_frames.to_string();
        let allocation_mark=ctx.program.arena.allocation_mark();let t=Instant::now();
        let result=decode_slice_body(&local,ctx,t,0.0,Some((&mut stream,head,head+kept,written)));
        let cleanup=ctx.program.arena.release_after(allocation_mark);cleanup?;
        let report=result?;fs::remove_file(&file)?;
        let event=json!({"event":"native.long_video.decode_slice_completed","groups":[first,last],
            "context_groups":[lo,hi],"kept_frames":kept,"first_frame":written,"wall_seconds":t.elapsed().as_secs_f64()});
        eprintln!("{event}");reports.push(json!({"geometry":event,"decode":report}));written+=kept;
    }
    if written!=frames{return Err("long decode omitted or repeated output frames".into());}
    let stream_report=stream.finish()?;
    Ok(json!({"passed":true,"mode":"decode-long","shape":[frames,h,w,3],"slices":reports,
        "streaming":stream_report,"load_seconds":model_load_seconds,
        "wall_seconds":started.elapsed().as_secs_f64(),"no_reference_activations_read":true}))
}
