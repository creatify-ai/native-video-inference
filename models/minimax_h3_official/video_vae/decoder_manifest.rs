// Native decoder layout from the model architecture, not a shape catalog.
// Pointwise dispatch receives the actual T/H/W separately; its pointer table
// continues to address the original maximum-tile allocations.
fn decoder_contiguous(shape:&[usize])->Vec<usize>{
    let mut stride=vec![1;shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev(){stride[i]=stride[i+1]*shape[i+1];}stride
}
fn decoder_view(spec:&mut Value,shape:Vec<usize>,stride:Vec<usize>)->Result<()>{
    if !spec.is_object() || spec["offset"]!=0{return Err("generic decoder requires the original zero-offset tensor views".into());}
    spec["shape"]=json!(shape);spec["stride"]=json!(stride);Ok(())
}
fn resize_decoder_manifest(base:&Value,pixel_h:usize,pixel_w:usize)->Result<Value>{
    if pixel_h<32 || pixel_w<32 || pixel_h>256 || pixel_w>256 || pixel_h%32!=0 || pixel_w%32!=0 {
        return Err("generic decoder tile requires 32..256 pixel axes on the 32-pixel lattice".into());
    }
    if pixel_h==256 && pixel_w==256{return Ok(base.clone());}
    if base["schema"]!="h3_original_inductor_v1" || base["buffers"][0]["shape"]!=json!([1,24,7,16,16]){
        return Err("generic decoder architecture template differs".into());
    }
    let (h,w)=(pixel_h/16,pixel_w/16);let patches=7*h*w;let rows=patches+5;
    let mut plan=base.clone();let mut counts=[0usize;4];
    for op in plan["ops"].as_array_mut().ok_or("decoder ops")?{
        // Old comparison activations are never validation targets for new shapes.
        op.as_object_mut().ok_or("decoder op object")?.remove("checks");
        match op["kind"].as_str().ok_or("decoder op kind")? {
            "kernel"=>{counts[0]+=1;},
            "mm"|"addmm"=>{
                let biased=op["kind"]=="addmm";let first=usize::from(biased);
                let row_count=if biased{rows}else{patches};
                let args=op["args"].as_array_mut().ok_or("linear arguments")?;
                let xshape=ints(&args[first]["shape"])?;let wshape=ints(&args[first+1]["shape"])?;
                if xshape.len()!=2 || wshape.len()!=2 || xshape[1]!=wshape[0] ||
                   xshape[0]!=(if biased{1797}else{1792}) {return Err("decoder linear template contract changed".into());}
                let inner=xshape[1] as usize;let columns=wshape[1] as usize;
                let stride=if biased{vec![inner,1]}else{vec![1,row_count]};
                decoder_view(&mut args[first],vec![row_count,inner],stride)?;
                decoder_view(&mut op["kwargs"]["out"],vec![row_count,columns],vec![columns,1])?;
                counts[if biased{1}else{2}]+=1;
            },
            "cudnn_sdpa"=>{
                let args=op["args"].as_array_mut().ok_or("attention arguments")?;
                for tensor in &mut args[..3]{
                    if tensor["shape"]!=json!([1,32,1797,64]) || tensor["stride"]!=json!([3680256,64,2048,1]){
                        return Err("decoder QKV template layout changed".into());
                    }
                    decoder_view(tensor,vec![1,32,rows,64],vec![rows*2048,64,2048,1])?;
                }
                let out=&mut op["outputs"][0];
                if out["shape"]!=json!([1,32,1797,64]) || out["stride"]!=json!([3680256,64,2048,1]){
                    return Err("decoder attention output layout changed".into());
                }
                decoder_view(out,vec![1,32,rows,64],vec![rows*2048,64,2048,1])?;
                counts[3]+=1;
            },
            _=>return Err("unrecognized decoder operator".into()),
        }
    }
    if counts!=[690,217,1,36]{return Err("decoder architecture operator count changed".into());}
    let input=vec![1,24,7,h,w];
    plan["buffers"][0]["shape"]=json!(input);plan["buffers"][0]["stride"]=json!(decoder_contiguous(&input));
    // bytes remains the allocation capacity; only the active logical view shrinks.
    let output=vec![1,3,28,pixel_h,pixel_w];
    decoder_view(&mut plan["outputs"][0],output.clone(),decoder_contiguous(&output))?;
    plan["gold"]["shape"]=json!(output);plan["gold"]["bytes"]=json!(3*28*pixel_h*pixel_w*2);
    plan["gold"]["stride"]=json!(decoder_contiguous(&output));
    plan["gold"]["file"]=Value::Null;plan["gold"]["sha256"]=Value::Null;
    Ok(plan)
}
