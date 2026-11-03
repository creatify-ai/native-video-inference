// Eager encoder shapes are propagated from the actual input through each op.
// The original maximum-tile allocations and all parameter views stay resident.
fn encoder_set_view(spec:&mut Value,shape:&[usize])->Result<()>{
    if spec["offset"]!=0 || spec["dtype"]!="torch.float32"{return Err("encoder tensor contract changed".into());}
    spec["shape"]=json!(shape);spec["stride"]=json!(decoder_contiguous(shape));Ok(())
}
fn encoder_arg_view(value:&mut Value,shapes:&HashMap<usize,Vec<usize>>)->Result<()>{
    match value {
        Value::Object(map) if map.contains_key("buffer")=>{
            let id=num(&value["buffer"])?;
            let shape=shapes.get(&id).ok_or("encoder input view has no producer")?;
            encoder_set_view(value,shape)?;
        },
        Value::Object(map)=>for v in map.values_mut(){encoder_arg_view(v,shapes)?;},
        Value::Array(values)=>for v in values{encoder_arg_view(v,shapes)?;},
        _=>{},
    }Ok(())
}
fn resize_encoder_manifest(base:&Value,pixel_h:usize,pixel_w:usize)->Result<Value>{
    if pixel_h<32 || pixel_w<32 || pixel_h>256 || pixel_w>256 || pixel_h%32!=0 || pixel_w%32!=0 {
        return Err("generic encoder tile requires 32..256 pixel axes on the 32-pixel lattice".into());
    }
    if pixel_h==256 && pixel_w==256{return Ok(base.clone());}
    if base["schema"]!="h3_original_eager_v1" || base["stage"]!="video_vae_encoder_tile"{
        return Err("generic encoder architecture template differs".into());
    }
    let mut plan=base.clone();let mut shapes=HashMap::new();let mut inputs=0;
    let capacities:Vec<usize>=arr(&base["buffers"])?.iter().map(|v|num(&v["bytes"])).collect::<Result<_>>()?;
    for buffer in plan["buffers"].as_array_mut().ok_or("encoder buffers")?{
        let id=num(&buffer["id"])?;
        match buffer["role"].as_str(){
            Some("parameter")=>{shapes.insert(id,ints(&buffer["shape"])?.into_iter().map(|v|v as usize).collect());},
            Some("input")=>{
                if buffer["shape"]!=json!([1,3,1,256,256]){return Err("encoder input template differs".into());}
                let dims=vec![1,3,1,pixel_h,pixel_w];buffer["shape"]=json!(dims);buffer["stride"]=json!(decoder_contiguous(&dims));
                shapes.insert(id,dims);inputs+=1;
            },Some("scratch")=>{},_=>return Err("unknown encoder buffer role".into()),
        }
    }
    if inputs!=1{return Err("encoder requires one runtime input".into());}
    let mut counts=[0usize;6];
    for (index,op) in plan["ops"].as_array_mut().ok_or("encoder ops")?.iter_mut().enumerate(){
        op.as_object_mut().ok_or("encoder op object")?.remove("checks");
        encoder_arg_view(&mut op["args"],&shapes)?;
        let args=arr(&op["args"])?;let x:Vec<usize>=ints(&args[0]["shape"])?.into_iter().map(|v|v as usize).collect();
        if x.len()!=5{return Err(format!("encoder op {index} input rank").into());}
        let kind=text(&op["kind"])?;
        let output:Vec<Vec<usize>>=match kind{
            "aten.reflection_pad3d.default"|"aten.constant_pad_nd.default"=>{
                let pad=ints(&args[1])?;if pad.len()!=6 || pad.iter().any(|&p|p<0){return Err("encoder padding contract".into());}
                let reflect=kind=="aten.reflection_pad3d.default";let mut y=x.clone();
                for d in 2..5{let p=(4-d)*2;
                    if reflect && (pad[p] as usize>=x[d] || pad[p+1] as usize>=x[d]){return Err("reflection padding exceeds the actual input axis".into());}
                    y[d]+=pad[p] as usize+pad[p+1] as usize;
                }counts[usize::from(!reflect)]+=1;vec![y]
            },
            "aten.convolution.default"=>{
                if args[6]!=false || args[8]!=1{return Err("encoder convolution architecture changed".into());}
                let weight=ints(&args[1]["shape"])?;let stride=ints(&args[3])?;let pad=ints(&args[4])?;let dil=ints(&args[5])?;
                if weight.len()!=5 || weight[1] as usize!=x[1] || stride.len()!=3 || pad.len()!=3 || dil.len()!=3{return Err("encoder convolution dimensions".into());}
                let mut y=vec![x[0],weight[0] as usize,0,0,0];
                for d in 0..3{
                    let extent=x[d+2] as i64+2*pad[d]-dil[d]*(weight[d+2]-1)-1;
                    if extent<0 || stride[d]<=0{return Err("encoder convolution has empty output".into());}
                    y[d+2]=(extent/stride[d]+1) as usize;
                }counts[2]+=1;vec![y]
            },
            "aten.native_group_norm.default"=>{
                let groups=num(&args[6])?;if groups==0 || x[1]%groups!=0{return Err("encoder group count".into());}
                counts[3]+=1;vec![x.clone(),vec![x[0],groups],vec![x[0],groups]]
            },
            "aten.silu.default"=>{counts[4]+=1;vec![x.clone()]},
            "aten.add.Tensor"=>{
                if args[0]["shape"]!=args[1]["shape"]{return Err("encoder residual shape mismatch".into());}
                counts[5]+=1;vec![x.clone()]
            },_=>return Err(format!("unsupported generic encoder op {kind}").into()),
        };
        if kind=="aten.native_group_norm.default"{
            op["args"][3]=json!(x[0]);op["args"][4]=json!(x[1]);op["args"][5]=json!(x[2..].iter().product::<usize>());
        }
        let specs=op["outputs"].as_array_mut().ok_or("encoder outputs")?;
        if specs.len()!=output.len(){return Err("encoder output count".into());}
        for (spec,dims) in specs.iter_mut().zip(output){
            let id=num(&spec["buffer"])?;
            if dims.iter().product::<usize>()*4>*capacities.get(id).ok_or("encoder allocation id")?{return Err("encoder shape exceeds maximum allocation".into());}
            encoder_set_view(spec,&dims)?;shapes.insert(id,dims);
        }
    }
    if counts!=[30,30,34,25,25,12]{return Err("encoder architecture operator counts changed".into());}
    encoder_arg_view(&mut plan["outputs"],&shapes)?;
    let expected=vec![1,48,1,pixel_h/16,pixel_w/16];
    if plan["outputs"][0]["shape"]!=json!(expected){return Err("encoder output architecture changed".into());}
    plan["gold"]["shape"]=json!(expected);plan["gold"]["stride"]=json!(decoder_contiguous(&expected));
    plan["gold"]["bytes"]=json!(expected.iter().product::<usize>()*4);
    plan["gold"]["file"]=Value::Null;plan["gold"]["sha256"]=Value::Null;
    Ok(plan)
}
