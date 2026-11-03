// Shape propagation follows the actual original decoder's ATen operator semantics.
// No interpolation, per-T coefficients, or exported graph lookup is used.
const GRAPH_LIMIT:usize=8;
const MAX_NODE_ELEMENTS:i64=536_870_912; // FP32 node <= 2 GiB; safe i32 depthwise count.
fn elements(shape:&[i64])->Result<i64>{
    if shape.is_empty()||shape.len()>3||shape.iter().any(|&x|x<=0){return Err(format!("invalid generic audio shape {shape:?}"));}
    let n=shape.iter().try_fold(1i64,|n,&x|n.checked_mul(x).ok_or("audio shape overflow".to_string()))?;
    if n>MAX_NODE_ELEMENTS{return Err("audio node exceeds admitted 2-GiB allocation limit".into());}Ok(n)
}
fn make_shape(spec:&mut Spec,shape:Vec<i64>)->Result<()>{spec.bytes=usize::try_from(elements(&shape)?).map_err(|e|e.to_string())?*4;spec.shape=shape;spec.sha256.clear();Ok(())}
fn shape_of<'a>(arg:&Value,shapes:&'a HashMap<String,Vec<i64>>)->Result<&'a Vec<i64>>{
    shapes.get(tensor_id(arg).ok_or("expected tensor shape argument")?).ok_or("missing preceding tensor shape".into())
}
fn broadcast_shape(a:&[i64],b:&[i64])->Result<Vec<i64>>{
    let n=a.len().max(b.len());let mut out=vec![1;n];
    for i in 0..n{let x=if i+a.len()>=n{a[i+a.len()-n]}else{1};let y=if i+b.len()>=n{b[i+b.len()-n]}else{1};
        if x!=y&&x!=1&&y!=1{return Err("invalid pointwise broadcast".into());}out[i]=x.max(y);}
    Ok(out)
}
fn infer_node(node:&Node,shapes:&HashMap<String,Vec<i64>>)->Result<Vec<i64>>{
    let a=shape_of(node.args.get(0).ok_or("missing first node argument")?,shapes)?;
    let mut out=a.clone();
    match node.op.as_str(){
        "aten.convolution.default"=>{
            let w=shape_of(&node.args[1],shapes)?;if a.len()!=3||w.len()!=3{return Err("audio convolution requires rank three".into());}
            let stride=first(&node.args[3])?;let padding=first(&node.args[4])?;let dilation=first(&node.args[5])?;
            let transposed=node.args[6].as_bool().ok_or("missing transpose flag")?;let output_padding=first(&node.args[7])?;let groups=integer(&node.args[8])?;
            if stride<=0||dilation<=0||groups<=0||padding<0||output_padding<0||output_padding>=stride{return Err("invalid audio convolution attributes".into());}
            let kernel=dilation.checked_mul(w[2]-1).ok_or("convolution overflow")?+1;
            if transposed {
                if a[1]!=w[0]{return Err("transposed convolution channels differ".into());}
                out=vec![a[0],w[1].checked_mul(groups).ok_or("channel overflow")?,(a[2]-1).checked_mul(stride).ok_or("temporal overflow")?-2*padding+kernel+output_padding];
            }else{
                if a[1]!=w[1]*groups||w[0]%groups!=0{return Err("convolution channels differ".into());}
                let numerator=a[2]+2*padding-kernel;if numerator<0{return Err("convolution kernel exceeds padded input".into());}
                out=vec![a[0],w[0],numerator/stride+1];
            }
            if !node.args[2].is_null(){let b=shape_of(&node.args[2],shapes)?;if b!=&[out[1]]{return Err("convolution bias channels differ".into());}}
        },
        "aten.replication_pad1d.default"=>{let p=list(&node.args[1])?;if a.len()!=3||p.len()!=2||p.iter().any(|&x|x<0){return Err("invalid replication padding".into());}out[2]=a[2]+p[0]+p[1];},
        "aten.slice.Tensor"=>{
            let rank=a.len() as i64;let dim=integer(&node.args[1])?;let dim=if dim<0{dim+rank}else{dim};if dim<0||dim>=rank{return Err("slice dimension".into());}
            let d=dim as usize;let length=a[d];let norm=|v:i64|if v<0{(v+length).max(0)}else{v.min(length)};
            let start=norm(integer(&node.args[2])?);let end=norm(integer(&node.args[3])?);let step=if node.args.len()>4{integer(&node.args[4])?}else{1};
            if step<=0||end<=start{return Err("unsupported empty/negative-step audio slice".into());}out[d]=(end-start-1)/step+1;
        },
        "aten.permute.default"=>{let order=list(&node.args[1])?;if order.len()!=a.len(){return Err("permutation rank".into());}let mut seen=std::collections::HashSet::new();out.clear();for d in order{let d=if d<0{d+a.len() as i64}else{d};if d<0||d>=a.len() as i64||!seen.insert(d){return Err("invalid permutation".into());}out.push(a[d as usize]);}},
        "aten.view.default"|"aten.reshape.default"=>{
            out=list(&node.args[1])?;let unknown:Vec<_>=out.iter().enumerate().filter(|(_,x)|**x==-1).map(|(i,_)|i).collect();if unknown.len()>1{return Err("multiple inferred reshape dimensions".into());}
            let n=elements(a)?;if let Some(&d)=unknown.first(){out[d]=1;let fixed=elements(&out)?;if n%fixed!=0{return Err("reshape not divisible".into());}out[d]=n/fixed;}if elements(&out)?!=n{return Err("reshape changes element count".into());}
        },
        "aten.unsqueeze.default"=>{let d=integer(&node.args[1])?;let d=if d<0{d+a.len() as i64+1}else{d};if d<0||d>a.len() as i64{return Err("unsqueeze dimension".into());}out.insert(d as usize,1);},
        "aten.squeeze.dim"=>{let d=integer(&node.args[1])?;let d=if d<0{d+a.len() as i64}else{d};if d<0||d>=a.len() as i64{return Err("squeeze dimension".into());}if out[d as usize]==1{out.remove(d as usize);}},
        "aten.mul.Tensor"|"aten.add.Tensor"|"aten.div.Tensor"|"aten.mul.Scalar"|"aten.add.Scalar"|"aten.div.Scalar"=>{if tensor_id(&node.args[1]).is_some(){out=broadcast_shape(a,shape_of(&node.args[1],shapes)?)?;}},
        "aten.sin.default"|"aten.pow.Tensor_Scalar"|"aten.clamp.default"|"aten.clone.default"|"aten._to_copy.default"|"aten.detach.default"|"aten.alias.default"=>{},
        op=>return Err(format!("unsupported semantic shape operator {op}")),
    }
    elements(&out)?;Ok(out)
}
fn infer_graph(base:&Manifest,tokens:i64)->Result<Manifest>{
    if tokens<=0||tokens>MAX_NODE_ELEMENTS/800/2{return Err("invalid or overflowing audio input length".into());}
    let mut manifest=base.clone();make_shape(&mut manifest.input,vec![2,32,tokens])?;
    let mut shapes:HashMap<String,Vec<i64>>=base.constants.iter().map(|(n,s)|(n.clone(),s.shape.clone())).collect();shapes.insert("input".into(),manifest.input.shape.clone());
    for (i,node) in manifest.nodes.iter_mut().enumerate(){
        if node.id!=format!("n{i:04}")||node.output.dtype!="torch.float32"{return Err("noncanonical audio architecture".into());}
        let output=infer_node(node,&shapes).map_err(|e|format!("{} {}: {e}",node.id,node.op))?;
        make_shape(&mut node.output,output.clone())?;shapes.insert(node.id.clone(),output);
    }
    if shapes.get(&manifest.output_id)!=Some(&vec![1,2,tokens*800]){return Err("decoder architecture does not produce stereo 800x waveform".into());}
    make_shape(&mut manifest.gold,vec![2,tokens*800])?;Ok(manifest)
}
fn read_template(root:&Path)->Result<(PathBuf,Manifest,String)>{
    let root=root.canonicalize().map_err(|e|e.to_string())?;let bytes=crate::startup_trace::read(root.join("manifest.json")).map_err(|e|e.to_string())?;
    let manifest:Manifest=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    if manifest.schema!="h3-audio-vae-graph-v1"||manifest.status!="complete"||manifest.input.dtype!="torch.float32"||manifest.input.shape.len()!=3||manifest.input.shape[..2]!=[2,32]||manifest.nodes.len()!=1621{return Err("unsupported audio architecture template".into());}
    for (name,spec) in &manifest.constants{if spec.file!=format!("constants/{name}.raw")||name.contains('/')||name.contains("..")||spec.dtype!="torch.float32"||elements(&spec.shape)? as usize*4!=spec.bytes{return Err("invalid static audio constant".into());}}
    let regenerated=infer_graph(&manifest,manifest.input.shape[2])?;
    for (expected,actual) in manifest.nodes.iter().zip(&regenerated.nodes){if expected.output.shape!=actual.output.shape||expected.output.bytes!=actual.output.bytes{return Err(format!("architecture semantic self-check failed {}",expected.id));}}
    Ok((root,manifest,digest(&bytes)))
}
pub fn describe_generic_graph(root:&Path,tokens:i64)->Result<Value>{
    let(_,template,manifest_sha)=read_template(root)?;let graph=infer_graph(&template,tokens)?;
    Ok(json!({"schema":"h3-generic-audio-shape-v1","template_manifest_sha256":manifest_sha,"input":graph.input,"gold":graph.gold,"output_id":graph.output_id,"nodes":graph.nodes,"constants":graph.constants,"semantics":"operator formulas; no per-length manifests or interpolation"}))
}
