// Eager encoder-specific operations. Decoder shape admission remains unchanged.
extern "C" {
 fn h3_audio_copy5(x:*const c_void,y:DevicePtr,rank:i32,shape:*const i64,stride:*const i64,offset:i64,stream:DevicePtr)->i32;
 fn h3_audio_layernorm(x:*const c_void,w:*const c_void,b:*const c_void,y:DevicePtr,rows:i32,cols:i32,eps:f32,stream:DevicePtr)->i32;
 fn h3_audio_encoder_op(op:i32,x:*const c_void,y:DevicePtr,count:i64,a:i32,b:i32,c:i32,stream:DevicePtr)->i32;
 fn h3_audio_encoder_attention(q:*const c_void,k:*const c_void,v:*const c_void,y:DevicePtr,batch:i32,heads:i32,tokens:i32,dim:i32,stream:DevicePtr)->i32;
 fn h3_audio_encoder_linear(x:*const c_void,w:*const c_void,b:*const c_void,y:DevicePtr,rows:i32,cols:i32,inner:i32,tf32:i32,stream:DevicePtr)->i32;
}
impl Decoder {
 pub fn load_encoder(root:impl AsRef<Path>)->Result<Self>{
  let root=root.as_ref().canonicalize().map_err(|e|e.to_string())?;
  let bytes=crate::startup_trace::read(root.join("manifest.json")).map_err(|e|e.to_string())?;
  let template:Manifest=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
  if template.schema!="h3-audio-encoder-graph-v1"||template.status!="complete"||template.provenance["no_intermediate_initializers"]!=true||template.input.shape.len()!=3||template.input.shape[..2]!=[2,1]||template.input.shape[2]<1||template.input.shape[2]>32_000*30{return Err("invalid audio encoder manifest".into());}
  let mut constants=HashMap::new();let mut shared_weights=HashMap::new();
  let valid=|s:&Spec|->bool{s.dtype=="torch.float32"&& !s.shape.is_empty()&&s.shape.len()<=5&&s.shape.iter().all(|x|*x>0)&&s.shape.iter().try_fold(1usize,|n,x|n.checked_mul(*x as usize)).and_then(|n|n.checked_mul(4))==Some(s.bytes)&&s.bytes<=2_147_483_648};
  let mut ordered:Vec<_>=template.constants.iter().collect();ordered.sort_by_key(|(name,_)|*name);
  for(name,spec)in &ordered {
   if name.contains('/')||name.contains("..")||spec.file!=format!("constants/{name}.raw")||!valid(spec){return Err("invalid encoder constant".into());}
  }
  if let Some(config)=crate::weight_loader::Config::from_env()? {
   // Start bounded reads before the first CUDA allocation. Reader leases stay
   // pinned and alive until their upload events complete, including errors.
   let jobs=ordered.iter().map(|(_,s)|crate::weight_loader::Job::new(&root,&s.file,s.bytes,Some(s.sha256.clone()))).collect::<Result<Vec<_>>>()?;
   let mut pool=if crate::weight_loader::flag("H3_MODEL_LOAD_ENCODER_READ_AHEAD")? {
    crate::weight_loader::ReaderPool::start_read_ahead(jobs,config)?
   }else{crate::weight_loader::ReaderPool::start(jobs,config)?};
   let tensors=ordered.iter().map(|(_,s)|Tensor::empty(&s.shape)).collect::<Result<Vec<_>>>()?;
   let mut upload=crate::weight_loader::device::Upload::new(ptr::null_mut(),config)?;
   unsafe{pool.consume_upload(&mut upload,|i,offset|tensors[i].pointer().cast::<u8>().add(offset).cast())}?;
   upload.finish()?;
   if crate::weight_loader::flag("H3_MODEL_LOAD_VERIFY_GPU")? {
    for (tensor,(_,s)) in tensors.iter().zip(&ordered){unsafe{upload.verify(tensor.pointer(),s.bytes,&s.sha256)}?;}
   }
   for ((name,spec),tensor) in ordered.into_iter().zip(tensors) {
    constants.insert(name.clone(),tensor.clone());shared_weights.insert(constant_key(spec),tensor);
   }
   eprintln!("audio encoder streaming load: {} tensors, config={config:?}",constants.len());
  }else{
   for(name,spec)in ordered {
    let data=crate::startup_trace::read(root.join(&spec.file)).map_err(|e|e.to_string())?;
    if data.len()!=spec.bytes||!weight_digest_matches(&data,&spec.sha256)?{return Err("encoder constant integrity".into());}
    let tensor=Tensor::from_bytes(&spec.shape,&data)?;constants.insert(name.clone(),tensor.clone());shared_weights.insert(constant_key(spec),tensor);
   }
  }
  let mut known:std::collections::HashSet<String>=constants.keys().cloned().collect();known.insert("input".into());
  let mut uses=HashMap::new();
  for node in &template.nodes {
   if !valid(&node.output)||!node.id.starts_with('n')||known.contains(&node.id){return Err("invalid encoder node".into());}
   for arg in &node.args{if let Some(id)=tensor_id(arg){if !known.contains(id){return Err(format!("missing encoder dependency {id}"));}*uses.entry(id.to_string()).or_insert(0)+=1;}}
   known.insert(node.id.clone());
  }
  if !known.contains(&template.output_id)||!valid(&template.input)||!valid(&template.gold){return Err("invalid encoder input/output".into());}
  *uses.entry(template.output_id.clone()).or_insert(0)+=1;
  let mut context=ptr::null_mut();unsafe{conv(h3_conv_create(&mut context))?;}
  let mut value=Self{root,manifest:template.clone(),template,template_sha:digest(&bytes),constants,context,uses,plans:HashMap::new(),allocation_pool:allocation_pool::Pool::default(),allocation_pool_enabled:false,
   shared_weights,shape_graphs:HashMap::new(),graph_prepare_count:0,graph_switch_count:0,graph_evictions:0,graph_clock:0,last_graph_prepare_seconds:0.,last_pool_released_bytes:0,last_graph_cache_hit:false,weight_identity:String::new()};
  value.weight_identity=value.current_weight_identity();Ok(value)
 }
 unsafe fn execute_encoder(&mut self,node:&Node,tensors:&HashMap<String,Tensor>,out:&Tensor,stream:DevicePtr)->Option<Result<()>>{
  if self.manifest.schema!="h3-audio-encoder-graph-v1"{return None;}
  let supported=matches!(node.op.as_str(),"aten.transpose.int"|"aten.permute.default"|"aten.unbind.int"|"aten.view.default"|"aten.unsqueeze.default"|"aten.squeeze.dim"|"aten.native_layer_norm.default"|"aten.addmm.default"|"aten._scaled_dot_product_efficient_attention.default"|"aten.mean.dim"|"aten._adaptive_avg_pool2d.default"|"aten.gelu.default"|"aten.constant_pad_nd.default");
  if !supported{return None;}
  Some((||->Result<()>{
   let get=|i:usize|->Result<&Tensor>{let id=tensor_id(node.args.get(i).ok_or("missing encoder argument")?).ok_or("expected encoder tensor")?;tensors.get(id).ok_or("missing encoder tensor".into())};
   let a=get(0)?;let n=out.bytes() as i64/4;
   let copy=|x:&Tensor,shape:&[i64],stride:&[i64],offset:i64,y:&Tensor|cuda(h3_audio_copy5(x.pointer(),y.pointer(),shape.len() as i32,shape.as_ptr(),stride.as_ptr(),offset,stream));
   match node.op.as_str(){
    "aten.transpose.int"|"aten.permute.default"|"aten.unbind.int"=>{
     let mut ss=strides(&a.shape);let mut offset=0;
     if node.op=="aten.transpose.int"{let d=integer(&node.args[1])? as usize;let e=integer(&node.args[2])? as usize;ss.swap(d,e);}
     else if node.op=="aten.permute.default"{ss=list(&node.args[1])?.iter().map(|d|ss[*d as usize]).collect();}
     else {
      // Dispatcher omits the default dim=0; exporter appends output index.
      let (d,index)=if node.args.len()==2{(0,integer(&node.args[1])?)}else{(integer(&node.args[1])? as usize,integer(&node.args[2])?)};
      if d>=ss.len()||index<0||index>=a.shape[d]{return Err("invalid encoder unbind".into());}
      offset=index*ss[d];ss.remove(d);
     }
     copy(a,&out.shape,&ss,offset,out)
    },
    "aten.view.default"|"aten.unsqueeze.default"|"aten.squeeze.dim"=>{
     if a.bytes()!=out.bytes(){return Err("encoder view changes byte count".into());}
     cuda(cudaMemcpyAsync(out.pointer(),a.pointer(),out.bytes(),3,stream))
    },
    "aten.native_layer_norm.default"=>{
     let w=get(2)?;let b=get(3)?;let cols=*a.shape.last().ok_or("norm dimensions")?;
     cuda(h3_audio_layernorm(a.pointer(),w.pointer(),b.pointer(),out.pointer(),(n/cols) as i32,cols as i32,node.args[4].as_f64().ok_or("norm eps")? as f32,stream))
    },
    "aten.addmm.default"=>{
     let x=get(1)?;let w=get(2)?;
     if x.shape.len()!=2||w.shape.len()!=2||x.shape[1]!=w.shape[0]||a.shape!=[w.shape[1]]{return Err("unsupported encoder addmm".into());}
     let packed=Tensor::empty(&[w.shape[1],w.shape[0]])?;
     copy(w,&packed.shape,&[1,w.shape[1]],0,&packed)?;
     let code=h3_audio_encoder_linear(x.pointer(),packed.pointer(),a.pointer(),out.pointer(),x.shape[0] as i32,w.shape[1] as i32,w.shape[0] as i32,self.manifest.provenance["allow_tf32"].as_bool().unwrap_or(false) as i32,stream);
     if code==0{Ok(())}else{Err(format!("audio linear status {code}"))}
    },
    "aten._scaled_dot_product_efficient_attention.default"=>{
     let k=get(1)?;let v=get(2)?;
     if a.shape.len()!=4||a.shape!=k.shape||a.shape!=v.shape||!node.args[3].is_null()||node.args[5].as_f64().unwrap_or(0.)!=0.||node.args[6]!=true{return Err("unsupported audio attention contract".into());}
     let s=&a.shape;let tmp=Tensor::empty(&[s[0],s[2],s[1],s[3]])?;
     cuda(h3_audio_encoder_attention(a.pointer(),k.pointer(),v.pointer(),tmp.pointer(),s[0] as i32,s[1] as i32,s[2] as i32,s[3] as i32,stream))?;
     copy(&tmp,&out.shape,&[s[2]*s[1]*s[3],s[3],s[1]*s[3],1],0,out)
    },
    "aten.mean.dim"=>{
     if a.shape.len()!=4||list(&node.args[1])?!=[2]||node.args.get(2).is_some_and(|v|v==true){return Err("unsupported head mean".into());}
     cuda(h3_audio_encoder_op(2,a.pointer(),out.pointer(),n,a.shape[2] as i32,a.shape[3] as i32,0,stream))
    },
    "aten._adaptive_avg_pool2d.default"=>{
     if a.shape.len()!=4||a.shape[2]!=1||out.shape[2]!=1{return Err("unsupported audio pool".into());}
     cuda(h3_audio_encoder_op(3,a.pointer(),out.pointer(),n,a.shape[3] as i32,out.shape[3] as i32,0,stream))
    },
    "aten.gelu.default"=>{
     if node.kwargs["approximate"]!="tanh"{return Err("unsupported audio GELU".into());}
     cuda(h3_audio_encoder_op(1,a.pointer(),out.pointer(),n,0,0,0,stream))
    },
    "aten.constant_pad_nd.default"=>{
     let pads=list(&node.args[1])?;if pads.len()!=2||a.shape.len()!=3||node.args.get(2).and_then(Value::as_f64).unwrap_or(0.)!=0.{return Err("unsupported audio zero pad".into());}
     cuda(h3_audio_encoder_op(0,a.pointer(),out.pointer(),n,a.shape[2] as i32,out.shape[2] as i32,pads[0] as i32,stream))
    },
    _=>unreachable!(),
   }
  })())
 }
}

impl Decoder {
 /// Infer temporal shapes from one model architecture. Padding is supplied by
 /// the caller, exactly as original encode() pads each channel to hop 800.
 pub fn select_encoder_samples(&mut self,samples:i64)->Result<()> {
  if self.manifest.schema!="h3-audio-encoder-graph-v1"||samples<=0||samples>960_000||samples%800!=0{return Err("invalid encoder sample count".into());}
  if self.manifest.input.shape[2]==samples{return Ok(());}
  if self.template.input.shape[2]==samples{self.manifest=self.template.clone();return Ok(());}
  if self.template.input.shape[2]%800!=0||self.template.nodes.iter().any(|n|n.op=="aten.constant_pad_nd.default"){return Err("dynamic audio encoder requires an unpadded hop-aligned architecture template".into());}
  let mut graph=self.template.clone();graph.input.shape=vec![2,1,samples];graph.input.bytes=(samples*8) as usize;graph.input.sha256.clear();
  let mut shapes:HashMap<String,Vec<i64>>=graph.constants.iter().map(|(n,s)|(n.clone(),s.shape.clone())).collect();shapes.insert("input".into(),graph.input.shape.clone());
  for node in &mut graph.nodes {
   let a=shape_of(&node.args[0],&shapes)?;let mut output=a.clone();
   match node.op.as_str(){
    "aten.transpose.int"=>output.swap(integer(&node.args[1])? as usize,integer(&node.args[2])? as usize),
    "aten.permute.default"=>output=list(&node.args[1])?.iter().map(|d|a[*d as usize]).collect(),
    "aten.unbind.int"=>{let d=if node.args.len()==2{0}else{integer(&node.args[1])? as usize};output.remove(d);},
    "aten.view.default"=>{
     let n=a.iter().product::<i64>();output=node.output.shape.clone();
     match (a.len(),output.len()) {
      (3,2)=>{if a[2]!=output[1]{return Err("unsupported encoder flatten".into());}output[0]=a[0]*a[1];},
      (2,3)=>{if output[0]!=2||output[2]!=a[1]||a[0]%2!=0{return Err("unsupported encoder unflatten".into());}output[1]=a[0]/2;},
      (3,5)=>{if output[0]!=2||output[2..]!=[3,8,256]||a[2]!=6144{return Err("unsupported QKV reshape".into());}output[1]=a[1];},
      _=>return Err("unsupported dynamic encoder reshape".into()),
     }
     if output.iter().product::<i64>()!=n{return Err("encoder reshape changes elements".into());}
    },
    "aten.native_layer_norm.default"|"aten.gelu.default"|"aten._scaled_dot_product_efficient_attention.default"=>{},
    "aten.addmm.default"=>{let x=shape_of(&node.args[1],&shapes)?;let w=shape_of(&node.args[2],&shapes)?;if x.len()!=2||w.len()!=2||x[1]!=w[0]{return Err("encoder projection dimensions".into());}output=vec![x[0],w[1]];},
    "aten.mean.dim"=>{if a.len()!=4||list(&node.args[1])?!=[2]{return Err("encoder head reduction dimensions".into());}output.remove(2);},
    "aten._adaptive_avg_pool2d.default"=>{let pool=list(&node.args[1])?;if a.len()!=4||pool!=[1,32]{return Err("encoder pooling dimensions".into());}output[2..].copy_from_slice(&pool);},
    "aten.unsqueeze.default"=>{let d=integer(&node.args[1])?;let d=if d<0{d+a.len() as i64+1}else{d};output.insert(d as usize,1);},
    "aten.squeeze.dim"=>{let d=integer(&node.args[1])?;let d=if d<0{d+a.len() as i64}else{d};if output[d as usize]!=1{return Err("encoder squeeze nonunit axis".into());}output.remove(d as usize);},
    _=>output=infer_node(node,&shapes)?,
   }
   let count=output.iter().try_fold(1i64,|n,x|if *x>0{n.checked_mul(*x)}else{None}).ok_or("encoder shape overflow")?;
   if count>536_870_912{return Err("encoder node memory budget".into());}
   node.output.shape=output.clone();node.output.bytes=count as usize*4;node.output.sha256.clear();shapes.insert(node.id.clone(),output);
  }
  graph.gold.shape=shapes.get(&graph.output_id).ok_or("encoder output missing")?.clone();
  if graph.gold.shape!=[2,32,samples/800]{return Err("encoder architecture does not downsample 800x".into());}
  graph.gold.bytes=(2*32*samples/800*4) as usize;graph.gold.sha256.clear();self.manifest=graph;Ok(())
 }
}
