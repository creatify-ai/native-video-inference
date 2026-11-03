// One immutable model/architecture; only CPU shape metadata is keyed by runtime T.
#[derive(Clone)]
struct ShapeGraph{manifest:Manifest,last_used:u64}
fn constant_key(spec:&Spec)->String{format!("{}:{}:{:?}:{}",spec.sha256,spec.dtype,spec.shape,spec.bytes)}
impl Decoder{
    pub fn load(root:impl AsRef<Path>)->Result<Self>{
        let(root,template,template_sha)=read_template(root.as_ref())?;
        let mut shared_weights=HashMap::new();let mut constants=HashMap::new();
        for(name,spec)in &template.constants{let key=constant_key(spec);if !shared_weights.contains_key(&key){let bytes=crate::startup_trace::read(root.join(&spec.file)).map_err(|e|e.to_string())?;if bytes.len()!=spec.bytes||!weight_digest_matches(&bytes,&spec.sha256)?{return Err(format!("constant integrity {name}"));}shared_weights.insert(key.clone(),Tensor::from_bytes(&spec.shape,&bytes)?);}constants.insert(name.clone(),shared_weights[&key].clone());}
        let mut context=ptr::null_mut();unsafe{conv(h3_conv_create(&mut context))?;}
        let mut uses=HashMap::new();for node in &template.nodes{for arg in &node.args{if let Some(id)=tensor_id(arg){*uses.entry(id.to_string()).or_insert(0)+=1;}}}*uses.entry(template.output_id.clone()).or_insert(0)+=1;
        let manifest=infer_graph(&template,template.input.shape[2])?;
        let mut value=Self{root,template,template_sha,manifest,uses,constants,context,plans:HashMap::new(),allocation_pool:allocation_pool::Pool::default(),allocation_pool_enabled:false,
            shared_weights,shape_graphs:HashMap::new(),graph_prepare_count:0,graph_switch_count:0,graph_evictions:0,graph_clock:0,last_graph_prepare_seconds:0.,last_pool_released_bytes:0,last_graph_cache_hit:false,weight_identity:String::new()};
        value.weight_identity=value.current_weight_identity();Ok(value)
    }
    fn current_weight_identity(&self)->String{let mut values:Vec<_>=self.shared_weights.iter().map(|(k,v)|format!("{k}:{:p}",v.pointer())).collect();values.sort();values.push(format!("context:{:p}",self.context));digest(values.join("\n").as_bytes())}
    pub fn select_tokens(&mut self,root:&Path,tokens:i64,debug:bool)->Result<()>{
        let started=std::time::Instant::now();let root=root.canonicalize().map_err(|e|e.to_string())?;
        if root!=self.root{return Err("generic audio remains bound to its original static model root".into());}
        if debug&&digest(&crate::startup_trace::read(root.join("manifest.json")).map_err(|e|e.to_string())?)!=self.template_sha{return Err("audio architecture manifest changed".into());}
        self.last_pool_released_bytes=0;self.last_graph_prepare_seconds=0.;self.graph_clock+=1;
        self.last_graph_cache_hit=self.shape_graphs.contains_key(&tokens);
        if !self.last_graph_cache_hit{
            let manifest=infer_graph(&self.template,tokens)?;
            if self.shape_graphs.len()>=GRAPH_LIMIT{let key=*self.shape_graphs.iter().min_by_key(|(_,g)|g.last_used).unwrap().0;self.shape_graphs.remove(&key);self.graph_evictions+=1;}
            self.shape_graphs.insert(tokens,ShapeGraph{manifest,last_used:self.graph_clock});self.graph_prepare_count+=1;self.last_graph_prepare_seconds=started.elapsed().as_secs_f64();
        }
        let graph=self.shape_graphs.get_mut(&tokens).unwrap();graph.last_used=self.graph_clock;
        if self.manifest.input.shape[2]!=tokens{
            if self.allocation_pool.snapshot().live_buffers!=0{return Err("audio shape switch with live pool leases".into());}
            self.allocation_pool.synchronize()?;self.last_pool_released_bytes=self.allocation_pool.snapshot().allocated_bytes;
            self.allocation_pool=allocation_pool::Pool::default();self.manifest=graph.manifest.clone();self.plans.clear();self.graph_switch_count+=1;
        }
        if self.current_weight_identity()!=self.weight_identity{return Err("audio resident weights/context identity changed".into());}Ok(())
    }
    pub fn graph_manifest_sha(&self)->&str{&self.template_sha}
    pub fn dynamic_graph_report(&self)->Value{
        let mut stats=[0u64;8];unsafe{h3_conv_cache_stats(self.context,stats.as_mut_ptr());}
        json!({"schema":"h3-generic-audio-graphs-v1","active_root":self.root,"active_tokens":self.manifest.input.shape[2],"resident_graphs":self.shape_graphs.len(),"graph_cache_limit":GRAPH_LIMIT,
            "graph_prepare_count":self.graph_prepare_count,"graph_switch_count":self.graph_switch_count,"graph_evictions":self.graph_evictions,"graph_cache_hit":self.last_graph_cache_hit,"graph_prepare_seconds":self.last_graph_prepare_seconds,
            "pool_released_on_switch_bytes":self.last_pool_released_bytes,"pool_retention":"active length only; synchronized release on switch",
            "shared_weights":self.shared_weights.len(),"weight_upload_count":self.shared_weights.len(),"shared_weight_bytes":self.shared_weights.values().map(Tensor::bytes).sum::<usize>(),
            "weights_and_context_identity_sha256":self.weight_identity,"model_load_count":1,"weight_reloaded":false,"cudnn_context_count":1,"per_length_manifest_reads":0,
            "shape_method":"native operator semantics from one architecture template","model_manifest_sha256":self.template_sha,
            "convolution_cache":{"entries":stats[0],"limit":stats[1],"hits":stats[2],"misses":stats[3],"evictions":stats[4],"workspace_bytes":stats[5],"workspace_limit_bytes":stats[6],"workspace_allocations":stats[7]}})
    }
}
