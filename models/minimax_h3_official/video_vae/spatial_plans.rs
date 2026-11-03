// Plans are shared (Arc): selecting one for a request no longer deep-copies the 1.8 MB program manifest.
struct SpatialPlans {
    base:std::sync::Arc<Value>,plans:Vec<((usize,usize),std::sync::Arc<Value>)>,active:(usize,usize),
    builds:usize,hits:usize,evictions:usize,
    /// Set once the decoder parameters were converted in place: every plan must then be hoisted.
    hoist:Option<std::sync::Arc<CastHoist>>,hoisted:Option<std::sync::Arc<Value>>,
}
impl SpatialPlans {
    const CAPACITY:usize=16;
    fn new(base:&std::sync::Arc<Value>,hoist:Option<std::sync::Arc<CastHoist>>)->Result<Self>{
        let hoisted=match &hoist{Some(h)=>Some(std::sync::Arc::new(h.apply(base)?)),None=>None};
        Ok(Self{base:base.clone(),plans:vec![],active:(256,256),builds:0,hits:0,evictions:0,hoist,hoisted})
    }
    /// The executable maximum-tile plan.
    fn top(&self)->std::sync::Arc<Value>{self.hoisted.clone().unwrap_or_else(||self.base.clone())}
    fn select(&mut self,stage:&str,h:usize,w:usize)->Result<std::sync::Arc<Value>>{
        let expected=match stage{"encoder"=>"video_vae_encoder_tile","decoder"=>"video_vae_decoder_tile",_=>return Err("unknown VAE stage".into())};
        if self.base["stage"]!=expected{return Err("VAE program stage mismatch".into());}
        if h<32 || w<32 || h>256 || w>256 || h%32!=0 || w%32!=0{return Err("unsupported spatial tile lattice".into());}
        if (h,w)==(256,256){self.active=(h,w);self.hits+=1;return Ok(self.top());}
        if let Some(index)=self.plans.iter().position(|(key,_)|*key==(h,w)){
            let entry=self.plans.remove(index);let plan=entry.1.clone();self.plans.push(entry);
            self.active=(h,w);self.hits+=1;return Ok(plan);
        }
        let plan=match stage{"encoder"=>resize_encoder_manifest(&self.base,h,w)?,
            "decoder"=>resize_decoder_manifest(&self.base,h,w)?,_=>return Err("unknown VAE stage".into())};
        let plan=std::sync::Arc::new(match &self.hoist{Some(h)=>h.apply(&plan)?,None=>plan});
        if self.plans.len()>=Self::CAPACITY{self.plans.remove(0);self.evictions+=1;}
        self.plans.push(((h,w),plan.clone()));self.builds+=1;self.active=(h,w);Ok(plan)
    }
    fn report(&self)->Value{json!({"shape_method":"native operator and decoder architecture semantics",
        "active_tile_height":self.active.0,"active_tile_width":self.active.1,
        "generic_small_tile":self.active!=(256,256),"capacity":Self::CAPACITY,
        "entries":self.plans.len(),"builds":self.builds,"hits":self.hits,"evictions":self.evictions,
        "weight_reload":false,"allocation_policy":"reuse original maximum-tile buffers",
        "weight_casts_hoisted":self.hoist.as_ref().map(|h|h.casts.len())})}
}
