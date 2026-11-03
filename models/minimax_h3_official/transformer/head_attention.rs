// Single-device counterpart to CP's full-sequence/local-head attention.
// Only exact copies are added; router, BSA and dense kernels are unchanged.
impl Runtime {
    fn grouped_attention(&self,q:Buffer,k:Buffer,v:Buffer,out:Buffer,sparse:bool)->Result<()> {
        let router=self.router.as_ref().ok_or("grouped attention router missing")?;
        let heads=router.plan.heads;
        if heads!=7 || self.seq<=crate::attention_heads::MAX_UNGROUPED_ROWS || self.cp.is_some() {
            return Err("invalid single-device head grouping".into());
        }
        let (gq,gk,gv,go)=(self.s("head_q"),self.s("head_k"),self.s("head_v"),self.s("head_out"));
        let scale=1.0f32/128.0f32.sqrt();
        for first in (0..HEADS).step_by(heads) {
            {
                let _host=HostSpan::new("attention.head_gather",0);
                let _gpu=GpuSpan::new("attention.head_gather",self.arena.stream)?;
                for (source,target) in [(q,gq),(k,gk),(v,gv)] {
                    cuda(unsafe {h3_attention_copy_heads(source.ptr,target.ptr,self.seq as i64,
                        first as c_int,heads as c_int,0,self.arena.stream)})?;
                }
            }
            if sparse {
                {
                    let _host=HostSpan::new("router",0);
                    let _gpu=GpuSpan::new("router",self.arena.stream)?;
                    router.launch(&self.route_buffers,self.arena.stream)?;
                }
                let _host=HostSpan::new("sparse_attention",0);
                let _gpu=GpuSpan::new("sparse_attention",self.arena.stream)?;
                let workspace=self.s("bsa_ws");
                cuda(unsafe {h3_bsa_forward_masked(gq.ptr,gk.ptr,gv.ptr,
                    self.route_buffers["indices"] as *const i32,self.block_sizes.ptr.cast(),self.query_block_mask.ptr.cast(),go.ptr,
                    workspace.ptr,workspace.bytes,self.seq as c_int,heads as c_int,self.kept as c_int,
                    scale,self.arena.stream)})?;
                BSA_CPP_CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                self.correct_dense_audio(gq,gk,gv,go,heads)?;
            } else {
                let _host=HostSpan::new("dense_attention.main",0);
                let _gpu=GpuSpan::new("dense_attention.main",self.arena.stream)?;
                self.dense.run_for_heads(gq,gk,gv,go,self.s("dense_ws"),self.seq,heads,0,scale,self.arena.stream)?;
            }
            let _host=HostSpan::new("attention.head_scatter",0);
            let _gpu=GpuSpan::new("attention.head_scatter",self.arena.stream)?;
            cuda(unsafe {h3_attention_copy_heads(go.ptr,out.ptr,self.seq as i64,
                first as c_int,heads as c_int,1,self.arena.stream)})?;
        }
        Ok(())
    }
}
