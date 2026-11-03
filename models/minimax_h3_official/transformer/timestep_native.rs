// Included by native.rs. Model-derived, request-independent T2V initial table.
// Computed once per resident model, using the original one-row GEMM dimensions.
fn zero_timestep_final_mod(model_arena: &mut Arena, weights: &HashMap<String,Tensor>) -> Result<Buffer> {
    let mut temporary=Arena::new()?;
    let mut projection=vec![0.0f32;256];projection[..128].fill(1.0);
    // H3 Timesteps(256, flip_sin_to_cos=true): t=0 is [cos(0),sin(0)].
    let bytes:Vec<u8>=projection.iter().flat_map(|v|v.to_le_bytes()).collect();
    let input=temporary.upload(&bytes)?;
    let hidden=temporary.alloc(HIDDEN*4)?;
    let temb=temporary.alloc((HIDDEN/2)*4)?;
    let activated=temporary.alloc((HIDDEN/2)*2)?;
    let output=model_arena.alloc(2*HIDDEN*2)?;
    let ws=temporary.alloc(32<<20)?;let lt=temporary.alloc(1<<20)?;
    struct Handle(*mut c_void);
    impl Drop for Handle{fn drop(&mut self){unsafe{h3_linear_destroy(self.0)}}}
    let mut raw=ptr::null_mut();
    blas(unsafe{h3_linear_create(temporary.stream,ws.ptr,ws.bytes,lt.ptr,lt.bytes,0,1,&mut raw)})?;
    let handle=Handle(raw);
    let linear=|name:&str,x:Buffer,y:Buffer,n:usize,k:usize,dtype:c_int|->Result<()> {
        let w=weights.get(&format!("{name}.weight")).ok_or("timestep weight missing")?;
        let b=weights.get(&format!("{name}.bias")).ok_or("timestep bias missing")?;
        let expected=if dtype==1{"torch.float32"}else{"torch.bfloat16"};
        if w.spec.shape!=[n,k] || b.spec.shape!=[n] || w.spec.dtype!=expected || b.spec.dtype!=expected {
            return Err("timestep weight layout differs from H3 contract".into());
        }
        blas(unsafe{h3_linear_forward(handle.0,x.ptr,w.buffer.ptr,b.buffer.ptr,y.ptr,1,n as c_int,k as c_int,dtype)})
    };
    // FP32 SiLU before linear_2; FP32 SiLU then BF16 cast before norm_out.linear.
    linear("time_embedder.linear_1",input,hidden,HIDDEN,256,1)?;
    cuda(unsafe{h3_silu_f32(hidden.ptr.cast(),hidden.ptr,HIDDEN as i64,0,temporary.stream)})?;
    linear("time_embedder.linear_2",hidden,temb,HIDDEN/2,HIDDEN,1)?;
    cuda(unsafe{h3_silu_f32(temb.ptr.cast(),activated.ptr,(HIDDEN/2) as i64,1,temporary.stream)})?;
    linear("norm_out.linear",activated,output,2*HIDDEN,HIDDEN/2,0)?;
    temporary.sync()?;
    Ok(output)
}
