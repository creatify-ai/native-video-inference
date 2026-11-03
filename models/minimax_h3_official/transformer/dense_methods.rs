// Additive APIs for Ulysses full-sequence/local-head attention. The ordinary
// size/run methods retain their accepted 56-head behavior byte for byte.
impl Dense {
    fn rect_bytes(&self, q_tokens:usize, kv_tokens:usize, heads:usize)->Result<usize> {
        Self::local_head_bytes(q_tokens,heads,0)?;Self::local_head_bytes(kv_tokens,heads,0)?;
        type Size=unsafe extern "C" fn(c_int,c_int,c_int,*mut usize)->c_int;
        let ptr=unsafe{dlsym(self.library,CString::new("h3_dense_rect_workspace_bytes").unwrap().as_ptr())};
        if ptr.is_null(){return Err("Ref2VA dense audio requires rectangular cuDNN ABI".into());}
        let function:Size=unsafe{std::mem::transmute(ptr)};let mut bytes=0;
        self.check(unsafe{function(q_tokens as c_int,kv_tokens as c_int,heads as c_int,&mut bytes)})?;
        Ok(bytes.max(1))
    }
    fn rect_run(&self,q:Buffer,k:Buffer,v:Buffer,out:Buffer,workspace:Buffer,
        q_tokens:usize,kv_tokens:usize,heads:usize,stream:Stream)->Result<()> {
        let qb=Self::local_head_bytes(q_tokens,heads,0)?;
        let kb=Self::local_head_bytes(kv_tokens,heads,0)?;
        if q.bytes<qb||out.bytes<qb||k.bytes<kb||v.bytes<kb{return Err("rectangular attention buffer too small".into());}
        type Run=unsafe extern "C" fn(*const c_void,*const c_void,*const c_void,*mut c_void,
            *mut c_void,usize,c_int,c_int,c_int,f32,Stream)->c_int;
        let ptr=unsafe{dlsym(self.library,CString::new("h3_dense_rect_forward").unwrap().as_ptr())};
        if ptr.is_null(){return Err("missing rectangular attention forward ABI".into());}
        let function:Run=unsafe{std::mem::transmute(ptr)};
        self.check(unsafe{function(q.ptr,k.ptr,v.ptr,out.ptr,workspace.ptr,workspace.bytes,
            q_tokens as c_int,kv_tokens as c_int,heads as c_int,1.0f32/128.0f32.sqrt(),stream)})
    }
    fn bytes_for_heads(&self, tokens: usize, heads: usize, layout: c_int) -> Result<usize> {
        Self::local_head_bytes(tokens, heads, layout)?;
        let function = self.workspace_v2.ok_or("local-head dense requires the explicit-layout v2 ABI")?;
        let mut bytes = 0;
        self.check(unsafe { function(tokens as c_int, heads as c_int, layout, &mut bytes) })?;
        Ok(bytes.max(1))
    }

    fn local_head_bytes(tokens: usize, heads: usize, layout: c_int) -> Result<usize> {
        if tokens == 0 || tokens > crate::limits::MAX_PACKED_ROWS || ![7, 14, 28, 56].contains(&heads) || ![0, 1].contains(&layout) {
            return Err("Ulysses dense requires 1<=S<=524288, H in {7,14,28,56}, D=128, layout 0/1".into());
        }
        tokens.checked_mul(heads).and_then(|n| n.checked_mul(128 * 2))
            .ok_or_else(|| "Ulysses dense tensor size overflow".into())
    }

    fn run_for_heads(&self, q: Buffer, k: Buffer, v: Buffer, out: Buffer,
        workspace: Buffer, tokens: usize, heads: usize, layout: c_int,
        scale: f32, stream: Stream) -> Result<()> {
        let bytes = Self::local_head_bytes(tokens, heads, layout)?;
        for buffer in [q, k, v, out] {
            if buffer.bytes < bytes || buffer.ptr.is_null() || (buffer.ptr as usize) % 16 != 0 {
                return Err("local-head dense buffer is too small, null, or unaligned".into());
            }
        }
        let function = self.forward_v2.ok_or("local-head dense requires the explicit-layout v2 ABI")?;
        self.check(unsafe { function(q.ptr, k.ptr, v.ptr, out.ptr, workspace.ptr, workspace.bytes,
            tokens as c_int, heads as c_int, layout, scale, stream) })
    }
}

impl Runtime {
    fn correct_dense_audio(&self,q:Buffer,k:Buffer,v:Buffer,out:Buffer,heads:usize)->Result<()> {
        let Some((lo,hi))=self.dense_audio_range else{return Ok(())};
        let stride=heads*128*2;
        self.dense.rect_run(q.slice(lo*stride,(hi-lo)*stride)?,k,v,
            out.slice(lo*stride,(hi-lo)*stride)?,self.s("dense_audio_ws"),hi-lo,self.seq,heads,self.arena.stream)
    }
}
