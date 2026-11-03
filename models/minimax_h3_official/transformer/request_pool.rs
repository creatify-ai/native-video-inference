// Only synchronized activation allocations enter this bounded pool. Model and
// reconstructed request weights never belong to it. CP layouts reuse capacity;
// single-device behavior remains exact-size and limited to small allocations.
struct RequestPool {
    cached:std::cell::RefCell<Vec<Buffer>>,
    cached_bytes:std::cell::Cell<usize>,
    hits:std::cell::Cell<usize>,misses:std::cell::Cell<usize>,
    released:std::cell::Cell<usize>,
    budget:usize,block_limit:usize,exact:bool,
    enabled:std::cell::Cell<bool>,
}
impl RequestPool {
    fn new()->Self {
        let cp=std::env::var("H3_CP_WORLD").as_deref()==Ok("8");
        Self{cached:Default::default(),cached_bytes:Default::default(),
            hits:Default::default(),misses:Default::default(),released:Default::default(),
            budget:if cp{16usize<<30}else{256<<20},block_limit:if cp{8usize<<30}else{64<<20},exact:!cp,
            enabled:std::cell::Cell::new(true)}
    }
    fn take(&self,bytes:usize)->Option<Buffer>{
        if !self.enabled.get(){self.misses.set(self.misses.get()+1);return None;}
        let mut pool=self.cached.borrow_mut();
        if let Some(i)=crate::allocation_policy::best_fit(pool.iter().map(|b|b.bytes),bytes,self.exact){
            let buffer=pool.remove(i);self.cached_bytes.set(self.cached_bytes.get()-buffer.bytes);
            self.hits.set(self.hits.get()+1);Some(buffer)
        }else{self.misses.set(self.misses.get()+1);None}
    }
    fn release(&self,buffer:Buffer){
        if !self.enabled.get() || buffer.bytes>self.block_limit {unsafe{cudaFree(buffer.ptr);}self.released.set(self.released.get()+1);return;}
        let mut pool=self.cached.borrow_mut();
        while self.cached_bytes.get()+buffer.bytes>self.budget {
            let old=pool.remove(0);unsafe{cudaFree(old.ptr);}
            self.cached_bytes.set(self.cached_bytes.get()-old.bytes);self.released.set(self.released.get()+1);
        }
        self.cached_bytes.set(self.cached_bytes.get()+buffer.bytes);pool.push(buffer);
    }
    fn clear(&self){
        for buffer in self.cached.borrow_mut().drain(..){
            unsafe{cudaFree(buffer.ptr);}self.released.set(self.released.get()+1);
        }
        self.cached_bytes.set(0);
    }
    fn report(&self)->Value{json!({"enabled":true,"exact_size":self.exact,"cached_bytes":self.cached_bytes.get(),
        "cached_buffers":self.cached.borrow().len(),"hits":self.hits.get(),"cuda_malloc_calls":self.misses.get(),
        "released_buffers":self.released.get(),"max_cached_bytes":self.budget,"max_cached_block_bytes":self.block_limit,
        "model_weights_pooled":false,"synchronize_before_return":true})}
}
impl Drop for RequestPool {
    fn drop(&mut self){for b in self.cached.get_mut().drain(..){unsafe{cudaFree(b.ptr);}}}
}
