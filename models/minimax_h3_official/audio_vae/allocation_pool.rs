//! Exact-size, single-stream cache for private decoder intermediates only.
//! Final returned tensors, input uploads and constants never acquire a lease.
use crate::{cuda, cudaFree, cudaMalloc, cudaStreamSynchronize, DevicePtr, Result};
use std::{cell::RefCell, collections::HashMap, ptr, rc::Rc};

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Counters {
    pub malloc_calls: usize,
    pub reuse_calls: usize,
    pub return_calls: usize,
    pub unpooled_node_allocations: usize,
}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Snapshot {
    pub enabled: bool,
    pub stream: Option<usize>,
    pub request: Counters,
    pub total: Counters,
    pub cached_buffers: usize,
    pub cached_bytes: usize,
    pub live_buffers: usize,
    pub live_bytes: usize,
    pub allocated_bytes: usize,
    pub high_water_bytes: usize,
}

#[derive(Default)]
struct State {
    stats: Snapshot,
    available: HashMap<usize, Vec<DevicePtr>>,
}

impl Drop for State {
    fn drop(&mut self) {
        // A Lease holds an Rc<State>, so the last state owner cannot disappear
        // while a checked-out buffer remains. Drain before freeing cached data.
        debug_assert_eq!(self.stats.live_buffers, 0);
        if let Some(stream) = self.stats.stream {
            let code = unsafe { cudaStreamSynchronize(stream as DevicePtr) };
            if code != 0 {
                // CUDA failure: do not release potentially live allocations.
                eprintln!("audio pool cleanup stream synchronization failed: {code}; cached allocations retained until process exit");
                return;
            }
        }
        for pointers in self.available.values_mut() {
            for pointer in pointers.drain(..) {
                let code = unsafe { cudaFree(pointer) };
                if code != 0 { eprintln!("audio pool cudaFree failed: {code}"); }
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Pool {
    state: Rc<RefCell<State>>,
}

impl Pool {
    pub fn begin(&self, enabled: bool, stream: DevicePtr) -> Result<()> {
        let mut state = self.state.borrow_mut();
        if state.stats.live_buffers != 0 {
            return Err("audio pool request started with outstanding intermediate leases".into());
        }
        if state.stats.stream.is_some_and(|bound| bound != stream as usize) {
            return Err("audio allocation pool is bound to a different CUDA stream".into());
        }
        if enabled { state.stats.stream = Some(stream as usize); }
        state.stats.enabled = enabled;
        state.stats.request = Counters::default();
        Ok(())
    }

    pub fn enabled(&self) -> bool { self.state.borrow().stats.enabled }

    pub fn acquire(&self, bytes: usize, stream: DevicePtr) -> Result<Lease> {
        let mut state = self.state.borrow_mut();
        if !state.stats.enabled || state.stats.stream != Some(stream as usize) {
            return Err("audio pooled allocation requires enabled mode and its bound stream".into());
        }
        if bytes == 0 { return Err("zero-size audio pooled allocation".into()); }
        let pointer = if let Some(pointer) = state.available.get_mut(&bytes).and_then(Vec::pop) {
            state.stats.request.reuse_calls += 1;
            state.stats.total.reuse_calls += 1;
            state.stats.cached_buffers -= 1;
            state.stats.cached_bytes -= bytes;
            pointer
        } else {
            let mut pointer = ptr::null_mut();
            unsafe { cuda(cudaMalloc(&mut pointer, bytes))?; }
            state.stats.request.malloc_calls += 1;
            state.stats.total.malloc_calls += 1;
            state.stats.allocated_bytes += bytes;
            state.stats.high_water_bytes = state.stats.high_water_bytes.max(state.stats.allocated_bytes);
            pointer
        };
        state.stats.live_buffers += 1;
        state.stats.live_bytes += bytes;
        Ok(Lease { state: self.state.clone(), pointer, bytes })
    }

    pub fn note_unpooled_node_allocation(&self) {
        let mut state = self.state.borrow_mut();
        state.stats.request.unpooled_node_allocations += 1;
        state.stats.total.unpooled_node_allocations += 1;
    }

    pub fn snapshot(&self) -> Snapshot { self.state.borrow().stats }

    /// At a serialized request boundary, return only idle intermediates to CUDA.
    /// Constants, final outputs and decoder/library plans are not pool-owned.
    pub fn trim(&self) -> Result<usize> {
        if self.state.borrow().stats.live_buffers != 0 {
            return Err("cannot trim audio pool with live intermediate leases".into());
        }
        self.synchronize()?;
        let mut state = self.state.borrow_mut();
        let State { stats, available } = &mut *state;
        let mut released = 0;
        for (&bytes, pointers) in available.iter_mut() {
            while let Some(&pointer) = pointers.last() {
                unsafe { cuda(cudaFree(pointer))?; }
                pointers.pop();
                stats.cached_buffers -= 1;
                stats.cached_bytes -= bytes;
                stats.allocated_bytes -= bytes;
                released += bytes;
            }
        }
        available.clear();
        Ok(released)
    }

    pub fn synchronize(&self) -> Result<()> {
        if let Some(stream) = self.state.borrow().stats.stream {
            unsafe { cuda(cudaStreamSynchronize(stream as DevicePtr))?; }
        }
        Ok(())
    }
}

// Not Clone: the enclosing Tensor Allocation is already reference counted.
// It returns to this exact-size bucket only when the final Rc<Tensor> drops.
pub(crate) struct Lease {
    state: Rc<RefCell<State>>,
    pointer: DevicePtr,
    bytes: usize,
}

impl Lease { pub fn pointer(&self) -> DevicePtr { self.pointer } }

impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        state.available.entry(self.bytes).or_default().push(self.pointer);
        state.stats.live_buffers -= 1;
        state.stats.live_bytes -= self.bytes;
        state.stats.cached_buffers += 1;
        state.stats.cached_bytes += self.bytes;
        state.stats.request.return_calls += 1;
        state.stats.total.return_calls += 1;
    }
}
