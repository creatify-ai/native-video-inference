//! Bounded reuse of owned allocations. No tensor values or model state.
//! The caller is responsible for context/stream ordering and reinitialization.
use std::collections::BTreeMap;

pub struct ReusePool<T> {
    capacity:usize,max_allocation:usize,max_entries:usize,
    bytes:usize,entries:usize,free:BTreeMap<usize,Vec<T>>,
}
impl<T> ReusePool<T> {
    pub fn new(capacity:usize,max_allocation:usize,max_entries:usize)->Self{
        Self{capacity,max_allocation,max_entries,bytes:0,entries:0,free:BTreeMap::new()}
    }
    pub fn bytes(&self)->usize{self.bytes}
    pub fn entries(&self)->usize{self.entries}
    pub fn take(&mut self,bytes:usize)->Option<T>{
        let values=self.free.get_mut(&bytes)?;let value=values.pop()?;
        self.bytes-=bytes;self.entries-=1;
        if values.is_empty(){self.free.remove(&bytes);}Some(value)
    }
    /// Return rejected allocations to their caller so ownership is never lost.
    pub fn put(&mut self,bytes:usize,value:T)->Option<T>{
        if bytes==0 || bytes>self.max_allocation || self.entries>=self.max_entries ||
            bytes>self.capacity.saturating_sub(self.bytes){return Some(value);}
        self.free.entry(bytes).or_default().push(value);self.bytes+=bytes;self.entries+=1;None
    }
    pub fn drain(&mut self)->Vec<T>{
        self.bytes=0;self.entries=0;
        std::mem::take(&mut self.free).into_values().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell,rc::Rc};
    struct Allocation(Rc<Cell<usize>>);
    impl Drop for Allocation{fn drop(&mut self){self.0.set(self.0.get()+1);}}
    #[test] fn retained_storage_keeps_ownership_until_taken_or_drained(){
        let drops=Rc::new(Cell::new(0));let mut pool=ReusePool::new(32,16,2);
        assert!(pool.put(16,Allocation(drops.clone())).is_none());
        assert!(pool.put(8,Allocation(drops.clone())).is_none());
        assert_eq!(drops.get(),0);assert!(pool.take(12).is_none());
        let value=pool.take(16).unwrap();assert_eq!((pool.bytes(),pool.entries()),(8,1));
        drop(value);assert_eq!(drops.get(),1);
        drop(pool.drain());assert_eq!(drops.get(),2);assert_eq!((pool.bytes(),pool.entries()),(0,0));
    }
    #[test] fn rejected_allocations_remain_owned_and_all_bounds_apply(){
        let mut pool=ReusePool::new(20,16,2);
        assert_eq!(pool.put(0,1),Some(1));assert_eq!(pool.put(17,2),Some(2));
        assert!(pool.put(16,3).is_none());assert_eq!(pool.put(8,4),Some(4));
        assert!(pool.put(1,5).is_none());assert_eq!(pool.put(1,6),Some(6));
        assert_eq!(pool.take(16),Some(3));assert_eq!(pool.take(1),Some(5));
    }
}
