//! Typed request inputs for native numerical callbacks. JSON declarations carry
//! the request key; large CPU tensors stay in this bounded, single-scope slot.
//! The declared executor, rather than this slot, schedules and waits for work.
use crate::declared_startup::Result;
use std::sync::{Arc, Mutex};

type Task<T> = Box<dyn FnOnce() -> Result<T> + Send>;
enum State<T> { Empty, Pending(String,Task<T>), Running(String), Ready(String,Arc<T>), Failed(String) }
pub struct RequestInput<T> { state: Mutex<State<T>> }
impl<T> Default for RequestInput<T> {fn default()->Self {Self {state:Mutex::new(State::Empty)}}}
impl<T> RequestInput<T> {
    pub fn submit(&self,key:&str,task:impl FnOnce()->Result<T>+Send+'static)->Result<()> {
        let mut state=self.state.lock().unwrap();
        if key.is_empty() || !matches!(*state,State::Empty) {return Err("native input slot is occupied or its scope is empty".into());}
        *state=State::Pending(key.into(),Box::new(task));Ok(())
    }
    /// Called only by the declared Preparation's CPU callback.
    pub fn prepare(&self,key:&str)->Result<()> {
        let task={
            let mut state=self.state.lock().unwrap();
            if !matches!(&*state,State::Pending(scope,_) if scope==key) {return Err("native CPU input belongs to another scope or already started".into());}
            match std::mem::replace(&mut *state,State::Running(key.into())) {State::Pending(_,task)=>task,_=>unreachable!()}
        };
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(task))
            .unwrap_or_else(|_|Err("native CPU input callback panicked".into()));
        let mut state=self.state.lock().unwrap();
        if !matches!(&*state,State::Running(scope) if scope==key) {return Err("native CPU scope changed while preparing".into());}
        match result {
            Ok(value)=>{*state=State::Ready(key.into(),Arc::new(value));Ok(())},
            Err(error)=>{*state=State::Failed(key.into());Err(error)},
        }
    }
    pub fn get(&self,key:&str)->Result<Arc<T>> {
        match &*self.state.lock().unwrap() {
            State::Ready(scope,value) if scope==key=>Ok(value.clone()),
            State::Failed(scope) if scope==key=>Err("native CPU input preparation failed".into()),
            _=>Err("native CPU input is not ready for this scope".into()),
        }
    }
    /// Release a completed scope without copying its CPU tensors. All callbacks
    /// and numerical consumers must have returned their borrowed Arc first.
    pub fn take(&self,key:&str)->Result<T> {
        let mut state=self.state.lock().unwrap();
        if !matches!(&*state,State::Ready(scope,_) if scope==key) {return Err("native input cannot retire an unready or foreign scope".into());}
        let State::Ready(scope,value)=std::mem::replace(&mut *state,State::Empty) else {unreachable!()};
        match Arc::try_unwrap(value) {
            Ok(value)=>Ok(value),
            Err(value)=>{*state=State::Ready(scope,value);Err("native input still has active consumers".into())},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_scope_is_bounded_and_large_inputs_move_without_copying() {
        let input=RequestInput::default();let value=vec![7u8;1024];let address=value.as_ptr() as usize;
        input.submit("one",move||Ok(value)).unwrap();
        assert!(input.submit("two",||Ok(vec![])).is_err());
        assert!(input.prepare("two").is_err());input.prepare("one").unwrap();
        let borrowed=input.get("one").unwrap();assert_eq!(borrowed.as_ptr() as usize,address);
        assert!(input.take("one").is_err());assert!(input.get("two").is_err());
        drop(borrowed);let value=input.take("one").unwrap();assert_eq!(value.as_ptr() as usize,address);
        input.submit("two",||Ok(vec![8])).unwrap();input.prepare("two").unwrap();assert_eq!(input.take("two").unwrap(),vec![8]);
        assert!(input.get("one").is_err());
    }
    #[test]
    fn failure_and_panic_cannot_publish_cpu_readiness() {
        for panic in [false,true] {
            let input=RequestInput::<()>::default();input.submit("one",move||{if panic {panic!("bad input");}Err("bad input".into())}).unwrap();
            assert!(input.prepare("one").is_err());assert!(input.get("one").is_err());assert!(input.take("one").is_err());
        }
    }
}
