//! Lazy, phase-owned working storage; parameters and numerical state live elsewhere.
//! A caller must finish its stream dependencies before retiring a populated slot.
use crate::Result;
use std::cell::{Ref, RefCell};

pub struct PhaseWorkspace<T> { value: RefCell<Option<T>> }
impl<T> Default for PhaseWorkspace<T> {
    fn default() -> Self { Self { value: RefCell::new(None) } }
}

impl<T> PhaseWorkspace<T> {
    pub fn get_or_try_init(&self, create: impl FnOnce() -> Result<T>) -> Result<Ref<'_, T>> {
        let current=self.value.try_borrow().map_err(|_|"workspace is being changed")?;
        if current.is_some() { return Ok(Ref::map(current,|v|v.as_ref().unwrap())); }
        drop(current);
        let mut current=self.value.try_borrow_mut().map_err(|_|"workspace is still borrowed")?;
        *current=Some(create()?);
        drop(current);
        Ok(Ref::map(self.value.borrow(),|v|v.as_ref().unwrap()))
    }

    pub fn release(&self, complete: impl FnOnce() -> Result<()>) -> Result<bool> {
        let mut current=self.value.try_borrow_mut().map_err(|_|"cannot release borrowed workspace")?;
        if current.is_none() { return Ok(false); }
        complete()?;
        drop(current.take());
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell,rc::Rc};
    struct Storage(Rc<Cell<usize>>);
    impl Drop for Storage {fn drop(&mut self){self.0.set(self.0.get()+1);}}

    #[test]
    fn completion_precedes_drop_and_the_next_phase_recreates_storage() {
        let slot=PhaseWorkspace::default();let drops=Rc::new(Cell::new(0));
        assert!(!slot.release(||panic!("empty phase has no pending GPU storage")).unwrap());
        let held=slot.get_or_try_init(||Ok(Storage(drops.clone()))).unwrap();
        let another=slot.get_or_try_init(||panic!("a live phase reuses its storage")).unwrap();
        assert!(std::ptr::eq(&*held,&*another));
        drop((held,another));
        assert!(slot.release(||{assert_eq!(drops.get(),0);Ok(())}).unwrap());
        assert_eq!(drops.get(),1);
        drop(slot.get_or_try_init(||Ok(Storage(drops.clone()))).unwrap());
        drop(slot);assert_eq!(drops.get(),2);
    }

    #[test]
    fn borrowers_and_failed_completion_cannot_retire_inflight_storage() {
        let slot=PhaseWorkspace::default();let drops=Rc::new(Cell::new(0));
        let held=slot.get_or_try_init(||Ok(Storage(drops.clone()))).unwrap();
        assert!(slot.release(||panic!("borrow is still active")).is_err());
        drop(held);
        assert!(slot.release(||Err("stream failed".into())).is_err());
        assert_eq!(drops.get(),0);
        drop(slot.get_or_try_init(||panic!("failed completion kept the original storage")).unwrap());
        assert!(slot.release(||Ok(())).unwrap());assert_eq!(drops.get(),1);
    }

    #[test]
    fn failed_preparation_does_not_publish_an_initialized_workspace() {
        let slot=PhaseWorkspace::<usize>::default();
        assert!(slot.get_or_try_init(||Err("allocation failed".into())).is_err());
        assert_eq!(*slot.get_or_try_init(||Ok(7)).unwrap(),7);
    }
}
