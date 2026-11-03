//! Choose a reusable allocation without exposing its unused tail to kernels.
pub fn best_fit(sizes:impl Iterator<Item=usize>,requested:usize,exact:bool)->Option<usize>{
    sizes.enumerate().filter(|(_,size)|*size>=requested && (!exact || *size==requested))
        .min_by_key(|(_,size)|*size).map(|(index,_)|index)
}

/// Restore the pool policy even if a compact rebuild returns an error.
pub struct Pause<'a>{enabled:&'a std::cell::Cell<bool>,previous:bool}
impl<'a> Pause<'a>{
    pub fn new(enabled:&'a std::cell::Cell<bool>)->Self{Self{enabled,previous:enabled.replace(false)}}
}
impl Drop for Pause<'_>{fn drop(&mut self){self.enabled.set(self.previous);}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn never_returns_an_undersized_buffer(){
        assert_eq!(best_fit([32,128,64].into_iter(),129,false),None);
        assert_eq!(best_fit([32,128,64].into_iter(),48,false),Some(2));
    }
    #[test] fn preserves_exact_single_device_policy(){
        assert_eq!(best_fit([32,128,64].into_iter(),48,true),None);
        assert_eq!(best_fit([32,128,64].into_iter(),64,true),Some(2));
    }
    #[test] fn large_sizes_do_not_truncate(){
        assert_eq!(best_fit([1usize<<34,1usize<<33].into_iter(),(1usize<<32)+1,false),Some(1));
    }
    #[test] fn compaction_restores_policy_after_error_and_nested_scope(){
        let enabled=std::cell::Cell::new(true);
        let rebuild=||->Result<(),()>{
            let _outer=Pause::new(&enabled);assert!(!enabled.get());
            {let _inner=Pause::new(&enabled);assert!(!enabled.get());}
            assert!(!enabled.get());Err(())
        };
        assert!(rebuild().is_err());assert!(enabled.get());
    }
}
