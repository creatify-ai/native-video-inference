//! Bounded context-owned resources that survive model weight eviction.
//! A backend owns the cache lease; the registry owns only weak references.
use crate::Result;
use std::{any::{Any, TypeId}, cell::{Cell, RefCell}, collections::BTreeMap, rc::{Rc, Weak}};

pub struct ResidentAssets<K, V> {
    limit: usize,
    max_entries: usize,
    used: Cell<usize>,
    entries: RefCell<BTreeMap<K, (usize, Rc<V>)>>,
}
impl<K: Ord, V> ResidentAssets<K, V> {
    pub fn new(limit: usize, max_entries: usize) -> Self {
        Self { limit, max_entries, used: Cell::new(0), entries: RefCell::new(BTreeMap::new()) }
    }
    pub fn bytes(&self) -> usize { self.used.get() }
    pub fn entries(&self) -> usize { self.entries.borrow().len() }
    pub fn get_or_try_insert_with(&self, key: K, bytes: usize, load: impl FnOnce() -> Result<V>) -> Result<Rc<V>> {
        if let Some((extent, value)) = self.entries.borrow().get(&key) {
            if *extent != bytes { return Err("resident asset extent changed".into()); }
            return Ok(value.clone());
        }
        let used = self.bytes().checked_add(bytes).ok_or("resident asset extent overflow")?;
        if used > self.limit || self.entries() >= self.max_entries {
            return Err("resident asset cache budget exceeded".into());
        }
        let value = Rc::new(load()?);
        self.entries.borrow_mut().insert(key, (bytes, value.clone()));
        self.used.set(used);
        Ok(value)
    }
}

type Registry = BTreeMap<(usize, TypeId), Weak<dyn Any>>;
thread_local! { static CONTEXT_ASSETS: RefCell<Registry> = RefCell::new(BTreeMap::new()); }

/// Context identities are never shared between threads or resource types. The
/// caller must hold this lease until all uses of the context have completed.
pub fn for_context<K: Ord + 'static, V: 'static>(context: usize, limit: usize, max_entries: usize)
    -> Result<Rc<ResidentAssets<K, V>>> {
    if context == 0 || limit == 0 || max_entries == 0 { return Err("invalid resident context asset budget".into()); }
    CONTEXT_ASSETS.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.retain(|_, cache| cache.strong_count() != 0);
        let key = (context, TypeId::of::<ResidentAssets<K, V>>());
        if let Some(cache) = registry.get(&key).and_then(Weak::upgrade) {
            let cache = cache.downcast::<ResidentAssets<K, V>>().map_err(|_| "resident asset type differs")?;
            if cache.limit != limit || cache.max_entries != max_entries {
                return Err("resident context asset budget changed".into());
            }
            return Ok(cache);
        }
        if registry.len() >= 64 { return Err("resident context registry budget exceeded".into()); }
        let cache = Rc::new(ResidentAssets::<K, V>::new(limit, max_entries));
        let erased: Rc<dyn Any> = cache.clone();
        registry.insert(key, Rc::downgrade(&erased));
        Ok(cache)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_identity_retains_resources_only_while_owned() {
        let first = for_context::<u32, String>(101, 8, 2).unwrap();
        let value = first.get_or_try_insert_with(1, 4, || Ok("code".into())).unwrap();
        let same = for_context::<u32, String>(101, 8, 2).unwrap();
        assert!(Rc::ptr_eq(&first, &same));
        assert!(Rc::ptr_eq(&value, &same.get_or_try_insert_with(1, 4, || panic!("reload")).unwrap()));
        assert!(!Rc::ptr_eq(&first, &for_context::<u32, String>(102, 8, 2).unwrap()));
        assert!(for_context::<u32, String>(101, 9, 2).is_err());
        let weak = Rc::downgrade(&first);
        drop(first); drop(same);
        assert!(weak.upgrade().is_none());
        assert_eq!(for_context::<u32, String>(101, 8, 2).unwrap().entries(), 0);
    }
    #[test]
    fn failure_and_budget_never_admit_partial_resources() {
        let cache = ResidentAssets::<u32, u32>::new(4, 1);
        assert!(cache.get_or_try_insert_with(1, 5, || panic!("over budget loader")).is_err());
        assert!(cache.get_or_try_insert_with(1, 4, || Err("load failed".into())).is_err());
        assert_eq!((cache.bytes(), cache.entries()), (0, 0));
        cache.get_or_try_insert_with(1, 4, || Ok(7)).unwrap();
        assert!(cache.get_or_try_insert_with(1, 3, || panic!("extent changed")).is_err());
        assert!(cache.get_or_try_insert_with(2, 0, || panic!("entry budget")).is_err());
        assert_eq!((cache.bytes(), cache.entries()), (4, 1));
        assert!(for_context::<u32, u32>(0, 4, 1).is_err());
    }
}
