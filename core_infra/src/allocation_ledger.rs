//! Physical allocations owned by a backend, including live and cached workspaces.
//!
//! Register once after allocation succeeds; move the lease with the allocation
//! through its pool and drop it only after the physical allocation is freed.
//! Borrowed parameters belong to their archive and must not acquire a lease.
use crate::Result;
use serde::Serialize;
use std::{collections::BTreeMap, sync::{Arc, Mutex}};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Totals {
    pub owned_bytes: usize,
    pub live_bytes: usize,
    pub cached_bytes: usize,
    pub peak_owned_bytes: usize,
    pub peak_live_bytes: usize,
}

#[derive(Clone, Default)]
pub struct AllocationLedger(Arc<Mutex<BTreeMap<i32, Totals>>>);

pub struct AllocationLease {
    ledger: AllocationLedger,
    device: i32,
    bytes: usize,
    cached: bool,
}

impl AllocationLedger {
    pub fn allocate(&self, device: i32, bytes: usize) -> Result<AllocationLease> {
        if device < 0 || bytes == 0 { return Err("invalid physical allocation extent".into()); }
        let mut rows = self.0.lock().unwrap();
        let row = rows.entry(device).or_default();
        // Do not change either count if the allocation cannot be represented.
        let owned = row.owned_bytes.checked_add(bytes).ok_or("allocation accounting overflow")?;
        let live = row.live_bytes.checked_add(bytes).ok_or("allocation accounting overflow")?;
        row.owned_bytes = owned;
        row.live_bytes = live;
        row.peak_owned_bytes = row.peak_owned_bytes.max(owned);
        row.peak_live_bytes = row.peak_live_bytes.max(live);
        Ok(AllocationLease { ledger: self.clone(), device, bytes, cached: false })
    }

    /// A CPU-only snapshot; reading another rank never changes its CUDA context.
    pub fn snapshot(&self, device: i32) -> Totals {
        self.0.lock().unwrap().get(&device).copied().unwrap_or_default()
    }
}

impl AllocationLease {
    pub fn cache(&mut self) {
        assert!(!self.cached, "allocation is already cached");
        let mut rows = self.ledger.0.lock().unwrap();
        let row = rows.get_mut(&self.device).unwrap();
        row.live_bytes -= self.bytes;
        row.cached_bytes += self.bytes;
        self.cached = true;
    }

    pub fn reuse(&mut self) {
        assert!(self.cached, "allocation is already live");
        let mut rows = self.ledger.0.lock().unwrap();
        let row = rows.get_mut(&self.device).unwrap();
        row.cached_bytes -= self.bytes;
        row.live_bytes += self.bytes;
        row.peak_live_bytes = row.peak_live_bytes.max(row.live_bytes);
        self.cached = false;
    }
}

impl Drop for AllocationLease {
    fn drop(&mut self) {
        let mut rows = self.ledger.0.lock().unwrap();
        let row = rows.get_mut(&self.device).unwrap();
        row.owned_bytes -= self.bytes;
        if self.cached { row.cached_bytes -= self.bytes; } else { row.live_bytes -= self.bytes; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_transfers_preserve_physical_ownership_and_do_not_double_credit() {
        let ledger = AllocationLedger::default();
        let mut first = ledger.allocate(0, 100).unwrap();
        let second = ledger.allocate(0, 60).unwrap();
        first.cache();
        assert_eq!(ledger.snapshot(0), Totals { owned_bytes: 160, live_bytes: 60, cached_bytes: 100,
            peak_owned_bytes: 160, peak_live_bytes: 160 });
        first.reuse();
        drop(second);
        first.cache();
        drop(first); // The pool's physical free, not its logical buffer drop.
        let end = ledger.snapshot(0);
        assert_eq!((end.owned_bytes, end.live_bytes, end.cached_bytes), (0, 0, 0));
        assert_eq!((end.peak_owned_bytes, end.peak_live_bytes), (160, 160));
    }

    #[test]
    fn independent_rank_threads_join_by_device_without_cuda_or_shared_leases() {
        let ledger = AllocationLedger::default();
        std::thread::scope(|scope| {
            let held = ledger.allocate(0, 17).unwrap();
            let other = ledger.clone();
            scope.spawn(move || {
                let mut allocation = other.allocate(1, 23).unwrap();
                allocation.cache();
                assert_eq!(other.snapshot(0).owned_bytes, 17);
                assert_eq!(other.snapshot(1).cached_bytes, 23);
            }).join().unwrap();
            assert_eq!(ledger.snapshot(1).owned_bytes, 0);
            assert_eq!(ledger.snapshot(0).live_bytes, 17);
            drop(held);
        });
    }

    #[test]
    fn invalid_extents_and_overflow_do_not_change_existing_ownership() {
        let ledger = AllocationLedger::default();
        assert!(ledger.allocate(-1, 1).is_err());
        assert!(ledger.allocate(0, 0).is_err());
        let held = ledger.allocate(0, usize::MAX).unwrap();
        assert!(ledger.allocate(0, 1).is_err());
        assert_eq!(ledger.snapshot(0).owned_bytes, usize::MAX);
        drop(held);
        assert_eq!(ledger.snapshot(0).owned_bytes, 0);
    }
}
