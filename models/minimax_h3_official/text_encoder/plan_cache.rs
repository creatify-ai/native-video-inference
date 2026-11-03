//! Bounded epoch cache policy; device plans use the unchanged vendor builder.
use std::collections::BTreeMap;
use crate::Result;
use serde_json::{json,Value};
// Single-rank and head-partitioned attention live in separate vendor caches.
// Include the attention world in their keys; convolution is shared (world 0).
pub struct Cache {pub capacity:usize,pub entries:BTreeMap<(i32,usize,usize),u64>,pub next:u64,pub builds:u64,pub hits:u64,pub evictions:u64,pub epochs:u64}
pub struct Decision {pub clear:bool,pub hits:usize,pub misses:usize,pub ids:Vec<u64>}
impl Cache {
    pub fn new(capacity:usize)->Result<Self>{if !(3..=48).contains(&capacity){return Err("H3_QWEN_PLAN_CACHE_ENTRIES must be 3..48".into());}Ok(Self{capacity,entries:BTreeMap::new(),next:1,builds:0,hits:0,evictions:0,epochs:0})}
    pub fn from_env()->Result<Self>{Self::new(std::env::var("H3_QWEN_PLAN_CACHE_ENTRIES").unwrap_or_else(|_|"12".into()).parse()?)}
    /// Vendor attention plans prepared before the first request must be counted
    /// in the same bounded epoch as request plans. Reserve three entries for an
    /// incoming image request; overflow still clears both host and device caches.
    pub fn register_preloaded_attention(&mut self,shapes:&[(i32,usize)])->Result<Decision>{
        if shapes.iter().any(|(kind,rows)|!(0..=1).contains(kind)||*rows==0){return Err("invalid preloaded attention shape".into());}
        let keys=shapes.iter().map(|(kind,rows)|(*kind,1,*rows)).collect::<std::collections::BTreeSet<_>>();
        if keys.len()>self.capacity-3{return Err("preloaded attention exceeds cache reserve".into());}
        Ok(self.admit_keys(keys.into_iter().collect()))
    }
    pub fn admit(&mut self,vision:usize,text:usize)->Decision {self.admit_images(vision,text,2)}
    pub fn register_prepared_images(&mut self,vision:usize,text:usize,images:usize)->Result<Decision>{
        if images>2 || text==0 || (images>0 && vision==0){return Err("invalid prepared image shape".into());}
        let keys=if images==0{vec![(1,1,text)]}else{vec![(0,1,vision),(1,1,text),(2,0,images*vision)]};
        if self.entries.len()+keys.iter().filter(|key|!self.entries.contains_key(key)).count()>self.capacity{
            return Err("prepared images exceed startup cache reserve".into());
        }
        Ok(self.admit_keys(keys))
    }
    pub fn admit_images(&mut self,vision:usize,text:usize,images:usize)->Decision{
        self.admit_images_for_world(vision,text,images,1)
    }
    pub fn admit_images_for_world(&mut self,vision:usize,text:usize,images:usize,world:usize)->Decision{
        // Conv patch rows depend only on vision sequence; allocations are always
        // CUDA aligned, so the original conv key's pointer-alignment is fixed.
        let keys=if images==0 {vec![(1,world,text)]}else{vec![(0,world,vision),(1,world,text),(2,0,images*vision)]};
        self.admit_keys(keys)
    }
    pub fn admit_references(&mut self,images:&[(usize,usize)],videos:&[(usize,usize)],text:usize)->Result<Decision>{
        self.admit_references_for_world(images,videos,text,1)
    }
    pub fn admit_references_for_world(&mut self,images:&[(usize,usize)],videos:&[(usize,usize)],text:usize,world:usize)->Result<Decision>{
        let mut keys=std::collections::BTreeSet::from([(1,world,text)]);
        for grids in [images,videos] {
            if grids.is_empty(){continue;}
            let mut rows=0usize;
            for &(h,w) in grids {
                let n=h.checked_mul(w).filter(|n|*n>0).ok_or("invalid vision grid")?;
                rows=rows.checked_add(n).ok_or("vision rows overflow")?;
                keys.insert((0,world,n));
            }
            keys.insert((2,0,rows));
        }
        if keys.len()>self.capacity{return Err("reference layout exceeds configured Qwen plan cache capacity".into());}
        Ok(self.admit_keys(keys.into_iter().collect()))
    }
    fn admit_keys(&mut self,keys:Vec<(i32,usize,usize)>)->Decision{
        let absent=keys.iter().filter(|k|!self.entries.contains_key(k)).count();
        let clear=self.entries.len()+absent>self.capacity;
        if clear{self.evictions+=self.entries.len() as u64;self.entries.clear();self.epochs+=1;}
        let mut decision=Decision{clear,hits:0,misses:0,ids:vec![]};
        for key in keys {
            if let Some(id)=self.entries.get(&key){decision.hits+=1;self.hits+=1;decision.ids.push(*id);}
            else{let id=self.next;self.next+=1;self.builds+=1;decision.misses+=1;self.entries.insert(key,id);decision.ids.push(id);}
        }
        decision
    }
    pub fn record(&self,d:&Decision)->Value{json!({"policy":"bounded_epoch_clear_on_capacity", "capacity":self.capacity,
        "entries":self.entries.len(),"builds_total":self.builds,"hits_total":self.hits,"evicted_entries_total":self.evictions,
        "epochs":self.epochs,"cleared_this_request":d.clear,"request_hits":d.hits,"request_misses":d.misses,
        "logical_plan_ids":d.ids,"plan_id_scope":"host-recorded creation identity, not a pointer; original DSO key cache independently counted"})}
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn return_reuses_and_capacity_evicts(){let mut c=Cache::new(6).unwrap();let a=c.admit(2304,1172);let b=c.admit(4032,2200);assert!(!a.clear&&!b.clear);let again=c.admit(2304,1172);assert_eq!(a.ids,again.ids);assert_eq!(again.hits,3);let d=c.admit(4096,2500);assert!(d.clear);assert_eq!(c.entries.len(),3);assert_eq!(c.evictions,6);assert_ne!(c.admit(2304,1172).ids,a.ids);}
 #[test] fn text_only_keeps_vision_and_conv(){let mut c=Cache::new(6).unwrap();let a=c.admit(2304,1000);let b=c.admit(2304,2000);assert_eq!(b.hits,2);assert_eq!(a.ids[0],b.ids[0]);assert_eq!(a.ids[2],b.ids[2]);assert_ne!(a.ids[1],b.ids[1]);}
 #[test] fn mode_changes_track_only_built_plans(){let mut c=Cache::new(12).unwrap();
   let text=c.admit_images(2304,40,0);assert_eq!(text.misses,1);assert_eq!(c.entries.len(),1);
   let one=c.admit_images(2304,620,1);assert_eq!(one.misses,3);
   let two=c.admit_images(2304,1200,2);assert_eq!(two.hits,1);assert_eq!(two.misses,2);
   let revisit=c.admit_images(2304,40,0);assert_eq!(revisit.hits,1);assert_eq!(revisit.ids,text.ids);
 }
 #[test] fn preloaded_plans_reuse_and_follow_the_same_eviction_epoch(){
   let mut c=Cache::new(6).unwrap();
   c.register_preloaded_attention(&[(0,2304),(1,620)]).unwrap();
   let first=c.admit_images(2304,620,1);assert_eq!((first.hits,first.misses,c.entries.len()),(2,1,3));
   c.admit_images(4096,1200,1);
   let changed=c.admit_images(5760,1500,1);assert!(changed.clear);assert_eq!(c.entries.len(),3);
   let revisit=c.admit_images(2304,620,1);assert_eq!(revisit.hits,0);assert_ne!(revisit.ids,first.ids);
 }
 #[test] fn excessive_preload_preserves_existing_cache_and_request_reserve(){
   let mut c=Cache::new(3).unwrap();c.admit_images(2304,620,1);let before=c.entries.clone();
   assert!(c.register_preloaded_attention(&[(1,620)]).is_err());assert_eq!(c.entries,before);
 }
 #[test] fn first_shape_preparation_registers_attention_and_patch_keys(){
   let mut c=Cache::new(12).unwrap();
   c.register_preloaded_attention(&[(0,4032),(1,1092)]).unwrap();
   let before=c.admit_images(4032,1092,1);assert_eq!((before.hits,before.misses),(2,1));
   let prepared=c.register_prepared_images(4128,2092,2).unwrap();assert_eq!(prepared.misses,3);
   let request=c.admit_images(4128,2092,2);assert_eq!((request.hits,request.misses),(3,0));
   assert_eq!(request.ids,prepared.ids);assert!(!request.clear);
 }
 #[test] fn preparation_cannot_evict_a_live_cache_or_register_invalid_shapes(){
   let mut c=Cache::new(3).unwrap();c.admit_images(4032,1092,1);let before=c.entries.clone();
   assert!(c.register_prepared_images(4128,2092,2).is_err());assert_eq!(c.entries,before);
   for (v,t,i) in [(4032,0,1),(0,1092,1),(4032,1092,3)] {
     assert!(c.register_prepared_images(v,t,i).is_err());assert_eq!(c.entries,before);
   }
 }
 #[test] fn one_reference_adopts_the_same_attention_and_patch_plans_as_its_real_forward(){
   let mut c=Cache::new(12).unwrap();
   c.register_preloaded_attention(&[(0,4032),(1,1092)]).unwrap();
   let prepared=c.register_prepared_images(172*96,4148,1).unwrap();
   let actual=c.admit_references(&[(172,96)],&[],4148).unwrap();
   assert_eq!((actual.hits,actual.misses),(3,0));assert_eq!(actual.ids,prepared.ids);
   assert!(!actual.clear);
 }
 #[test] fn invalid_capacity(){assert!(Cache::new(2).is_err());assert!(Cache::new(49).is_err());}
 #[test] fn mixed_references_share_only_identical_operator_shapes(){
   let mut c=Cache::new(12).unwrap();
   let a=c.admit_references(&[(48,48),(48,64)],&[(32,48),(32,48)],2000).unwrap();
   assert_eq!(a.misses,6); // three attention sizes, text, two modality conv batches
   let b=c.admit_references(&[(48,64)],&[],1000).unwrap();
   assert_eq!(b.hits,2); // 3072 vision rows also match the prior video conv batch
   assert_eq!(c.admit_references(&[(48,48),(48,64)],&[(32,48),(32,48)],2000).unwrap().ids,a.ids);
 }
 #[test] fn excessive_reference_plan_count_does_not_mutate_cache(){
   let mut c=Cache::new(3).unwrap();c.admit_images(256,50,1);let previous=c.entries.clone();
   assert!(c.admit_references(&[(16,16),(16,32)],&[],100).is_err());assert_eq!(c.entries,previous);
 }
 #[test] fn fallback_and_parallel_attention_share_only_the_convolution_plan(){
   let mut c=Cache::new(12).unwrap();
   let single=c.admit_images_for_world(2880,797,2,1);
   let parallel=c.admit_images_for_world(2880,1520,2,8);
   // The same vision shape has a full-head plan and a head-partitioned plan.
   assert_eq!((parallel.hits,parallel.misses,c.entries.len()),(1,2,5));
   assert_ne!(single.ids[0],parallel.ids[0]);
   assert_eq!(single.ids[2],parallel.ids[2]);
   let again=c.admit_images_for_world(2880,797,2,1);
   assert_eq!(again.hits,3);assert_eq!(again.ids,single.ids);
 }
 #[test] fn fl_ref_backend_switches_keep_the_physical_cache_bounded(){
   let mut c=Cache::new(6).unwrap();
   c.admit_images_for_world(2880,797,2,1);
   c.admit_references_for_world(&[(48,60),(48,60)],&[],1520,8).unwrap();
   assert_eq!(c.entries.len(),5);
   // Another parallel text length fills capacity; switching to full-head
   // attention at that same length must clear instead of aliasing the plan.
   c.admit_images_for_world(2880,1600,2,8);
   let switch=c.admit_references_for_world(&[(48,60),(48,60)],&[],1600,1).unwrap();
   assert!(switch.clear);assert_eq!(c.entries.len(),3);assert_eq!(c.evictions,6);
 }
}
