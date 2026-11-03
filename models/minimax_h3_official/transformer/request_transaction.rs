//! A serialized outer request may issue several DiT phases/windows.
//! Ordering is checked before CUDA and duplicated by each shard owner.
use crate::Result;
use serde_json::Value;

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Phase { pub id:String, pub index:usize, pub count:usize }
impl Phase {
    pub fn from_request(request:&Value)->Result<Option<Self>> {
        let Some(value)=request.get("cache_transaction") else{return Ok(None);};
        let fields=value.as_object().ok_or("cache transaction must be an object")?;
        let id=value["id"].as_str().ok_or("cache transaction id missing")?;
        let index=usize::try_from(value["index"].as_u64().ok_or("cache transaction index missing")?)?;
        let count=usize::try_from(value["count"].as_u64().ok_or("cache transaction count missing")?)?;
        if fields.len()!=3 || id.is_empty() || id.len()>128 ||
           !id.bytes().all(|b|b.is_ascii_alphanumeric() || b"-_.".contains(&b)) ||
           !(2..=4096).contains(&count) || index>=count {
            return Err("invalid cache transaction identity/order".into());
        }
        Ok(Some(Self{id:id.into(),index,count}))
    }
    pub fn last(&self)->bool{self.index+1==self.count}
}

#[derive(Default)]
pub struct Sequence { active:Option<(String,String,usize,usize)> }
impl Sequence {
    pub fn is_idle(&self)->bool{self.active.is_none()}
    pub fn check(&self,workflow:&str,phase:Option<&Phase>)->Result<()> {
        match (&self.active,phase) {
            (None,None)=>Ok(()),
            (None,Some(p)) if p.index==0=>Ok(()),
            (Some((id,mode,next,count)),Some(p)) if id==&p.id && mode==workflow && *next==p.index && *count==p.count=>Ok(()),
            _=>Err("cache transaction changed mode/id, skipped/replayed a phase, or was not completed".into()),
        }
    }
    pub fn advance(&mut self,workflow:&str,phase:Option<&Phase>)->Result<()> {
        self.check(workflow,phase)?;
        self.active=phase.filter(|p|!p.last()).map(|p|(p.id.clone(),workflow.into(),p.index+1,p.count));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn one_outer_request_owns_all_phases_and_switches_only_after_release() {
        let mut sequence=Sequence::default();
        let mut p=Phase{id:"request-1".into(),index:0,count:3};
        sequence.advance("fl2va",Some(&p)).unwrap();assert!(!sequence.is_idle());
        assert!(sequence.check("fl2va",None).is_err());
        assert!(sequence.check("fl2va",Some(&p)).is_err());
        p.index=2;assert!(sequence.check("fl2va",Some(&p)).is_err());
        p.index=1;assert!(sequence.check("ref2va",Some(&p)).is_err());
        p.id="request-2".into();assert!(sequence.check("fl2va",Some(&p)).is_err());
        p.id="request-1".into();sequence.advance("fl2va",Some(&p)).unwrap();
        p.index=2;sequence.advance("fl2va",Some(&p)).unwrap();assert!(sequence.is_idle());
        assert!(sequence.check("fl2va",Some(&p)).is_err());
        sequence.advance("ref2va",None).unwrap();
    }
    #[test]
    fn transaction_admission_is_bounded_and_explicit() {
        for value in [json!(null),json!({}),json!({"id":"a/b","index":0,"count":2}),
            json!({"id":"a","index":2,"count":2}),json!({"id":"a","index":0,"count":1}),
            json!({"id":"a","index":0,"count":4097}),json!({"id":"a","index":0,"count":2,"x":1})] {
            assert!(Phase::from_request(&json!({"cache_transaction":value})).is_err());
        }
        assert!(Phase::from_request(&json!({})).unwrap().is_none());
        assert_eq!(Phase::from_request(&json!({"cache_transaction":{"id":"req-1","index":1,"count":2}})).unwrap().unwrap().last(),true);
    }
}
