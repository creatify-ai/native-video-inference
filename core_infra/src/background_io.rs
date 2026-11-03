//! Cooperative, opt-in host IO. Foreground work can take over retained reads.
use std::{cell::RefCell,path::PathBuf,sync::{Arc,atomic::{AtomicBool,Ordering}},thread,time::Duration};

pub const PAUSED:&str="background host IO paused";
#[derive(Clone)]
pub struct Policy {pub gate:PathBuf,pub cancel:PathBuf,pub stop:Option<Arc<AtomicBool>>,pub pin_gate:Option<PathBuf>}
thread_local! {static CURRENT:RefCell<Option<Policy>>=const{RefCell::new(None)};}
pub struct Scope(Option<Policy>);
impl Scope {pub fn enter(policy:Policy)->Self{Self(CURRENT.with(|slot|slot.replace(Some(policy))))}}
impl Drop for Scope {fn drop(&mut self){CURRENT.with(|slot|slot.replace(self.0.take()));}}
pub fn current()->Option<Policy>{CURRENT.with(|slot|slot.borrow().clone())}
pub struct Control {policy:Policy,foreground:AtomicBool,closed:AtomicBool}
impl Control {
    pub fn current()->Option<Arc<Self>>{current().map(|policy|Arc::new(Self{
        policy,foreground:AtomicBool::new(false),closed:AtomicBool::new(false)}))}
    pub fn foreground(&self,value:bool){self.foreground.store(value,Ordering::Release);}
    pub fn checkpoint(&self)->Result<(),String>{
        if self.closed.load(Ordering::Acquire)||self.policy.cancel.exists()||
            self.policy.stop.as_ref().is_some_and(|s|s.load(Ordering::Acquire)){
            return Err("background host IO cancelled".into());
        }
        if !self.foreground.load(Ordering::Acquire)&&!self.policy.gate.exists(){return Err(PAUSED.into());}
        Ok(())
    }
    pub fn wait(&self)->Result<(),String>{
        loop{match self.checkpoint(){Err(e) if e==PAUSED=>thread::sleep(Duration::from_millis(2)),result=>return result}}
    }
    pub fn registration_checkpoint(&self)->Result<(),String>{
        loop{
            self.checkpoint()?;
            if self.foreground.load(Ordering::Acquire)||self.policy.pin_gate.as_ref().is_none_or(|p|p.exists()){return Ok(());}
            thread::sleep(Duration::from_millis(2));
        }
    }
    pub fn close(&self){self.closed.store(true,Ordering::Release);}
}

#[cfg(feature="cuda")]
thread_local! {static PINS:RefCell<Vec<Arc<crate::host_buffer::HostBuffer>>>=const{RefCell::new(Vec::new())};}
#[cfg(feature="cuda")]
pub fn defer_pin(buffer:Arc<crate::host_buffer::HostBuffer>){PINS.with(|pins|{
    let mut pins=pins.borrow_mut();if !pins.iter().any(|p|Arc::ptr_eq(p,&buffer)){pins.push(buffer);}
});}
pub fn finish_pins()->crate::Result<()>{
    #[cfg(feature="cuda")]
    PINS.with(|pins|->crate::Result<()>{
        let mut pins=pins.borrow_mut();
        while let Some(buffer)=pins.last(){
            let control=Control::current();
            buffer.pin_with(||match &control{Some(control)=>control.registration_checkpoint().map_err(Into::into),None=>Ok(())})?;
            pins.pop();
        }Ok(())
    })?;
    Ok(())
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn foreground_takes_over_a_paused_reader_without_opening_the_background_gate(){
        let root=std::env::temp_dir().join(format!("host-io-{}-{:?}",std::process::id(),thread::current().id()));
        std::fs::create_dir_all(&root).unwrap();
        let _scope=Scope::enter(Policy{gate:root.join("gate"),cancel:root.join("cancel"),stop:None,pin_gate:None});
        let control=Control::current().unwrap();assert_eq!(control.checkpoint().unwrap_err(),PAUSED);
        control.foreground(true);control.checkpoint().unwrap();assert!(!root.join("gate").exists());
        control.foreground(false);std::fs::write(root.join("gate"),b"").unwrap();control.checkpoint().unwrap();
        control.close();assert!(control.wait().unwrap_err().contains("cancelled"));std::fs::remove_dir_all(root).unwrap();
    }
    #[test]fn registration_waits_for_idle_and_yields_to_foreground_arrival(){
        let root=std::env::temp_dir().join(format!("host-pin-{}-{:?}",std::process::id(),thread::current().id()));
        std::fs::create_dir_all(&root).unwrap();std::fs::write(root.join("io"),b"").unwrap();
        let _scope=Scope::enter(Policy{gate:root.join("io"),cancel:root.join("cancel"),stop:None,pin_gate:Some(root.join("idle"))});
        let control=Control::current().unwrap();control.checkpoint().unwrap();
        let waiting=control.clone();let task=thread::spawn(move||waiting.registration_checkpoint());
        thread::sleep(Duration::from_millis(15));assert!(!task.is_finished());
        std::fs::remove_file(root.join("io")).unwrap();assert_eq!(task.join().unwrap().unwrap_err(),PAUSED);
        std::fs::write(root.join("io"),b"").unwrap();std::fs::write(root.join("idle"),b"").unwrap();
        control.registration_checkpoint().unwrap();std::fs::remove_dir_all(root).unwrap();
    }
}
