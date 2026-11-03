//! Transport for an existing owner-thread archive cohort. The cohort publishes
//! verified uploads; the one DeclaredStartup executor schedules constructors,
//! independent preparation, validation and consumer gates around those facts.
use crate::declared_startup::{Bindings,Result};
use crate::startup_owner::OwnerExecutor;
use serde_json::Value;
use std::{cell::RefCell,collections::BTreeMap,sync::{Arc,Condvar,Mutex},time::{Duration,Instant}};

#[derive(Default)]
struct State {values:BTreeMap<String,Value>,closed:bool}
#[derive(Clone)]
pub struct CohortLoads {shared:Arc<(Mutex<State>,Condvar)>,names:Vec<String>,timeout:Duration}
impl CohortLoads {
    /// Bind readiness only. Full declaration admission still occurs before the
    /// transport starts readers, allocates buffers, or invokes preparation.
    /// Each model must supply its ordinary validate_load receipt validator.
    pub fn bind(names:Vec<String>,bindings:&mut Bindings,timeout_seconds:f64)->Result<Self>{
        if names.is_empty() || names.iter().any(|n|n.is_empty()||n.trim()!=n)
            || names.iter().collect::<std::collections::BTreeSet<_>>().len()!=names.len()
            || !timeout_seconds.is_finite() || timeout_seconds<=0. || bindings.has("cancel","") {
            return Err("invalid declared cohort transport".into());
        }
        let result=Self {shared:Arc::new((Mutex::new(State::default()),Condvar::new())),
            names:names.clone(),timeout:Duration::from_secs_f64(timeout_seconds)};
        for name in names {
            if bindings.has("load",&name){return Err("duplicate cohort load binding".into());}
            let (loads,key)=(result.clone(),name.clone());
            bindings.insert("load",&name,Arc::new(move |_|loads.wait(&key)));
        }
        let loads=result.clone();
        bindings.insert("cancel","",Arc::new(move |_|{loads.cancel();Ok(Value::Null)}));
        Ok(result)
    }
    fn wait(&self,name:&str)->Result<Value>{
        let deadline=Instant::now()+self.timeout;
        let (lock,changed)=&*self.shared;
        let mut state=lock.lock().unwrap();
        loop {
            if state.closed{return Err("declared cohort cancelled".into());}
            if let Some(value)=state.values.get(name){return Ok(value.clone());}
            let remaining=deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero(){return Err("declared cohort timed out".into());}
            state=changed.wait_timeout(state,remaining).unwrap().0;
        }
    }
    pub fn publish(&self,name:&str,receipt:Value)->Result<()>{
        if !self.names.iter().any(|n|n==name){return Err("undeclared cohort component".into());}
        let (lock,changed)=&*self.shared;let mut state=lock.lock().unwrap();
        if state.closed || state.values.contains_key(name){return Err("closed or duplicate cohort publication".into());}
        state.values.insert(name.into(),receipt);changed.notify_all();Ok(())
    }
    pub fn cancel(&self){
        let(lock,changed)=&*self.shared;lock.lock().unwrap().closed=true;changed.notify_all();
    }
    /// Drive a legacy physical reader/owner transport without handing startup
    /// policy back to a model. IO continues while core dispatches owner work.
    /// `prepare` is called after readers start; `ready` follows verified upload.
    pub fn drive<T>(&self,runtime:&mut OwnerExecutor<'_>,device:&str,device_receipt:Value,
        transport:impl FnOnce(&mut dyn FnMut()->Result<()>,&mut dyn FnMut(&str,Value)->Result<()>)->Result<T>)->Result<T>{
        runtime.start()?;
        let owner=RefCell::new(runtime);
        let mut prepare=||{
            self.publish(device,device_receipt.clone())?;
            owner.borrow_mut().before_use(None,device)
        };
        let mut ready=|name:&str,receipt:Value|{
            self.publish(name,receipt)?;
            let published=self.shared.0.lock().unwrap().values.keys().cloned().collect::<Vec<_>>();
            owner.borrow_mut().finish_publications(&published)
        };
        let result=transport(&mut prepare,&mut ready);
        match result {
            Ok(value)=>{
                if let Err(error)=owner.borrow_mut().preload(){self.cancel();return Err(error);}
                Ok(value)
            },
            Err(error)=>{self.cancel();let _=owner.borrow_mut().close();Err(error)},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{declared_startup::{Component,Declaration,Preparation,AtUse},startup_owner::OwnerBindings};
    use serde_json::json;
    use std::{cell::Cell,rc::Rc};
    fn component(name:&str,after:&[&str],required:&[&str])->Component {
        Component{name:name.into(),after:after.iter().map(|s|s.to_string()).collect(),
            required_preparations:Some(required.iter().map(|s|s.to_string()).collect()),
            reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false}
    }
    #[test]
    fn owner_construction_is_scheduled_and_validated_before_each_consumer(){
        let mut bindings=Bindings::default();
        let loads=CohortLoads::bind(vec!["device".into(),"encoder".into()],&mut bindings,2.).unwrap();
        for name in ["device","encoder"]{bindings.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));}
        let built=Rc::new(Cell::new(false));let state=built.clone();let owner_thread=std::thread::current().id();
        let mut local=OwnerBindings::default();
        local.insert("run","model",Box::new(move |_|{
            assert_eq!(std::thread::current().id(),owner_thread);state.set(true);Ok(json!(true))
        }));
        bindings.insert("validate_run","model",Arc::new(|v|Ok(json!(v==true))));
        let declaration=Declaration {components:vec![component("device",&[],&[]),component("encoder",&["device"],&["model"])],
            preparations:vec![Preparation{name:"model".into(),before:vec!["encoder".into()],inputs:vec![],
                weights:vec!["encoder".into()],after:vec![],cpu:false,at_use:AtUse::Never,..Preparation::default()}],
            workers:4,reader_budget_mib:1,timeout_seconds:2.,consumers:None};
        let mut runtime=OwnerExecutor::new(declaration,bindings,local).unwrap();
        loads.drive(&mut runtime,"device",json!(true),|prepare,ready|{
            prepare()?;assert!(!built.get());ready("encoder",json!(true))?;Ok(())
        }).unwrap();
        assert!(built.get());
        assert_eq!(runtime.snapshot()["preload_ready"],true);
        runtime.before_use(None,"encoder").unwrap();runtime.close().unwrap();
    }
    #[test]
    fn missing_or_duplicate_components_and_cancellation_fail_closed(){
        let mut bindings=Bindings::default();let loads=CohortLoads::bind(vec!["weights".into()],&mut bindings,1.).unwrap();
        assert!(loads.publish("other",json!(true)).is_err());
        loads.publish("weights",json!(true)).unwrap();assert!(loads.publish("weights",json!(true)).is_err());
        loads.cancel();assert!(bindings.call("load","weights",&Value::Null).is_err());
    }
    #[test]
    fn publication_finishes_released_owner_work_before_transport_blocks_again(){
        let mut bindings=Bindings::default();
        let loads=CohortLoads::bind(vec!["device".into(),"encoder".into(),"dit".into()],&mut bindings,2.).unwrap();
        for name in ["device","encoder","dit"]{
            bindings.insert("validate_load",name,Arc::new(move |v|{
                // Receipt processing is asynchronous and may complete after
                // the transport's immediate nonblocking poll has returned.
                if name=="encoder"{std::thread::sleep(Duration::from_millis(20));}
                Ok(json!(v==true))
            }));
        }
        let built=Rc::new(Cell::new(false));let flag=built.clone();let mut local=OwnerBindings::default();
        local.insert("run","model",Box::new(move |_|{flag.set(true);Ok(json!(true))}));
        bindings.insert("validate_run","model",Arc::new(|v|Ok(json!(v==true))));
        let declaration=Declaration {components:vec![component("device",&[],&[]),
            component("encoder",&["device"],&["model"]),component("dit",&["device"],&[])],
            preparations:vec![Preparation{name:"model".into(),before:vec!["encoder".into()],
                weights:vec!["encoder".into()],after:vec![],inputs:vec![],cpu:false,at_use:AtUse::Never,..Preparation::default()}],
            workers:4,reader_budget_mib:1,timeout_seconds:2.,consumers:None};
        let mut runtime=OwnerExecutor::new(declaration,bindings,local).unwrap();
        loads.drive(&mut runtime,"device",json!(true),|prepare,ready|{
            prepare()?;ready("encoder",json!(true))?;
            assert!(built.get(),"released constructor was left behind while transport resumed blocking IO");
            ready("dit",json!(true))?;Ok(())
        }).unwrap();runtime.close().unwrap();
    }
    #[test]
    fn physical_completions_may_precede_unpublished_load_dependencies(){
        let mut bindings=Bindings::default();
        let loads=CohortLoads::bind(vec!["device".into(),"first".into(),"later".into()],&mut bindings,2.).unwrap();
        for name in ["device","first","later"]{bindings.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));}
        let declaration=Declaration {components:vec![component("device",&[],&[]),
            component("first",&["device"],&[]),component("later",&["first"],&[])],
            preparations:vec![],workers:4,reader_budget_mib:1,timeout_seconds:2.,consumers:None};
        let mut runtime=OwnerExecutor::new(declaration,bindings,OwnerBindings::default()).unwrap();
        loads.drive(&mut runtime,"device",json!(true),|prepare,ready|{
            prepare()?;
            // Waiting for first here would stop the physical transport from
            // ever reaching the upload which releases this dependency.
            ready("later",json!(true))?;ready("first",json!(true))?;Ok(())
        }).unwrap();runtime.close().unwrap();
    }
    #[test]
    fn uploaded_archive_does_not_stop_reader_drain_while_backend_is_pending(){
        let mut bindings=Bindings::default();
        let loads=CohortLoads::bind(vec!["device".into(),"encoder".into(),"dit".into()],&mut bindings,3.).unwrap();
        for name in ["device","encoder","dit"]{
            bindings.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));
        }
        let (release,receive)=std::sync::mpsc::channel();
        let receive=Mutex::new(receive);
        bindings.insert("run","backend",Arc::new(move |_|{
            receive.lock().unwrap().recv_timeout(Duration::from_secs(2))
                .map_err(|_|"reader drain blocked behind independent backend preparation")?;
            Ok(json!(true))
        }));
        bindings.insert("validate_run","backend",Arc::new(|v|Ok(json!(v==true))));
        let declaration=Declaration {components:vec![component("device",&[],&[]),
            component("encoder",&["device"],&["backend"]),component("dit",&["device"],&[])],
            preparations:vec![Preparation{name:"backend".into(),before:vec!["encoder".into()],
                weights:vec!["device".into()],after:vec![],inputs:vec![],cpu:false,at_use:AtUse::Never,..Preparation::default()}],
            workers:4,reader_budget_mib:1,timeout_seconds:3.,consumers:None};
        let mut runtime=OwnerExecutor::new(declaration,bindings,OwnerBindings::default()).unwrap();
        loads.drive(&mut runtime,"device",json!(true),|prepare,ready|{
            prepare()?;
            ready("encoder",json!(true))?;
            // This upload must remain reachable before its peer's backend is ready.
            ready("dit",json!(true))?;
            release.send(()).map_err(|e|e.to_string())?;Ok(())
        }).unwrap();
        runtime.before_use(None,"encoder").unwrap();
        assert_eq!(runtime.snapshot()["preload_ready"],true);runtime.close().unwrap();
    }
}
