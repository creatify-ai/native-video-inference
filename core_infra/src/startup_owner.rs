//! A thread-affine transport for the same declared executor. CUDA/model objects
//! can remain non-Send on their owner thread. Core dispatches their callbacks
//! and drives waits from notifications; models never implement a polling loop.
use crate::declared_startup::{Bindings, ConsumerGuard, Declaration, DeclaredStartup, Result, ResultHandle};
use serde_json::Value;
use std::collections::{BTreeMap,BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

type LocalWork<'a> = Box<dyn FnMut(&Value) -> Result<Value> + 'a>;
#[derive(Default)]
pub struct OwnerBindings<'a> { work: BTreeMap<(String,String), LocalWork<'a>>, scoped:BTreeSet<(String,String)> }
impl<'a> OwnerBindings<'a> {
    pub fn insert(&mut self, operation:&str, name:&str, work:LocalWork<'a>) {
        self.work.insert((operation.into(),name.into()),work);
    }
    /// Admit a callback whose borrowed request data will be supplied while
    /// driving its result. No request borrow is stored in the resident owner.
    pub fn reserve(&mut self,operation:&str,name:&str)->Result<()> {
        let key=(operation.into(),name.into());
        if self.work.contains_key(&key) || !self.scoped.insert(key) {return Err("duplicate scoped owner callback".into());}
        Ok(())
    }
    fn call(&mut self,operation:&str,name:&str,inputs:&Value)->Result<Value> {
        let work=self.work.get_mut(&(operation.into(),name.into())).ok_or("startup owner callback binding missing")?;
        catch_unwind(AssertUnwindSafe(||work(inputs))).unwrap_or_else(|_|Err("startup owner callback panicked".into()))
    }
}
enum Event {
    Call { operation:String,name:String,inputs:Value,reply:mpsc::Sender<Result<Value>> },
    Changed,
    Closed(Result<()>),
}
pub struct OwnerExecutor<'a> {
    executor:Arc<DeclaredStartup>, local:OwnerBindings<'a>,
    sender:mpsc::Sender<Event>, receiver:mpsc::Receiver<Event>, closed:Option<Result<()>>,
}
impl<'a> OwnerExecutor<'a> {
    pub fn new(declaration:Declaration, mut bindings:Bindings, local:OwnerBindings<'a>) -> Result<Self> {
        let (sender,receiver)=mpsc::channel();
        for (operation,name) in local.work.keys().chain(local.scoped.iter()) {
            if !["load","cpu","initialize","run","validate_load","validate_cpu","validate_initialize","validate_run"].contains(&operation.as_str()) {
                return Err("owner lane only accepts asynchronous work and receipt validators".into());
            }
            if bindings.has(operation,name) { return Err("duplicate owner/shared startup callback".into()); }
            let (tx,op,key)=(sender.clone(),operation.clone(),name.clone());
            bindings.insert(operation,name,Arc::new(move |inputs| {
                let (reply,result)=mpsc::channel();
                tx.send(Event::Call {operation:op.clone(),name:key.clone(),inputs:inputs.clone(),reply})
                    .map_err(|_|"startup owner closed")?;
                result.recv().map_err(|_|"startup owner discarded required work")?
            }));
        }
        let executor=Arc::new(DeclaredStartup::new(declaration,bindings)?);
        let tx=sender.clone();executor.set_waker(Arc::new(move || {let _=tx.send(Event::Changed);}));
        Ok(Self {executor,local,sender,receiver,closed:None})
    }
    pub fn start(&self) -> Result<()> { self.executor.start() }
    pub fn request(&self, scope:&str, inputs:Value) -> Result<BTreeMap<String,ResultHandle>> { self.executor.request(scope,inputs) }
    pub fn snapshot(&self) -> Value { self.executor.snapshot() }
    pub fn elapsed_seconds(&self) -> f64 { self.executor.elapsed_seconds() }
    fn dispatch(&mut self,event:Event) -> Result<()> {
        self.dispatch_with(event,None)
    }
    fn dispatch_with(&mut self,event:Event,scoped:Option<&mut OwnerBindings<'_>>)->Result<()> {
        match event {
            Event::Call {operation,name,inputs,reply} => {
                let _trace=crate::startup_trace::Span::new("startup_owner",format!("{operation}/{name}"),0);
                let result=if self.local.scoped.contains(&(operation.clone(),name.clone())) {
                    match scoped {Some(bindings)=>bindings.call(&operation,&name,&inputs),
                        None=>Err("startup owner callback requires its request bindings".into())}
                }else {
                    self.local.call(&operation,&name,&inputs)
                };
                let _=reply.send(result);Ok(())
            },
            Event::Changed=>Ok(()),
            Event::Closed(result)=>result,
        }
    }
    fn wait(&mut self,deadline:Instant) -> Result<()> {
        let remaining=deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err("startup owner timed out".into()); }
        let event=self.receiver.recv_timeout(remaining).map_err(|e|format!("startup owner wait: {e}"))?;
        self.dispatch(event)
    }
    /// Drive the same declared result with request-local, non-Send callbacks
    /// (for example a borrowed audit observer). Ordering still comes solely
    /// from Component/Preparation dependencies. Every callback stays on owner.
    pub fn wait_result_with(&mut self,handle:ResultHandle,bindings:&mut OwnerBindings<'_>)->Result<Value> {
        if bindings.work.keys().any(|key|!self.local.scoped.contains(key)) || !bindings.scoped.is_empty() {
            return Err("request bindings must fill admitted scoped owner callbacks".into());
        }
        let deadline=Instant::now()+Duration::from_secs_f64(self.executor.declaration().timeout_seconds);
        loop {
            self.executor.poll()?;
            if let Some(result)=self.executor.result(handle,false,Some(0.))? {return Ok(result);}
            let remaining=deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {return Err("startup owner result timed out".into());}
            let event=self.receiver.recv_timeout(remaining).map_err(|e|format!("startup owner result wait: {e}"))?;
            self.dispatch_with(event,Some(&mut *bindings))?;
        }
    }
    /// Dispatch only work already released by the declaration. A transport's
    /// readiness notification is not itself a numerical consumer and must not
    /// wait for a still-running independent preparation or a later upload.
    pub fn poll(&mut self) -> Result<()> {
        self.executor.poll()?;
        while let Ok(event)=self.receiver.try_recv(){self.dispatch(event)?;}
        self.executor.poll()
    }
    pub fn before_use(&mut self,scope:Option<&str>,name:&str) -> Result<()> {
        self.before_use_many(scope,&[name])
    }
    pub fn before_use_many(&mut self,scope:Option<&str>,names:&[&str])->Result<()> {
        let deadline=Instant::now()+Duration::from_secs_f64(self.executor.declaration().timeout_seconds);
        while !self.executor.check_before_use_many(scope,names)? {self.wait(deadline)?;}
        Ok(())
    }
    pub fn invoke<T>(&mut self,scope:Option<&str>,name:&str,compute:impl FnOnce()->Result<T>) -> Result<T> {
        let _consumer=self.enter(scope,name)?;compute()
    }
    pub fn enter(&mut self,scope:Option<&str>,name:&str)->Result<ConsumerGuard> {
        self.before_use(scope,name)?;self.executor.enter(scope,name)
    }
    /// Enter the same consumer gate with borrowed request callbacks. The core
    /// activates the admitted at-use closure; the owner only dispatches work.
    pub fn enter_with(&mut self,scope:Option<&str>,name:&str,bindings:&mut OwnerBindings<'_>)->Result<ConsumerGuard> {
        if bindings.work.keys().any(|key|!self.local.scoped.contains(key)) || !bindings.scoped.is_empty() {
            return Err("request bindings must fill admitted scoped owner callbacks".into());
        }
        let deadline=Instant::now()+Duration::from_secs_f64(self.executor.declaration().timeout_seconds);
        while !self.executor.check_before_use(scope,name)? {
            let remaining=deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero(){return Err("startup owner consumer timed out".into());}
            let event=self.receiver.recv_timeout(remaining).map_err(|e|format!("startup owner consumer wait: {e}"))?;
            self.dispatch_with(event,Some(&mut *bindings))?;
        }
        self.executor.enter(scope,name)
    }
    pub fn enter_many(&mut self,scope:Option<&str>,names:&[&str])->Result<ConsumerGuard> {
        self.before_use_many(scope,names)?;self.executor.enter_many(scope,names)
    }
    pub fn invoke_many<T>(&mut self,scope:Option<&str>,names:&[&str],compute:impl FnOnce()->Result<T>)->Result<T> {
        let _consumer=self.enter_many(scope,names)?;compute()
    }
    pub fn preload(&mut self) -> Result<()> {
        let deadline=Instant::now()+Duration::from_secs_f64(self.executor.declaration().timeout_seconds);
        loop {
            self.executor.poll()?;
            if self.executor.snapshot()["preload_ready"]==true {return Ok(());}
            self.wait(deadline)?;
        }
    }
    pub fn close(&mut self) -> Result<()> {
        if let Some(result)=&self.closed {return result.clone();}
        // Close interrupts native jobs before joining them. Reply to queued
        // owner callbacks with cancellation while that join is in progress.
        let (executor,tx)=(self.executor.clone(),self.sender.clone());
        let closer=std::thread::spawn(move || {
            let result=catch_unwind(AssertUnwindSafe(||executor.close()))
                .unwrap_or_else(|_|Err("startup owner close panicked".into()));
            let _=tx.send(Event::Closed(result));
        });
        let result=loop {
            match self.receiver.recv() {
                Ok(Event::Call {reply,..})=>{let _=reply.send(Err("startup owner cancelled".into()));},
                Ok(Event::Changed)=>{},Ok(Event::Closed(result))=>break result,
                Err(_)=>break Err("startup owner close result missing".into()),
            }
        };
        let result=closer.join().map_err(|_|"startup owner closer panicked".to_string()).and(result);
        self.closed=Some(result.clone());result
    }
}
impl Drop for OwnerExecutor<'_> {fn drop(&mut self){let _=self.close();}}

#[cfg(test)]
mod scoped_consumer_tests {
    use super::*;
    use serde_json::json;
    use std::{rc::Rc,cell::RefCell};
    #[test]
    fn upload_only_scope_retires_untouched_actions_without_running_them() {
        let declaration=serde_json::from_value(json!({"components":[
            {"name":"weights","required_preparations":[]},
            {"name":"dit","after":["weights"],"required_preparations":["layout"]}],
            "preparations":[{"name":"layout","before":["dit"],"weights":["weights"],"inputs":["request"],"at_use":"always"}],
            "workers":2,"timeout_seconds":2})).unwrap();
        let mut shared=Bindings::default();
        for name in ["weights","dit"] {
            shared.insert("load",name,Arc::new(|_|Ok(json!(true))));
            shared.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));
        }
        let mut local=OwnerBindings::default();
        local.reserve("run","layout").unwrap();local.reserve("validate_run","layout").unwrap();
        let mut runtime=OwnerExecutor::new(declaration,shared,local).unwrap();runtime.start().unwrap();
        let old=runtime.request("upload",json!({"request":"upload"})).unwrap();
        drop(runtime.enter(Some("upload"),"weights").unwrap());
        // No request-local binding existed. Only the numerical consumer in
        // the next scope may activate layout, with a fresh result identity.
        runtime.request("inference",json!({"request":"inference"})).unwrap();
        assert!(runtime.executor.result(old["layout"],false,Some(0.)).is_err());
        let mut observed=Vec::new();
        {
            let mut scoped=OwnerBindings::default();let calls=&mut observed;
            scoped.insert("run","layout",Box::new(move |v|{calls.push(v["request"].clone());Ok(json!(true))}));
            scoped.insert("validate_run","layout",Box::new(|v|Ok(json!(v==true))));
            drop(runtime.enter_with(Some("inference"),"dit",&mut scoped).unwrap());
        }
        assert_eq!(observed,vec![json!("inference")]);runtime.close().unwrap();
    }
    #[test]fn scoped_gate_activates_at_use_work_and_rejects_foreign_scope_and_false_receipts(){
        for valid in [false,true] {
            let declaration=serde_json::from_value(json!({"components":[{"name":"dit","required_preparations":["layout"]}],
                "preparations":[{"name":"layout","before":["dit"],"weights":["dit"],"inputs":["request"],"at_use":"always"}],
                "workers":2,"reader_budget_mib":0,"timeout_seconds":2.})).unwrap();
            let mut shared=Bindings::default();shared.insert("load","dit",Arc::new(|_|Ok(json!(true))));
            shared.insert("validate_load","dit",Arc::new(|v|Ok(v.clone())));
            let mut local=OwnerBindings::default();local.reserve("run","layout").unwrap();local.reserve("validate_run","layout").unwrap();
            let mut runtime=OwnerExecutor::new(declaration,shared,local).unwrap();runtime.start().unwrap();
            runtime.request("one",json!({"request":"one"})).unwrap();
            let observed=Rc::new(RefCell::new(Vec::new()));let owner=std::thread::current().id();let copy=observed.clone();
            let mut scoped=OwnerBindings::default();scoped.insert("run","layout",Box::new(move |inputs|{
                assert_eq!(std::thread::current().id(),owner);copy.borrow_mut().push(inputs["request"].clone());Ok(json!(true))
            }));scoped.insert("validate_run","layout",Box::new(move |_|Ok(json!(valid))));
            assert!(runtime.enter_with(Some("other"),"dit",&mut scoped).is_err());assert!(observed.borrow().is_empty());
            if valid {
                let guard=runtime.enter_with(Some("one"),"dit",&mut scoped).unwrap();
                assert_eq!(*observed.borrow(),vec![json!("one")]);assert_eq!(runtime.snapshot()["active_invocations"],1);
                drop(guard);assert_eq!(runtime.snapshot()["active_invocations"],0);
            }else{assert!(runtime.enter_with(Some("one"),"dit",&mut scoped).is_err());assert_eq!(runtime.snapshot()["active_invocations"],0);}
            runtime.close().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn one_serial_lane_consumes_encoders_in_readiness_order_without_a_wait_all_barrier() {
        let declaration:Declaration=serde_json::from_value(json!({"components":[
            {"name":"t5","required_preparations":[]},{"name":"clip","required_preparations":[]},
            {"name":"vae","required_preparations":[]},{"name":"dit","required_preparations":["t5-encode","clip-encode","vae-encode","join"]}],
            "preparations":[
                {"name":"t5-encode","before":["dit"],"inputs":["request"],"weights":["t5"]},
                {"name":"clip-encode","before":["dit"],"inputs":["request"],"weights":["clip"]},
                {"name":"vae-encode","before":["dit"],"inputs":["request"],"weights":["vae"]},
                {"name":"join","before":["dit"],"inputs":["request"],"after":["t5-encode","clip-encode","vae-encode"]}],
            "workers":2,"timeout_seconds":3})).unwrap();
        let (published,receipts)=mpsc::channel();let mut shared=Bindings::default();
        for name in ["t5","clip","vae"] {
            let published=published.clone();shared.insert_async("load",name,Arc::new(move |_,done|{published.send((name,done)).unwrap();}));
        }
        shared.insert("load","dit",Arc::new(|_|Ok(json!(true))));
        for name in ["t5","clip","vae","dit"]{shared.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));}
        shared.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
        let mut local=OwnerBindings::default();
        for name in ["t5-encode","clip-encode","vae-encode","join"]{
            local.reserve("run",name).unwrap();local.reserve("validate_run",name).unwrap();
        }
        let mut runtime=OwnerExecutor::new(declaration,shared,local).unwrap();runtime.start().unwrap();
        let handles=runtime.request("one",json!({"request":"one"})).unwrap();
        let mut pending=BTreeMap::new();
        for _ in 0..3 {let (name,done)=receipts.recv_timeout(Duration::from_secs(2)).unwrap();pending.insert(name,done);}
        let seen=Rc::new(RefCell::new(Vec::new()));let owner=std::thread::current().id();let mut scoped=OwnerBindings::default();
        for name in ["t5-encode","clip-encode","vae-encode","join"]{
            let seen=seen.clone();scoped.insert("run",name,Box::new(move |_|{
                assert_eq!(std::thread::current().id(),owner);seen.borrow_mut().push(name);Ok(json!(true))
            }));
            scoped.insert("validate_run",name,Box::new(|v|Ok(json!(v==true))));
        }
        // Declaration order is deliberately different from weight readiness.
        for (index,name) in ["clip","vae","t5"].into_iter().enumerate(){
            pending.remove(name).unwrap().send(Ok(json!(true)));
            runtime.wait_result_with(handles[&format!("{name}-encode")],&mut scoped).unwrap();
            assert_eq!(&seen.borrow()[..index+1],&["clip-encode","vae-encode","t5-encode"][..index+1]);
        }
        runtime.wait_result_with(handles["join"],&mut scoped).unwrap();
        assert_eq!(*seen.borrow(),vec!["clip-encode","vae-encode","t5-encode","join"]);
        runtime.before_use(Some("one"),"dit").unwrap();runtime.close().unwrap();
    }

    #[test]
    fn request_callbacks_borrow_stack_state_and_finish_on_the_existing_owner() {
        let owner=std::thread::current().id();let mut observed=Vec::new();
        let mut local=OwnerBindings::default();local.reserve("run","layout").unwrap();local.reserve("validate_run","layout").unwrap();
        let mut shared=Bindings::default();
        for (operation,name) in [("load","model"),("cpu","layout"),("validate_load","model"),("validate_cpu","layout")] {
            shared.insert(operation,name,Arc::new(|_|Ok(json!(true))));
        }
        let mut runtime=OwnerExecutor::new(declaration(),shared,local).unwrap();runtime.start().unwrap();
        let handles=runtime.request("one",json!({"request":true})).unwrap();
        {
            let mut scoped=OwnerBindings::default();let observer=&mut observed;
            scoped.insert("run","layout",Box::new(move |_|{assert_eq!(std::thread::current().id(),owner);observer.push("layout");Ok(json!(true))}));
            scoped.insert("validate_run","layout",Box::new(|value|Ok(json!(value==true))));
            assert_eq!(runtime.wait_result_with(handles["layout"],&mut scoped).unwrap(),json!(true));
        }
        assert_eq!(observed,vec!["layout"]);
        runtime.invoke(Some("one"),"model",||Ok(())).unwrap();runtime.close().unwrap();
    }

    #[test]
    fn scoped_preparation_runs_at_its_own_weights_and_assembly_can_finish_before_dit() {
        let declaration=serde_json::from_value(json!({"components":[
            {"name":"t5","required_preparations":[]},{"name":"vae","required_preparations":[]},
            {"name":"dit","required_preparations":["encode-t5","encode-vae","conditioning"]}],
            "preparations":[
                {"name":"encode-t5","before":["dit"],"inputs":["request"],"weights":["t5"]},
                {"name":"encode-vae","before":["dit"],"inputs":["request"],"weights":["vae"]},
                {"name":"conditioning","before":["dit"],"inputs":["request"],"after":["encode-t5","encode-vae"]}],
            "workers":4,"timeout_seconds":2})).unwrap();
        let t5=Arc::new(std::sync::Mutex::new(None));let dit=Arc::new(std::sync::Mutex::new(None));
        let mut shared=Bindings::default();let mut local=OwnerBindings::default();
        for (name,slot) in [("t5",t5.clone()),("dit",dit.clone())] {
            shared.insert_async("load",name,Arc::new(move |_,done|{*slot.lock().unwrap()=Some(done);}));
        }
        shared.insert("load","vae",Arc::new(|_|Ok(json!(true))));
        for name in ["t5","vae","dit"] {shared.insert("validate_load",name,Arc::new(|value|Ok(value.clone())));}
        let (t,d)=(t5.clone(),dit.clone());shared.insert("cancel","",Arc::new(move |_|{t.lock().unwrap().take();d.lock().unwrap().take();Ok(Value::Null)}));
        for name in ["encode-t5","encode-vae","conditioning"] {
            local.reserve("run",name).unwrap();local.reserve("validate_run",name).unwrap();
        }
        let mut runtime=OwnerExecutor::new(declaration,shared,local).unwrap();runtime.start().unwrap();
        let handles=runtime.request("one",json!({"request":true})).unwrap();let observed=Rc::new(RefCell::new(Vec::new()));
        let mut scoped=OwnerBindings::default();
        for name in ["encode-t5","encode-vae","conditioning"] {
            let observed=observed.clone();scoped.insert("run",name,Box::new(move |_|{observed.borrow_mut().push(name);Ok(json!(true))}));
            scoped.insert("validate_run",name,Box::new(|value|Ok(value.clone())));
        }
        runtime.wait_result_with(handles["encode-vae"],&mut scoped).unwrap();
        assert_eq!(*observed.borrow(),vec!["encode-vae"]);
        // The asynchronous submissions may still be queued; drive until both
        // roots have published their pending producers, without releasing IO.
        let deadline=Instant::now()+Duration::from_secs(2);
        while t5.lock().unwrap().is_none() || dit.lock().unwrap().is_none() {
            if Instant::now()>deadline {panic!("independent IO did not start");}std::thread::yield_now();
        }
        t5.lock().unwrap().take().unwrap().send(Ok(json!(true)));
        runtime.wait_result_with(handles["conditioning"],&mut scoped).unwrap();
        assert_eq!(*observed.borrow(),vec!["encode-vae","encode-t5","conditioning"]);
        assert!(!runtime.executor.check_before_use(Some("one"),"dit").unwrap());
        dit.lock().unwrap().take().unwrap().send(Ok(json!(true)));
        runtime.before_use(Some("one"),"dit").unwrap();runtime.close().unwrap();
    }

    fn declaration()->Declaration {
        serde_json::from_value(json!({"components":[{"name":"model","required_preparations":["layout"]}],
            "preparations":[{"name":"layout","before":["model"],"inputs":["request"],"weights":["model"],"cpu":true}],
            "workers":2,"timeout_seconds":2})).unwrap()
    }
    #[test]
    fn non_send_model_state_stays_on_owner_while_cpu_work_runs_in_parallel() {
        let owner=std::thread::current().id();let state=Rc::new(RefCell::new(vec![]));
        let mut local=OwnerBindings::default();
        for operation in ["load","run"] {
            let state=state.clone();local.insert(operation,if operation=="load"{"model"}else{"layout"},Box::new(move |_| {
                assert_eq!(std::thread::current().id(),owner);
                state.borrow_mut().push(operation);Ok(json!(true))
            }));
        }
        let mut shared=Bindings::default();
        for (operation,name) in [("validate_load","model"),("validate_run","layout"),("validate_cpu","layout")] {
            shared.insert(operation,name,Arc::new(|v|Ok(json!(v==true))));
        }
        shared.insert("cpu","layout",Arc::new(move |_|{assert_ne!(std::thread::current().id(),owner);Ok(json!(true))}));
        let mut runtime=OwnerExecutor::new(declaration(),shared,local).unwrap();
        runtime.start().unwrap();runtime.request("one",json!({"request":true})).unwrap();
        runtime.invoke(Some("one"),"model",||{assert_eq!(std::thread::current().id(),owner);Ok(())}).unwrap();
        assert_eq!(*state.borrow(),vec!["load","run"]);runtime.close().unwrap();
    }
    #[test]
    fn closing_before_driving_owner_jobs_joins_every_callback_without_executing_it() {
        let mut shared=Bindings::default();let mut local=OwnerBindings::default();
        for (operation,name) in [("load","model"),("run","layout"),("cpu","layout")] {
            local.insert(operation,name,Box::new(|_|panic!("cancelled owner callback executed")));
        }
        for (operation,name) in [("validate_load","model"),("validate_run","layout"),("validate_cpu","layout")] {
            shared.insert(operation,name,Arc::new(|_|Ok(json!(true))));
        }
        let mut runtime=OwnerExecutor::new(declaration(),shared,local).unwrap();runtime.start().unwrap();
        runtime.request("one",json!({"request":true})).unwrap();runtime.close().unwrap();
        assert!(runtime.before_use(Some("one"),"model").is_err());
    }
    #[test]
    fn owner_failure_and_panic_block_the_actual_consumer() {
        for panic in [false,true] {
            let mut shared=Bindings::default();let mut local=OwnerBindings::default();
            for (operation,name) in [("load","model"),("cpu","layout"),("validate_load","model"),("validate_run","layout"),("validate_cpu","layout")] {
                shared.insert(operation,name,Arc::new(|_|Ok(json!(true))));
            }
            local.insert("run","layout",Box::new(move |_|{if panic {panic!("owner initializer failed");}Err("owner initializer failed".into())}));
            let mut runtime=OwnerExecutor::new(declaration(),shared,local).unwrap();runtime.start().unwrap();
            runtime.request("one",json!({"request":true})).unwrap();
            assert!(runtime.invoke::<()>(Some("one"),"model",||panic!("invalid model executed")).is_err());
            runtime.close().unwrap();
        }
    }
    #[test]
    fn nested_numerical_consumers_keep_scope_active_without_borrowing_the_owner() {
        let mut shared=Bindings::default();
        for (operation,name) in [("load","model"),("run","layout"),("cpu","layout"),
            ("validate_load","model"),("validate_run","layout"),("validate_cpu","layout")] {
            shared.insert(operation,name,Arc::new(|_|Ok(json!(true))));
        }
        let mut runtime=OwnerExecutor::new(declaration(),shared,OwnerBindings::default()).unwrap();
        runtime.start().unwrap();runtime.request("one",json!({"request":true})).unwrap();
        let outer=runtime.enter(Some("one"),"model").unwrap();
        runtime.invoke(Some("one"),"model",||Ok(())).unwrap();
        assert_eq!(runtime.snapshot()["active_invocations"],1);
        assert!(runtime.request("two",json!({"request":true})).is_err());
        drop(outer);runtime.request("two",json!({"request":true})).unwrap();
        runtime.invoke(Some("two"),"model",||Ok(())).unwrap();runtime.close().unwrap();
    }
}


impl OwnerExecutor<'_> {
    /// Settle published load receipts and runnable owner callbacks before an
    /// external transport blocks again. Never join a later upload or a still
    /// unresolved preparation merely because it gates a published component.
    pub(crate) fn finish_publications(&mut self,names:&[String])->Result<()> {
        let _trace=crate::startup_trace::Span::new("declared_cohort_progress","published",0);
        let deadline=Instant::now()+Duration::from_secs_f64(self.executor.declaration().timeout_seconds);
        loop {
            self.poll()?;
            let mut receipts=false;
            for name in names {receipts|=self.executor.published_load_pending(name)?;}
            let callbacks=self.executor.released_owner_pending(|operation,name|
                self.local.work.keys().any(|(op,key)|
                    matches!(op.as_str(),"load"|"cpu"|"run") && op==operation && key==name))?;
            if !receipts && !callbacks {return Ok(());}
            self.wait(deadline)?;
        }
    }
}
