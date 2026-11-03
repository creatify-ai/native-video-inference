//! The complete, model-independent startup policy. Models supply work and
//! validators; this module owns admission, dependency release, request scopes,
//! bounded execution, cancellation and consumer readiness. No CUDA dependency.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub type Result<T> = std::result::Result<T, String>;
pub type Work = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;
/// An existing transport submits work and publishes its one validated result
/// without reserving an executor worker for the duration of native I/O.
pub type AsyncWork = Arc<dyn Fn(&Value, crate::startup_result::ResultSender<Value>) + Send + Sync>;

/// Bind numerical/transport functions without giving the model a scheduler.
#[derive(Clone, Default)]
pub struct Bindings { work: BTreeMap<(String, String), Work>, asynchronous:BTreeMap<(String,String),AsyncWork> }
impl Bindings {
    pub fn insert(&mut self, operation: &str, name: &str, work: Work) {
        assert!(!self.asynchronous.contains_key(&(operation.into(),name.into())),"duplicate startup callback");
        self.work.insert((operation.into(), name.into()), work);
    }
    /// The transport must also bind synchronous `cancel/` to join its producers.
    pub fn insert_async(&mut self,operation:&str,name:&str,work:AsyncWork) {
        assert!(!self.has(operation,name),"duplicate startup callback");
        self.asynchronous.insert((operation.into(),name.into()),work);
    }
    pub fn has(&self, operation: &str, name: &str) -> bool {
        self.work.contains_key(&(operation.into(), name.into()))
            || self.asynchronous.contains_key(&(operation.into(),name.into()))
    }
    pub fn call(&self, operation: &str, name: &str, inputs: &Value) -> Result<Value> {
        let work = self.work.get(&(operation.into(), name.into()))
            .ok_or_else(|| format!("missing startup callback: {operation}/{name}"))?;
        catch_unwind(AssertUnwindSafe(|| work(inputs)))
            .map_err(|_| format!("startup callback panicked: {operation}/{name}"))?
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Component {
    pub name: String,
    /// Load dependencies: preceding components or validated static preparations.
    #[serde(default)] pub after: Vec<String>,
    /// Omission is different from explicitly requiring no preparation.
    pub required_preparations: Option<Vec<String>>,
    #[serde(default)] pub reader_limit_mib: usize,
    #[serde(default)] pub prefetch: bool,
    /// Start this loader only when a consumer needs its dependency closure.
    /// CPU readers can be a separate eager component without allocating VRAM.
    #[serde(default)] pub load_at_use: bool,
    /// Validate an early native initializer before final load dependencies.
    /// It may start owned readers/device/uploads; it never publishes weights.
    #[serde(default)] pub initialize: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AtUse { #[default] Never, Always, Predicate }

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Preparation {
    pub name: String,
    pub before: Vec<String>,
    #[serde(default)] pub inputs: Vec<String>,
    #[serde(default)] pub weights: Vec<String>,
    #[serde(default)] pub after: Vec<String>,
    #[serde(default)] pub cpu: bool,
    #[serde(default)] pub at_use: AtUse,
    /// Ready actions in one named resource group execute serially, including
    /// initialization and receipt validation. Waiting actions take no worker.
    #[serde(default)] pub serial_group: Option<String>,
    /// Invoke and validate initialize/<name> immediately before run/<name>,
    /// within the same resource reservation. Model callbacks ensure/cache
    /// their numerical objects; core owns when this work may execute.
    #[serde(default)] pub initialize: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Declaration {
    pub components: Vec<Component>,
    pub preparations: Vec<Preparation>,
    #[serde(default = "workers")] pub workers: usize,
    #[serde(default = "reader_budget")] pub reader_budget_mib: usize,
    #[serde(default = "timeout")] pub timeout_seconds: f64,
    #[serde(default)] pub consumers: Option<Vec<String>>,
}
fn workers() -> usize { 8 }
fn reader_budget() -> usize { 6144 }
fn timeout() -> f64 { 900.0 }
fn unique(values: &[String]) -> bool {
    values.iter().all(|n| !n.is_empty() && n.trim() == n)
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

impl Declaration {
    /// Admit the *whole* declaration before projecting a registered capability.
    /// No owner, worker, file or model callback is created during admission.
    pub fn admit(mut self, bindings: &Bindings) -> Result<Self> {
        if !bindings.asynchronous.is_empty() && !bindings.work.contains_key(&("cancel".into(),String::new())) {
            return Err("asynchronous startup transport must join its producers on cancel".into());
        }
        if bindings.asynchronous.keys().any(|(operation,_)| !["load","cpu","run","prefetch","component_initialize"].contains(&operation.as_str())) {
            return Err("startup validators and cancellation must be synchronous".into());
        }
        if self.components.is_empty() || self.workers == 0 || self.workers > 256
            || !self.timeout_seconds.is_finite() || self.timeout_seconds <= 0.0 {
            return Err("invalid startup resources".into());
        }
        let names: Vec<_> = self.components.iter().map(|c| c.name.clone()).collect();
        let preparations: Vec<_> = self.preparations.iter().map(|p| p.name.clone()).collect();
        if !unique(&names) || !unique(&preparations) {
            return Err("startup needs unique components and preparations".into());
        }
        let mut seen = BTreeSet::new();
        let mut readers = 0usize;
        for c in &self.components {
            if !unique(&c.after) || c.after.iter().any(|n| {
                if names.contains(n) { !seen.contains(n) }
                else { !self.preparations.iter().any(|p| &p.name==n && p.inputs.is_empty()) }
            }) {
                return Err("startup dependencies must precede their component".into());
            }
            seen.insert(c.name.clone());
            let required = c.required_preparations.as_ref()
                .ok_or_else(|| format!("component must explicitly declare required preparations: {}", c.name))?;
            if !unique(required) || required.iter().any(|n| !preparations.contains(n)) {
                return Err(format!("missing or invalid required startup preparation: {}", c.name));
            }
            for operation in ["load", "validate_load"] {
                if !bindings.has(operation, &c.name) { return Err(format!("missing startup callback: {operation}/{}", c.name)); }
            }
            if c.initialize {
                for operation in ["component_initialize", "validate_component_initialize"] {
                    if !bindings.has(operation, &c.name) { return Err(format!("missing startup callback: {operation}/{}", c.name)); }
                }
            }
            if c.prefetch && !bindings.has("prefetch", &c.name) {
                return Err(format!("missing startup prefetch callback: {}", c.name));
            }
            readers = readers.checked_add(c.reader_limit_mib).ok_or("reader budget overflow")?;
        }
        if readers > self.reader_budget_mib { return Err("component readers exceed container budget".into()); }
        let mut preceding = BTreeSet::new();
        for p in &self.preparations {
            // An explicitly released at-use preparation can be independent
            // of numerical consumers (e.g. optional host assets after first
            // compute). It stays dormant without occupying a worker.
            if (p.before.is_empty() && p.at_use != AtUse::Always) || !unique(&p.before) || p.before.iter().any(|n| !names.contains(n))
                || !unique(&p.inputs) || p.inputs.iter().any(|n| n == "prepared" || n == "preparations" || (p.initialize && n=="initialized"))
                || !unique(&p.weights) || p.weights.iter().any(|n| !names.contains(n))
                || !unique(&p.after) || p.after.iter().any(|n| !preceding.contains(n))
                || (p.inputs.is_empty() && p.after.iter().any(|n| !self.preparations.iter().find(|d| &d.name == n).unwrap().inputs.is_empty()))
                || (p.inputs.is_empty() && p.at_use == AtUse::Predicate) {
                return Err(format!("invalid declared startup preparation: {}", p.name));
            }
            if p.serial_group.as_ref().is_some_and(|g|g.is_empty() || g.trim()!=g) {
                return Err(format!("invalid startup serial group: {}",p.name));
            }
            if p.initialize {
                for operation in ["initialize","validate_initialize"] {
                    if !bindings.work.contains_key(&(operation.into(),p.name.clone())) {
                        return Err(format!("missing synchronous startup callback: {operation}/{}",p.name));
                    }
                }
            }
            preceding.insert(p.name.clone());
            let actual: BTreeSet<_> = self.components.iter()
                .filter(|c| c.required_preparations.as_ref().unwrap().contains(&p.name))
                .map(|c| c.name.clone()).collect();
            if actual != p.before.iter().cloned().collect() {
                return Err(format!("preparation consumers differ from the component declaration: {}", p.name));
            }
            for operation in ["run", "validate_run"].into_iter()
                .chain(if p.cpu { vec!["cpu", "validate_cpu"] } else { vec![] })
                .chain(if p.at_use == AtUse::Predicate { vec!["at_use"] } else { vec![] }) {
                if !bindings.has(operation, &p.name) { return Err(format!("missing startup callback: {operation}/{}", p.name)); }
            }
        }
        self.validate_load_graph(&names)?;
        if let Some(consumers) = &self.consumers {
            if consumers.is_empty() || !unique(consumers) || consumers.iter().any(|n| !names.contains(n)) {
                return Err("declare existing unique startup consumers".into());
            }
            let mut selected: BTreeSet<_> = consumers.iter().cloned().collect();
            loop {
                let previous = selected.clone();
                for c in &self.components {
                    if previous.contains(&c.name) {
                        selected.extend(c.after.iter().filter(|n|names.contains(n)).cloned());
                        let ids = self.preparation_closure(c.required_preparations.as_ref().unwrap().iter()
                            .chain(c.after.iter().filter(|n|!names.contains(n))).cloned().collect());
                        for p in &self.preparations {
                            if ids.contains(&p.name) {
                                selected.extend(p.weights.iter().cloned());
                            }
                        }
                    }
                }
                if selected == previous { break; }
            }
            self.components.retain(|c| selected.contains(&c.name));
            let ids = self.preparation_closure(self.components.iter().flat_map(|c|
                c.required_preparations.as_ref().unwrap().iter()
                    .chain(c.after.iter().filter(|n|!names.contains(n))).cloned()).collect());
            self.preparations.retain(|p| ids.contains(&p.name));
            for p in &mut self.preparations { p.before.retain(|n| selected.contains(n)); }
        }
        Ok(self)
    }
    fn validate_load_graph(&self,components:&[String])->Result<()> {
        // Gate dependencies are separate from load dependencies. Only the
        // latter can form a new cycle through a preparation's weight inputs.
        let load=|n:&str|format!("load/{n}");
        let prepare=|n:&str|format!("prepare/{n}");
        let mut graph:BTreeMap<String,BTreeSet<String>>=BTreeMap::new();
        for c in &self.components {
            graph.insert(load(&c.name),c.after.iter().map(|n|
                if components.contains(n){load(n)}else{prepare(n)}).collect());
        }
        for p in self.preparations.iter().filter(|p|p.inputs.is_empty()) {
            graph.insert(prepare(&p.name),p.weights.iter().map(|n|load(n))
                .chain(p.after.iter().map(|n|prepare(n))).collect());
        }
        while !graph.is_empty() {
            let ready:BTreeSet<_>=graph.iter().filter(|(_,deps)|deps.is_empty()).map(|(n,_)|n.clone()).collect();
            if ready.is_empty(){return Err("startup load/preparation dependencies contain a cycle".into());}
            graph.retain(|n,_|!ready.contains(n));
            for dependencies in graph.values_mut(){dependencies.retain(|n|!ready.contains(n));}
        }
        Ok(())
    }
    fn preparation_closure(&self, mut ids: BTreeSet<String>) -> BTreeSet<String> {
        loop {
            let previous=ids.clone();
            for p in &self.preparations { if previous.contains(&p.name) { ids.extend(p.after.iter().cloned()); } }
            if ids==previous { return ids; }
        }
    }
}

#[derive(Clone, Debug)]
enum Status { Waiting, Running, Validating, Ready(Value), Failed(String), Cancelled }
impl Status {
    fn done(&self) -> bool { !matches!(self, Self::Waiting | Self::Running | Self::Validating) }
    /// Dependency checks borrow readiness; result payloads move across a
    /// boundary only when a consumer actually asks for the value.
    fn ready(&self) -> Result<bool> {
        match self {
            Self::Ready(_)=>Ok(true),Self::Failed(error)=>Err(error.clone()),
            Self::Cancelled=>Err("startup work cancelled".into()),_=>Ok(false),
        }
    }
    fn result(&self) -> Result<Option<Value>> {
        match self {
            Self::Ready(v) => Ok(Some(v.clone())), Self::Failed(e) => Err(e.clone()),
            Self::Cancelled => Err("startup work cancelled".into()), _ => Ok(None),
        }
    }
}
struct Node {
    name: String, operation: &'static str, inputs: Value, dependencies: Vec<usize>,
    prepared: Option<usize>, at_use: bool, used: bool, optional: bool, status: Status, generation: u64,
    preparations: BTreeMap<String, usize>,
    serial_group:Option<String>,initialize:bool,
    created_at:Instant,completed_at:Option<Instant>,used_at:Option<Instant>,
}
/// Generational result identity: a retired request can never name a new result.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ResultHandle { pub node: usize, pub generation: u64 }
#[derive(Clone, Serialize)]
pub struct Event { pub component: String, pub phase: String, pub elapsed_seconds: f64 }
struct State {
    started: bool, workers_ready: bool, closed: bool, nodes: Vec<Node>, weights: BTreeMap<String, usize>,
    static_work: BTreeMap<String, usize>, scope: Option<String>, request_work: BTreeMap<String, usize>,
    events: Vec<Event>, warnings: BTreeMap<String, String>,
    permanent_nodes: usize, generation: u64,
    active_invocations: usize,
}
struct Shared {
    state: Mutex<State>, changed: Condvar, started: Instant,
    waker: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}
impl Shared {
    fn notify(&self) {
        self.changed.notify_all();
        if let Some(wake)=&*self.waker.lock().unwrap() { wake(); }
    }
}
/// An admitted numerical call. Its owned token keeps the request active while
/// a nested consumer drives thread-affine preparation on the same owner lane.
pub struct ConsumerGuard { shared: Arc<Shared>, _caller: std::marker::PhantomData<std::rc::Rc<()>> }
impl Drop for ConsumerGuard {
    fn drop(&mut self) {
        let mut state=self.shared.state.lock().unwrap();state.active_invocations-=1;
        self.shared.notify();
    }
}

/// One execution policy for in-process functions and native process transports.
/// Waiting nodes occupy no worker. Model callbacks never advance stages or join.
pub struct DeclaredStartup {
    declaration: Declaration, bindings: Arc<Bindings>, shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>, request_lock: Mutex<()>,
    close_lock: Mutex<()>,
}
impl DeclaredStartup {
    pub fn new(declaration: Declaration, bindings: Bindings) -> Result<Self> {
        let declaration = declaration.admit(&bindings)?;
        crate::startup_trace::configure_from_env();
        Ok(Self { declaration, bindings: Arc::new(bindings),
            shared: Arc::new(Shared { state: Mutex::new(State { started: false, workers_ready: false, closed: false,
                nodes: vec![], weights: BTreeMap::new(), static_work: BTreeMap::new(),
                scope: None, request_work: BTreeMap::new(), events: vec![], warnings: BTreeMap::new(),
                permanent_nodes: 0, generation: 0, active_invocations: 0 }),
                changed: Condvar::new(), started: Instant::now(), waker: Mutex::new(None) }),
            workers: Mutex::new(vec![]), request_lock: Mutex::new(()), close_lock: Mutex::new(()) })
    }
    pub fn declaration(&self) -> &Declaration { &self.declaration }
    pub fn elapsed_seconds(&self) -> f64 { self.shared.started.elapsed().as_secs_f64() }
    /// Owner-lane transports use this notification to drive thread-affine work.
    /// The framework waker must enqueue an event without calling the executor.
    pub(crate) fn set_waker(&self, wake: Arc<dyn Fn() + Send + Sync>) { *self.shared.waker.lock().unwrap()=Some(wake); }
    fn event(shared: &Shared, state: &mut State, name: &str, phase: &str) {
        if state.events.len() >= 4096 { state.events.drain(..2048); }
        state.events.push(Event { component: name.into(), phase: phase.into(), elapsed_seconds: shared.started.elapsed().as_secs_f64() });
    }
    fn node(state: &mut State, name: &str, operation: &'static str, inputs: Value,
            dependencies: Vec<usize>, prepared: Option<usize>, at_use: bool, optional: bool) -> usize {
        let id = state.nodes.len();
        state.nodes.push(Node { name: name.into(), operation, inputs, dependencies, prepared,
            at_use, used: false, optional, status: Status::Waiting, generation: state.generation,
            preparations: BTreeMap::new(),serial_group:None,initialize:false,
            created_at:Instant::now(),completed_at:None,used_at:None });
        id
    }
    fn preparation(state: &mut State, p: &Preparation, inputs: Value, at_use: bool) -> usize {
        let cpu = if p.cpu { Some(Self::node(state, &p.name, "cpu", inputs.clone(), vec![], None, false, false)) } else { None };
        let mut dependencies: Vec<_> = p.weights.iter().map(|n| state.weights[n]).collect();
        let preparations: BTreeMap<_,_> = p.after.iter().map(|n| (n.clone(),
            *state.request_work.get(n).or_else(||state.static_work.get(n)).expect("admitted preparation dependency"))).collect();
        dependencies.extend(preparations.values().copied());
        dependencies.extend(cpu);
        let id=Self::node(state, &p.name, "run", inputs, dependencies, cpu, at_use, false);
        state.nodes[id].preparations=preparations;
        state.nodes[id].serial_group=p.serial_group.clone();state.nodes[id].initialize=p.initialize;id
    }
    pub fn start(&self) -> Result<()> {
        let _trace=crate::startup_trace::Span::new("declared_startup_setup","workers",0);
        let mut state = self.shared.state.lock().unwrap();
        if state.started || state.closed { return Err("declared startup can only start once while open".into()); }
        state.started = true;
        // Weight IDs are admitted together; execution order comes only from edges.
        for c in &self.declaration.components {
            let id = Self::node(&mut state, &c.name, "load", json!({}), vec![], None, c.load_at_use, false);
            state.weights.insert(c.name.clone(), id);
        }
        for c in &self.declaration.components {
            let id = state.weights[&c.name];
            if c.prefetch { Self::node(&mut state, &c.name, "prefetch", json!({}), vec![], None, false, true); }
        }
        for p in self.declaration.preparations.iter().filter(|p| p.inputs.is_empty()) {
            let id = Self::preparation(&mut state, p, json!({}), p.at_use == AtUse::Always);
            state.static_work.insert(p.name.clone(), id);
        }
        let mut initializers=BTreeMap::new();
        for c in self.declaration.components.iter().filter(|c|c.initialize) {
            let id=Self::node(&mut state,&c.name,"component_initialize",json!({}),vec![],None,c.load_at_use,false);
            initializers.insert(c.name.clone(),id);
        }
        for c in &self.declaration.components {
            let id=state.weights[&c.name];
            state.nodes[id].dependencies=c.after.iter().map(|n|
                *state.weights.get(n).or_else(||state.static_work.get(n)).expect("admitted load dependency")).collect();
            state.nodes[id].dependencies.extend(initializers.get(&c.name));
        }
        state.permanent_nodes = state.nodes.len();
        drop(state);
        let mut workers = self.workers.lock().unwrap();
        for index in 0..self.declaration.workers {
            let shared = self.shared.clone(); let bindings = self.bindings.clone();
            match thread::Builder::new().name(format!("declared-startup-{index}")).spawn(move || Self::worker(shared, bindings)) {
                Ok(worker) => workers.push(worker),
                Err(error) => { drop(workers); self.close()?; return Err(format!("startup worker: {error}")); }
            }
        }
        // A first callback may start readers that fault large host allocations.
        // Keep that work from racing creation of the rest of its executor cohort.
        drop(workers);
        self.shared.state.lock().unwrap().workers_ready=true;
        self.shared.notify();
        Ok(())
    }
    fn worker(shared: Arc<Shared>, bindings: Arc<Bindings>) {
        loop {
            let (id, generation, operation, name, mut inputs, optional,initialize) = {
                let mut state = shared.state.lock().unwrap();
                loop {
                    if state.closed { return; }
                    if !state.workers_ready {
                        state=shared.changed.wait(state).unwrap();
                        continue;
                    }
                    let mut selected = None;
                    // Metadata and independent asset admission have priority;
                    // blocked dependency nodes never reserve a worker.
                    let order = (0..state.nodes.len()).filter(|&i| state.nodes[i].operation != "load")
                        .chain((0..state.nodes.len()).filter(|&i| state.nodes[i].operation == "load")).collect::<Vec<_>>();
                    for &id in &order {
                        let node = &state.nodes[id];
                        if !matches!(node.status, Status::Waiting) { continue; }
                        let mut pending = node.at_use && !node.used; let mut error = None;
                        for dependency in &node.dependencies {
                            match state.nodes[*dependency].status.ready() {
                                Err(e) => { error = Some(e); break; }, Ok(false) => pending = true, _ => {},
                            }
                        }
                        if let Some(error) = error {
                            state.nodes[id].status = Status::Failed(error);
                            shared.notify();
                            continue;
                        }
                        if !pending && !Self::group_busy(&state,id) { selected = Some(id); break; }
                    }
                    // Preserve the existing policy for ungrouped work. Within
                    // a chosen serial group, use actual dependency completion
                    // times, even when all workers were busy as weights arrived.
                    if let Some(id)=selected {
                        if let Some(group)=&state.nodes[id].serial_group {
                            selected=order.iter().copied().filter(|&other| {
                                let n=&state.nodes[other];
                                n.serial_group.as_ref()==Some(group) && matches!(n.status,Status::Waiting)
                                    && (!n.at_use||n.used) && n.dependencies.iter().all(|&d|matches!(state.nodes[d].status,Status::Ready(_)))
                            }).min_by_key(|&other|(Self::ready_at(&state,other),other));
                        }
                    }
                    if let Some(id) = selected {
                        let node = &state.nodes[id];
                        let (operation, name, optional,generation,initialize) = (node.operation, node.name.clone(), node.optional,node.generation,node.initialize);
                        let mut inputs = node.inputs.clone();
                        if let Some(cpu) = node.prepared {
                            inputs["prepared"] = state.nodes[cpu].status.result().unwrap().unwrap();
                        }
                        if !node.preparations.is_empty() {
                            inputs["preparations"] = node.preparations.iter().map(|(name,id)|
                                (name.clone(),state.nodes[*id].status.result().unwrap().unwrap())).collect();
                        }
                        state.nodes[id].status = Status::Running;
                        let phase = match operation { "load" => "start", "component_initialize"=>"initialize_start", "cpu" => "cpu_prepare_start", "prefetch" => "prefetch_start", _ => "prepare_start" };
                        Self::event(&shared, &mut state, &name, phase);
                        break (id, generation, operation, name, inputs, optional,initialize);
                    }
                    state = shared.changed.wait(state).unwrap();
                }
            };
            let kind=match operation {"load"=>"declared_load","component_initialize"=>"declared_component_initialize","cpu"=>"declared_cpu",
                "prefetch"=>"declared_prefetch",_=>"declared_prepare"};
            let asynchronous=bindings.asynchronous.get(&(operation.into(),name.clone())).cloned();
            let _sync_trace=if asynchronous.is_none(){Some(crate::startup_trace::Span::new(kind,&name,0))}else{None};
            let async_trace=if asynchronous.is_some(){Some(crate::startup_trace::AsyncSpan::new(kind,&name))}else{None};
            if initialize {
                let initialized={
                    let _trace=crate::startup_trace::Span::new("declared_initialize",&name,0);
                    bindings.call("initialize",&name,&inputs).and_then(|receipt| {
                        if bindings.call("validate_initialize",&name,&receipt)?!=Value::Bool(true) {
                            return Err(format!("startup initializer receipt rejected: {name}"));
                        }
                        Ok(receipt)
                    })
                };
                match initialized {Ok(receipt)=>inputs["initialized"]=receipt,Err(error)=>{
                    Self::finish(&shared,&bindings,id,generation,operation,&name,optional,Err(error));continue;
                }}
                if shared.state.lock().unwrap().closed {continue;}
            }
            if let Some(work)=asynchronous {
                let (sender,receiver)=crate::startup_result::channel();
                let (target,callbacks,label)=(shared.clone(),bindings.clone(),name.clone());
                receiver.subscribe(move |result| {
                    Self::finish(&target,&callbacks,id,generation,operation,&label,optional,result);
                    drop(async_trace);
                });
                if catch_unwind(AssertUnwindSafe(||work(&inputs,sender))).is_err() {
                    Self::finish(&shared,&bindings,id,generation,operation,&name,optional,
                        Err(format!("startup asynchronous callback panicked: {operation}/{name}")));
                }
            } else {
                let result=bindings.call(operation,&name,&inputs);
                Self::finish(&shared,&bindings,id,generation,operation,&name,optional,result);
            }
        }
    }
    fn group_busy(state:&State,id:usize)->bool {
        let Some(group)=&state.nodes[id].serial_group else{return false;};
        state.nodes.iter().any(|n|n.serial_group.as_ref()==Some(group) && matches!(n.status,Status::Running|Status::Validating))
    }
    fn ready_at(state:&State,id:usize)->Instant {
        let node=&state.nodes[id];
        let mut ready=node.created_at;
        for &dependency in &node.dependencies {if let Some(done)=state.nodes[dependency].completed_at{ready=ready.max(done);}}
        if node.at_use {if let Some(used)=node.used_at{ready=ready.max(used);}}
        ready
    }
    fn finish(shared:&Shared,bindings:&Bindings,id:usize,generation:u64,operation:&str,name:&str,optional:bool,result:Result<Value>) {
        // A cancelled or already completed producer cannot validate or publish
        // into a replacement request. Validation runs outside the state lock.
        {
            let mut state=shared.state.lock().unwrap();
            if state.closed || !state.nodes.get(id).is_some_and(|n|n.generation==generation && matches!(n.status,Status::Running)) {return;}
            state.nodes[id].status=Status::Validating;
        }
        let result=result.and_then(|value| {
            if !optional && bindings.call(&format!("validate_{operation}"),name,&value)?!=Value::Bool(true) {
                return Err(format!("startup {operation} receipt rejected: {name}"));
            }
            Ok(value)
        });
        let mut state=shared.state.lock().unwrap();
        if state.closed || !state.nodes.get(id).is_some_and(|n|n.generation==generation && matches!(n.status,Status::Validating)) {return;}
        match result {
            Ok(value)=>{
                state.nodes[id].completed_at=Some(Instant::now());
                state.nodes[id].status=Status::Ready(value);
                let phase=match operation {"load"=>"ready","component_initialize"=>"initialize_ready","cpu"=>"cpu_prepare_ready","prefetch"=>"prefetch_ready",_=>"prepare_ready"};
                Self::event(shared,&mut state,name,phase);
            },
            Err(error) if optional=>{state.warnings.insert(name.into(),error);state.nodes[id].completed_at=Some(Instant::now());state.nodes[id].status=Status::Ready(Value::Null);},
            Err(error)=>state.nodes[id].status=Status::Failed(error),
        }
        shared.notify();
    }
    pub fn request(&self, scope: &str, facts: Value) -> Result<BTreeMap<String, ResultHandle>> {
        let _request = self.request_lock.lock().unwrap();
        if scope.is_empty() { return Err("request scope must be nonempty".into()); }
        let expected: BTreeSet<_> = self.declaration.preparations.iter().flat_map(|p| p.inputs.iter().cloned()).collect();
        if facts.as_object().map(|v| v.keys().cloned().collect::<BTreeSet<_>>()) != Some(expected) {
            return Err("request inputs differ from the startup declaration".into());
        }
        {
            let state = self.shared.state.lock().unwrap();
            if !state.started || state.closed { return Err("request preparation requires an open started plan".into()); }
            if state.active_invocations != 0 { return Err("previous request computation is still active".into()); }
            if state.scope.is_some() {
                if state.scope.as_deref() == Some(scope) { return Err("request scope cannot be reused".into()); }
                for id in state.static_work.values().chain(state.request_work.values()) {
                    let node=&state.nodes[*id];
                    let independent=self.declaration.preparations.iter().any(|p|
                        p.name==node.name && p.inputs.is_empty()
                            && p.before.is_empty() && p.at_use==AtUse::Always);
                    // Independent owner-lifetime work is not a request barrier,
                    // but its failed/cancelled result must still propagate.
                    if independent { let _=node.status.ready()?;continue; }
                    // Unentered at-use actions own no compute. Their eager CPU
                    // phase must still finish before retiring the request scope.
                    let unused=node.at_use&&!node.used&&matches!(node.status,Status::Waiting);
                    if unused {
                        if let Some(cpu)=node.prepared {
                            if !state.nodes[cpu].status.ready()? {return Err("previous request preparation is still active".into());}
                        }
                    }else if !node.status.ready()? {return Err("previous request preparation is still active".into());}
                }
            }
        }
        let mut jobs = vec![];
        for p in self.declaration.preparations.iter().filter(|p| !p.inputs.is_empty()) {
            let inputs: Value = p.inputs.iter().map(|n| (n.clone(), facts[n].clone())).collect();
            let at_use = match p.at_use {
                AtUse::Never => false, AtUse::Always => true,
                AtUse::Predicate => self.bindings.call("at_use", &p.name, &inputs)?.as_bool()
                    .ok_or_else(|| format!("preparation dependency must return bool: {}", p.name))?,
            };
            jobs.push((p, inputs, at_use));
        }
        let mut state = self.shared.state.lock().unwrap();
        if state.closed { return Err("request preparation requires an open started plan".into()); }
        if state.active_invocations != 0 { return Err("previous request computation is still active".into()); }
        state.generation = state.generation.checked_add(1).ok_or("startup request generation overflow")?;
        let permanent = state.permanent_nodes;
        state.nodes.truncate(permanent);
        state.request_work.clear(); state.scope = Some(scope.into());
        for (p, inputs, at_use) in jobs {
            let id = Self::preparation(&mut state, p, inputs, at_use);
            state.request_work.insert(p.name.clone(), id);
        }
        let result = state.static_work.iter().chain(state.request_work.iter()).map(|(n, id)|
            (n.clone(), ResultHandle { node: *id, generation: state.nodes[*id].generation })).collect();
        self.shared.notify();
        Ok(result)
    }
    fn consumer(&self, state: &mut State, scope: Option<&str>, name: &str) -> Result<Vec<usize>> {
        if !state.started || state.closed { return Err("component use requires an open started declaration".into()); }
        if let Some(scope) = scope {
            if state.scope.as_deref() != Some(scope) { return Err("component belongs to another request scope".into()); }
        }
        let c = self.declaration.components.iter().find(|c| c.name == name)
            .ok_or_else(|| format!("unknown declared startup component: {name}"))?;
        let mut ids = vec![state.weights[name]];
        for required in c.required_preparations.as_ref().unwrap() {
            let id = *state.request_work.get(required).or_else(|| state.static_work.get(required))
                .ok_or_else(|| format!("required startup request input is missing: {required}"))?;
            ids.push(id);
        }
        // An at-use initializer can itself depend on another at-use
        // initializer or lazy loader. Activate the full admitted closure.
        let mut pending=ids.clone();let mut changed=false;
        while let Some(id)=pending.pop() {
            let node=&mut state.nodes[id];
            if node.used {continue;}
            node.used=true;node.used_at=Some(Instant::now());changed=true;
            pending.extend(node.dependencies.iter().copied());
        }
        if changed {self.shared.notify();}
        Ok(ids)
    }
    /// Release an owner-lifetime, explicitly deferred preparation. This
    /// schedules its admitted closure without blocking the releasing thread.
    /// Model adapters validate the external event before making this call.
    pub fn release_preparation(&self, name: &str) -> Result<ResultHandle> {
        let definition=self.declaration.preparations.iter().find(|p|p.name==name)
            .ok_or_else(||format!("unknown declared startup preparation: {name}"))?;
        if !definition.inputs.is_empty() || !definition.before.is_empty() || definition.at_use!=AtUse::Always {
            return Err("release requires an independent static at-use preparation".into());
        }
        let mut state=self.shared.state.lock().unwrap();
        if !state.started || state.closed {
            return Err("preparation release requires an open started declaration".into());
        }
        let id=state.static_work[name];
        if matches!(state.nodes[id].status,Status::Cancelled) {
            return Err("released preparation was cancelled".into());
        }
        let mut pending=vec![id];let mut changed=false;
        while let Some(index)=pending.pop() {
            let node=&mut state.nodes[index];
            if node.used {continue;}
            node.used=true;node.used_at=Some(Instant::now());changed=true;
            pending.extend(node.dependencies.iter().copied());
        }
        if changed {self.shared.notify();}
        Ok(ResultHandle{node:id,generation:state.nodes[id].generation})
    }
    pub fn check_before_use(&self, scope: Option<&str>, name: &str) -> Result<bool> {
        self.check_before_use_many(scope,&[name])
    }
    fn consumers(&self,state:&mut State,scope:Option<&str>,names:&[&str])->Result<Vec<usize>> {
        if names.is_empty() || names.iter().collect::<BTreeSet<_>>().len()!=names.len() {
            return Err("declare existing unique components for a numerical consumer".into());
        }
        for name in names {
            if !self.declaration.components.iter().any(|c|c.name==*name) {
                return Err(format!("unknown declared startup component: {name}"));
            }
        }
        let mut ids=BTreeSet::new();
        for name in names {ids.extend(self.consumer(state,scope,name)?);}
        Ok(ids.into_iter().collect())
    }
    /// Release the entire dependency closure before waiting for any member.
    pub fn check_before_use_many(&self,scope:Option<&str>,names:&[&str])->Result<bool> {
        let mut state = self.shared.state.lock().unwrap();
        let ids = self.consumers(&mut state, scope, names)?;
        let mut ready = true;
        for id in ids { if !state.nodes[id].status.ready()? { ready = false; } }
        Ok(ready)
    }
    pub fn before_use(&self, scope: Option<&str>, name: &str) -> Result<()> {
        self.before_use_interruptible(scope, name, None)
    }
    pub fn before_use_many(&self,scope:Option<&str>,names:&[&str])->Result<()> {
        self.wait_for_consumers(scope,names,None)
    }
    pub fn wake(&self) {
        // Coordinate with the condition-variable predicate to avoid a lost
        // cancellation notification between the predicate check and wait.
        let _state = self.shared.state.lock().unwrap();
        self.shared.notify();
    }
    pub fn before_use_interruptible(&self, scope: Option<&str>, name: &str, cancelled: Option<&AtomicBool>) -> Result<()> {
        self.wait_for_consumers(scope,&[name],cancelled)
    }
    fn wait_for_consumers(&self,scope:Option<&str>,names:&[&str],cancelled:Option<&AtomicBool>)->Result<()> {
        let deadline = Instant::now() + Duration::from_secs_f64(self.declaration.timeout_seconds);
        let mut state = self.shared.state.lock().unwrap();
        let ids = self.consumers(&mut state, scope, names)?;
        loop {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) { return Err("startup gate closed".into()); }
            if state.closed { return Err("component use requires an open started declaration".into()); }
            if let Some(scope) = scope { if state.scope.as_deref() != Some(scope) { return Err("component belongs to another request scope".into()); } }
            let mut ready = true;
            for id in &ids { if !state.nodes[*id].status.ready()? { ready = false; } }
            if ready { return Ok(()); }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return Err(format!("startup consumer timed out: {}",names.join(","))); }
            state = self.shared.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }
    /// Guard the actual consumer, including request lifetime, on its caller's
    /// thread. Numerical functions may own non-Send GPU objects on that thread.
    pub fn invoke<T>(&self, scope: Option<&str>, name: &str, compute: impl FnOnce() -> Result<T>) -> Result<T> {
        let _invocation=self.enter(scope,name)?;
        compute()
    }
    pub fn enter(&self,scope:Option<&str>,name:&str)->Result<ConsumerGuard> {
        self.enter_many(scope,&[name])
    }
    pub fn invoke_many<T>(&self,scope:Option<&str>,names:&[&str],compute:impl FnOnce()->Result<T>)->Result<T> {
        let _invocation=self.enter_many(scope,names)?;compute()
    }
    pub fn enter_many(&self,scope:Option<&str>,names:&[&str])->Result<ConsumerGuard> {
        let expected=scope.map(str::to_owned).or_else(||self.shared.state.lock().unwrap().scope.clone());
        self.before_use_many(expected.as_deref(),names)?;
        {
            let mut state=self.shared.state.lock().unwrap();
            if state.scope!=expected { return Err("component belongs to another request scope".into()); }
            let ids=self.consumers(&mut state,expected.as_deref(),names)?;
            for id in ids { if !state.nodes[id].status.ready()? { return Err("consumer readiness changed before invocation".into()); } }
            state.active_invocations+=1;
        }
        Ok(ConsumerGuard{shared:self.shared.clone(),_caller:std::marker::PhantomData})
    }
    pub fn invoke_bound(&self, scope: Option<&str>, name: &str, inputs: &Value) -> Result<Value> {
        self.invoke(scope,name,||self.bindings.call("invoke",name,inputs))
    }
    pub fn result(&self, handle: ResultHandle, wait: bool, timeout_seconds: Option<f64>) -> Result<Option<Value>> {
        let seconds = timeout_seconds.unwrap_or(self.declaration.timeout_seconds);
        if !seconds.is_finite() || seconds < 0.0 { return Err("invalid startup result timeout".into()); }
        let deadline = Instant::now() + Duration::from_secs_f64(seconds);
        let mut state = self.shared.state.lock().unwrap();
        loop {
            let node = state.nodes.get(handle.node).filter(|n| n.generation == handle.generation)
                .ok_or("startup result belongs to an expired request scope")?;
            let result = node.status.result()?;
            if result.is_some() || !wait { return Ok(result); }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return Err("startup result timed out".into()); }
            state = self.shared.changed.wait_timeout(state, remaining).unwrap().0;
        }
    }
    pub fn cancel(&self, handle: ResultHandle) -> Result<bool> {
        let mut state = self.shared.state.lock().unwrap();
        let node = state.nodes.get_mut(handle.node).filter(|n| n.generation == handle.generation)
            .ok_or("startup result belongs to an expired request scope")?;
        if !matches!(node.status, Status::Waiting) { return Ok(false); }
        let cpu = node.prepared;
        node.status = Status::Cancelled;
        if let Some(cpu) = cpu {
            if matches!(state.nodes[cpu].status, Status::Waiting) { state.nodes[cpu].status = Status::Cancelled; }
        }
        self.shared.notify(); Ok(true)
    }
    pub fn poll(&self) -> Result<()> {
        let state = self.shared.state.lock().unwrap();
        for id in state.weights.values().chain(state.static_work.values()).chain(state.request_work.values()) {
            state.nodes[*id].status.ready()?;
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Value {
        let state = self.shared.state.lock().unwrap();
        let ready = |ids: Vec<usize>| ids.iter().all(|id| matches!(state.nodes[*id].status, Status::Ready(_)));
        let weights_ready = state.started && ready(state.weights.values().copied().collect());
        let nodes: Vec<_> = state.nodes.iter().enumerate().map(|(id,n)| json!({"name":n.name,"operation":n.operation,
            "generation":n.generation,"done":n.status.done(),"running":matches!(n.status,Status::Running|Status::Validating),"cancelled":matches!(n.status,Status::Cancelled),
            "serial_group":n.serial_group,"initialize":n.initialize,
            "ready_elapsed_seconds":if (!n.at_use||n.used) && n.dependencies.iter().all(|&d|matches!(state.nodes[d].status,Status::Ready(_))) {
                Some(Self::ready_at(&state,id).duration_since(self.shared.started).as_secs_f64())}else{None},
            "waiting_for_serial_group":matches!(n.status,Status::Waiting) && (!n.at_use||n.used)
                && n.dependencies.iter().all(|&d|matches!(state.nodes[d].status,Status::Ready(_))) && Self::group_busy(&state,id),
            "state":match n.status {Status::Waiting=>"waiting",Status::Running|Status::Validating=>"pending",_=>"ready"}})).collect();
        json!({"executor":"rust","started":state.started,"closed":state.closed,"scope":state.scope,
            "weights":state.weights,"static":state.static_work,"request":state.request_work,"nodes":nodes,
            "initializers":state.nodes.iter().enumerate().filter(|(_,n)|n.operation=="component_initialize").map(|(id,n)|(n.name.clone(),id)).collect::<BTreeMap<_,_>>(),
            "active_invocations":state.active_invocations,"weights_ready":weights_ready,
            "preload_ready":weights_ready && ready(state.static_work.values().copied()
                .filter(|&id|!state.nodes[id].at_use || state.nodes[id].used).collect()),
            "events":state.events,"warnings":state.warnings})
    }
    pub fn close(&self) -> Result<()> {
        let _close=self.close_lock.lock().unwrap();
        {
            let mut state = self.shared.state.lock().unwrap();
            if state.closed { return Ok(()); }
            state.closed = true;
            for n in &mut state.nodes { if !n.status.done() { n.status = Status::Cancelled; } }
            self.shared.notify();
        }
        // Interrupt native operations *before* joining workers which may be in RPC.
        let mut failure = if self.bindings.has("cancel", "") { self.bindings.call("cancel", "", &Value::Null).err() } else { None };
        for worker in self.workers.lock().unwrap().drain(..) {
            if worker.join().is_err() { failure.get_or_insert("startup worker panicked during close".into()); }
        }
        let mut state=self.shared.state.lock().unwrap();
        while state.active_invocations!=0 { state=self.shared.changed.wait(state).unwrap(); }
        match failure { Some(error)=>Err(error),None=>Ok(()) }
    }
}
impl Drop for DeclaredStartup { fn drop(&mut self) { let _ = self.close(); } }

#[cfg(test)]
mod tests {
    use super::*;
    fn released_host_fixture(counter:Arc<std::sync::atomic::AtomicUsize>)->(Declaration,Bindings) {
        let declaration:Declaration=serde_json::from_value(json!({
            "components":[{"name":"dit","required_preparations":[]}],
            "preparations":[{"name":"host","before":[],"weights":["dit"],"at_use":"always"}],
            "workers":1,"timeout_seconds":2})).unwrap();
        let mut bindings=Bindings::default();
        bindings.insert("load","dit",Arc::new(|_|Ok(json!(true))));
        bindings.insert("validate_load","dit",Arc::new(|v|Ok(v.clone())));
        bindings.insert("run","host",Arc::new(move |_|{counter.fetch_add(1,Ordering::SeqCst);Ok(json!(true))}));
        bindings.insert("validate_run","host",Arc::new(|v|Ok(v.clone())));
        (declaration,bindings)
    }
    #[test]
    fn released_preparation_does_not_run_or_reserve_worker_until_released() {
        use std::sync::atomic::AtomicUsize;
        let counter=Arc::new(AtomicUsize::new(0));let(d,b)=released_host_fixture(counter.clone());
        let runtime=DeclaredStartup::new(d,b).unwrap();
        assert!(runtime.release_preparation("host").is_err());
        runtime.start().unwrap();runtime.before_use(None,"dit").unwrap();
        assert_eq!(counter.load(Ordering::SeqCst),0);
        assert_eq!(runtime.snapshot()["preload_ready"],true);
        let first=runtime.release_preparation("host").unwrap();
        assert_eq!(runtime.result(first,true,None).unwrap(),Some(json!(true)));
        let second=runtime.release_preparation("host").unwrap();
        assert_eq!(first.node,second.node);assert_eq!(first.generation,second.generation);
        assert_eq!(counter.load(Ordering::SeqCst),1);
        runtime.close().unwrap();assert!(runtime.release_preparation("host").is_err());
    }
    #[test]
    fn released_preparation_rejects_implicit_or_consumer_gating_work() {
        use std::sync::atomic::AtomicUsize;
        let(d,b)=released_host_fixture(Arc::new(AtomicUsize::new(0)));
        let mut implicit=d.clone();implicit.preparations[0].at_use=AtUse::Never;
        assert!(DeclaredStartup::new(implicit,b.clone()).is_err());
        let mut gating=d;gating.preparations[0].before=vec!["dit".into()];
        gating.components[0].required_preparations=Some(vec!["host".into()]);
        let runtime=DeclaredStartup::new(gating,b).unwrap();runtime.start().unwrap();
        assert!(runtime.release_preparation("host").is_err());runtime.close().unwrap();
    }
    #[test]
    fn released_preparation_rejects_cancelled_work_and_validates_receipts() {
        use std::sync::atomic::AtomicUsize;
        let(d,b)=released_host_fixture(Arc::new(AtomicUsize::new(0)));
        let runtime=DeclaredStartup::new(d.clone(),b.clone()).unwrap();runtime.start().unwrap();
        let id=runtime.shared.state.lock().unwrap().static_work["host"];
        assert!(runtime.cancel(ResultHandle{node:id,generation:0}).unwrap());
        assert!(runtime.release_preparation("host").is_err());runtime.close().unwrap();
        let mut invalid=b;invalid.insert("validate_run","host",Arc::new(|_|Ok(json!(false))));
        let runtime=DeclaredStartup::new(d,invalid).unwrap();runtime.start().unwrap();
        let result=runtime.release_preparation("host").unwrap();
        assert!(runtime.result(result,true,None).is_err());runtime.close().unwrap();
    }
    #[test]
    fn released_owner_lifetime_work_survives_request_scope_changes() {
        use std::sync::{atomic::AtomicUsize,mpsc};
        let(d,mut b)=released_host_fixture(Arc::new(AtomicUsize::new(0)));
        let(tx,rx)=mpsc::channel();b.work.remove(&("run".into(),"host".into()));b.insert_async("run","host",Arc::new(move|_,done|{tx.send(done).unwrap()}));
        b.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
        let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();
        let handle=runtime.release_preparation("host").unwrap();let done=rx.recv_timeout(Duration::from_secs(2)).unwrap();
        runtime.request("one",json!({})).unwrap();runtime.request("two",json!({})).unwrap();
        done.send(Ok(json!(true)));
        assert_eq!(runtime.result(handle,true,None).unwrap(),Some(json!(true)));runtime.close().unwrap();
    }
    #[test]
    fn early_component_initializer_runs_before_load_dependencies_and_validates_before_weights() {
        use std::sync::mpsc;
        for accepted in [true,false] {
            let d:Declaration=serde_json::from_value(json!({"components":[
                {"name":"dit","initialize":true,"after":["backend"],"required_preparations":["backend"]}],
                "preparations":[{"name":"backend","before":["dit"]}],"workers":1,"timeout_seconds":3})).unwrap();
            let mut b=Bindings::default();let (asset_send,asset_recv)=mpsc::channel();
            b.insert_async("run","backend",Arc::new(move |_,done|{asset_send.send(done).unwrap();}));
            b.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
            b.insert("validate_run","backend",Arc::new(|v|Ok(v.clone())));
            let (initialized,seen)=mpsc::channel();
            b.insert("component_initialize","dit",Arc::new(move |_|{initialized.send(()).unwrap();Ok(json!({"cuda_ready":true}))}));
            b.insert("validate_component_initialize","dit",Arc::new(move |v|Ok(json!(accepted && v["cuda_ready"]==true))));
            let loaded=Arc::new(AtomicBool::new(false));let marker=loaded.clone();
            b.insert("load","dit",Arc::new(move |_|{marker.store(true,Ordering::SeqCst);Ok(json!(true))}));
            b.insert("validate_load","dit",Arc::new(|v|Ok(v.clone())));
            let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();
            let assets=asset_recv.recv_timeout(Duration::from_secs(2)).unwrap();
            seen.recv_timeout(Duration::from_secs(2)).expect("backend wait delayed the initializer");
            assert!(!loaded.load(Ordering::SeqCst));
            assets.send(Ok(json!(true)));
            assert_eq!(runtime.before_use(None,"dit").is_ok(),accepted);
            assert_eq!(loaded.load(Ordering::SeqCst),accepted);runtime.close().unwrap();
        }
    }
    #[test]
    fn component_initializer_requires_both_bindings_before_any_work() {
        let mut d=declaration();d.components[0].initialize=true;
        let mut b=bindings();
        assert!(DeclaredStartup::new(d.clone(),b.clone()).err().unwrap().contains("component_initialize"));
        b.insert("component_initialize","dit",Arc::new(|_|panic!("admission ran initializer")));
        assert!(DeclaredStartup::new(d,b).err().unwrap().contains("validate_component_initialize"));
    }
    #[test]
    fn lazy_component_keeps_its_initializer_lazy_after_projection() {
        let d:Declaration=serde_json::from_value(json!({"components":[
            {"name":"dit","initialize":true,"load_at_use":true,"required_preparations":[]},
            {"name":"other","initialize":true,"required_preparations":[]}],
            "preparations":[],"consumers":["dit"],"workers":1,"timeout_seconds":3})).unwrap();
        let count=Arc::new(std::sync::atomic::AtomicUsize::new(0));let mut b=Bindings::default();
        for name in ["dit","other"] {
            let count=count.clone();b.insert("component_initialize",name,Arc::new(move |_|{
                assert_eq!(name,"dit");count.fetch_add(1,Ordering::SeqCst);Ok(json!(true))
            }));
            for op in ["load","validate_load","validate_component_initialize"] {b.insert(op,name,Arc::new(|_|Ok(json!(true))));}
        }
        let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();
        assert_eq!(runtime.snapshot()["initializers"].as_object().unwrap().len(),1);
        assert_eq!(count.load(Ordering::SeqCst),0);
        runtime.before_use(None,"dit").unwrap();assert_eq!(count.load(Ordering::SeqCst),1);runtime.close().unwrap();
    }
    #[test]
    fn component_load_waits_for_validated_static_preparation_with_one_worker() {
        for accepted in [true,false] {
            let d:Declaration=serde_json::from_value(json!({"components":[
                {"name":"reader","required_preparations":[]},
                {"name":"dit","after":["backend"],"required_preparations":["backend"]}],
                "preparations":[{"name":"backend","before":["dit"],"weights":["reader"]}],
                "workers":1,"timeout_seconds":2})).unwrap();
            let calls=Arc::new(Mutex::new(Vec::new()));let mut b=Bindings::default();
            for (op,name) in [("load","reader"),("run","backend"),("load","dit")] {
                let calls=calls.clone();b.insert(op,name,Arc::new(move |_|{
                    calls.lock().unwrap().push(format!("{op}/{name}"));Ok(json!(true))
                }));
            }
            for name in ["reader","dit"] {b.insert("validate_load",name,Arc::new(|v|Ok(v.clone())));}
            let seen=calls.clone();b.insert("validate_run","backend",Arc::new(move |_|{
                seen.lock().unwrap().push("validated/backend".into());Ok(json!(accepted))
            }));
            let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();
            assert_eq!(runtime.before_use(None,"dit").is_ok(),accepted);runtime.close().unwrap();
            let mut expected=vec!["load/reader","run/backend","validated/backend"];
            if accepted {expected.push("load/dit");}
            assert_eq!(*calls.lock().unwrap(),expected);
        }
    }
    #[test]
    fn load_preparation_cycles_and_request_dependencies_fail_during_admission() {
        for request_dependent in [true,false] {
            let mut d=declaration();d.components[0].after=vec!["backend".into()];
            d.components[0].required_preparations.as_mut().unwrap().push("backend".into());
            d.preparations.insert(0,Preparation{name:"backend".into(),before:vec!["dit".into()],
                weights:vec!["dit".into()],inputs:if request_dependent{vec!["request".into()]}else{vec![]},..Preparation::default()});
            let mut b=bindings();b.insert("run","backend",Arc::new(|_|panic!("admission ran work")));
            b.insert("validate_run","backend",Arc::new(|_|panic!("admission validated work")));
            assert!(DeclaredStartup::new(d,b).is_err());
        }
    }
    #[test]
    fn capability_projection_keeps_load_preparations_and_their_weight_closure() {
        let d:Declaration=serde_json::from_value(json!({"components":[
            {"name":"encoder","required_preparations":[]},
            {"name":"dit","after":["backend"],"required_preparations":[]},
            {"name":"other","required_preparations":["backend"]}],
            "preparations":[{"name":"backend","before":["other"],"weights":["encoder"]}],
            "consumers":["dit"],"workers":1,"timeout_seconds":2})).unwrap();
        let mut b=Bindings::default();
        for name in ["encoder","dit","other"] {for op in ["load","validate_load"] {
            b.insert(op,name,Arc::new(|_|Ok(json!(true))));
        }}
        for op in ["run","validate_run"] {b.insert(op,"backend",Arc::new(|_|Ok(json!(true))));}
        let runtime=DeclaredStartup::new(d,b).unwrap();
        assert_eq!(runtime.declaration().components.iter().map(|c|c.name.as_str()).collect::<Vec<_>>(),["encoder","dit"]);
        assert_eq!(runtime.declaration().preparations[0].name,"backend");
        runtime.start().unwrap();runtime.before_use(None,"dit").unwrap();runtime.close().unwrap();
    }
    #[test]
    fn serial_resource_uses_dependency_readiness_and_does_not_reserve_waiting_workers() {
        use std::sync::{mpsc,atomic::AtomicUsize};
        let declaration:Declaration=serde_json::from_value(json!({"components":[
            {"name":"a","required_preparations":[]},{"name":"b","required_preparations":[]},
            {"name":"c","required_preparations":[]},{"name":"metadata","required_preparations":[]},
            {"name":"result","required_preparations":["first","second","third","progress"]}],
            "preparations":[
                {"name":"first","before":["result"],"inputs":["request"],"weights":["a"],"serial_group":"device","initialize":true},
                {"name":"second","before":["result"],"inputs":["request"],"weights":["b"],"serial_group":"device","initialize":true},
                {"name":"third","before":["result"],"inputs":["request"],"weights":["c"],"serial_group":"device","initialize":true},
                {"name":"progress","before":["result"],"inputs":["request"],"weights":["metadata"]}],
            "workers":2,"timeout_seconds":5})).unwrap();
        let mut bindings=Bindings::default();let (publish,receipts)=mpsc::channel();
        for name in ["a","b","c","metadata"] {
            let publish=publish.clone();bindings.insert_async("load",name,Arc::new(move |_,done|{publish.send((name,done)).unwrap();}));
        }
        bindings.insert("load","result",Arc::new(|_|Ok(json!(true))));
        for name in ["a","b","c","metadata","result"]{bindings.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));}
        bindings.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
        let order=Arc::new(Mutex::new(Vec::<String>::new()));let active=Arc::new(AtomicUsize::new(0));
        let (entered,busy)=mpsc::channel();let (release,wait)=mpsc::channel();let wait=Arc::new(Mutex::new(wait));
        for name in ["first","second","third"] {
            let (seen,count)=(order.clone(),active.clone());bindings.insert("initialize",name,Arc::new(move |_| {
                assert_eq!(count.fetch_add(1,Ordering::SeqCst),0);seen.lock().unwrap().push(format!("initialize/{name}"));Ok(json!({"resource":name}))
            }));
            bindings.insert("validate_initialize",name,Arc::new(move |v|Ok(json!(v["resource"]==name))));
            let (seen,entered,wait)=(order.clone(),entered.clone(),wait.clone());
            bindings.insert("run",name,Arc::new(move |inputs| {
                assert_eq!(inputs["initialized"]["resource"],name);seen.lock().unwrap().push(format!("run/{name}"));
                if name=="first" {entered.send(()).unwrap();wait.lock().unwrap().recv_timeout(Duration::from_secs(3)).map_err(|_|"test did not release first action")?;}
                Ok(json!(true))
            }));
            let (seen,count)=(order.clone(),active.clone());bindings.insert("validate_run",name,Arc::new(move |v| {
                seen.lock().unwrap().push(format!("validate/{name}"));assert_eq!(count.fetch_sub(1,Ordering::SeqCst),1);Ok(json!(v==true))
            }));
        }
        let (progress,advanced)=mpsc::channel();bindings.insert("run","progress",Arc::new(move |_|{progress.send(()).unwrap();Ok(json!(true))}));
        bindings.insert("validate_run","progress",Arc::new(|v|Ok(json!(v==true))));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();runtime.request("one",json!({"request":1})).unwrap();
        let mut pending=BTreeMap::new();for _ in 0..4{let (name,done)=receipts.recv_timeout(Duration::from_secs(2)).unwrap();pending.insert(name,done);}
        pending.remove("a").unwrap().send(Ok(json!(true)));busy.recv_timeout(Duration::from_secs(2)).unwrap();
        // Reverse declaration order while the serial resource is occupied.
        pending.remove("c").unwrap().send(Ok(json!(true)));pending.remove("b").unwrap().send(Ok(json!(true)));
        pending.remove("metadata").unwrap().send(Ok(json!(true)));
        let progress=advanced.recv_timeout(Duration::from_secs(2));release.send(()).unwrap();
        progress.expect("waiting serial actions blocked unrelated metadata/read progress");
        runtime.before_use(Some("one"),"result").unwrap();
        assert_eq!(*order.lock().unwrap(),["initialize/first","run/first","validate/first","initialize/third","run/third","validate/third","initialize/second","run/second","validate/second"]);
        runtime.close().unwrap();
    }
    #[test]
    fn rejected_initializer_never_runs_compute_and_initializer_bindings_are_admitted() {
        let declaration:Declaration=serde_json::from_value(json!({"components":[{"name":"model","required_preparations":["action"]}],
            "preparations":[{"name":"action","before":["model"],"weights":["model"],"initialize":true,"serial_group":"device"}],
            "workers":2,"timeout_seconds":2})).unwrap();
        let called=Arc::new(AtomicBool::new(false));let marker=called.clone();let mut bindings=Bindings::default();
        bindings.insert("load","model",Arc::new(|_|Ok(json!(true))));bindings.insert("validate_load","model",Arc::new(|v|Ok(v.clone())));
        bindings.insert("run","action",Arc::new(move |_|{marker.store(true,Ordering::SeqCst);Ok(json!(true))}));bindings.insert("validate_run","action",Arc::new(|v|Ok(v.clone())));
        assert!(DeclaredStartup::new(declaration.clone(),bindings.clone()).err().unwrap().contains("initialize/action"));
        bindings.insert("initialize","action",Arc::new(|_|Ok(json!({"invalid":true}))));
        assert!(DeclaredStartup::new(declaration.clone(),bindings.clone()).err().unwrap().contains("validate_initialize/action"));
        bindings.insert("validate_initialize","action",Arc::new(|_|Ok(json!(false))));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
        assert!(runtime.before_use(None,"model").unwrap_err().contains("initializer receipt rejected"));
        assert!(!called.load(Ordering::SeqCst));runtime.close().unwrap();
    }
    #[test]
    fn asynchronous_run_holds_its_serial_resource_without_holding_a_worker() {
        use std::sync::mpsc;
        let declaration:Declaration=serde_json::from_value(json!({"components":[{"name":"model","required_preparations":["first","second","cpu-progress"]}],
            "preparations":[{"name":"first","before":["model"],"serial_group":"device"},
                {"name":"second","before":["model"],"serial_group":"device"},
                {"name":"cpu-progress","before":["model"]}],"workers":1,"timeout_seconds":3})).unwrap();
        let mut bindings=Bindings::default();bindings.insert("load","model",Arc::new(|_|Ok(json!(true))));bindings.insert("validate_load","model",Arc::new(|v|Ok(v.clone())));
        let (started,pending)=mpsc::channel();bindings.insert_async("run","first",Arc::new(move |_,done|{started.send(done).unwrap();}));
        let second=Arc::new(AtomicBool::new(false));let ran=second.clone();bindings.insert("run","second",Arc::new(move |_|{ran.store(true,Ordering::SeqCst);Ok(json!(true))}));
        let (cpu,progress)=mpsc::channel();bindings.insert("run","cpu-progress",Arc::new(move |_|{cpu.send(()).unwrap();Ok(json!(true))}));
        for name in ["first","second","cpu-progress"]{bindings.insert("validate_run",name,Arc::new(|v|Ok(v.clone())));}
        bindings.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
        let done=pending.recv_timeout(Duration::from_secs(2)).unwrap();progress.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(!second.load(Ordering::SeqCst));done.send(Ok(json!(true)));
        runtime.before_use(None,"model").unwrap();assert!(second.load(Ordering::SeqCst));runtime.close().unwrap();
    }
    #[test]
    fn a_multi_component_consumer_releases_all_lazy_loaders_before_waiting() {
        let declaration:Declaration=serde_json::from_value(json!({"components":[
            {"name":"a","required_preparations":[],"load_at_use":true},
            {"name":"b","required_preparations":[],"load_at_use":true}],
            "preparations":[],"workers":1,"timeout_seconds":2})).unwrap();
        let pending=Arc::new(Mutex::new(Vec::new()));let mut bindings=Bindings::default();
        for name in ["a","b"] {
            let pending=pending.clone();
            bindings.insert_async("load",name,Arc::new(move |_,done| {
                let mut pending=pending.lock().unwrap();pending.push(done);
                if pending.len()==2 {for done in pending.drain(..){done.send(Ok(json!(true)));}}
            }));
            bindings.insert("validate_load",name,Arc::new(|v|Ok(v.clone())));
        }
        let pending=pending.clone();bindings.insert("cancel","",Arc::new(move |_|{pending.lock().unwrap().clear();Ok(Value::Null)}));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
        assert!(runtime.enter_many(None,&[]).is_err());
        assert_eq!(runtime.check_before_use(None,"missing").unwrap_err(),"unknown declared startup component: missing");
        let guard=runtime.enter_many(None,&["a","b"]).unwrap();
        assert_eq!(runtime.snapshot()["active_invocations"],1);drop(guard);runtime.close().unwrap();
    }
    #[test]
    fn independent_async_completion_releases_preparation_with_one_worker() {
        let declaration:Declaration=serde_json::from_value(json!({"components":[
            {"name":"slow","required_preparations":[]},
            {"name":"fast","required_preparations":["fast-init"]}],
            "preparations":[{"name":"fast-init","before":["fast"],"weights":["fast"]}],
            "workers":1,"timeout_seconds":2})).unwrap();
        let pending=Arc::new(Mutex::new(None));let mut bindings=Bindings::default();
        let slot=pending.clone();
        bindings.insert_async("load","slow",Arc::new(move |_,done|{*slot.lock().unwrap()=Some(done);}));
        bindings.insert_async("load","fast",Arc::new(|_,done|done.send(Ok(json!(true)))));
        for (operation,name) in [("validate_load","slow"),("validate_load","fast"),("run","fast-init"),("validate_run","fast-init")] {
            bindings.insert(operation,name,Arc::new(|_|Ok(json!(true))));
        }
        let slot=pending.clone();bindings.insert("cancel","",Arc::new(move |_|{slot.lock().unwrap().take();Ok(Value::Null)}));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
        runtime.before_use(None,"fast").unwrap();
        assert!(!runtime.check_before_use(None,"slow").unwrap());
        pending.lock().unwrap().take().unwrap().send(Ok(json!(true)));
        runtime.before_use(None,"slow").unwrap();runtime.close().unwrap();
    }
    #[test]
    fn asynchronous_receipts_require_validation_and_a_joining_transport() {
        for dropped in [false,true] {
            let declaration:Declaration=serde_json::from_value(json!({"components":[
                {"name":"model","required_preparations":[]}],"preparations":[],"workers":1,"timeout_seconds":2})).unwrap();
            let mut bindings=Bindings::default();
            bindings.insert_async("load","model",Arc::new(move |_,done|{if !dropped {done.send(Ok(json!(false)));}}));
            bindings.insert("validate_load","model",Arc::new(|v|Ok(v.clone())));
            assert!(DeclaredStartup::new(declaration.clone(),bindings.clone()).is_err());
            bindings.insert("cancel","",Arc::new(|_|Ok(Value::Null)));
            let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
            assert!(runtime.before_use(None,"model").is_err());runtime.close().unwrap();
        }
    }
    #[test]
    fn cancelled_async_results_cannot_validate_or_release_consumers() {
        let declaration:Declaration=serde_json::from_value(json!({"components":[
            {"name":"model","required_preparations":[]}],"preparations":[],"workers":1,"timeout_seconds":2})).unwrap();
        let pending=Arc::new(Mutex::new(None));let slot=pending.clone();let mut bindings=Bindings::default();
        let (send,receive)=std::sync::mpsc::channel();let send=Mutex::new(send);
        bindings.insert_async("load","model",Arc::new(move |_,done|{*slot.lock().unwrap()=Some(done);send.lock().unwrap().send(()).unwrap();}));
        bindings.insert("validate_load","model",Arc::new(|_|panic!("cancelled result was validated")));
        let slot=pending.clone();bindings.insert("cancel","",Arc::new(move |_|{
            slot.lock().unwrap().take().unwrap().send(Ok(json!(true)));Ok(Value::Null)
        }));
        let runtime=DeclaredStartup::new(declaration,bindings).unwrap();runtime.start().unwrap();
        receive.recv_timeout(Duration::from_secs(2)).unwrap();runtime.close().unwrap();
        assert!(runtime.check_before_use(None,"model").is_err());
    }
    use std::sync::mpsc;

    fn declaration() -> Declaration {
        serde_json::from_value(json!({"components":[{"name":"dit","required_preparations":["layout"]}],
            "preparations":[{"name":"layout","before":["dit"],"inputs":["request"],"weights":["dit"],"cpu":true}],
            "workers":2,"timeout_seconds":2})).unwrap()
    }
    fn bindings() -> Bindings {
        let mut b = Bindings::default();
        for (op, name) in [("load","dit"),("cpu","layout"),("run","layout"),
            ("validate_load","dit"),("validate_cpu","layout"),("validate_run","layout")] {
            b.insert(op, name, Arc::new(|_| Ok(Value::Bool(true))));
        }
        b
    }
    fn wait_node(runtime: &DeclaredStartup, name: &str, section: &str) {
        let snapshot = runtime.snapshot();
        let id = snapshot[section][name].as_u64().unwrap() as usize;
        runtime.result(ResultHandle { node:id, generation:snapshot["nodes"][id]["generation"].as_u64().unwrap() }, true, Some(2.)).unwrap();
    }
    #[test]
    fn callbacks_start_only_after_the_whole_worker_cohort_is_created() {
        let slot=Arc::new(Mutex::new(None::<std::sync::Weak<DeclaredStartup>>));
        let observed=Arc::new(AtomicBool::new(false));
        let mut b=bindings();let (owner,seen)=(slot.clone(),observed.clone());
        b.insert("load","dit",Arc::new(move |_| {
            let runtime=owner.lock().unwrap().as_ref().unwrap().upgrade().unwrap();
            let workers=runtime.workers.try_lock().map_err(|_|"callback raced worker creation")?;
            if workers.len()!=runtime.declaration.workers {return Err("callback saw an incomplete worker cohort".into());}
            seen.store(true,Ordering::Release);Ok(json!(true))
        }));
        let mut d=declaration();d.workers=32;
        let runtime=Arc::new(DeclaredStartup::new(d,b).unwrap());
        *slot.lock().unwrap()=Some(Arc::downgrade(&runtime));
        runtime.start().unwrap();wait_node(&runtime,"dit","weights");
        assert!(observed.load(Ordering::Acquire));runtime.close().unwrap();
    }
    #[test]
    fn cancellation_wakes_workers_of_an_unreleased_cohort_without_callbacks() {
        let invoked=Arc::new(AtomicBool::new(false));let seen=invoked.clone();
        let mut b=bindings();b.insert("load","dit",Arc::new(move |_|{
            seen.store(true,Ordering::Release);Ok(json!(true))
        }));
        let runtime=DeclaredStartup::new(declaration(),b.clone()).unwrap();
        {
            let mut state=runtime.shared.state.lock().unwrap();state.started=true;
            let id=DeclaredStartup::node(&mut state,"dit","load",json!({}),vec![],None,false,false);
            state.weights.insert("dit".into(),id);
        }
        let shared=runtime.shared.clone();let (entered,receive)=mpsc::channel();
        let worker=thread::spawn(move||{entered.send(()).unwrap();DeclaredStartup::worker(shared,Arc::new(b));});
        receive.recv().unwrap();runtime.close().unwrap();worker.join().unwrap();
        assert!(!invoked.load(Ordering::Acquire));
    }
    #[test]
    fn admission_rejects_missing_work_validator_dependencies_and_resources_without_callbacks() {
        let calls = Arc::new(Mutex::new(0));
        let mut b = bindings(); let c = calls.clone();
        b.insert("load","dit",Arc::new(move |_| { *c.lock().unwrap() += 1; Ok(json!(true)) }));
        for mutation in 0..6 {
            let mut d = declaration();
            match mutation {
                0 => d.preparations.clear(),
                1 => d.components[0].required_preparations = None,
                2 => d.components[0].after = vec!["dit".into()],
                3 => d.preparations[0].before = vec!["unknown".into()],
                4 => { d.components[0].reader_limit_mib=2; d.reader_budget_mib=1; },
                _ => d.preparations[0].weights=vec!["unknown".into()],
            }
            assert!(DeclaredStartup::new(d,b.clone()).is_err());
        }
        b.work.remove(&("validate_run".into(),"layout".into()));
        assert!(DeclaredStartup::new(declaration(), b).is_err());
        assert_eq!(*calls.lock().unwrap(),0);
    }
    #[test]
    fn cpu_overlaps_weight_io_and_gpu_waits_for_both() {
        let (cpu_sent,cpu_received)=mpsc::channel();
        let (release,wait)=mpsc::channel(); let wait=Mutex::new(wait);
        let mut b=bindings();
        b.insert("load","dit",Arc::new(move |_| { wait.lock().unwrap().recv_timeout(Duration::from_secs(2)).map_err(|e|e.to_string())?; Ok(json!("weights")) }));
        b.insert("cpu","layout",Arc::new(move |_| { cpu_sent.send(()).unwrap(); Ok(json!("structure")) }));
        b.insert("run","layout",Arc::new(|inputs| { assert_eq!(inputs["prepared"],"structure"); Ok(json!(true)) }));
        let runtime=DeclaredStartup::new(declaration(),b).unwrap(); runtime.start().unwrap();
        runtime.request("request-one",json!({"request":{}})).unwrap();
        cpu_received.recv_timeout(Duration::from_secs(1)).expect("CPU preparation was serialized behind weights");
        assert!(!runtime.check_before_use(Some("request-one"),"dit").unwrap());
        release.send(()).unwrap(); runtime.before_use(Some("request-one"),"dit").unwrap();
        let e=runtime.snapshot()["events"].as_array().unwrap().clone();
        let position=|phase:&str| e.iter().position(|e|e["phase"]==phase).unwrap();
        assert!(position("cpu_prepare_ready")<position("prepare_start"));
        assert!(position("ready")<position("prepare_start"));
        runtime.close().unwrap();
    }
    #[test]
    fn errors_false_receipts_non_bool_and_panics_never_grant_execution() {
        for mode in 0..4 {
            let mut b=bindings();
            b.insert("validate_run","layout",Arc::new(move |_| match mode {
                0=>Err("backend failed".into()),1=>Ok(json!(false)),2=>Ok(json!(1)),_=>panic!("model validator panicked"),
            }));
            let runtime=DeclaredStartup::new(declaration(),b).unwrap(); runtime.start().unwrap();
            runtime.request("one",json!({"request":{}})).unwrap();
            assert!(runtime.before_use(Some("one"),"dit").is_err());
            assert!(runtime.check_before_use(Some("one"),"dit").is_err());
            runtime.close().unwrap();
        }
    }
    #[test]
    fn missing_input_and_foreign_request_never_borrow_readiness() {
        let runtime=DeclaredStartup::new(declaration(),bindings()).unwrap(); runtime.start().unwrap();
        assert!(runtime.before_use(None,"dit").unwrap_err().contains("input is missing"));
        assert!(runtime.request("one",json!({})).is_err());
        runtime.request("one",json!({"request":1})).unwrap(); runtime.before_use(Some("one"),"dit").unwrap();
        assert!(runtime.before_use(Some("other"),"dit").is_err());
        assert!(runtime.request("one",json!({"request":2})).is_err());
        runtime.request("two",json!({"request":2})).unwrap();
        assert!(runtime.before_use(Some("one"),"dit").is_err());
        runtime.before_use(Some("two"),"dit").unwrap(); runtime.close().unwrap();
    }
    #[test]
    fn lazy_loader_keeps_cpu_frontend_eager_and_activates_only_its_consumer_closure() {
        let d:Declaration=serde_json::from_value(json!({"components":[
            {"name":"readers","required_preparations":[]},
            {"name":"weights","after":["readers"],"required_preparations":[],"load_at_use":true},
            {"name":"model","after":["weights"],"required_preparations":["layout"]},
            {"name":"unused","required_preparations":[],"load_at_use":true}],
            "preparations":[{"name":"layout","before":["model"],"inputs":["request"],"weights":["weights"],"cpu":true,"at_use":"always"}],
            "workers":1,"timeout_seconds":2})).unwrap();
        let calls=Arc::new(Mutex::new(Vec::<String>::new()));let mut b=bindings();
        for name in ["readers","weights","model","unused"] {
            let (calls,key)=(calls.clone(),name.to_string());
            b.insert("load",name,Arc::new(move |_|{calls.lock().unwrap().push(key.clone());Ok(json!(true))}));
            b.insert("validate_load",name,Arc::new(|v|Ok(json!(v==true))));
        }
        let (sent,cpu)=mpsc::channel();b.insert("cpu","layout",Arc::new(move |_|{sent.send(()).unwrap();Ok(json!(true))}));
        let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();
        runtime.request("one",json!({"request":1})).unwrap();cpu.recv_timeout(Duration::from_secs(1)).unwrap();
        runtime.before_use(None,"readers").unwrap();assert_eq!(*calls.lock().unwrap(),["readers"]);
        runtime.invoke(Some("one"),"model",||Ok(())).unwrap();
        assert_eq!(*calls.lock().unwrap(),["readers","weights","model"]);
        runtime.close().unwrap();assert!(!calls.lock().unwrap().iter().any(|name|name=="unused"));
    }
    #[test]
    fn at_use_initializer_dependencies_activate_transitively() {
        let d:Declaration=serde_json::from_value(json!({"components":[
            {"name":"first","required_preparations":["first-layout"]},
            {"name":"model","required_preparations":["layout"]}],
            "preparations":[
                {"name":"first-layout","before":["first"],"inputs":["request"],"at_use":"always"},
                {"name":"layout","before":["model"],"inputs":["request"],"after":["first-layout"],"at_use":"always"}],
            "workers":1,"timeout_seconds":2})).unwrap();
        let calls=Arc::new(Mutex::new(Vec::new()));let mut b=Bindings::default();
        for name in ["first","model"] {for operation in ["load","validate_load"] {b.insert(operation,name,Arc::new(|_|Ok(json!(true))));}}
        for name in ["first-layout","layout"] {
            let (calls,key)=(calls.clone(),name.to_string());
            b.insert("run",name,Arc::new(move |_|{calls.lock().unwrap().push(key.clone());Ok(json!(true))}));
            b.insert("validate_run",name,Arc::new(|v|Ok(json!(v==true))));
        }
        let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();runtime.request("one",json!({"request":1})).unwrap();
        runtime.invoke(Some("one"),"model",||Ok(())).unwrap();
        assert_eq!(*calls.lock().unwrap(),["first-layout","layout"]);runtime.close().unwrap();
    }
    #[test]
    fn at_use_defers_gpu_but_not_cpu_and_cancel_prevents_dispatch() {
        let mut d=declaration(); d.preparations[0].at_use=AtUse::Always;
        let gpu=Arc::new(Mutex::new(0)); let observed=gpu.clone();
        let mut b=bindings(); b.insert("run","layout",Arc::new(move |_|{*observed.lock().unwrap()+=1;Ok(json!(true))}));
        let runtime=DeclaredStartup::new(d,b).unwrap(); runtime.start().unwrap();
        let nodes=runtime.request("one",json!({"request":1})).unwrap(); wait_node(&runtime,"dit","weights");
        let cpu=runtime.snapshot()["nodes"].as_array().unwrap().iter().position(|n|n["operation"]=="cpu").unwrap();
        runtime.result(ResultHandle { node:cpu,generation:1 },true,Some(2.)).unwrap();
        assert_eq!(*gpu.lock().unwrap(),0);
        assert!(runtime.cancel(nodes["layout"]).unwrap());
        assert!(runtime.before_use(Some("one"),"dit").unwrap_err().contains("cancelled"));
        runtime.close().unwrap(); assert_eq!(*gpu.lock().unwrap(),0);
    }
    #[test]
    fn static_preload_readiness_does_not_claim_request_readiness() {
        let mut d=declaration(); d.preparations.push(Preparation {name:"backend".into(),before:vec!["dit".into()],inputs:vec![],weights:vec![],after:vec![],cpu:false,at_use:AtUse::Never,serial_group:None,initialize:false});
        d.components[0].required_preparations.as_mut().unwrap().push("backend".into());
        let mut b=bindings(); for op in ["run","validate_run"] { b.insert(op,"backend",Arc::new(|_|Ok(json!(true)))); }
        let runtime=DeclaredStartup::new(d,b).unwrap(); runtime.start().unwrap();
        wait_node(&runtime,"dit","weights");wait_node(&runtime,"backend","static");
        assert_eq!(runtime.snapshot()["preload_ready"],true);
        assert!(runtime.before_use(None,"dit").is_err()); runtime.close().unwrap();
    }
    #[test]
    fn capability_projection_keeps_transitive_initializer_weights() {
        let mut d=declaration();
        d.components.insert(0,Component{name:"encoder".into(),after:vec![],required_preparations:Some(vec![]),reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false});
        d.components.push(Component{name:"unused".into(),after:vec![],required_preparations:Some(vec![]),reader_limit_mib:0,prefetch:false,load_at_use:false,initialize:false});
        d.preparations[0].weights.push("encoder".into());d.consumers=Some(vec!["dit".into()]);
        let mut b=bindings();for name in ["encoder","unused"] {for op in ["load","validate_load"]{b.insert(op,name,Arc::new(|_|Ok(json!(true))));}}
        let runtime=DeclaredStartup::new(d,b).unwrap();
        assert_eq!(runtime.declaration.components.iter().map(|c|c.name.as_str()).collect::<Vec<_>>(),vec!["encoder","dit"]);
    }
    #[test]
    fn close_interrupts_transport_before_joining_running_work() {
        let (running,observed)=mpsc::channel();let (cancel,released)=mpsc::channel();let released=Mutex::new(released);
        let mut b=bindings();b.insert("load","dit",Arc::new(move |_|{running.send(()).unwrap();released.lock().unwrap().recv_timeout(Duration::from_secs(2)).map_err(|e|e.to_string())?;Ok(json!(true))}));
        b.insert("cancel","",Arc::new(move |_|{cancel.send(()).unwrap();Ok(Value::Null)}));
        let runtime=DeclaredStartup::new(declaration(),b).unwrap();runtime.start().unwrap();observed.recv_timeout(Duration::from_secs(1)).unwrap();
        runtime.close().unwrap(); assert!(runtime.before_use(None,"dit").is_err());
    }
    #[test]
    fn repeated_requests_release_old_nodes_and_reject_stale_result_handles() {
        let runtime=DeclaredStartup::new(declaration(),bindings()).unwrap();runtime.start().unwrap();
        let first=runtime.request("one",json!({"request":1})).unwrap()["layout"];
        runtime.before_use(Some("one"),"dit").unwrap();
        let bound=runtime.snapshot()["nodes"].as_array().unwrap().len();
        for index in 2..40 {
            let scope=index.to_string();runtime.request(&scope,json!({"request":index})).unwrap();
            runtime.before_use(Some(&scope),"dit").unwrap();
            assert_eq!(runtime.snapshot()["nodes"].as_array().unwrap().len(),bound);
            assert!(runtime.result(first,false,None).unwrap_err().contains("expired"));
            assert!(runtime.cancel(first).is_err());
        }
        runtime.close().unwrap();
    }
    #[test]
    fn preparation_dependencies_wait_for_validated_results_without_delaying_cpu() {
        let mut d=declaration();
        d.preparations.insert(0,Preparation{name:"backend".into(),before:vec!["dit".into()],inputs:vec![],weights:vec![],after:vec![],cpu:false,at_use:AtUse::Never,serial_group:None,initialize:false});
        d.preparations[1].after=vec!["backend".into()];
        d.components[0].required_preparations.as_mut().unwrap().push("backend".into());
        let (release,wait)=mpsc::channel();let wait=Mutex::new(wait);
        let (cpu_sent,cpu_received)=mpsc::channel();
        let mut b=bindings();
        b.insert("run","backend",Arc::new(move |_|{wait.lock().unwrap().recv_timeout(Duration::from_secs(2)).map_err(|e|e.to_string())?;Ok(json!("backend"))}));
        b.insert("validate_run","backend",Arc::new(|v|Ok(json!(v=="backend"))));
        b.insert("cpu","layout",Arc::new(move |_|{cpu_sent.send(()).unwrap();Ok(json!(true))}));
        b.insert("run","layout",Arc::new(|inputs|{assert_eq!(inputs["preparations"]["backend"],"backend");Ok(json!(true))}));
        let runtime=DeclaredStartup::new(d,b).unwrap();runtime.start().unwrap();runtime.request("one",json!({"request":1})).unwrap();
        cpu_received.recv_timeout(Duration::from_secs(1)).unwrap();assert!(!runtime.check_before_use(Some("one"),"dit").unwrap());
        release.send(()).unwrap();runtime.invoke(Some("one"),"dit",||Ok(())).unwrap();runtime.close().unwrap();
    }
    #[test]
    fn request_scope_cannot_change_during_an_actual_invocation() {
        let runtime=Arc::new(DeclaredStartup::new(declaration(),bindings()).unwrap());runtime.start().unwrap();
        runtime.request("one",json!({"request":1})).unwrap();runtime.before_use(Some("one"),"dit").unwrap();
        let (entered,observed)=mpsc::channel();let (release,wait)=mpsc::channel();let compute=runtime.clone();
        let task=thread::spawn(move ||compute.invoke(Some("one"),"dit",||{entered.send(()).unwrap();wait.recv_timeout(Duration::from_secs(2)).unwrap();Ok(())}));
        observed.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(runtime.request("two",json!({"request":2})).unwrap_err().contains("computation is still active"));
        release.send(()).unwrap();task.join().unwrap().unwrap();
        runtime.request("two",json!({"request":2})).unwrap();runtime.invoke(Some("two"),"dit",||Ok(())).unwrap();runtime.close().unwrap();
    }
}


impl DeclaredStartup {
    // Only receipts already published by the physical transport participate.
    // An unpublished dependency or unused lazy loader must leave IO running.
    pub(crate) fn published_load_pending(&self,name:&str)->Result<bool> {
        let state=self.shared.state.lock().unwrap();
        if state.closed || !state.started {return Err("cohort publication requires an open started declaration".into());}
        let id=*state.weights.get(name).ok_or_else(||format!("unknown declared startup component: {name}"))?;
        Self::released_node_pending(&state,id)
    }
    fn released_node_pending(state:&State,id:usize)->Result<bool> {
        let node=&state.nodes[id];
        match &node.status {
            Status::Running|Status::Validating=>Ok(true),Status::Ready(_)=>Ok(false),
            Status::Failed(error)=>Err(error.clone()),Status::Cancelled=>Err("startup work cancelled".into()),
            Status::Waiting=>{
                if node.at_use && !node.used {return Ok(false);}
                for dependency in &node.dependencies {if !state.nodes[*dependency].status.ready()? {return Ok(false);}}
                Ok(true)
            },
        }
    }
    pub(crate) fn released_owner_pending(&self,owned:impl Fn(&str,&str)->bool)->Result<bool> {
        let state=self.shared.state.lock().unwrap();
        if state.closed {return Err("startup owner cancelled".into());}
        for (id,node) in state.nodes.iter().enumerate() {
            if owned(node.operation,&node.name) && Self::released_node_pending(&state,id)? {return Ok(true);}
        }
        Ok(false)
    }
}
