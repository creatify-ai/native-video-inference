//! Thin C ABI for the same Rust executor used by native callers. The host binds
//! model/transport callbacks; no host scheduler or readiness state is required.
use crate::declared_startup::{Bindings, Declaration, DeclaredStartup, Result, ResultHandle};
use crate::declared_startup_gate::StartupGate;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, atomic::{AtomicU64, Ordering}};

pub type Callback = unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_char);
struct Handle { executor: Arc<DeclaredStartup>, gates: Mutex<BTreeMap<String, StartupGate>>, callback: Callback, context: usize }
static HANDLES: OnceLock<Mutex<BTreeMap<u64, Arc<Handle>>>> = OnceLock::new();
static NEXT: AtomicU64 = AtomicU64::new(1);
fn handles() -> &'static Mutex<BTreeMap<u64, Arc<Handle>>> { HANDLES.get_or_init(|| Mutex::new(BTreeMap::new())) }
fn invoke(callback: Callback, context: usize, operation: &str, name: &str, inputs: &Value) -> Result<Value> {
    let request = CString::new(json!({"operation":operation,"name":name,"inputs":inputs}).to_string()).unwrap();
    let mut reply = std::ptr::null_mut();
    unsafe { callback(context as *mut c_void, request.as_ptr(), &mut reply); }
    if reply.is_null() { return Err("model callback returned no result".into()); }
    let owned = unsafe { CString::from_raw(reply) };
    let response: Value = serde_json::from_slice(owned.as_bytes()).map_err(|e| e.to_string())?;
    if response["ok"] != true { return Err(response["error"].as_str().unwrap_or("model callback failed").into()); }
    Ok(response["value"].clone())
}
fn output(operation: impl FnOnce() -> Result<Value>) -> *mut c_char {
    let value = match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => json!({"ok":true,"value":value}),
        Ok(Err(error)) => json!({"ok":false,"error":error}),
        Err(_) => json!({"ok":false,"error":"Rust startup ABI panicked"}),
    };
    CString::new(value.to_string()).unwrap().into_raw()
}
unsafe fn input(raw: *const c_char) -> Result<Value> {
    if raw.is_null() { return Err("startup ABI input missing".into()); }
    serde_json::from_slice(CStr::from_ptr(raw).to_bytes()).map_err(|e| e.to_string())
}
/// Copy a callback reply into Rust-owned storage before returning to Rust.
#[no_mangle]
pub unsafe extern "C" fn native_startup_reply(out: *mut *mut c_char, raw: *const c_char) {
    if !out.is_null() && !raw.is_null() { *out = CStr::from_ptr(raw).to_owned().into_raw(); }
}
#[no_mangle]
pub unsafe extern "C" fn native_startup_free(raw: *mut c_char) { if !raw.is_null() { drop(CString::from_raw(raw)); } }
#[no_mangle]
pub unsafe extern "C" fn native_startup_create(raw: *const c_char, callback: Option<Callback>, context: *mut c_void) -> *mut c_char {
    output(|| {
        let spec = input(raw)?; let callback = callback.ok_or("startup callback missing")?; let context = context as usize;
        let declaration: Declaration = serde_json::from_value(spec["declaration"].clone()).map_err(|e| e.to_string())?;
        let mut bindings = Bindings::default();
        for binding in spec["bindings"].as_array().ok_or("startup bindings missing")? {
            let operation = binding["operation"].as_str().ok_or("callback operation missing")?.to_owned();
            let name = binding["name"].as_str().ok_or("callback name missing")?.to_owned();
            let (op, key) = (operation.clone(), name.clone());
            bindings.insert(&operation, &name, Arc::new(move |inputs| invoke(callback, context, &op, &key, inputs)));
        }
        let executor = Arc::new(DeclaredStartup::new(declaration, bindings)?);
        let admitted = serde_json::to_value(executor.declaration()).map_err(|e| e.to_string())?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let age = executor.elapsed_seconds();
        handles().lock().unwrap().insert(id, Arc::new(Handle { executor, gates: Mutex::new(BTreeMap::new()), callback, context }));
        Ok(json!({"handle":id,"declaration":admitted,"abi":5,"elapsed_seconds":age,
            "capabilities":["preparation_serial_group","preparation_initializer","component_after_preparation","component_initializer","preparation_release"]}))
    })
}
#[no_mangle]
pub unsafe extern "C" fn native_startup_call(id: u64, raw: *const c_char) -> *mut c_char {
    output(|| {
        let command = input(raw)?;
        let handle = handles().lock().unwrap().get(&id).cloned().ok_or("unknown startup handle")?;
        let executor = &handle.executor;
        let name = command["name"].as_str().unwrap_or("");
        let scope = command["scope"].as_str();
        match command["operation"].as_str().ok_or("startup operation missing")? {
            "start" => { executor.start()?; Ok(executor.snapshot()) },
            "request" => Ok(json!(executor.request(scope.ok_or("request scope missing")?, command["inputs"].clone())?)),
            "release_preparation" => Ok(json!(executor.release_preparation(name)?)),
            "check_before_use" => Ok(json!(executor.check_before_use(scope, name)?)),
            "before_use" => { executor.before_use(scope, name)?; Ok(Value::Null) },
            "invoke" => executor.invoke_bound(scope,name,&command["inputs"]),
            "snapshot" => Ok(executor.snapshot()),
            "poll" => { executor.poll()?; Ok(Value::Null) },
            "result" => Ok(json!(executor.result(ResultHandle {
                node:command["node"].as_u64().ok_or("startup node missing")? as usize,
                generation:command["generation"].as_u64().ok_or("startup result generation missing")? },
                command["wait"].as_bool().unwrap_or(true), command["timeout"].as_f64())?)),
            "cancel" => Ok(json!(executor.cancel(ResultHandle {
                node:command["node"].as_u64().ok_or("startup node missing")? as usize,
                generation:command["generation"].as_u64().ok_or("startup result generation missing")? })?)),
            "gate_open" => {
                let token = command["token"].as_str().ok_or("gate token missing")?.to_owned();
                let path = command["path"].as_str().ok_or("gate path missing")?;
                let mut gates = handle.gates.lock().unwrap();
                if gates.contains_key(&token) { return Err("duplicate startup gate".into()); }
                let callback = handle.callback; let context = handle.context; let key = token.clone();
                let publish = Arc::new(move |data: &Value| invoke(callback, context, "publish", &key, data));
                let gate = StartupGate::open(executor.clone(), Path::new(path), &token, publish)?;
                let environment = gate.environment(); gates.insert(token, gate); Ok(environment)
            },
            "gate_status" => handle.gates.lock().unwrap().get(name).ok_or("unknown startup gate")?.status(),
            "gate_poll" => {
                handle.gates.lock().unwrap().get(name).ok_or("unknown startup gate")?.poll()?;
                Ok(Value::Null)
            },
            "gate_close" => {
                let gate = handle.gates.lock().unwrap().remove(name);
                if let Some(mut gate) = gate { gate.close()?; }
                Ok(Value::Null)
            },
            "close" => {
                let gates = std::mem::take(&mut *handle.gates.lock().unwrap());
                for (_, mut gate) in gates { gate.close()?; }
                executor.close()?; Ok(Value::Null)
            },
            "destroy" => {
                executor.close()?;
                handles().lock().unwrap().remove(&id); Ok(Value::Null)
            },
            _ => Err("unknown startup ABI operation".into()),
        }
    })
}
