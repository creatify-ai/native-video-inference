//! Exercise a native backend with an older primary CUDA closure already loaded.
#[cfg(target_os="linux")]
fn run()->Result<(),Box<dyn std::error::Error>>{
    use native_core_infra::native_backend::{Pin,Registry};
    use serde::Deserialize;
    use serde_json::{json,Value};
    use std::{ffi::{c_char,c_int,c_void,CStr,CString},io::Write,time::{Duration,Instant}};
    unsafe extern "C" {fn dlopen(path:*const c_char,flags:c_int)->*mut c_void;fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;}
    #[derive(Deserialize)]
    struct Step {backend:String,input:String,await_field:Option<String>,await_value:Option<Value>,timeout_seconds:Option<u64>}
    #[derive(Deserialize)]
    struct Spec {primary_runtime:Pin,#[serde(default)] primary_libraries:Vec<Pin>,registry:Pin,steps:Vec<Step>,
        #[serde(default)] metadata_only:bool}
    unsafe fn open(pin:&Pin)->Result<*mut c_void,Box<dyn std::error::Error>>{
        pin.verify()?;let name=CString::new(pin.path.as_os_str().as_encoded_bytes())?;
        let handle=dlopen(name.as_ptr(),2|256); // NOW | GLOBAL, like the primary executable
        if handle.is_null(){return Err(format!("primary dlopen failed: {}",pin.path.display()).into());}Ok(handle)
    }
    unsafe fn symbol(handle:*mut c_void,name:&CStr)->Result<*mut c_void,Box<dyn std::error::Error>>{
        let result=dlsym(handle,name.as_ptr());if result.is_null(){return Err("primary runtime symbol missing".into());}Ok(result)
    }
    fn check(code:i32)->Result<(),Box<dyn std::error::Error>>{
        if code==0{Ok(())}else{Err(format!("primary CUDA error {code}").into())}
    }
    let path=std::env::args().nth(1).ok_or("probe specification required")?;
    let spec:Spec=serde_json::from_slice(&std::fs::read(path)?)?;
    if spec.steps.len()>128{return Err("too many probe commands".into());}
    let primary=unsafe{open(&spec.primary_runtime)?};
    for library in &spec.primary_libraries{unsafe{open(library)?;}}
    if spec.metadata_only{
        let registry=Registry::read(spec.registry)?;
        let metadata=registry.inspect("ltx25")?;
        println!("{}",json!({"passed":true,"metadata":metadata,"gpu_requested":false,
            "process_maps":std::fs::read_to_string("/proc/self/maps")?}));
        return Ok(());
    }
    let set:unsafe extern "C" fn(i32)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaSetDevice")?)};
    let version:unsafe extern "C" fn(*mut i32)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaRuntimeGetVersion")?)};
    let malloc:unsafe extern "C" fn(*mut *mut c_void,usize)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaMalloc")?)};
    let memset:unsafe extern "C" fn(*mut c_void,i32,usize)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaMemset")?)};
    let memcpy:unsafe extern "C" fn(*mut c_void,*const c_void,usize,i32)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaMemcpy")?)};
    let free:unsafe extern "C" fn(*mut c_void)->i32=unsafe{std::mem::transmute(symbol(primary,c"cudaFree")?)};
    let mut primary_version=0;check(unsafe{version(&mut primary_version)})?;check(unsafe{set(0)})?;
    let mut allocation=std::ptr::null_mut();check(unsafe{malloc(&mut allocation,1<<20)})?;
    check(unsafe{memset(allocation,0x31,1<<20)})?;
    let mut registry=Registry::read(spec.registry)?;let mut responses=Vec::new();
    for step in spec.steps{
        let started=Instant::now();let deadline=started+Duration::from_secs(step.timeout_seconds.unwrap_or(1800));
        let response=loop{
            let value=registry.call(&step.backend,&step.input)?;
            if value["failed"]==true{return Err(format!("backend failed: {}",value["error"]).into());}
            match &step.await_field{
                None=>break value,
                Some(field) if value.get(field)==step.await_value.as_ref()=>break value,
                _ if Instant::now()>=deadline=>return Err("backend probe phase timed out".into()),
                _=>std::thread::sleep(Duration::from_millis(5)),
            }
        };
        eprintln!("{}",json!({"event":"native-backend-probe-step","seconds":started.elapsed().as_secs_f64(),"input":step.input}));
        responses.push(json!({"seconds":started.elapsed().as_secs_f64(),"details":response}));
    }
    let mut bytes=[0u8;64];check(unsafe{memcpy(bytes.as_mut_ptr().cast(),allocation,bytes.len(),2)})?;
    if bytes.iter().any(|v|*v!=0x31){return Err("primary owner allocation changed".into());}
    check(unsafe{free(allocation)})?;
    let mut after=0;check(unsafe{version(&mut after)})?;
    if after!=primary_version{return Err("primary runtime binding changed".into());}
    let report=json!({"passed":true,"owner_pid":std::process::id(),"owner_instance":native_core_infra::process_identity::current(),"primary_cuda_runtime":primary_version,
        "primary_allocation_preserved":true,"responses":responses,
        "process_maps":std::fs::read_to_string("/proc/self/maps")?});
    let mut out=std::io::stdout().lock();writeln!(out,"{report}")?;out.flush()?;Ok(())
}
fn main(){
    #[cfg(target_os="linux")]
    if let Err(error)=run(){eprintln!("{error}");std::process::exit(1);}
    #[cfg(not(target_os="linux"))]
    panic!("native backend probes run on Modal Linux");
}
