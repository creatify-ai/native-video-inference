//! Runtime-scalar launch of offline-compiled, unchanged reference router kernels.
//! This file has no dependency on Python, Torch, Triton, or another Rust crate.
use std::{collections::HashMap, ffi::{c_char,c_int,c_uint,c_void,CStr,CString},fs,path::Path,ptr};
pub type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
pub type Stream = *mut c_void;
pub type DevicePtr = u64;
type Module = *mut c_void;
type Function = *mut c_void;
unsafe extern "C" {
    fn cuModuleLoadData(out:*mut Module,image:*const c_void)->c_int;
    fn cuModuleUnload(module:Module)->c_int;
    fn cuModuleGetFunction(out:*mut Function,module:Module,name:*const c_char)->c_int;
    fn cuFuncSetAttribute(function:Function,attribute:c_int,value:c_int)->c_int;
    fn cuLaunchKernel(function:Function,gx:c_uint,gy:c_uint,gz:c_uint,bx:c_uint,by:c_uint,bz:c_uint,
        shared:c_uint,stream:Stream,params:*mut *mut c_void,extra:*mut *mut c_void)->c_int;
    fn cuCtxSynchronize()->c_int;
    fn cuGetErrorString(code:c_int,text:*mut *const c_char)->c_int;
}
fn check(code:c_int)->Result<()> {
    if code==0{return Ok(());}
    let mut text=ptr::null();unsafe{cuGetErrorString(code,&mut text);}
    let message=if text.is_null(){"unknown CUDA error".into()}else{unsafe{CStr::from_ptr(text)}.to_string_lossy()};
    Err(format!("CUDA driver {code}: {message}").into())
}
#[derive(Debug,Clone)]
pub struct BufferSpec {pub name:String,pub input:bool,pub file:String,pub bytes:usize}
#[derive(Debug,Clone,Copy)]
pub struct Plan {pub sequence:usize,pub heads:usize,pub groups:usize,pub kept:usize,pub topk_block:usize,pub topk_warps:usize}
impl Plan {
    pub fn new(sequence:usize,heads:usize)->Result<Self> {
        if sequence==0||sequence>524288||![7,14,28,56].contains(&heads){return Err("dynamic router requires 1<=S<=524288 and heads in {7,14,28,56}".into());}
        // Original router address arithmetic is int32: larger layouts use
        // CP local heads or independent head groups on one GPU.
        if sequence>262144 && heads==56{return Err("router above 262144 requires local or grouped heads".into());}
        let groups=sequence.div_ceil(64);
        // Exactly ceil((1-.75)*G), round up to eight, capped to G.
        let kept=groups.div_ceil(4).div_ceil(8)*8;
        Ok(Self{sequence,heads,groups,kept:kept.min(groups),topk_block:groups.next_power_of_two(),topk_warps:if groups>=1024{4}else{2}})
    }
    pub fn buffers(self)->Vec<BufferSpec> {
        [("q",true,self.sequence*self.heads*128*2),("k",true,self.sequence*self.heads*128*2),
         ("qp",false,self.heads*self.groups*4*128*2),("kp",false,self.heads*self.groups*4*128*2),
         ("scores",false,self.heads*self.groups*self.groups*4),("indices",false,self.heads*self.groups*self.kept*4)]
            .into_iter().map(|(name,input,bytes)|BufferSpec{name:name.into(),input,file:format!("{name}.raw"),bytes}).collect()
    }
    pub fn block_sizes(self)->Vec<i32> {(0..self.groups).map(|g|(self.sequence-g*64).min(64) as i32).collect()}
    fn specialization(values:&[usize])->String {values.iter().map(|v|if *v==1{'o'}else if v%16==0{'a'}else{'x'}).collect()}
    pub fn variants(self)->[String;4] {
        let g=self.groups;let s=self.sequence;let h=self.heads;
        let pool=Self::specialization(&[s*h*128,h*128,128,g*4*128,128,s,h]);
        let score=Self::specialization(&[128,g*4*128,128,g*4*128,g,g*g,g*4,s.div_ceil(16),s.div_ceil(16),g,g]);
        let topk=Self::specialization(&[g,self.kept]);
        [format!("pool_q-{pool}"),format!("pool_k-{pool}"),format!("scores-{score}"),format!("topk-b{}-w{}-{topk}",self.topk_block,self.topk_warps)]
    }
    fn scalar(self,kernel:&str,name:&str)->Result<u64> {
        let kernel=kernel.split('-').next().unwrap();
        let g=self.groups;let s=self.sequence;let h=self.heads;
        let value=match (kernel,name) {
            ("pool_q"|"pool_k","stride_xb")=>s*h*128,
            ("pool_q"|"pool_k","stride_xt")=>h*128,
            ("pool_q"|"pool_k","stride_xh"|"stride_yn")=>128,
            ("pool_q"|"pool_k","stride_yl")=>g*4*128,
            ("pool_q"|"pool_k","S")=>s,
            ("pool_q"|"pool_k","H")=>h,
            ("pool_q","SCALE")=>return Ok(((128f64.powf(-0.5)*1.4426950408889634) as f32).to_bits() as u64),
            ("pool_k","SCALE")=>return Ok(1f32.to_bits() as u64),
            ("scores","stride_qm"|"stride_kn")=>128,
            ("scores","stride_ql"|"stride_kl")=>g*4*128,
            ("scores","stride_om"|"NOUT"|"MOUT")=>g,
            ("scores","stride_ol")=>g*g,
            ("scores","M")=>g*4,
            ("scores","M_VALID"|"N_VALID")=>s.div_ceil(16),
            ("topk","G")=>g,
            ("topk","K")=>self.kept,
            _=>return Err(format!("unknown dynamic router scalar {kernel}.{name}").into()),
        };
        if value>i32::MAX as usize{return Err("dynamic router scalar exceeds int32".into());}
        Ok(value as u64)
    }
}
enum Arg {Pointer(String),Dynamic(String),Bits(u64)}
struct Kernel {name:String,module:Module,function:Function,threads:u32,shared:u32,args:Vec<Arg>}
impl Drop for Kernel {fn drop(&mut self){unsafe{cuModuleUnload(self.module);}}}
pub struct DynamicProgram {pub buffers:Vec<BufferSpec>,pub plan:Plan,kernels:Vec<Kernel>}
impl DynamicProgram {
    /// The caller owns CUDA context and buffers; loading selects four fixed-code variants.
    pub fn load_for_shape(root:&Path,sequence:usize,heads:usize)->Result<Self> {
        let plan=Plan::new(sequence,heads)?;
        let manifest=fs::read_to_string(root.join("launch.tsv"))?;
        let mut lines=manifest.lines();
        if lines.next()!=Some("H3_DYNAMIC_ROUTER_V1"){return Err("invalid dynamic router manifest".into());}
        let bound=lines.next().and_then(|s|s.strip_prefix("max_sequence\t"))
            .and_then(|s|s.parse::<usize>().ok()).ok_or("missing dynamic router bounds")?;
        if ![262144,524288].contains(&bound) || sequence>bound {return Err("request exceeds compiled dynamic router package bounds".into());}
        let selected=plan.variants();
        let mut result=Self{buffers:plan.buffers(),plan,kernels:vec![]};
        let mut active=false;
        for line in lines {
            let f:Vec<_>=line.split('\t').collect();
            match f.as_slice() {
                ["kernel",name,file,symbol,threads,shared]=>{
                    active=selected.iter().any(|label|label==name);
                    if !active{continue;}
                    if Path::new(file).components().count()!=1{return Err("unsafe router CUBIN filename".into());}
                    let data=crate::startup_trace::read(root.join(file))?;let mut module=ptr::null_mut();
                    check(unsafe{cuModuleLoadData(&mut module,data.as_ptr().cast())})?;
                    let mut kernel=Kernel{name:name.to_string(),module,function:ptr::null_mut(),threads:threads.parse()?,shared:shared.parse()?,args:vec![]};
                    let symbol=CString::new(*symbol)?;
                    check(unsafe{cuModuleGetFunction(&mut kernel.function,module,symbol.as_ptr())})?;
                    if kernel.shared>0{check(unsafe{cuFuncSetAttribute(kernel.function,8,kernel.shared as i32)})?;}
                    if kernel.threads==0||kernel.threads>1024{return Err("invalid router threads".into());}
                    result.kernels.push(kernel);
                },
                ["arg",encoding,value]=>{
                    if !active{continue;}
                    let arg=match *encoding{"ptr"=>Arg::Pointer(value.to_string()),"dynamic"=>Arg::Dynamic(value.to_string()),"bits"=>Arg::Bits(value.parse()?),_=>return Err("unknown router argument encoding".into())};
                    result.kernels.last_mut().ok_or("argument before kernel")?.args.push(arg);
                },
                _=>return Err(format!("invalid dynamic router manifest line {line}").into()),
            }
        }
        if result.kernels.len()!=4{return Err("dynamic router requires exactly four selected variants".into());}
        // Inventory ordering is independent of any request; enforce stage order here.
        result.kernels.sort_by_key(|k|selected.iter().position(|name|name==&k.name).unwrap());
        if result.kernels.iter().map(|k|k.name.as_str()).collect::<Vec<_>>() != selected.iter().map(String::as_str).collect::<Vec<_>>() {return Err("duplicate or missing router stage variant".into());}
        Ok(result)
    }
    pub fn launch(&self,pointers:&HashMap<String,DevicePtr>,stream:Stream)->Result<()> {
        for b in &self.buffers {
            let p=*pointers.get(&b.name).ok_or("missing router buffer")?;
            if p==0||p%16!=0{return Err(format!("router buffer {} must be 16-byte aligned",b.name).into());}
        }
        let p=self.plan;
        for kernel in &self.kernels {
            let grid=match kernel.name.split('-').next().unwrap(){"pool_q"|"pool_k"=>[p.groups*4,p.heads,1],"scores"=>[(p.groups*4).div_ceil(128),(p.groups*4).div_ceil(128),p.heads],_=>[p.heads*p.groups,1,1]};
            let mut values:Vec<u64>=kernel.args.iter().map(|arg|match arg{
                Arg::Pointer(name)=>pointers.get(name).copied().ok_or_else(||format!("missing {name}").into()),
                Arg::Dynamic(name)=>p.scalar(&kernel.name,name),Arg::Bits(value)=>Ok(*value),
            }).collect::<Result<_>>()?;
            let mut args:Vec<*mut c_void>=values.iter_mut().map(|v|(v as *mut u64).cast()).collect();
            check(unsafe{cuLaunchKernel(kernel.function,grid[0] as u32,grid[1] as u32,grid[2] as u32,kernel.threads,1,1,kernel.shared,stream,args.as_mut_ptr(),ptr::null_mut())})
                .map_err(|e|format!("{}: {e}",kernel.name))?;
        }
        Ok(())
    }
}
impl Drop for DynamicProgram {fn drop(&mut self){unsafe{cuCtxSynchronize();}}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn plan_boundaries(){
        for s in [1,16,17,63,64,65,16384,16385,65472,65473,65536,65537,262144] {
            let p=Plan::new(s,56).unwrap();
            assert_eq!(p.groups,(s+63)/64);assert!(p.kept>0&&p.kept<=p.groups);
            assert_eq!(p.block_sizes().iter().map(|x|*x as usize).sum::<usize>(),s);
            assert_eq!(p.topk_warps,if p.groups>=1024{4}else{2});
        }
        assert!(Plan::new(0,56).is_err());assert!(Plan::new(262145,56).is_err());assert!(Plan::new(63510,55).is_err());
        for sequence in [262145,376538,524288] {for heads in [7,14,28] {
            let p=Plan::new(sequence,heads).unwrap();assert_eq!(p.topk_block,8192);
            assert!(p.scalar("pool_q","stride_xb").unwrap()<=i32::MAX as u64);
            assert!(heads*p.groups*p.groups<=i32::MAX as usize);
        }}
        assert!(Plan::new(524289,7).is_err());
        assert_eq!(Plan::new(63510,56).unwrap().kept,256);
        assert_eq!(Plan::new(43610,56).unwrap().kept,176);
        assert_eq!(Plan::new(107098,56).unwrap().kept,424);
    }
}
