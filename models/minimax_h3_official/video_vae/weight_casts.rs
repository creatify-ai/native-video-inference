// Decoder weight casts hoisted out of the per-tile replay (H3_VAE_HOIST_WEIGHT_CASTS=0 keeps them).
// The compiled decoder converts every FP32 linear weight and bias to FP16 right before its GEMM,
// on every tile: 435 casts and 4.8 GB of FP16 per 256x256 tile. Each parameter is read by
// nothing but its cast, so at load the same CUBIN converts it once into the first half of its
// own buffer, and the GEMM reads that buffer where it read the cast's scratch output. The GEMM
// operands are the same FP16 bits as before; no memory is added.
#[derive(Clone,Copy,PartialEq,Debug)]
enum SpecPath{Arg(usize),Out,Output(usize)}
struct CastHoist{
    /// Hoisted cast op index -> (FP32 parameter buffer, FP16 scratch buffer, elements).
    casts:HashMap<usize,(usize,usize,usize)>,
    /// (consumer op index, operand, scratch buffer it read, parameter buffer it reads now).
    redirects:Vec<(usize,SpecPath,usize,usize)>,
}
const CAST_SIGNATURE:[(&str,&str);4]=[("in_ptr0","*fp32"),("out_ptr0","*fp16"),("xnumel","i32"),("XBLOCK","constexpr")];
fn elem_bytes(dtype:&str)->Result<usize>{Ok(match dtype{"torch.float16"|"torch.bfloat16"=>2,"torch.float32"|"torch.int32"=>4,
    "torch.int64"=>8,"torch.uint8"|"torch.bool"|"torch.int8"=>1,_=>return Err(format!("unknown dtype {dtype}").into())})}
fn spec_at<'a>(op:&'a Value,path:SpecPath)->&'a Value{match path{SpecPath::Arg(i)=>&op["args"][i],SpecPath::Out=>&op["kwargs"]["out"],SpecPath::Output(i)=>&op["outputs"][i]}}
/// Every tensor operand of an op as (operand, buffer, reads, writes). Kernel operands follow the
/// CUBIN signature names; an unrecognized name counts as both read and written.
fn op_operands(op:&Value,kernels:&Value)->Result<Vec<(SpecPath,usize,bool,bool)>>{
    let mut out=Vec::new();
    let tensor=|v:&Value|v.is_object()&&v.get("buffer").is_some();
    if op["kind"]=="kernel" {
        // serde_json maps are key-sorted here, so rebuild Inductor's parameter order: pointers lead,
        // in_out_ptr*, then in_ptr*, then out_ptr*, each by index.
        let signature=kernels[text(&op["kernel"])?]["signature"].as_object().ok_or("kernel signature")?;
        let rank=|name:&str|->Option<(u8,u32)>{
            let (kind,index)=if let Some(i)=name.strip_prefix("in_out_ptr"){(0,i)}else if let Some(i)=name.strip_prefix("in_ptr"){(1,i)}
                else if let Some(i)=name.strip_prefix("out_ptr"){(2,i)}else{return None;};
            Some((kind,index.parse().ok()?))};
        let mut names:Vec<(u8,u32,&str)>=Vec::new();let mut known=true;
        for (name,kind) in signature{
            if !kind.as_str().is_some_and(|t|t.starts_with('*')){continue;}
            match rank(name.as_str()){Some((k,i))=>names.push((k,i,name.as_str())),None=>known=false}
        }
        names.sort_unstable();
        let args=arr(&op["args"])?;
        let leading=args.iter().take_while(|a|a["type"]=="pointer").count();
        if leading!=names.len() || args.iter().filter(|a|a["type"]=="pointer").count()!=leading{known=false;}
        for (i,arg) in args.iter().enumerate(){
            if arg["type"]!="pointer"{continue;}
            let name=if known{names[i].2}else{""};
            let (read,write)=if name.starts_with("in_out_ptr"){(true,true)}else if name.starts_with("in_ptr"){(true,false)}
                else if name.starts_with("out_ptr"){(false,true)}else{(true,true)};
            out.push((SpecPath::Arg(i),num(&arg["buffer"])?,read,write));
        }
        return Ok(out);
    }
    if let Some(args)=op["args"].as_array(){for (i,arg) in args.iter().enumerate(){if tensor(arg){out.push((SpecPath::Arg(i),num(&arg["buffer"])?,true,false));}}}
    if tensor(&op["kwargs"]["out"]){out.push((SpecPath::Out,num(&op["kwargs"]["out"]["buffer"])?,false,true));}
    if let Some(outputs)=op["outputs"].as_array(){for (i,o) in outputs.iter().enumerate(){if tensor(o){out.push((SpecPath::Output(i),num(&o["buffer"])?,false,true));}}}
    Ok(out)
}
/// Last byte (exclusive) a strided view addresses.
fn view_end(spec:&Value)->Result<usize>{
    let shape=ints(&spec["shape"])?;let stride=ints(&spec["stride"])?;
    if shape.iter().any(|&d|d==0){return Ok(0);}
    let span:i64=shape.iter().zip(&stride).map(|(d,s)|(d-1)*s).sum();
    if span<0 || stride.iter().any(|&s|s<0){return Err("negative view stride".into());}
    Ok(num(&spec["offset"])?+(span as usize+1)*elem_bytes(text(&spec["dtype"])?)?)
}
impl CastHoist {
    fn enabled(manifest:&Value)->bool{
        std::env::var("H3_VAE_HOIST_WEIGHT_CASTS").as_deref()!=Ok("0") && manifest["stage"]=="video_vae_decoder_tile"
    }
    fn plan(manifest:&Value)->Result<Self>{
        let ops=arr(&manifest["ops"])?;let buffers=arr(&manifest["buffers"])?;let kernels=&manifest["kernels"];
        let role=|id:usize|buffers.get(id).and_then(|b|b["role"].as_str()).unwrap_or("");
        let contiguous=|spec:&Value|->Result<bool>{Ok(num(&spec["offset"])?==0 && ints(&spec["stride"])?==decoder_contiguous(&ints(&spec["shape"])?.iter().map(|&d|d as usize).collect::<Vec<_>>()).iter().map(|&d|d as i64).collect::<Vec<_>>())};
        let mut casts=HashMap::new();
        for (i,op) in ops.iter().enumerate(){
            if op["kind"]!="kernel"{continue;}
            let kernel=&kernels[text(&op["kernel"])?];
            if !text(&kernel["symbol"])?.starts_with("triton_poi_fused__to_copy_"){continue;}
            let signature=kernel["signature"].as_object().ok_or("kernel signature")?;
            if signature.len()!=CAST_SIGNATURE.len() || !CAST_SIGNATURE.iter().all(|(n,t)|signature.get(*n).and_then(Value::as_str)==Some(*t)){continue;}
            let args=arr(&op["args"])?;
            if args.len()!=5 || args[0]["type"]!="pointer" || args[1]["type"]!="pointer" || args[2]["type"]!="i32" || args[3]["bits"]!=0 || args[4]["bits"]!=0{continue;}
            let (input,output)=(&args[0],&args[1]);
            let elements:usize=ints(&input["shape"])?.iter().map(|&d|d as usize).product();
            if input["dtype"]!="torch.float32" || output["dtype"]!="torch.float16" || input["shape"]!=output["shape"] ||
               !contiguous(input)? || !contiguous(output)? || args[2]["bits"].as_u64()!=Some(elements as u64){continue;}
            let (parameter,scratch)=(num(&input["buffer"])?,num(&output["buffer"])?);
            if role(parameter)!="parameter" || role(scratch)!="scratch" || num(&buffers[parameter]["bytes"])?<elements*4{continue;}
            casts.insert(i,(parameter,scratch,elements));
        }
        // A parameter converted in place must have no reader but its casts.
        let parameters:std::collections::HashSet<usize>=casts.values().map(|c|c.0).collect();
        let mut rejected:std::collections::HashSet<usize>=std::collections::HashSet::new();
        for (i,op) in ops.iter().enumerate(){
            for (path,buffer,_,_) in op_operands(op,kernels)?{
                if parameters.contains(&buffer) && !(casts.contains_key(&i) && path==SpecPath::Arg(0)){rejected.insert(buffer);}
            }
        }
        // Each cast's scratch output: every use until the scratch is written again must be a
        // read inside the converted range; redirect those reads to the parameter.
        let mut redirects=Vec::new();
        let mut order:Vec<usize>=casts.keys().copied().collect();order.sort_unstable();
        for &i in &order {
            let (parameter,scratch,elements)=casts[&i];
            if rejected.contains(&parameter){continue;}
            let mut mine=Vec::new();let mut ok=true;
            for (j,op) in ops.iter().enumerate().skip(i+1){
                let touches:Vec<_>=op_operands(op,kernels)?.into_iter().filter(|t|t.1==scratch).collect();
                if touches.is_empty(){continue;}
                if casts.get(&j).is_some_and(|c|c.1==scratch){break;}
                let (reads,writes)=(touches.iter().any(|t|t.2),touches.iter().any(|t|t.3));
                if reads && writes{ok=false;break;}
                if writes{break;}
                for (path,_,_,_) in touches{
                    if view_end(spec_at(op,path))?>elements*2{ok=false;}
                    mine.push((j,path,scratch,parameter));
                }
                if !ok{break;}
            }
            if ok{redirects.extend(mine);}else{rejected.insert(parameter);}
        }
        casts.retain(|_,c|!rejected.contains(&c.0));
        redirects.retain(|r|!rejected.contains(&r.3));
        Ok(Self{casts,redirects})
    }
    /// The same plan with the hoisted casts removed and their consumers reading the converted
    /// parameters. Op indices and operand positions are those of the base plan; a resized plan
    /// keeps both, which is checked through the scratch buffer each redirect expects.
    fn apply(&self,plan:&Value)->Result<Value>{
        let mut out=plan.clone();
        let ops=out["ops"].as_array_mut().ok_or("plan ops")?;
        for &(j,path,scratch,parameter) in &self.redirects{
            let spec=match path{SpecPath::Arg(i)=>&mut ops[j]["args"][i],SpecPath::Out=>&mut ops[j]["kwargs"]["out"],SpecPath::Output(i)=>&mut ops[j]["outputs"][i]};
            if spec["buffer"].as_u64()!=Some(scratch as u64){return Err("weight-cast redirect does not match the plan".into());}
            spec["buffer"]=json!(parameter);
        }
        let mut index=0;
        ops.retain(|_|{let keep=!self.casts.contains_key(&index);index+=1;keep});
        out["weight_casts_hoisted"]=json!(self.casts.len());
        Ok(out)
    }
    /// Convert each hoisted parameter once with its own cast kernel: FP32 -> FP16 into a staging
    /// buffer, then into the first half of the parameter's allocation.
    fn convert(&self,manifest:&Value,arena:&mut Arena,buffers:&[DevicePtr],kernels:&HashMap<String,Kernel>)->Result<()>{
        let ops=arr(&manifest["ops"])?;
        let largest=self.casts.values().map(|c|c.2*2).max().unwrap_or(0);
        if largest==0{return Ok(());}
        let staging=arena.alloc(largest)?;
        let mut done=std::collections::HashSet::new();
        let mut order:Vec<usize>=self.casts.keys().copied().collect();order.sort_unstable();
        for i in order {
            let (parameter,_,elements)=self.casts[&i];
            if !done.insert(parameter){continue;}
            let op=&ops[i];let kernel=kernels.get(text(&op["kernel"])?).ok_or("unknown CUBIN")?;
            let grid=ints(&op["grid"])?;
            let source=buffers[parameter];
            if source==0{return Err("weight-cast parameter is not resident".into());}
            let mut values:[u64;5]=[source,staging,elements as u64,0,0];
            if kernel.params!=values.len(){return Err("weight-cast kernel ABI changed".into());}
            let mut pointers:Vec<*mut c_void>=values.iter_mut().map(|v|(v as *mut u64).cast()).collect();
            check(unsafe{cuLaunchKernel(kernel.function,grid[0] as _,grid[1] as _,grid[2] as _,kernel.threads,1,1,kernel.shared,arena.stream,pointers.as_mut_ptr(),ptr::null_mut())})?;
            check(unsafe{cuMemcpyDtoDAsync_v2(source,staging,elements*2,arena.stream)})?;
        }
        arena.sync()
    }
}
