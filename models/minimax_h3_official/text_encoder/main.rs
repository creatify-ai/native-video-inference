#![recursion_limit = "256"]
#[path="../support/startup_trace.rs"]
pub mod startup_trace;
#[path="../support/debug_audit.rs"]
pub mod debug_audit;
mod pixels;
mod prepare;
mod ref2va;
mod ref2va_mixed;
mod dynamic_request;
mod resize;
mod plan_cache;
mod packed_weights;
#[path="../support/weight_loader.rs"]
mod weight_loader;
mod text_shards;
#[cfg(feature = "cuda")]
mod native;
#[cfg(feature = "cuda")]
mod group;

use std::{env, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len()==9 && arguments[1]=="prepare-structure" {
        let model=Path::new(&arguments[2]);dynamic_request::checked_model(model)?;
        let canvases=arguments[4..6].iter().filter(|s|s.as_str()!="-").map(Path::new).collect::<Vec<_>>();
        let (report,_)=prepare::plan_with_size(model,Path::new(&arguments[3]),&canvases,
            Path::new(&arguments[6]),arguments[7].parse()?,arguments[8].parse()?)?;
        println!("{}",serde_json::to_string(&report)?);return Ok(());
    }
    if arguments.len()==6 && matches!(arguments[1].as_str(),"prepare-ref2va"|"prepare-ref2va-structure") {
        let result=if arguments[1]=="prepare-ref2va-structure" {
            ref2va::plan(Path::new(&arguments[2]),Path::new(&arguments[3]),Path::new(&arguments[4]),Path::new(&arguments[5]))?.0
        }else{ref2va::run(Path::new(&arguments[2]),Path::new(&arguments[3]),Path::new(&arguments[4]),Path::new(&arguments[5]))?};
        println!("{}",serde_json::to_string(&result)?);return Ok(());
    }
    #[cfg(feature = "cuda")]
    if (arguments.len()==6 && arguments[1]=="infer-ref2va") || (arguments.len()==7 && arguments[1]=="validate-ref2va") {
        let result=native::infer_ref2va(Path::new(&arguments[2]),Path::new(&arguments[3]),Path::new(&arguments[4]),Path::new(&arguments[5]),
            arguments.get(6).map(Path::new))?;
        println!("{}",serde_json::to_string(&result)?);return Ok(());
    }
    if arguments.len()==7 && arguments[1]=="resize-rgb" {
        let w=arguments[4].parse()?;let h=arguments[5].parse()?;
        let mode=&arguments[6];if mode!="smart"{return Err("resize-rgb only accepts smart".into());}
        crate::dynamic_request::geometry(w,h)?;
        let (rw,rh)=crate::resize::smart(w,h,32,65536,16777216)?;
        let data=crate::resize::rgb(&crate::startup_trace::read(&arguments[2])?,w,h,rw,rh)?;
        std::fs::write(&arguments[3],&data)?;
        println!("{}",serde_json::json!({"width":rw,"height":rh,"sha256":prepare::hash(&data),"bytes":data.len()}));
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    if arguments.len()==3 && arguments[1]=="serve" {
        return native::serve(Path::new(&arguments[2]));
    }
    // One sequence-parallel rank per GPU behind a single socket.
    #[cfg(feature = "cuda")]
    if arguments.len()==4 && arguments[1]=="serve-group" {
        return group::serve(Path::new(&arguments[2]),arguments[3].parse()?);
    }

    #[cfg(feature = "cuda")]
    if arguments.get(1).is_some_and(|mode|mode=="infer"||mode=="validate-model") {
        let validation=arguments[1]=="validate-model";
        let base=if validation{8}else{7};
        if arguments.len()!=base && arguments.len()!=base+2 {return Err("usage: h3-conditioner infer MODEL PROMPT FIRST_RGB LAST_RGB OUTPUT [WIDTH HEIGHT]; validate-model adds GOLD before dimensions".into());}
        let (width,height)=if arguments.len()==base+2 {(arguments[base].parse()?,arguments[base+1].parse()?)}else{(768,768)};
        let report = native::infer_with_size(Path::new(&arguments[2]), Path::new(&arguments[3]),
            &arguments[4..6].iter().filter(|s|s.as_str()!="-").map(Path::new).collect::<Vec<_>>(),Path::new(&arguments[6]),
            if validation {Some(Path::new(&arguments[7]))} else {None},width,height)?;
        println!("{}",serde_json::to_string(&report)?);
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    if arguments.len() == 4 && arguments[1] == "replay-ops" {
        let report = native::replay_ops(Path::new(&arguments[2]), Path::new(&arguments[3]))?;
        println!("{}", serde_json::to_string(&report)?);
        if report["passed"] != true { return Err("native op comparison failed".into()); }
        return Ok(());
    }
    if (arguments.len() != 7 && arguments.len()!=9) || arguments[1] != "prepare" {
        return Err("usage: h3-conditioner prepare MODEL_ROOT PROMPT_TXT CANVAS0_RGB CANVAS1_RGB OUTPUT [WIDTH HEIGHT]".into());
    }
    let (width,height)=if arguments.len()==9 {(arguments[7].parse()?,arguments[8].parse()?)}else{(768,768)};
    let report = prepare::run_with_size(Path::new(&arguments[2]), Path::new(&arguments[3]),
                              &arguments[4..6].iter().filter(|s|s.as_str()!="-").map(Path::new).collect::<Vec<_>>(),
                              Path::new(&arguments[6]),width,height)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
