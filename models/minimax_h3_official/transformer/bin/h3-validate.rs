//! Experiment-only continuous replay: gold inputs at step zero, then native state.
//! Does not establish native request preparation or complete-product acceptance.
use h3_official_native_run::{native::Runtime, Result};
use std::{fs, path::Path};

fn run()->Result<()> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=5 && args.len()!=6 {return Err("usage: h3-validate MODEL ROUTER DENSE OUTPUT [NATIVE_REQUEST]".into());}
    h3_official_native_run::debug_audit::begin()?;
    h3_official_native_run::native::cp_select_device_from_environment()?;
    h3_official_native_run::pointwise_opt::begin()?;
    h3_official_native_run::bsa_pack::begin()?;
    let mut runtime=Runtime::new_with_request(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),args.get(5).map(Path::new))?;
    let nfe=runtime.nfe();
    let mut report=runtime.execute("infer",nfe)?;
    report["bsa_pack"]=h3_official_native_run::bsa_pack::finish(runtime.sparse_calls_expected()?)?;
    report["context_parallel"]=runtime.cp_report();
    report["scope"]=serde_json::json!(if args.len()==6 {"continuous DiT using native request tensors; gold tensors are comparison-only"} else {"continuous DiT replay from original initial tensors; preparation and decode not validated"});
    fs::write(Path::new(&args[4]).join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    if report["passed"]!=true {return Err("continuous native trajectory differs from independent PyTorch gold".into());}
    Ok(())
}
fn main(){if let Err(error)=run(){eprintln!("{error}");std::process::exit(1);}}
