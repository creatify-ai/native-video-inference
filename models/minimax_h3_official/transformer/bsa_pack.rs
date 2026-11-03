//! Recipe/phase/layer SubblockH3 dispatch; S<4096 always dense.
//! Public sparse calls use the heterogeneous FlashInfer kernel and count every dispatch.
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
/// Every h3_bsa_forward call in this request, whichever kernel the bridge ran.
pub static CPP_CALLS: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static EXACT: Cell<bool> = const { Cell::new(false) };
}
#[cfg(feature="cuda")]
extern "C" {
    fn h3_bsa_pack_set_mode(mode: i32) -> i32;
    fn h3_bsa_pack_counts(baseline: *mut u64, optimized: *mut u64, fallback: *mut u64) -> i32;
    fn h3_bsa_set_kernel(kernel: i32) -> i32;
    fn h3_bsa_kernel_counts(served: *mut u64, exact: *mut u64) -> i32;
    fn h3_bsa_exact_available() -> i32;
}
fn parse(raw: &str) -> Result<bool,String> {
    match raw.trim() { "0"=>Ok(false),"1"=>Ok(true),
        _=>Err("H3_BSA_PACK_MODE_FILE/H3_BSA_PACK must contain 0 or 1".into()) }
}
fn parse_kernel(raw: &str) -> Result<bool,String> {
    match raw.trim() { "served"=>Ok(false),"exact"=>Ok(true),
        _=>Err("H3_BSA_KERNEL must be served or exact".into()) }
}
/// Whether this build links the exact kernel (B200 builds with H3_BSA_EXACT_ROOT).
pub fn exact_available()->bool {
    #[cfg(feature="cuda")] { unsafe {h3_bsa_exact_available()!=0} }
    #[cfg(not(feature="cuda"))] { false }
}
pub fn exact_selected()->bool {EXACT.with(Cell::get)}
/// (served, exact) kernel calls since begin().
pub fn kernel_counts()->Result<[u64;2],String> {
    #[allow(unused_mut)] let mut counts=[0u64;2];
    #[cfg(feature="cuda")] {
        let status=unsafe {h3_bsa_kernel_counts(&mut counts[0],&mut counts[1])};
        if status!=0 {return Err(format!("BSA kernel counters failed: {status}"));}
    }
    Ok(counts)
}
pub fn begin()->Result<(),String> {
    let raw=match std::env::var_os("H3_BSA_PACK_MODE_FILE") {
        Some(path)=>std::fs::read_to_string(path).map_err(|e|e.to_string())?,
        None=>match std::env::var("H3_BSA_PACK") {
            Ok(value)=>value,Err(std::env::VarError::NotPresent)=>"1".into(),
            Err(error)=>return Err(error.to_string()),
        },
    };
    let enabled=parse(&raw)?;
    let exact=match std::env::var("H3_BSA_KERNEL") {
        Ok(value)=>parse_kernel(&value)?,Err(std::env::VarError::NotPresent)=>false,
        Err(error)=>return Err(error.to_string()),
    };
    if exact {return Err("public H3 uses the heterogeneous FlashInfer BF16 SubBlock kernel (H3_BSA_KERNEL=served)".into());}
    #[cfg(feature="cuda")] {
        let status=unsafe {h3_bsa_pack_set_mode(enabled as i32)};
        if status!=0 {return Err(format!("BSA pack mode reset failed: {status}"));}
        let status=unsafe {h3_bsa_set_kernel(exact as i32)};
        if status!=0 {return Err(format!("BSA kernel selection failed: {status}"));}
    }
    ENABLED.with(|state|state.set(enabled));EXACT.with(|state|state.set(exact));
    CPP_CALLS.store(0,Ordering::Relaxed);Ok(())
}
fn validate(enabled:bool,baseline:u64,optimized:u64,fallback:u64,expected:u64)->Result<(),String> {
    // Validate the complete phase/layer/head-group dispatch count.
    let total=baseline.checked_add(optimized).ok_or("BSA pack count overflow")?;
    let valid=total==expected && if enabled {
        baseline==fallback && optimized.checked_add(fallback)==Some(expected)
    } else {baseline==expected && optimized==0 && fallback==0};
    if !valid {return Err(format!("BSA pack dispatch mismatch: expected={expected} enabled={enabled} baseline={baseline} optimized={optimized} fallback={fallback}"));}
    Ok(())
}
/// In exact mode every sparse call must run the exact kernel; otherwise every call runs served.
fn validate_kernels(exact_mode:bool,served:u64,exact:u64,expected:u64)->Result<(),String> {
    let valid=served.checked_add(exact)==Some(expected) && if exact_mode {served==0} else {exact==0};
    if !valid {return Err(format!("BSA kernel dispatch mismatch: expected={expected} exact_mode={exact_mode} served={served} exact={exact}"));}
    Ok(())
}
pub fn finish(expected:u64)->Result<serde_json::Value,String> {
    #[allow(unused_mut)] let mut counts=[0u64;3];
    #[cfg(feature="cuda")] {
        let status=unsafe {h3_bsa_pack_counts(&mut counts[0],&mut counts[1],&mut counts[2])};
        if status!=0 {return Err(format!("BSA pack counters failed: {status}"));}
    }
    let enabled=ENABLED.with(Cell::get);let exact_mode=exact_selected();
    let [served,exact]=kernel_counts()?;
    validate_kernels(exact_mode,served,exact,expected)?;
    // Only served-kernel calls pack; the exact kernel reads the caller's tensors directly.
    validate(enabled,counts[0],counts[1],counts[2],served)?;
    Ok(serde_json::json!({"enabled":enabled,"baseline_dispatch_calls":counts[0],
        "optimized_dispatch_calls":counts[1],"fallback_dispatch_calls":counts[2],
        "total_dispatch_calls":counts[0]+counts[1],"expected_sparse_calls":expected,"packing_only":true,
        "kernel":if exact_mode {"exact"} else {"served"},"served_kernel_calls":served,"exact_kernel_calls":exact,
        "dense_below_min_seq_len":expected==0,"sparse_min_seq_len":4096}))
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn mode_is_explicit() {
        assert!(!parse(" 0\n").unwrap());assert!(parse("1").unwrap());
        for value in ["","true","false","2","-1"]{assert!(parse(value).is_err());}
        assert!(!parse_kernel("served").unwrap());assert!(parse_kernel(" exact\n").unwrap());
        for value in ["","0","1","cute","Exact","cpp"]{assert!(parse_kernel(value).is_err());}
    }
    #[test] fn exact_mode_requires_every_sparse_call_on_the_exact_kernel() {
        for expected in [0,50,96,144,200,300,1600,2400] {
            assert!(validate_kernels(false,expected,0,expected).is_ok());
            assert!(validate_kernels(true,0,expected,expected).is_ok());
            // The exact kernel's calls pack nothing; the pack gate then sees zero calls.
            assert!(validate(true,0,0,0,0).is_ok());assert!(validate(false,0,0,0,0).is_ok());
        }
        for expected in [50,200,300] {
            for (served,exact) in [(1,expected-1),(0,expected-1),(0,expected+1),(expected,0)] {
                assert!(validate_kernels(true,served,exact,expected).is_err());
            }
            for (served,exact) in [(expected-1,1),(expected-1,0),(0,expected)] {
                assert!(validate_kernels(false,served,exact,expected).is_err());
            }
        }
        assert!(validate_kernels(true,u64::MAX,1,0).is_err());
    }
    #[test] fn counts_remain_strict_for_both_dispatch_paths() {
        for mode in [false,true] {
            assert!(validate(mode,0,0,0,0).is_ok());
            for counts in [(1,0,0),(0,1,0),(0,0,1),(0,200,0)] {
                assert!(validate(mode,counts.0,counts.1,counts.2,0).is_err());
            }
            assert!(validate(mode,u64::MAX,1,0,200).is_err());
        }
        assert!(validate(false,200,0,0,200).is_ok());assert!(validate(true,0,200,0,200).is_ok());
        assert!(validate(true,1,199,1,200).is_ok());
        for counts in [(0,0,0),(0,199,0),(1,200,0),(0,200,1),(200,0,0)] {
            assert!(validate(true,counts.0,counts.1,counts.2,200).is_err());
        }
        assert!(validate(false,199,1,0,200).is_err());assert!(validate(true,0,0,0,1).is_err());
    }
    #[test] fn sparse_evaluation_is_counted_without_relaxing_the_gate() {
        assert!(validate(false,50,0,0,50).is_ok());
        assert!(validate(true,0,50,0,50).is_ok());
        assert!(validate(true,1,49,1,50).is_ok());
        for counts in [(0,0,0),(0,49,0),(0,51,0),(1,50,1),(50,0,0)] {
            assert!(validate(true,counts.0,counts.1,counts.2,50).is_err());
        }
    }
    #[test] fn sparse_layers_are_counted_without_relaxing_the_gate() {
        for expected in [144,1152] {
            assert!(validate(false,expected,0,0,expected).is_ok());
            assert!(validate(true,0,expected,0,expected).is_ok());
            assert!(validate(true,1,expected-1,1,expected).is_ok());
            for counts in [(0,0,0),(0,expected-1,0),(0,expected+1,0),(1,expected,1),(expected,0,0)] {
                assert!(validate(true,counts.0,counts.1,counts.2,expected).is_err());
            }
        }
    }
    #[test] fn grouped_heads_count_every_dispatch_and_keep_fallback_accounting() {
        for expected in [400,1600,2400] {
            assert!(validate(false,expected,0,0,expected).is_ok());
            assert!(validate(true,7,expected-7,7,expected).is_ok());
            assert!(validate(true,0,expected/8,0,expected).is_err());
            assert!(validate(true,7,expected-7,0,expected).is_err());
            assert!(validate(false,expected-1,0,0,expected).is_err());
        }
    }
    #[test] fn multiple_windows_sum_full_sparse_evaluations() {
        for expected in [100,150,600,800,3200,16000] {
            assert!(validate(true,0,expected,0,expected).is_ok());
            assert!(validate(true,3,expected-3,3,expected).is_ok());
            assert!(validate(true,0,expected-50,0,expected).is_err());
            assert!(validate(true,0,expected+50,0,expected).is_err());
            assert!(validate(true,0,expected,0,expected+1).is_err());
        }
    }
}
