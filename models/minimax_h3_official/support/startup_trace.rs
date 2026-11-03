//! Compatibility policy for the existing H3 trace environment.
use std::path::Path;
use std::sync::Once;
pub fn initialize() {
    static INITIALIZED:Once=Once::new();
    INITIALIZED.call_once(||native_core_infra::startup_trace::configure(native_core_infra::startup_trace::Configuration {
        directory:std::env::var_os("H3_STARTUP_TRACE_DIR").map(Into::into),
        stop_file:std::env::var_os("H3_STARTUP_TRACE_STOP_FILE").map(Into::into),
    }));
}
pub fn timestamp()->Option<f64>{initialize();native_core_infra::startup_trace::timestamp()}
pub struct Span { _span:native_core_infra::startup_trace::Span }
impl Span {pub fn new(kind:&'static str,label:impl AsRef<str>,bytes:usize)->Self {
    initialize();Self{_span:native_core_infra::startup_trace::Span::new(kind,label,bytes)}
}}
pub fn measure<T>(kind:&'static str,label:impl AsRef<str>,bytes:usize,f:impl FnOnce()->T)->T {
    let _span=Span::new(kind,label,bytes);f()
}
pub fn read(path:impl AsRef<Path>)->std::io::Result<Vec<u8>>{initialize();native_core_infra::startup_trace::read(path)}
