//! Native inference mechanisms with no model or application dependencies.
pub mod checkpoint;
pub mod allocation_ledger;
pub mod phase_workspace;
pub mod integrity;
pub mod guard_table;
pub mod startup_policy;
pub mod declared_startup;
pub mod startup_owner;
pub mod startup_cohort;
pub mod startup_input;
pub mod startup_result;
#[cfg(all(feature="model-load",target_os="linux"))]
pub mod startup_native;
#[cfg(unix)]
pub mod declared_startup_gate;
#[cfg(unix)]
pub mod startup_ffi;
pub mod immutable_assets;
pub mod backend_plans;
pub mod resident_assets;
pub mod reuse_pool;
#[cfg(unix)]
pub mod resident;
pub mod rng;
pub mod startup_trace;
pub mod cpu_priority;
pub mod tensor;
pub mod weight_loader;
pub mod background_io;
pub mod backend_abi;
pub mod process_identity;
#[cfg(unix)]
pub mod phase_gate;
#[cfg(target_os="linux")]
pub mod execution_start;
pub mod switch_benchmark;
#[cfg(target_os="linux")]
pub mod native_backend;
#[cfg(target_os="linux")]
pub mod backend_service;
#[cfg(all(feature="cuda",target_os="linux"))]
pub mod device_context;
#[cfg(all(feature="cuda",target_os="linux"))]
pub mod pinned_weights;
#[cfg(all(feature="cuda",target_os="linux"))]
pub mod host_buffer;
#[cfg(all(feature="cuda",target_os="linux"))]
pub mod resident_weights;
#[cfg(feature="model-load")]
pub mod model_load;
#[cfg(feature="cuda")]
pub mod startup;
pub type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;

#[cfg(feature="cuda")]
pub mod archive_reader;

#[cfg(unix)]
pub mod startup_gate;
