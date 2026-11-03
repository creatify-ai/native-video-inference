//! Replay of Triton CUBINs recorded from the PyTorch reference during profile preparation.
//!
//! The binary, entry point, warp count and shared memory are the reference's own;
//! callers supply the grid and parameters in kernel signature order. Triton 3.7
//! appends the global and profile scratch pointers after the user parameters.
use crate::cuda::Stream;
use crate::Result;
use serde_json::Value;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::Path;

type Module = *mut c_void;
type Function = *mut c_void;

extern "C" {
    fn cuModuleLoadData(module: *mut Module, image: *const c_void) -> c_int;
    fn cuModuleUnload(module: Module) -> c_int;
    fn cuModuleGetFunction(function: *mut Function, module: Module, name: *const c_char) -> c_int;
    fn cuFuncSetAttribute(function: Function, attribute: c_int, value: c_int) -> c_int;
    fn cuLaunchKernel(function: Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
                      shared: c_uint, stream: Stream, params: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
    fn cuGetErrorString(code: c_int, text: *mut *const c_char) -> c_int;
    fn cuTensorMapEncodeTiled(map: *mut TensorMap, data_type: c_uint, rank: c_uint, address: *mut c_void,
                              dims: *const u64, strides: *const u64, box_dims: *const u32,
                              element_strides: *const u32, interleave: c_uint, swizzle: c_uint,
                              l2_promotion: c_uint, oob_fill: c_uint) -> c_int;
}

/// CUtensorMap: 128 opaque bytes, 64-byte aligned, passed by value as a kernel parameter.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap([u8; 128]);

/// Triton's host tensordesc metadata (compiled into the kernel's TMA lowering).
#[derive(Clone, Debug)]
struct TensorDescMeta { swizzle: u32, elem_size: u64, elem_type: u32, block: Vec<u32> }

fn check(code: c_int, what: &str) -> Result<()> {
    if code == 0 { return Ok(()); }
    let mut text = std::ptr::null();
    unsafe { cuGetErrorString(code, &mut text) };
    let text = if text.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(text) }.to_string_lossy() };
    Err(format!("{what}: CUDA driver {code}: {text}").into())
}

#[derive(Clone, Debug)]
pub enum Arg {
    Ptr(u64), I32(i32), I64(i64), F32(f32),
    /// A host TensorDescriptor over a contiguous-last-dim tensor (shape/strides in elements).
    TensorDesc { base: u64, shape: Vec<i64>, strides: Vec<i64> },
}

pub struct Kernel {
    pub name: String,
    pub sha256: String,
    module: Module,
    function: Function,
    threads: u32,
    shared: u32,
    /// Non-constexpr parameter types in signature order (e.g. "*bf16", "i32", "fp32", "tensordesc<...>").
    pub params: Vec<String>,
    descs: Vec<TensorDescMeta>,
    /// Constexpr values keyed by signature index (e.g. XBLOCK of Inductor kernels).
    pub constants: Value,
}

impl Kernel {
    /// `root` is a recorded `triton/` directory: `launches.json` plus `cubin/<sha>.cubin`.
    pub fn load(root: &Path, sha256: &str) -> Result<Self> {
        let _trace=native_core_infra::startup_trace::Span::new("backend_module_prepare",root.to_string_lossy(),0);
        let document: Value = serde_json::from_slice(&std::fs::read(root.join("launches.json"))?)?;
        let kernel = &document["kernels"][sha256];
        if kernel.is_null() { return Err(format!("no recorded kernel {sha256}").into()); }
        let meta = &kernel["metadata"];
        let name = meta["name"].as_str().ok_or("kernel metadata lacks name")?.to_string();
        let threads = 32 * meta["num_warps"].as_u64().ok_or("kernel metadata lacks num_warps")? as u32;
        let shared = meta["shared"].as_u64().unwrap_or(0) as u32;
        if meta["num_ctas"].as_u64().unwrap_or(1) != 1 { return Err(format!("{name}: clustered launch unsupported").into()); }
        for key in ["global_scratch_size", "profile_scratch_size"] {
            if meta[key].as_u64().unwrap_or(0) != 0 { return Err(format!("{name}: {key} != 0 unsupported").into()); }
        }
        // serde_json maps do not keep key order; the recorder stores the ordered types.
        let params = kernel["signature_order"].as_array().map(|order| order.iter()
            .filter_map(|v| v.as_str()).map(String::from).collect::<Vec<_>>())
            .ok_or_else(|| format!("{name}: launches.json lacks signature_order"))?
            .into_iter().filter(|t| t != "constexpr").collect();
        let descs: Vec<TensorDescMeta> = meta["tensordesc_meta"].as_array().map(|items| items.iter().map(|m| TensorDescMeta {
            swizzle: m["swizzle"].as_u64().unwrap_or(0) as u32, elem_size: m["elem_size"].as_u64().unwrap_or(0),
            elem_type: m["elem_type"].as_u64().unwrap_or(0) as u32,
            block: m["block_size"].as_array().map(|b| b.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect())
                .unwrap_or_default(),
        }).collect()).unwrap_or_default();
        if descs.iter().any(|d| d.elem_size == 0 || d.block.is_empty()) || meta["tensordesc_meta"].as_array()
            .is_some_and(|items| items.iter().any(|m| m["fp4_padded"] == true)) {
            return Err(format!("{name}: unsupported tensordesc metadata").into());
        }
        let cubin = std::fs::read(root.join("cubin").join(format!("{sha256}.cubin")))?;
        if crate::gold::sha(&cubin) != sha256 { return Err(format!("{name}: CUBIN identity differs").into()); }
        let mut module = std::ptr::null_mut();
        check(unsafe { cuModuleLoadData(&mut module, cubin.as_ptr().cast()) }, "cuModuleLoadData")?;
        let entry = CString::new(name.clone())?;
        let mut function = std::ptr::null_mut();
        check(unsafe { cuModuleGetFunction(&mut function, module, entry.as_ptr()) }, "cuModuleGetFunction")?;
        if shared > 48 * 1024 {
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, as Triton's loader sets it.
            check(unsafe { cuFuncSetAttribute(function, 8, shared as c_int) }, "cuFuncSetAttribute")?;
        }
        Ok(Self { name, sha256: sha256.into(), module, function, threads, shared, params, descs,
                  constants: kernel["constants"].clone() })
    }

    /// Launch with parameters in signature order. Scratch pointers are null:
    /// recorded kernels must report zero global/profile scratch.
    pub fn launch(&self, grid: [u32; 3], args: &[Arg], stream: Stream) -> Result<()> {
        if args.len() != self.params.len() {
            return Err(format!("{}: {} arguments for {} parameters", self.name, args.len(), self.params.len()).into());
        }
        // Parameter storage: 8-byte scalars/pointers and 64-byte-aligned tensor maps.
        let mut scalars: Vec<[u8; 8]> = Vec::new();
        let mut maps: Vec<TensorMap> = Vec::new();
        enum Slot { Scalar(usize), Map(usize) }
        let mut slots = Vec::new();
        let mut desc_index = 0;
        let push = |scalars: &mut Vec<[u8; 8]>, slots: &mut Vec<Slot>, bytes: [u8; 8]| {
            scalars.push(bytes); slots.push(Slot::Scalar(scalars.len() - 1));
        };
        for (arg, ty) in args.iter().zip(&self.params) {
            match arg {
                Arg::Ptr(p) if ty.starts_with('*') && p % 16 == 0 => push(&mut scalars, &mut slots, p.to_le_bytes()),
                Arg::I64(v) if ty == "i64" => push(&mut scalars, &mut slots, v.to_le_bytes()),
                Arg::I32(v) if ty == "i32" => push(&mut scalars, &mut slots, low4(&v.to_le_bytes())),
                Arg::F32(v) if ty == "fp32" => push(&mut scalars, &mut slots, low4(&v.to_le_bytes())),
                Arg::TensorDesc { base, shape, strides } if ty.starts_with("tensordesc") => {
                    let meta = self.descs.get(desc_index).ok_or_else(|| format!("{}: tensordesc without metadata", self.name))?;
                    desc_index += 1;
                    maps.push(encode_tensor_map(*base, shape, strides, meta)?);
                    slots.push(Slot::Map(maps.len() - 1));
                    // Triton 3.7 passes the descriptor's shape (i32) and strides (i64) after the map.
                    for v in shape { push(&mut scalars, &mut slots, low4(&i32::try_from(*v)?.to_le_bytes())); }
                    for v in strides { push(&mut scalars, &mut slots, v.to_le_bytes()); }
                }
                _ => return Err(format!("{}: argument {arg:?} does not match {ty} (pointers 16-byte aligned)", self.name).into()),
            }
        }
        // Triton 3.7 appends global and profile scratch pointers (both unused here).
        for _ in 0..2 { push(&mut scalars, &mut slots, [0u8; 8]); }
        let mut params: Vec<*mut c_void> = slots.iter().map(|slot| match slot {
            Slot::Scalar(i) => scalars[*i].as_mut_ptr().cast(),
            Slot::Map(i) => (&mut maps[*i] as *mut TensorMap).cast(),
        }).collect();
        native_core_infra::execution_start::enqueue_if_armed(stream)?;
        check(unsafe { cuLaunchKernel(self.function, grid[0], grid[1], grid[2], self.threads, 1, 1, self.shared,
                                      stream, params.as_mut_ptr(), std::ptr::null_mut()) }, &self.name)
    }
}

fn low4(bytes: &[u8; 4]) -> [u8; 8] { let mut b = [0u8; 8]; b[..4].copy_from_slice(bytes); b }

/// Triton 3.7.1 fill_tma_descriptor_tiled: reversed dims and box, byte strides,
/// unit element strides, no interleave, recorded swizzle, L2 128B promotion,
/// no OOB fill (zero padding), device->host data-type remap.
fn encode_tensor_map(base: u64, shape: &[i64], strides: &[i64], meta: &TensorDescMeta) -> Result<TensorMap> {
    let rank = shape.len();
    if rank == 0 || rank > 5 || strides.len() != rank || meta.block.len() != rank || strides[rank - 1] != 1 {
        return Err("tensor descriptor rank/stride mismatch".into());
    }
    let mut dims = [0u64; 5];
    let mut byte_strides = [0u64; 5];
    let mut box_dims = [0u32; 5];
    for i in 0..rank {
        dims[rank - 1 - i] = u64::try_from(shape[i])?;
        box_dims[rank - 1 - i] = meta.block[i];
    }
    for i in 0..rank - 1 { byte_strides[rank - 2 - i] = meta.elem_size * u64::try_from(strides[i])?; }
    let host_type = match meta.elem_type { 8 => 10, 9 => 8, 10 => 9, other => other };
    let element_strides = [1u32; 5];
    let mut map = TensorMap([0u8; 128]);
    // CU_TENSOR_MAP_INTERLEAVE_NONE = 0, CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2, OOB_FILL_NONE = 0.
    check(unsafe { cuTensorMapEncodeTiled(&mut map, host_type, rank as c_uint, base as *mut c_void, dims.as_ptr(),
        byte_strides.as_ptr(), box_dims.as_ptr(), element_strides.as_ptr(), 0, meta.swizzle, 2, 0) },
        "cuTensorMapEncodeTiled")?;
    Ok(map)
}

impl Kernel {
    /// Integer constexpr at signature index `index`.
    pub fn constant_u32(&self, index: usize) -> Result<u32> {
        self.constants[format!("({index},)")].as_u64().map(|v| v as u32)
            .ok_or_else(|| format!("{}: no integer constexpr at {index}", self.name).into())
    }
}

impl Drop for Kernel {
    fn drop(&mut self) { unsafe { cuModuleUnload(self.module); } }
}
