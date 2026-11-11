//! The exact element-wise ops of kernels/exact_ops.cu (single IEEE operations per element, so they
//! write ATen's bits; see that file) and torch's fp32 cuBLAS GEMM (kernels/cublas.cpp).
use crate::cuda::{Function, Module, Stream};
use crate::runtime::{p_f32, p_i32, p_i64, p_u64};
use crate::Result;
use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/exact_ops.cubin"));
#[cfg(not(feature = "cuda"))]
static IMAGE: &[u8] = &[];

static MODULE: OnceLock<std::result::Result<Module, String>> = OnceLock::new();

fn function(name: &str) -> Result<Function> {
    let m = MODULE.get_or_init(|| Module::load(IMAGE)).as_ref().map_err(|e| e.clone())?;
    m.function(name)
}

fn run(name: &str, n: usize, s: Stream, params: Vec<Vec<u8>>) -> Result<()> {
    let blocks = n.div_ceil(256).clamp(1, 148 * 16) as u32;
    function(name)?.launch([blocks, 1, 1], [256, 1, 1], 0, s, &params, &[])
}

/// out = a * a (fp32, n elements).
pub fn square_f32(s: Stream, a: u64, out: u64, n: usize) -> Result<()> {
    run("dsv4_square_f32", n, s, vec![p_u64(a), p_u64(out), p_i64(n as i64)])
}
/// out = a + v (fp32 scalar).
pub fn add_scalar_f32(s: Stream, a: u64, v: f32, out: u64, n: usize) -> Result<()> {
    run("dsv4_add_scalar_f32", n, s, vec![p_u64(a), p_f32(v), p_u64(out), p_i64(n as i64)])
}
/// out[r, j] = a[r, j] * b[r].
pub fn mul_rows_f32(s: Stream, a: u64, b: u64, out: u64, rows: usize, cols: usize) -> Result<()> {
    run("dsv4_mul_rows_f32", rows * cols, s, vec![p_u64(a), p_u64(b), p_u64(out), p_i64(rows as i64), p_i32(cols as i32)])
}
/// out = a * s[0] (a one-element device tensor).
pub fn mul_dev_scalar_f32(s: Stream, a: u64, v: u64, out: u64, n: usize) -> Result<()> {
    run("dsv4_mul_dev_scalar_f32", n, s, vec![p_u64(a), p_u64(v), p_u64(out), p_i64(n as i64)])
}
/// out[r, j] = a[r, j] + b[j].
pub fn add_cols_f32(s: Stream, a: u64, b: u64, out: u64, rows: usize, cols: usize) -> Result<()> {
    run("dsv4_add_cols_f32", rows * cols, s, vec![p_u64(a), p_u64(b), p_u64(out), p_i64(rows as i64), p_i32(cols as i32)])
}
/// out[q, k] = p[q] * x[q, k] for q < rc, k < d.
pub fn mul_bcast_last_f32(s: Stream, p: u64, x: u64, out: u64, rc: usize, d: usize) -> Result<()> {
    run("dsv4_mul_bcast_last_f32", rc * d, s, vec![p_u64(p), p_u64(x), p_u64(out), p_i64(rc as i64), p_i32(d as i32)])
}
/// out[r, c, k] = e[r, k] + h[r, c, k] (bf16, added in fp32).
pub fn add_bcast_hc_bf16(s: Stream, e: u64, h: u64, out: u64, rows: usize, c: usize, d: usize) -> Result<()> {
    run("dsv4_add_bcast_hc_bf16", rows * c * d, s, vec![p_u64(e), p_u64(h), p_u64(out), p_i64(rows as i64), p_i32(c as i32), p_i32(d as i32)])
}
/// out = bf16(a) (round to nearest even).
pub fn f32_to_bf16(s: Stream, a: u64, out: u64, n: usize) -> Result<()> {
    run("dsv4_f32_to_bf16", n, s, vec![p_u64(a), p_u64(out), p_i64(n as i64)])
}

/// dst row dst_idx[r] = src row src_idx[r] (r < n, `row_bytes` a multiple of 8; an index of 0
/// means the identity).
pub fn copy_rows(s: Stream, src: u64, src_idx: u64, dst: u64, dst_idx: u64, n: usize, row_bytes: usize) -> Result<()> {
    if n == 0 { return Ok(()); }
    let words = row_bytes / 8;
    run("dsv4_copy_rows", n * words, s, vec![p_u64(src), p_u64(src_idx), p_u64(dst), p_u64(dst_idx), p_i32(n as i32), p_i32(words as i32)])
}
/// dst[i * stride + offset] = src[i] (int64, i < n).
pub fn scatter_strided_i64(s: Stream, src: u64, dst: u64, n: usize, stride: usize, offset: usize) -> Result<()> {
    run("dsv4_scatter_strided_i64", n, s, vec![p_u64(src), p_u64(dst), p_i32(n as i32), p_i32(stride as i32), p_i32(offset as i32)])
}

/// SGLang's clear_unaccepted_c128_draft_states over every C128 layer and request in one launch:
/// `args` (device int32) holds the layer ids [nlayers], then per request rid, seq_len, accepted
/// rows ([nreq] each); see kernels/exact_ops.cu.
pub fn clear_c128_rejected(s: Stream, state: u64, layer_stride_floats: usize, args: u64, nlayers: usize, nreq: usize, ring: usize, drafts: usize, half: usize) -> Result<()> {
    if nreq == 0 { return Ok(()); }
    let (rid, seq, acc) = (args + 4 * nlayers as u64, args + 4 * (nlayers + nreq) as u64, args + 4 * (nlayers + 2 * nreq) as u64);
    function("dsv4_clear_c128_rejected")?.launch([half.div_ceil(256) as u32, (nreq * drafts) as u32, nlayers as u32], [256, 1, 1], 0, s,
        &[p_u64(state), p_u64(args), p_i64(layer_stride_floats as i64), p_u64(rid), p_u64(seq), p_u64(acc), p_i32(ring as i32), p_i32(drafts as i32), p_i32(half as i32)], &[])
}

/// `kernels/exact_ops.cu` dsv4_fill_decode_ids: c4_ids, c128_ids and the page table of `b` decode
/// rows from the per-row parameters at `fill`.
pub fn fill_decode_ids(s: Stream, fill: u64, c4_ids: u64, c128_ids: u64, table: u64, b: usize, c4_width: usize, c128_width: usize, w: usize) -> Result<()> {
    if b == 0 { return Ok(()); }
    let n = c4_width.max(c128_width).max(w);
    function("dsv4_fill_decode_ids")?.launch([n.div_ceil(256).min(64) as u32, b as u32, 1], [256, 1, 1], 0, s,
        &[p_u64(fill), p_u64(c4_ids), p_u64(c128_ids), p_u64(table), p_i32(c4_width as i32), p_i32(c128_width as i32), p_i32(w as i32)], &[])
}

extern "C" {
    fn dsv4_cublas_open(path: *const c_char, err: *mut c_char, cap: usize) -> i32;
    fn dsv4_sgemm(trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32, a: u64, lda: i32, b: u64, ldb: i32, c: u64, ldc: i32,
                  workspace: u64, workspace_bytes: usize, stream: *mut c_void, err: *mut c_char, cap: usize) -> i32;
}

fn ffi_err(err: &[u8]) -> String {
    let n = err.iter().position(|&b| b == 0).unwrap_or(err.len());
    String::from_utf8_lossy(&err[..n]).into_owned()
}

/// torch's cuBLAS workspace on Hopper and later (CUBLAS_WORKSPACE_CONFIG default ":4096:8").
pub const CUBLAS_WORKSPACE_BYTES: usize = 4096 * 8 * 1024;

/// Open the reference libcublas (the cuBLASLt library's directory, `lt_path`'s sibling).
pub fn cublas_open(lt_path: &str) -> Result<()> {
    let path = std::path::Path::new(lt_path).with_file_name("libcublas.so.13");
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
    let mut err = vec![0u8; 1024];
    if unsafe { dsv4_cublas_open(c.as_ptr(), err.as_mut_ptr().cast(), err.len()) } == 0 { return Err(ffi_err(&err)); }
    Ok(())
}

/// F.linear(x, w) in fp32 as torch runs it: out[rows, n] = x[rows, k] @ w[n, k]^T, i.e. cuBLAS
/// column-major C[n, rows] = W^T... (transa T, transb N, m = n, n = rows).
pub fn linear_f32(s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, workspace: u64) -> Result<()> {
    let mut err = vec![0u8; 1024];
    let ok = unsafe { dsv4_sgemm(1, 0, n as i32, rows as i32, k as i32, w, k as i32, x, k as i32, out, n as i32, workspace,
                                 CUBLAS_WORKSPACE_BYTES, s.0, err.as_mut_ptr().cast(), err.len()) };
    if ok == 0 { return Err(ffi_err(&err)); }
    Ok(())
}
