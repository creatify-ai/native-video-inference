//! PyTorch CUDA kernels the reference runs between library kernels, launched from the CUBINs of
//! its libtorch_cuda (in the kernel bundle) with the argument structs ATen builds.
use crate::cuda::Stream;
use crate::runtime::{p_i32, p_i64, p_u32, p_u64, Kernels, Launch, Pod};
use crate::Result;

/// at::cuda::detail::TensorInfo<T, unsigned int>: data, sizes[25], strides[25], dims (216 bytes).
pub fn tensor_info(data: u64, sizes: &[u32], strides: &[u32]) -> Vec<u8> {
    let mut p = Pod::default();
    p.u64(data);
    for i in 0..25 { p.u32(*sizes.get(i).unwrap_or(&0)); }
    for i in 0..25 { p.u32(*strides.get(i).unwrap_or(&0)); }
    p.i32(sizes.len() as i32);
    p.pad_to(216);
    p.take()
}

/// `embedding` forward: index_select(weight, 0, ids) with indexSelectSmallIndex<bf16, int64, uint32>.
pub fn embedding_bf16(k: &Kernels, s: Stream, out: u64, weight: u64, ids: u64, n: usize, vocab: usize, dim: usize) -> Result<()> {
    let name = "_ZN2at6native44_GLOBAL__N__b23ef953_11_Indexing_cu_89862edb21indexSelectSmallIndexIN3c108BFloat16EljLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS7_IKS8_S9_EENS7_IKT0_S9_EEiiS9_l";
    Launch {
        name, grid: [dim.div_ceil(128).min(148 * 8) as u32, 1, 1], block: [dim.min(128) as u32, 1, 1], smem: 0, attrs: vec![],
        // ATen reduces the indexed dimension of dst/src to size 1 (TensorInfo::reduceDim).
        params: vec![tensor_info(out, &[1, dim as u32], &[dim as u32, 1]),
                     tensor_info(weight, &[1, dim as u32], &[dim as u32, 1]),
                     tensor_info(ids, &[n as u32], &[1]),
                     p_i32(0), p_i32(0), p_u32(dim as u32), p_i64(vocab as i64)],
    }.run(k, s)
}

/// ATen `vectorized_elementwise_kernel` grid: one block of 128 threads per 128 * vec elements.
fn elementwise_grid(n: usize, vec: usize) -> [u32; 3] { [n.div_ceil(128 * vec) as u32, 1, 1] }

/// `out = input * scalar` on float32 (AUnaryFunctor<float, float, float, MulFunctor>, vec 4).
pub fn mul_scalar_f32(k: &Kernels, s: Stream, out: u64, input: u64, n: usize, scalar: f32) -> Result<()> {
    let name = "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13AUnaryFunctorIfffNS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm2EELb0EEEviT0_T1_";
    // The functor: an empty MulFunctor (padded to 4 bytes), then the float operand.
    let mut f = Pod::default();
    f.u32(0).f32(scalar);
    let mut a = Pod::default();
    a.u64(out).u64(input);
    Launch { name, grid: elementwise_grid(n, 4), block: [128, 1, 1], smem: 0, attrs: vec![],
             params: vec![p_i32(n as i32), f.take(), a.take()] }.run(k, s)
}

/// `out = a + b` on bf16 (CUDAFunctor_add<BFloat16> with alpha 1, vec 8).
pub fn add_bf16(k: &Kernels, s: Stream, out: u64, a: u64, b: u64, n: usize) -> Result<()> {
    let name = "_ZN2at6native29vectorized_elementwise_kernelILi8ENS0_15CUDAFunctor_addIN3c108BFloat16EEESt5arrayIPcLm3EELb0EEEviT0_T1_";
    let mut arr = Pod::default();
    arr.u64(out).u64(a).u64(b);
    Launch { name, grid: elementwise_grid(n, 8), block: [128, 1, 1], smem: 0, attrs: vec![],
             params: vec![p_i32(n as i32), 1.0f32.to_le_bytes().to_vec(), arr.take()] }.run(k, s)
}

/// `out.copy_(input)` from bf16 to float32 (`logits.float()`): direct_copy_kernel_cuda through
/// unrolled_elementwise_kernel with LoadWithCast<1> / StoreWithCast<1>, 4 elements per thread.
pub fn copy_bf16_to_f32(k: &Kernels, s: Stream, out: u64, input: u64, n: usize) -> Result<()> {
    let name = "_ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE5_clEvEUlfE_St5arrayIPcLm2EELi4E23TrivialOffsetCalculatorILi1EjESB_NS0_6memory12LoadWithCastILi1EEENSC_13StoreWithCastILi1EEEEEviT_T0_T2_T3_T4_T5_";
    let mut arr = Pod::default();
    arr.u64(out).u64(input);
    // LoadWithCast<1>: dtypes {ScalarType::BFloat16 = 15}, element sizes {2};
    // StoreWithCast<1>: {ScalarType::Float = 6}, {4}. The offset calculators are empty.
    let mut load = Pod::default();
    load.u32(15).u32(2);
    let mut store = Pod::default();
    store.u32(6).u32(4);
    Launch { name, grid: [n.div_ceil(128 * 4) as u32, 1, 1], block: [128, 1, 1], smem: 0, attrs: vec![],
             params: vec![p_i32(n as i32), vec![0], arr.take(), vec![0], vec![0], load.take(), store.take()] }.run(k, s)
}

/// `F.log_softmax(input, dim=-1)` on float32 rows: cunn_SoftMaxForward<ILP 4, float, float, float,
/// LogSoftMaxForwardEpilogue>, one block per row of SoftMaxForward_getBlockSize threads
/// (min(classes, 1024) rounded to warps), shared memory one float per warp.
pub fn log_softmax_f32(k: &Kernels, s: Stream, out: u64, input: u64, rows: usize, classes: usize) -> Result<()> {
    let name = "_ZN2at6native43_GLOBAL__N__f71e7945_10_SoftMax_cu_9f978f6319cunn_SoftMaxForwardILi4EfffNS1_25LogSoftMaxForwardEpilogueEEEvPT2_PKT0_i";
    let block = classes.min(1024).div_ceil(32) * 32;
    Launch { name, grid: [rows as u32, 1, 1], block: [block as u32, 1, 1], smem: (block / 32 * 4) as u32, attrs: vec![],
             params: vec![p_u64(out), p_u64(input), p_i32(classes as i32)] }.run(k, s)
}

/// `torch.argmax(logits, -1)` of `rows` float32 rows of `n` columns (SGLang's greedy sampler):
/// reduce_kernel<512, 1, ReduceOp<float, ArgMaxOps<float>, uint32, int64, 4, 4>>, writing each row's
/// int64 index of the first maximum to `out[row]`. The ReduceOp (ATen Reduce.cuh) is built as
/// ATen's setReduceConfig builds it for a contiguous [rows, n] tensor reduced over its last dim:
/// arg_t (float, int64) identity (-inf, 0); ReduceConfig (element size 16, n inputs per output,
/// vectorized input, block and splits from set_block_dimension, inputs split across lanes then
/// warps); the input OffsetCalculator<1, uint32> walks the row (stride 4 bytes); the output
/// calculator has no dims for one row, else the row dim (output stride 8, input stride 4n bytes).
/// Fields ATen leaves unset stay zero (the kernel does not read them).
pub fn argmax_f32_rows(k: &Kernels, s: Stream, input: u64, rows: usize, n: usize, out: u64) -> Result<()> {
    let name = k.names_containing("reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_9ArgMaxOpsIfEEjlLi4ELi4EEEEEvT1_").into_iter().next()
        .ok_or("kernel bundle has no ATen argmax reduce kernel")?;
    // setReduceConfig: reduction on the fastest dim, dim0 = n / input_vec_size (4), dim1 = rows.
    let last_pow2 = |v: usize| -> usize { if v == 0 { 1 } else { 1 << (usize::BITS - 1 - v.leading_zeros()) } };
    let (max_threads, warp) = (512usize, 32usize);
    let (dim0, dim1) = (n / 4, rows);
    let d0 = if dim0 < max_threads { last_pow2(dim0) } else { max_threads };
    let d1 = if dim1 < max_threads { last_pow2(dim1) } else { max_threads };
    let mut bw = d0.min(warp);
    let bh = d1.min(max_threads / bw);
    bw = d0.min(max_threads / bh);
    let threads = bw * bh;
    let mut step_input = 1usize;
    let mut step_output = 1usize;
    let mut input_mult = [0usize; 3];
    let mut output_mult = [0usize; 2];
    input_mult[0] = step_input; step_input *= bw;
    let values_per_thread = |step: usize| n.div_ceil(step);
    if values_per_thread(step_input) >= (bh * 16).min(256) { input_mult[1] = step_input; step_input *= bh; }
    else { output_mult[1] = step_output; step_output *= bh; }
    // A global (cross-CTA) split needs >= 256 values per thread: n = 129280 gives 253.
    if input_mult[1] != 0 && values_per_thread(step_input) >= 256 { return Err("argmax: cross-CTA reduce config not ported".into()); }
    let grid = rows.div_ceil(step_output);
    let divider = |d: usize| -> (u32, u32, u32) {
        let shift = (0..32).find(|&s| (1u64 << s) >= d as u64).unwrap_or(32);
        (d as u32, (((1u64 << 32) * ((1u64 << shift) - d as u64)) / d as u64 + 1) as u32, shift as u32)
    };
    let mut b = vec![0u8; 1064];
    let mut put = |off: usize, bytes: &[u8]| b[off..off + bytes.len()].copy_from_slice(bytes);
    put(8, &f32::NEG_INFINITY.to_le_bytes());
    let cfg = [16, n, rows, step_input, step_output, 1, input_mult[0], input_mult[1], input_mult[2], output_mult[0], output_mult[1], bw, bh, threads];
    for (i, v) in cfg.iter().enumerate() { put(24 + 4 * i, &(*v as i32).to_le_bytes()); }
    put(80, &[1]);                                       // vectorize_input
    put(84, &1i32.to_le_bytes());                        // output_vec_size
    let (dv, m1, sh) = divider(n);
    put(88, &1i32.to_le_bytes());                        // input_calc: dims, sizes_[0], strides_[0][0]
    put(92, &dv.to_le_bytes()); put(96, &m1.to_le_bytes()); put(100, &sh.to_le_bytes());
    put(392, &4u32.to_le_bytes());
    if rows > 1 {                                        // output_calc (at 492): the row dim
        let (dv, m1, sh) = divider(rows);
        put(492, &1i32.to_le_bytes());
        put(496, &dv.to_le_bytes()); put(500, &m1.to_le_bytes()); put(504, &sh.to_le_bytes());
        put(796, &8u32.to_le_bytes());                   // strides_[0][0]: output bytes per row
        put(800, &((4 * n) as u32).to_le_bytes());       // strides_[0][1]: input bytes per row
    }
    put(1000, &input.to_le_bytes());                     // src
    put(1008, &out.to_le_bytes());                       // dst[0]
    put(1057, &[1]);                                     // final_output
    put(1060, &1i32.to_le_bytes());                      // noutputs
    Launch { name: &name, grid: [grid as u32, 1, 1], block: [bw as u32, bh as u32, 1], smem: (16 * threads) as u32, attrs: vec![], params: vec![b] }.run(k, s)
}

/// One row of `argmax_f32_rows`.
pub fn argmax_f32_row(k: &Kernels, s: Stream, input: u64, n: usize, out: u64) -> Result<()> { argmax_f32_rows(k, s, input, 1, n, out) }

/// setReduceConfig for a reduction over the fastest dim (rows x n fp32, input vectorized by 4):
/// (block width, block height, step_input, step_output, input_mult, output_mult), as ATen picks
/// them for reduce_kernel<512, 1> (the argmax and mean launches).
fn lastdim_reduce_config(rows: usize, n: usize) -> Result<(usize, usize, usize, usize, [usize; 3], [usize; 2])> {
    let last_pow2 = |v: usize| -> usize { if v == 0 { 1 } else { 1 << (usize::BITS - 1 - v.leading_zeros()) } };
    let (max_threads, warp) = (512usize, 32usize);
    let (dim0, dim1) = (n / 4, rows);
    let d0 = if dim0 < max_threads { last_pow2(dim0) } else { max_threads };
    let d1 = if dim1 < max_threads { last_pow2(dim1) } else { max_threads };
    let mut bw = d0.min(warp);
    let bh = d1.min(max_threads / bw);
    bw = d0.min(max_threads / bh);
    let (mut step_input, mut step_output) = (1usize, 1usize);
    let (mut input_mult, mut output_mult) = ([0usize; 3], [0usize; 2]);
    input_mult[0] = step_input;
    step_input *= bw;
    let values_per_thread = |step: usize| n.div_ceil(step);
    if values_per_thread(step_input) >= (bh * 16).min(256) { input_mult[1] = step_input; step_input *= bh; }
    else { output_mult[1] = step_output; step_output *= bh; }
    if input_mult[1] != 0 && values_per_thread(step_input) >= 256 { return Err("reduce: cross-CTA config not ported".into()); }
    Ok((bw, bh, step_input, step_output, input_mult, output_mult))
}

/// IntDivider<uint32> fields (divisor, m1, shift).
fn int_divider(d: usize) -> (u32, u32, u32) {
    let shift = (0..32).find(|&s| (1u64 << s) >= d as u64).unwrap_or(32);
    (d as u32, (((1u64 << 32) * ((1u64 << shift) - d as u64)) / d as u64 + 1) as u32, shift as u32)
}

/// `x.mean(-1)` of fp32 rows: ATen reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float,
/// float, float>, uint32, float, 4, 4>> (ReduceOp layout: ops.factor @0, ident @4, ReduceConfig
/// @8, input OffsetCalculator<1> @72, output OffsetCalculator<2> @476, src @984, dst @992,
/// final_output @1041, noutputs @1044; 1048 bytes). `out` fp32 [rows].
pub fn mean_rows_f32(k: &Kernels, s: Stream, input: u64, rows: usize, n: usize, out: u64) -> Result<()> {
    let name = k.names_containing("reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffffEEjfLi4ELi4EEEEEvT1_").into_iter().next()
        .ok_or("kernel bundle has no ATen mean reduce kernel")?;
    let (bw, bh, step_input, step_output, input_mult, output_mult) = lastdim_reduce_config(rows, n)?;
    let threads = bw * bh;
    let mut b = vec![0u8; 1048];
    let mut put = |off: usize, bytes: &[u8]| b[off..off + bytes.len()].copy_from_slice(bytes);
    put(0, &(1.0f32 / n as f32).to_le_bytes());          // MeanOps::factor (float(1) / num_inputs)
    let cfg = [4, n, rows, step_input, step_output, 1, input_mult[0], input_mult[1], input_mult[2], output_mult[0], output_mult[1], bw, bh, threads];
    for (i, v) in cfg.iter().enumerate() { put(8 + 4 * i, &(*v as i32).to_le_bytes()); }
    put(64, &[1]);                                       // vectorize_input
    put(68, &1i32.to_le_bytes());                        // output_vec_size
    let (dv, m1, sh) = int_divider(n);
    put(72, &1i32.to_le_bytes());
    put(76, &dv.to_le_bytes()); put(80, &m1.to_le_bytes()); put(84, &sh.to_le_bytes());
    put(376, &4u32.to_le_bytes());                       // input strides_[0][0]
    if rows > 1 {
        let (dv, m1, sh) = int_divider(rows);
        put(476, &1i32.to_le_bytes());
        put(480, &dv.to_le_bytes()); put(484, &m1.to_le_bytes()); put(488, &sh.to_le_bytes());
        put(780, &4u32.to_le_bytes());                   // strides_[0][0]: output bytes per row
        put(784, &((4 * n) as u32).to_le_bytes());       // strides_[0][1]: input bytes per row
    }
    put(984, &input.to_le_bytes());
    put(992, &out.to_le_bytes());
    put(1041, &[1]);                                     // final_output
    put(1044, &1i32.to_le_bytes());                      // noutputs
    let grid = rows.div_ceil(step_output);
    Launch { name: &name, grid: [grid as u32, 1, 1], block: [bw as u32, bh as u32, 1], smem: (4 * threads) as u32, attrs: vec![], params: vec![b] }.run(k, s)
}

/// `x.sum(dim=1)` of fp32 [rows, c, d] -> [rows, d]: ATen reduce_kernel<128, 4, ReduceOp<float,
/// func_wrapper_t<float, sum_functor>, ...>> with the config it picks for a reduction over a
/// non-contiguous dim (block 32 x 4 along the outputs, output vectorized by 4, each thread adding
/// its c inputs in order). Same ReduceOp layout as `mean_rows_f32`.
pub fn sum_dim1_f32(k: &Kernels, s: Stream, input: u64, rows: usize, c: usize, d: usize, out: u64) -> Result<()> {
    let name = k.names_containing("reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_11sum_functor").into_iter().next()
        .ok_or("kernel bundle has no ATen sum reduce kernel")?;
    if d % 512 != 0 { return Err("sum_dim1: d must be a multiple of 512".into()); }
    let outputs = rows * d;
    let mut b = vec![0u8; 1048];
    let mut put = |off: usize, bytes: &[u8]| b[off..off + bytes.len()].copy_from_slice(bytes);
    let cfg = [4, c, outputs, 1, 128, 1, 0, 0, 0, 1, 32, 32, 4, 128];
    for (i, v) in cfg.iter().enumerate() { put(8 + 4 * i, &(*v as i32).to_le_bytes()); }
    put(68, &4i32.to_le_bytes());                        // output_vec_size (vectorize_input false)
    let (dv, m1, sh) = int_divider(c);
    put(72, &1i32.to_le_bytes());
    put(76, &dv.to_le_bytes()); put(80, &m1.to_le_bytes()); put(84, &sh.to_le_bytes());
    put(376, &((4 * d) as u32).to_le_bytes());           // input stride of the reduced dim
    let (dv, m1, sh) = int_divider(d);
    put(476, &(if rows > 1 { 2i32 } else { 1 }).to_le_bytes());
    put(480, &dv.to_le_bytes()); put(484, &m1.to_le_bytes()); put(488, &sh.to_le_bytes());
    put(780, &4u32.to_le_bytes()); put(784, &4u32.to_le_bytes());
    if rows > 1 {
        let (dv, m1, sh) = int_divider(rows);
        put(492, &dv.to_le_bytes()); put(496, &m1.to_le_bytes()); put(500, &sh.to_le_bytes());
        put(788, &((4 * d) as u32).to_le_bytes());       // output bytes per row
        put(792, &((4 * c * d) as u32).to_le_bytes());   // input bytes per row
    }
    put(984, &input.to_le_bytes());
    put(992, &out.to_le_bytes());
    put(1041, &[1]);
    put(1044, &1i32.to_le_bytes());
    Launch { name: &name, grid: [(outputs / 512) as u32, 1, 1], block: [32, 4, 1], smem: 0, attrs: vec![], params: vec![b] }.run(k, s)
}

/// An ATen vectorized unary fp32 kernel (vectorized_elementwise_kernel<4, f, array<char*, 2>>:
/// numel, the empty functor, [out, in]; 128 threads x 8 elements per block).
fn unary_f32(k: &Kernels, s: Stream, fragment: &str, input: u64, out: u64, n: usize) -> Result<()> {
    let name = k.names_containing(fragment).into_iter().find(|n| n.contains("vectorized_elementwise_kernelILi4E"))
        .ok_or_else(|| format!("kernel bundle has no ATen {fragment} kernel"))?;
    let mut arr = out.to_le_bytes().to_vec();
    arr.extend(input.to_le_bytes());
    Launch { name: &name, grid: [n.div_ceil(1024) as u32, 1, 1], block: [128, 1, 1], smem: 0, attrs: vec![],
             params: vec![(n as i32).to_le_bytes().to_vec(), vec![0u8], arr] }.run(k, s)
}

/// torch.rsqrt (fp32).
pub fn rsqrt_f32(k: &Kernels, s: Stream, input: u64, out: u64, n: usize) -> Result<()> { unary_f32(k, s, "rsqrt_kernel_cuda", input, out, n) }
/// torch.sigmoid (fp32).
pub fn sigmoid_f32(k: &Kernels, s: Stream, input: u64, out: u64, n: usize) -> Result<()> { unary_f32(k, s, "sigmoid_kernel_cuda", input, out, n) }
