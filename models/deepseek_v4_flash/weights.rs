//! SGLang's loaded parameters, rebuilt from the published checkpoint.
//!
//! `prepare(name)` returns the bytes of SGLang's parameter `name` in logical row-major order (what
//! the reference's params SHA-256 covers), or None where the recipe is not ported yet. The memory
//! layout the kernels read is a separate concern (e.g. MN-major packed scales).
use crate::dense_prepare::{cat_rows, e4m3_to_f32, f32_to_e4m3, pack_block_scales, requant_ue8m0, u32_bytes};
use crate::moe_prepare::{gather_rows, gemm1_rows, interleave_scales, shuffle_rows};
use crate::safetensors::Checkpoint;
use crate::Result;

pub struct Source<'a> { pub ckpt: &'a Checkpoint }

fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    Some((rest[..dot].parse().ok()?, &rest[dot + 1..]))
}

impl<'a> Source<'a> {
    fn read(&self, name: &str) -> Result<Vec<u8>> { self.ckpt.read(name) }

    fn shape(&self, name: &str) -> Result<Vec<usize>> { Ok(self.ckpt.entry(name)?.shape.clone()) }

    fn dtype(&self, name: &str) -> Result<String> { Ok(self.ckpt.entry(name)?.dtype.clone()) }

    /// Compressor `ape` [ratio, coff * d] as SGLang's `_apply_ape_hotfix` leaves it: for the
    /// overlapping ratio-4 compressors the two column halves are stacked as rows, then viewed back.
    fn ape(&self, name: &str) -> Result<Vec<u8>> {
        let data = self.as_f32(name)?;
        let shape = self.shape(name)?;
        let (ratio, width) = (shape[0], shape[1]);
        if ratio != 4 { return Ok(data); }
        let half = width / 2 * 4; // bytes per half row
        let mut out = Vec::with_capacity(data.len());
        for h in 0..2 {
            for r in 0..ratio {
                let row = &data[r * width * 4..(r + 1) * width * 4];
                out.extend_from_slice(&row[h * half..(h + 1) * half]);
            }
        }
        Ok(out)
    }

    /// Integer checkpoint tensor narrowed to int32 (torch `.to(int32)` keeps the low 32 bits).
    fn int32(&self, name: &str) -> Result<Vec<u8>> {
        let data = self.read(name)?;
        match self.dtype(name)?.as_str() {
            "I32" => Ok(data),
            "I64" => Ok(data.chunks_exact(8).flat_map(|c| [c[0], c[1], c[2], c[3]]).collect()),
            d => Err(format!("{name}: cannot narrow {d} to int32")),
        }
    }

    /// BF16 checkpoint tensor as FP32 (exact widening), passing FP32 through.
    fn as_f32(&self, name: &str) -> Result<Vec<u8>> {
        let data = self.read(name)?;
        match self.dtype(name)?.as_str() {
            "F32" => Ok(data),
            "BF16" => Ok(data.chunks_exact(2).flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes()).collect()),
            d => Err(format!("{name}: cannot widen {d} to f32")),
        }
    }

    /// Packed UE8M0 scales of FP8 block-quantized weights fused by rows from `parts` (prefixes).
    fn packed_scales(&self, parts: &[String]) -> Result<Vec<u8>> {
        let mut scales = Vec::new();
        let mut mn = 0;
        let mut sf_k = 0;
        for p in parts {
            let s = self.read(&format!("{p}.scale"))?;
            let w = self.shape(&format!("{p}.weight"))?;
            let ss = self.shape(&format!("{p}.scale"))?;
            if w[0] % 128 != 0 { return Err(format!("{p}: rows not a multiple of 128")); }
            mn += w[0];
            sf_k = ss[1];
            scales.extend_from_slice(&s);
        }
        Ok(u32_bytes(&pack_block_scales(&scales, mn, sf_k, 128).0))
    }

    /// FP8 block-quantized weights fused by rows, requantized for UE8M0 DeepGEMM as SGLang does at
    /// load time. Returns (weight bytes, packed scales as logical int32 bytes).
    fn requantized(&self, parts: &[String]) -> Result<(Vec<u8>, Vec<u8>)> {
        let weight = self.fused_weight(parts)?;
        let mut scales = Vec::new();
        let mut n = 0;
        let mut k = 0;
        for p in parts {
            let w = self.shape(&format!("{p}.weight"))?;
            n += w[0];
            k = w[1];
            scales.extend_from_slice(&self.read(&format!("{p}.scale"))?);
        }
        let (w, s) = requant_ue8m0(&weight, &scales, n, k);
        Ok((w, u32_bytes(&pack_block_scales(&s, n, k.div_ceil(128), 128).0)))
    }

    fn fused_weight(&self, parts: &[String]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for p in parts { out = cat_rows(&out, &self.read(&format!("{p}.weight"))?); }
        Ok(out)
    }

    pub fn prepare(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let direct = |src: &str| self.read(src).map(Some);
        match name {
            "model.embed_tokens.weight" => return direct("embed.weight"),
            "lm_head.weight" => return direct("head.weight"),
            "model.norm.weight" => return direct("norm.weight"),
            "model.hc_head_fn" | "model.hc_head_base" | "model.hc_head_scale" => return direct(&name["model.".len()..]),
            _ => {}
        }
        let Some((l, rest)) = layer_of(name) else { return Ok(None) };
        self.layer_param(&format!("layers.{l}"), rest)
    }

    /// The MTP draft model's parameter `name` (SGLang's DeepseekV4ForCausalLMNextN, built from the
    /// checkpoint's `mtp.0.*`): its decoder is a layer like the target's (compress ratio 0) whose
    /// routed experts are MXFP4; embedding and LM head are the target's.
    pub fn prepare_draft(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let ck = "mtp.0";
        let direct = |src: &str| self.read(src).map(Some);
        let v = |p: &str| vec![p.to_string()];
        match name {
            "model.embed_tokens.weight" => return direct("embed.weight"),
            "lm_head.weight" => return direct("head.weight"),
            "model.enorm.weight" | "model.hnorm.weight" => return direct(&format!("{ck}.{}", &name["model.".len()..])),
            "model.shared_head.norm.weight" => return direct(&format!("{ck}.norm.weight")),
            "model.hc_head_fn" | "model.hc_head_base" | "model.hc_head_scale" => return direct(&format!("{ck}.{}", &name["model.".len()..])),
            "model.e_proj.weight" => return Ok(Some(self.requantized(&v(&format!("{ck}.e_proj")))?.0)),
            "model.e_proj.weight_scale_inv" => return Ok(Some(self.requantized(&v(&format!("{ck}.e_proj")))?.1)),
            "model.h_proj.weight" => return Ok(Some(self.requantized(&v(&format!("{ck}.h_proj")))?.0)),
            "model.h_proj.weight_scale_inv" => return Ok(Some(self.requantized(&v(&format!("{ck}.h_proj")))?.1)),
            _ => {}
        }
        let Some(rest) = name.strip_prefix("model.decoder.") else { return Ok(None) };
        if let Some(what) = rest.strip_prefix("mlp.experts.") { return self.mxfp4_expert_param(ck, what).map(Some); }
        self.layer_param(ck, rest)
    }

    /// One routed-experts parameter of the MXFP4 experts under `ck` (all 256), in SGLang's
    /// Mxfp4FlashinferTrtllmMoEMethod layout: w13 = [w3; w1] (reorder_w1w3_to_w3w1), the gated row
    /// interleave and 32-row shuffle (official permute, epilogue tile 128) on weights and E8M0
    /// scales alike, the scales then block_scale_interleave'd (128x4); w2 shuffled the same way.
    fn mxfp4_expert_param(&self, ck: &str, what: &str) -> Result<Vec<u8>> {
        let n = 256;
        let x = |i: usize, w: &str, t: &str| format!("{ck}.ffn.experts.{i}.{w}.{t}");
        let ones = || (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<u8>>();
        Ok(match what {
            "w13_weight" | "w13_weight_scale_inv" => {
                let t = if what == "w13_weight" { "weight" } else { "scale" };
                let mut out = Vec::new();
                for i in 0..n {
                    let (up, gate) = (self.read(&x(i, "w3", t))?, self.read(&x(i, "w1", t))?);
                    let rows = self.shape(&x(i, "w3", t))?[0] * 2;
                    let fused = cat_rows(&up, &gate);
                    let row_bytes = fused.len() / rows;
                    let permuted = gather_rows(&fused, row_bytes, &gemm1_rows(rows, 128));
                    out.extend(if t == "weight" { permuted } else { interleave_scales(&permuted, rows, row_bytes) });
                }
                out
            }
            "w2_weight" | "w2_weight_scale_inv" => {
                let t = if what == "w2_weight" { "weight" } else { "scale" };
                let mut out = Vec::new();
                for i in 0..n {
                    let w = self.read(&x(i, "w2", t))?;
                    let rows = self.shape(&x(i, "w2", t))?[0];
                    let permuted = gather_rows(&w, w.len() / rows, &shuffle_rows(rows, 128));
                    out.extend(if t == "weight" { permuted } else { interleave_scales(&permuted, rows, w.len() / rows) });
                }
                out
            }
            "output1_scale_scalar" | "output1_scale_gate_scalar" | "output2_scale_scalar" => ones(),
            other => return Err(format!("MXFP4 experts parameter {other} is not ported")),
        })
    }

    /// Parameter `rest` of a decoder layer (SGLang's names under `model.layers.N.`) whose
    /// checkpoint tensors sit under `ck` (`layers.N`, or `mtp.0` for the MTP layer).
    pub fn layer_param(&self, ck: &str, rest: &str) -> Result<Option<Vec<u8>>> {
        let a = format!("{ck}.attn");
        let f = format!("{ck}.ffn");
        let v = |parts: &[&str]| parts.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        Ok(Some(match rest {
            "hc_attn_fn" | "hc_attn_base" | "hc_attn_scale" | "hc_ffn_fn" | "hc_ffn_base" | "hc_ffn_scale" =>
                self.read(&format!("{ck}.{rest}"))?,
            "input_layernorm.weight" => self.read(&format!("{ck}.attn_norm.weight"))?,
            "post_attention_layernorm.weight" => self.read(&format!("{ck}.ffn_norm.weight"))?,
            "self_attn.attn_sink" => self.read(&format!("{a}.attn_sink"))?,
            "self_attn.wqkv_a.weight" => self.requantized(&v(&[&format!("{a}.wq_a"), &format!("{a}.wkv")]))?.0,
            "self_attn.wqkv_a.weight_scale_inv" => self.requantized(&v(&[&format!("{a}.wq_a"), &format!("{a}.wkv")]))?.1,
            "self_attn.q_norm.weight" => self.read(&format!("{a}.q_norm.weight"))?,
            "self_attn.kv_norm.weight" => self.read(&format!("{a}.kv_norm.weight"))?,
            "self_attn.wo_a.weight" => self.read(&format!("{a}.wo_a.weight"))?,
            "self_attn.wq_b.weight" => self.requantized(&v(&[&format!("{a}.wq_b")]))?.0,
            "self_attn.wq_b.weight_scale_inv" => self.requantized(&v(&[&format!("{a}.wq_b")]))?.1,
            "self_attn.wo_b.weight" => self.requantized(&v(&[&format!("{a}.wo_b")]))?.0,
            "self_attn.wo_b.weight_scale_inv" => self.requantized(&v(&[&format!("{a}.wo_b")]))?.1,
            "self_attn.wo_a.weight_scale_inv" => {
                // 8 groups of 1024 rows, each packed on its own: logical [8, 1024, 8].
                let s = self.read(&format!("{a}.wo_a.scale"))?;
                let ss = self.shape(&format!("{a}.wo_a.scale"))?;
                let groups = 8;
                let rows = ss[0] / groups;
                let mut out = Vec::new();
                for g in 0..groups {
                    let part = &s[g * rows * ss[1]..(g + 1) * rows * ss[1]];
                    out.extend(u32_bytes(&pack_block_scales(part, rows * 128, ss[1], 128).0));
                }
                out
            }
            "self_attn.attn_mqa.k_scale" | "self_attn.attn_mqa.v_scale" => 1.0f32.to_le_bytes().to_vec(),
            "self_attn.compressor.ape" => self.ape(&format!("{a}.compressor.ape"))?,
            "self_attn.compressor.norm.weight" => self.as_f32(&format!("{a}.compressor.norm.weight"))?,
            "self_attn.compressor.wkv_gate.weight" =>
                self.fused_weight(&v(&[&format!("{a}.compressor.wkv"), &format!("{a}.compressor.wgate")]))?,
            "self_attn.indexer.wq_b.weight" => self.requantized(&v(&[&format!("{a}.indexer.wq_b")]))?.0,
            "self_attn.indexer.wq_b.weight_scale_inv" => self.requantized(&v(&[&format!("{a}.indexer.wq_b")]))?.1,
            "self_attn.indexer.weights_proj.weight" => self.read(&format!("{a}.indexer.weights_proj.weight"))?,
            "self_attn.indexer.compressor.ape" => self.ape(&format!("{a}.indexer.compressor.ape"))?,
            "self_attn.indexer.compressor.norm.weight" => self.as_f32(&format!("{a}.indexer.compressor.norm.weight"))?,
            "self_attn.indexer.compressor.wkv_gate.weight" => self.fused_weight(&v(&[
                &format!("{a}.indexer.compressor.wkv"), &format!("{a}.indexer.compressor.wgate")]))?,
            "mlp.gate.weight" => self.read(&format!("{f}.gate.weight"))?,
            "mlp.gate.e_score_correction_bias" => self.read(&format!("{f}.gate.bias"))?,
            "mlp.topk.tid2eid" => self.int32(&format!("{f}.gate.tid2eid"))?,
            "mlp.shared_experts.gate_up_proj.weight" =>
                self.requantized(&v(&[&format!("{f}.shared_experts.w1"), &format!("{f}.shared_experts.w3")]))?.0,
            "mlp.shared_experts.gate_up_proj.weight_scale_inv" =>
                self.requantized(&v(&[&format!("{f}.shared_experts.w1"), &format!("{f}.shared_experts.w3")]))?.1,
            "mlp.shared_experts.down_proj.weight" => self.requantized(&v(&[&format!("{f}.shared_experts.w2")]))?.0,
            "mlp.shared_experts.down_proj.weight_scale_inv" => self.requantized(&v(&[&format!("{f}.shared_experts.w2")]))?.1,
            r if r.starts_with("mlp.experts.") => return self.expert_param(ck, &r["mlp.experts.".len()..]).map(Some),
            _ => return Ok(None),
        }))
    }

    fn scalar_f32(&self, name: &str) -> Result<f32> {
        let d = self.read(name)?;
        if d.len() != 4 { return Err(format!("{name}: expected one f32")); }
        Ok(f32::from_le_bytes([d[0], d[1], d[2], d[3]]))
    }

    /// Per-expert global scales and the layer scalars SGLang derives from them.
    fn expert_scales(&self, ck: &str) -> Result<ExpertScales> {
        let n = 256;
        let e = |i: usize, w: &str, what: &str| self.scalar_f32(&format!("{ck}.ffn.experts.{i}.{w}.{what}"));
        let mut s = ExpertScales::default();
        for i in 0..n {
            s.w13_scale_2.push([e(i, "w1", "weight_scale_2")?, e(i, "w3", "weight_scale_2")?]);
            s.w2_scale_2.push(e(i, "w2", "weight_scale_2")?);
            s.w13_input.push([e(i, "w1", "input_scale")?, e(i, "w3", "input_scale")?]);
            s.w2_input.push(e(i, "w2", "input_scale")?);
        }
        // `.max()` of the [E, 2] / [E] input scales (NaN-free checkpoint).
        s.w13_is = s.w13_input.iter().flat_map(|p| p.iter().copied()).fold(f32::NEG_INFINITY, f32::max);
        s.w2_is = s.w2_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for i in 0..n {
            let gate = s.w13_is * s.w13_scale_2[i][0];
            let up = s.w13_is * s.w13_scale_2[i][1];
            s.g1_gate_raw.push(gate);
            s.g1_up_raw.push(up);
            let shared = if gate != up { gate.max(up) } else { gate };
            s.g1.push(shared);
            s.g2.push(s.w2_is * s.w2_scale_2[i]);
        }
        s.w13_isq = 1.0f32 / s.w13_is;
        s.w2_isq = 1.0f32 / s.w2_is;
        Ok(s)
    }

    /// One routed-experts parameter of layer `l` (all 256 experts), in SGLang's TRT-LLM layout.
    fn expert_param(&self, ck: &str, what: &str) -> Result<Vec<u8>> {
        let n = 256;
        let x = |i: usize, w: &str, t: &str| format!("{ck}.ffn.experts.{i}.{w}.{t}");
        let f32s = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        Ok(match what {
            "w13_weight" => {
                let mut out = Vec::new();
                for i in 0..n {
                    let up = self.read(&x(i, "w3", "weight"))?;
                    let gate = self.read(&x(i, "w1", "weight"))?;
                    let rows = self.shape(&x(i, "w3", "weight"))?[0] * 2;
                    let fused = cat_rows(&up, &gate);
                    out.extend(gather_rows(&fused, fused.len() / rows, &gemm1_rows(rows, 128)));
                }
                out
            }
            "w2_weight" => {
                let mut out = Vec::new();
                for i in 0..n {
                    let w = self.read(&x(i, "w2", "weight"))?;
                    let rows = self.shape(&x(i, "w2", "weight"))?[0];
                    out.extend(gather_rows(&w, w.len() / rows, &shuffle_rows(rows, 128)));
                }
                out
            }
            "w13_weight_scale" => {
                let s = self.expert_scales(ck)?;
                let mut out = Vec::new();
                for i in 0..n {
                    let mut up = self.read(&x(i, "w3", "weight_scale"))?;
                    let mut gate = self.read(&x(i, "w1", "weight_scale"))?;
                    let shape = self.shape(&x(i, "w3", "weight_scale"))?;
                    let (gate_a, up_a) = (s.g1_gate_raw[i], s.g1_up_raw[i]);
                    if gate_a != up_a {
                        // Rows are [up, gate]; ratios = [up, gate] / max(gate, up), then back to E4M3.
                        let shared = gate_a.max(up_a);
                        for (half, ratio) in [(&mut up, up_a / shared), (&mut gate, gate_a / shared)] {
                            for b in half.iter_mut() { *b = f32_to_e4m3(e4m3_to_f32(*b) * ratio); }
                        }
                    }
                    let fused = cat_rows(&up, &gate);
                    let (m, k) = (shape[0] * 2, shape[1]);
                    let permuted = gather_rows(&fused, k, &gemm1_rows(m, 128));
                    out.extend(interleave_scales(&permuted, m, k));
                }
                out
            }
            "w2_weight_scale" => {
                let mut out = Vec::new();
                for i in 0..n {
                    let w = self.read(&x(i, "w2", "weight_scale"))?;
                    let shape = self.shape(&x(i, "w2", "weight_scale"))?;
                    let permuted = gather_rows(&w, shape[1], &shuffle_rows(shape[0], 128));
                    out.extend(interleave_scales(&permuted, shape[0], shape[1]));
                }
                out
            }
            "w13_weight_scale_2" => { let s = self.expert_scales(ck)?; f32s(&s.w13_scale_2.concat()) }
            "w2_weight_scale_2" => f32s(&self.expert_scales(ck)?.w2_scale_2),
            "w13_input_scale" => { let s = self.expert_scales(ck)?; f32s(&s.w13_input.concat()) }
            "w2_input_scale" => f32s(&self.expert_scales(ck)?.w2_input),
            "g1_alphas" | "g1_alphas_up" => f32s(&self.expert_scales(ck)?.g1),
            "g2_alphas" => f32s(&self.expert_scales(ck)?.g2),
            "w13_input_scale_quant" => f32s(&[self.expert_scales(ck)?.w13_isq]),
            "w2_input_scale_quant" => f32s(&[self.expert_scales(ck)?.w2_isq]),
            "gemm1_clamp_limit" => {
                let s = self.expert_scales(ck)?;
                // torch `scalar / tensor` is `tensor.reciprocal() * scalar` (two roundings).
                f32s(&s.g1.iter().map(|a| (1.0f32 / a) * 10.0f32).collect::<Vec<_>>())
            }
            "g1_scale_c" => {
                let s = self.expert_scales(ck)?;
                f32s(&s.g1.iter().map(|a| s.w2_isq * a).collect::<Vec<_>>())
            }
            other => return Err(format!("experts parameter {other} is not ported")),
        })
    }
}

#[derive(Default)]
struct ExpertScales {
    w13_scale_2: Vec<[f32; 2]>,
    w2_scale_2: Vec<f32>,
    w13_input: Vec<[f32; 2]>,
    w2_input: Vec<f32>,
    w13_is: f32,
    w2_is: f32,
    g1_gate_raw: Vec<f32>,
    g1_up_raw: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    w13_isq: f32,
    w2_isq: f32,
}

/// Packed UE8M0 scales are MN-major in memory (int32 [.., N, P] stored as [.., P, align4(N)]).
pub fn memory_layout(name: &str, logical: Vec<u8>) -> Vec<u8> {
    // Routed-expert scales (MXFP4's E8M0 bytes) are already in the kernels' swizzled layout.
    if !name.ends_with("weight_scale_inv") || name.contains("experts.w") { return logical; }
    // Shapes follow from the parameter: groups x N x P int32. Recover N and P from the packing rule
    // through the tensor's bytes and the known K blocks of each projection.
    let (groups, n, p) = scale_shape(name, logical.len() / 4);
    let aligned = n.div_ceil(4) * 4;
    let words: Vec<u32> = logical.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut out = vec![0u32; groups * p * aligned];
    for g in 0..groups {
        for i in 0..n {
            for j in 0..p { out[g * p * aligned + j * aligned + i] = words[(g * n + i) * p + j]; }
        }
    }
    out.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn scale_shape(name: &str, words: usize) -> (usize, usize, usize) {
    // (groups, N rows, packed K columns) of each packed-scale parameter in this model.
    let table: &[(&str, usize, usize, usize)] = &[
        ("self_attn.wqkv_a.weight_scale_inv", 1, 1536, 8),
        ("self_attn.wq_b.weight_scale_inv", 1, 32768, 2),
        ("self_attn.wo_a.weight_scale_inv", 8, 1024, 8),
        ("self_attn.wo_b.weight_scale_inv", 1, 4096, 16),
        ("self_attn.indexer.wq_b.weight_scale_inv", 1, 8192, 2),
        ("shared_experts.gate_up_proj.weight_scale_inv", 1, 4096, 8),
        ("shared_experts.down_proj.weight_scale_inv", 1, 4096, 4),
        ("e_proj.weight_scale_inv", 1, 4096, 8),
        ("h_proj.weight_scale_inv", 1, 4096, 8),
    ];
    for (suffix, g, n, p) in table {
        if name.ends_with(suffix) { assert_eq!(g * n * p, words, "{name}"); return (*g, *n, *p); }
    }
    panic!("unknown packed scale {name}");
}
