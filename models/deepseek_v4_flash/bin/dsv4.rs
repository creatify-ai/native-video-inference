//! `dsv4`: parity rungs and tools for the native DeepSeek-V4-Flash port.
//!
//!   dsv4 verify-weights <checkpoint dir> <params.json> [name prefix]
//!       Rebuild SGLang's loaded parameters from the checkpoint and compare each SHA-256 with the
//!       reference's params dump. Prints one JSON line per parameter and a summary.
//!
//!   dsv4 run <checkpoint dir | prepared dir> <kernel bundle> <tables dir> <prompts.json> <max new tokens> <out.json>
//!       Serve each prompt greedily (SGLang /generate with temperature 0, return_logprob,
//!       top_logprobs_num 5, logprob_start_len 0) and write the responses in the reference's
//!       outputs schema. `prompts.json` is a list of strings or of reference responses; for the
//!       latter each native response is also compared with the reference's. A prepared package
//!       (DSV4_MANIFEST_SHA256 pins its manifest) loads every weight up front through core_infra's
//!       streaming loader, its readers started before the CUDA context; the startup timeline is printed.
//!
//!   dsv4 prepare <checkpoint dir> <params.json> <tables dir> <out dir> [workers]
//!       Write the serving package for core_infra's streaming loader (prepare.rs): every
//!       parameter, checked against the reference's params dump, in 64 MiB safetensors shards.
//!       `python -m core_infra.runtime.checkpoint manifest --root <out>/weights --file '*.safetensors'
//!       --out <out>/manifest.json` then writes its manifest.
//!
//!   dsv4 bench-long <prepared dir> <kernel bundle> <long gold run dir> <prompt indices> <decode> <reps>
//!       Long-context throughput: time to first token (chunked prefill) and decode inter-token
//!       latency on the run's pre-tokenized prompts, with every output token and logprob checked
//!       against the reference server's outputs.
//!
//!   dsv4 bench-concurrent <prepared dir> <kernel bundle> <concurrent gold run dir> <decode> <reps>
//!       Concurrent serving: the run's prompts all prefilled, then decoded as one batch (CUDA graph
//!       per batch size); wall time and throughput, each request's tokens checked against the
//!       reference server's.
//!
//!   dsv4 mtp-target <prepared dir> <kernel bundle> <MTP gold run dir>
//!       MTP (speculative decoding): the target model's prefill and each verify forward of the
//!       reference run, fed its tokens, compared forward by forward.
//!
//!   dsv4 mtp-generate <prepared dir> <kernel bundle> <MTP run dir> <checkpoint> [max new tokens]
//!       Greedy MTP generation of the run's first prompt (draft + verify loop), checked against
//!       the reference server's tokens.
//!
//!   dsv4 mtp-serve <prepared dir> <kernel bundle> <concurrent MTP gold run> <checkpoint> [reps]
//!       Concurrent MTP serving on SGLang's schedule, each request checked against the reference.
//!
//!   dsv4 tokenize <checkpoint dir> <outputs.json>
//!       Encode each reference prompt and decode its generated ids natively; compare with the
//!       reference server's input ids and output text.
use dsv4_flash_native::{gold::sha, safetensors::Checkpoint, weights::Source};
use serde_json::{json, Value};
use std::path::Path;

fn verify_weights(args: &[String]) -> Result<(), String> {
    let [root, params] = [&args[0], &args[1]];
    let prefix = args.get(2).cloned().unwrap_or_default();
    let ckpt = Checkpoint::open(Path::new(root))?;
    let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(params).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let source = Source { ckpt: &ckpt };
    let (mut equal, mut differ, mut missing, mut failed) = (0, 0, 0, 0);
    for row in rows {
        let name = row["name"].as_str().unwrap_or("");
        if !name.starts_with(&prefix) { continue; }
        let want = row["sha256"].as_str().unwrap_or("");
        // A draft model's dump (params-pidN-draft.json): the MTP layer's recipes.
        let prepared = if params.ends_with("-draft.json") { source.prepare_draft(name) } else { source.prepare(name) };
        match prepared {
            Ok(Some(bytes)) => {
                let got = sha(&bytes);
                let ok = got == want && bytes.len() as u64 == row["bytes"].as_u64().unwrap_or(0);
                if ok { equal += 1 } else { differ += 1 }
                if !ok {
                    println!("{}", json!({"name": name, "equal": false, "native_bytes": bytes.len(), "ref_bytes": row["bytes"],
                                          "native": got, "ref": want, "dtype": row["dtype"], "shape": row["shape"]}));
                }
            }
            Ok(None) => { missing += 1; }
            Err(e) => { failed += 1; println!("{}", json!({"name": name, "error": e})); }
        }
    }
    println!("{}", json!({"summary": {"equal": equal, "differ": differ, "not_ported": missing, "errors": failed}}));
    if differ + failed > 0 { Err("weights differ".into()) } else { Ok(()) }
}

fn tokenize(args: &[String]) -> Result<(), String> {
    let tok = dsv4_flash_native::tokenize::Tokenizer::open(Path::new(&args[0]))?;
    let outputs: Value = serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut all = true;
    for (i, o) in outputs.as_array().ok_or("outputs is not a list")?.iter().enumerate() {
        let ids = tok.encode(o["prompt"].as_str().unwrap_or(""))?;
        let want_ids: Vec<u32> = o["input_token_logprobs"].as_array().map(|a| a.iter().map(|e| e[1].as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
        let out_ids: Vec<u32> = o["output_token_logprobs"].as_array().map(|a| a.iter().map(|e| e[1].as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
        let text = tok.decode(&out_ids)?;
        let (ids_ok, text_ok) = (ids == want_ids, text == o["text"].as_str().unwrap_or(""));
        all &= ids_ok && text_ok;
        println!("{}", json!({"prompt": i, "input_ids_equal": ids_ok, "text_equal": text_ok, "ids": ids,
                              "text": if text_ok { Value::Null } else { Value::String(text) }}));
    }
    if all { Ok(()) } else { Err("tokenization differs".into()) }
}

#[cfg(feature = "cuda")]
fn run(args: &[String], t0: std::time::Instant) -> Result<(), String> {
    use dsv4_flash_native::{model::Model, tokenize::Tokenizer};
    let [model_dir, bundle, tables, prompts_path, max_new, out_path] = [&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]];
    let model_dir = Path::new(model_dir);
    let mut timeline: Vec<(&'static str, f64)> = Vec::new();
    // A prepared package (manifest.json): start its readers first, before anything touches CUDA.
    // The package's CUDA compute cache: the reference cuBLASLt ships its sm_103 GEMM kernels as PTX,
    // which the driver would otherwise JIT-compile on first use (seconds per kernel). Set before
    // any CUDA call; an explicit CUDA_CACHE_PATH wins.
    if model_dir.join("cuda-cache").is_dir() && std::env::var_os("CUDA_CACHE_PATH").is_none() {
        std::env::set_var("CUDA_CACHE_PATH", model_dir.join("cuda-cache"));
        if std::env::var_os("CUDA_CACHE_MAXSIZE").is_none() { std::env::set_var("CUDA_CACHE_MAXSIZE", "4294967296"); }
    }
    // Reader policy measured on B300 (Modal Volume, warm host): direct reads (buffered reads through
    // the sandbox stop near 5 GB/s), 64 readers, 16 MiB chunks (26.7 GB/s; 32 MiB: 21.5 GB/s, with
    // a slower reader start). Explicit NATIVE_LOAD_* settings win.
    if model_dir.join("manifest.json").exists() {
        for (key, value) in [("NATIVE_LOAD_FILE_IO", "direct"), ("NATIVE_LOAD_WORKERS", "64"), ("NATIVE_LOAD_CHUNK_MIB", "16")] {
            if std::env::var_os(key).is_none() { std::env::set_var(key, value); }
        }
    }
    let startup = if model_dir.join("manifest.json").exists() {
        let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "a prepared package needs DSV4_MANIFEST_SHA256 (its manifest's pin)")?;
        let startup = Model::start_prepared(model_dir, &sha, Path::new(bundle))?;
        timeline.push(("readers_started", t0.elapsed().as_secs_f64()));
        Some(startup)
    } else { None };
    let max_new: usize = max_new.parse().map_err(|e| format!("max new tokens: {e}"))?;
    let prompts: Value = serde_json::from_slice(&std::fs::read(prompts_path).map_err(|e| format!("{prompts_path}: {e}"))?).map_err(|e| e.to_string())?;
    let tok = Tokenizer::open(model_dir)?;
    // generation_config.json eos_token_id (an id or a list of ids).
    let gen: Value = serde_json::from_slice(&std::fs::read(model_dir.join("generation_config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let eos: Vec<usize> = match &gen["eos_token_id"] { Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect(),
                                                       v => v.as_u64().map(|v| vec![v as usize]).unwrap_or_default() };
    timeline.push(("tokenizer", t0.elapsed().as_secs_f64()));
    let ckpt = if startup.is_none() { Some(Checkpoint::open(model_dir)?) } else { None };
    let mut model = match (startup, &ckpt) {
        (Some(startup), _) => startup.finish(t0, &mut timeline)?,
        (None, Some(ckpt)) => Model::open(ckpt, Path::new(bundle), Path::new(tables))?,
        _ => unreachable!(),
    };
    let mut responses = Vec::new();
    let mut all_equal = true;
    for (i, p) in prompts.as_array().ok_or("prompts is not a list")?.iter().enumerate() {
        let text = p.as_str().or_else(|| p["prompt"].as_str()).ok_or("prompt is not a string")?;
        let ids: Vec<i64> = tok.encode(text)?.iter().map(|&v| v as i64).collect();
        let request_start = t0.elapsed().as_secs_f64();
        let g = model.generate(&ids, max_new, &eos)?;
        if i == 0 {
            timeline.push(("first_token", request_start + g.ms[0] as f64 / 1e3));
            let epoch0: f64 = std::env::var("DSV4_EPOCH0").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let mut line = json!({"startup_s": timeline.iter().map(|(k, v)| json!([k, (v * 1e3).round() / 1e3])).collect::<Vec<_>>(),
                                  "epoch_process_start": epoch0, "epoch_first_token": epoch0 + request_start + g.ms[0] as f64 / 1e3});
            if let Some(s) = model.w.load_stats() {
                line["weights"] = json!({"tensors": s.tensors, "read_gb": s.read_bytes as f64 / 1e9, "device_gb": s.device_bytes as f64 / 1e9,
                                         "load_s": s.seconds, "gb_per_s": s.read_bytes as f64 / 1e9 / s.seconds});
            }
            println!("{line}");
        }
        let entry = |lp: Option<f32>, id: usize| json!([lp.map(|v| v as f64), id, Value::Null]);
        let mut input = vec![entry(None, ids[0] as usize)];
        input.extend(ids[1..].iter().zip(&g.logprobs[0].input).map(|(&id, &lp)| entry(Some(lp), id as usize)));
        let response = json!({
            "prompt": text,
            "text": tok.decode(&g.tokens.iter().map(|&t| t as u32).collect::<Vec<_>>())?,
            "output_token_logprobs": g.tokens.iter().zip(&g.logprobs).map(|(&t, lp)| entry(Some(lp.token), t)).collect::<Vec<_>>(),
            "input_token_logprobs": input,
            "output_top_logprobs": g.logprobs.iter().map(|lp| lp.top.iter().map(|&(v, id)| entry(Some(v), id)).collect::<Vec<_>>()).collect::<Vec<_>>(),
        });
        let decode_ms: Vec<u128> = g.ms.iter().skip(1).copied().collect();
        // Steady-state per-token decode latency without logprobs (steps 3+: the graph replays).
        let itl: Vec<f64> = g.token_ms.iter().skip(3).copied().collect();
        let fwd: Vec<f64> = g.forward_ms.iter().skip(3).copied().collect();
        let fwd_ms = (fwd.iter().sum::<f64>() / fwd.len().max(1) as f64 * 1000.0).round() / 1000.0;
        let mut line = json!({"prompt": i, "prompt_tokens": ids.len(), "new_tokens": g.tokens.len(), "prefill_ms": g.ms.first(),
                              "decode_ms_mean": decode_ms.iter().sum::<u128>() as f64 / decode_ms.len().max(1) as f64,
                              "decode_itl_ms": (itl.iter().sum::<f64>() / itl.len().max(1) as f64 * 1000.0).round() / 1000.0,
                              "decode_forward_ms": fwd_ms});
        if p.is_object() {
            let equal = p == &response;
            all_equal &= equal;
            line["equal_to_reference"] = json!(equal);
            if !equal {
                for key in ["text", "output_token_logprobs", "input_token_logprobs", "output_top_logprobs"] {
                    line[format!("{key}_equal")] = json!(p[key] == response[key]);
                }
            }
        }
        println!("{line}");
        responses.push(response);
    }
    std::fs::write(out_path, serde_json::to_vec(&responses).map_err(|e| e.to_string())?).map_err(|e| format!("{out_path}: {e}"))?;
    if all_equal { Ok(()) } else { Err("native responses differ from the reference".into()) }
}

/// `dsv4 bench-long <prepared> <bundle> <long gold run> <prompt indices> <decode> <reps>`: serve the
/// run's pre-tokenized prompts as SGLang's long-context bench does (greedy, no input logprobs,
/// decode + 1 new tokens, chunked prefill), timing the time to first token (from the first chunk's
/// launch to the token on the host, KV cache preallocated as SGLang's pool is) and the decode
/// inter-token latency (graph replays), and check every token and output logprob against the run's
/// outputs-a.json (the reference server with CUDA graphs).
#[cfg(feature = "cuda")]
fn bench_long(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::Model;
    let [prepared, bundle, run_dir, which, decode, reps] = [&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]];
    let decode: usize = decode.parse().map_err(|e| format!("decode: {e}"))?;
    let reps: usize = reps.parse().map_err(|e| format!("reps: {e}"))?;
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    m.last_logits_only = true;
    let run = Path::new(run_dir);
    let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let reference: Value = serde_json::from_slice(&std::fs::read(run.join("outputs-a.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let entry = |lp: f32, id: usize| json!([lp as f64, id, Value::Null]);
    let mut all_equal = true;
    for idx in which.split(',') {
        let i: usize = idx.parse().map_err(|e| format!("prompt index: {e}"))?;
        let ids = &prompts[i];
        for rep in 0..reps {
            let want = &reference[i];
            let want_tokens: Option<Vec<Value>> = want["output_token_logprobs"].as_array().map(|w| w.iter().map(|e| e[1].clone()).collect());
            if rep == 0 {
                // Parity pass: tokens and output logprobs (per-step host sampling, as `run`).
                let n_ref = want_tokens.as_ref().map(|w| w.len()).unwrap_or(decode + 1);
                let g = m.generate(ids, n_ref, &[])?;
                let tokens: Vec<Value> = g.tokens.iter().zip(&g.logprobs).map(|(&t, lp)| entry(lp.token, t)).collect();
                let top: Vec<Value> = g.logprobs.iter().map(|lp| Value::Array(lp.top.iter().map(|&(v, id)| entry(v, id)).collect())).collect();
                let tokens_equal = want_tokens == Some(g.tokens.iter().map(|&t| json!(t)).collect());
                let logprobs_equal = want["output_token_logprobs"] == Value::Array(tokens) && want["output_top_logprobs"] == Value::Array(top);
                all_equal &= tokens_equal && logprobs_equal;
                println!("{}", json!({"prompt": i, "rep": rep, "mode": "logprobs", "prompt_tokens": ids.len(), "ttft_s": (g.token_ms[0] / 1e3 * 1e4).round() / 1e4,
                                      "tokens_equal": tokens_equal, "logprobs_equal": logprobs_equal}));
            } else {
                // Timing passes as SGLang's bench serves them (no logprobs): device sampling, steps
                // enqueued back to back; TTFT and ITL = (total - TTFT) / decode.
                let (tokens, ttft, itl) = m.generate_overlapped(ids, decode + 1)?;
                // The reference's tokens must be this run's first tokens (it served fewer).
                let tokens_equal = want_tokens.as_ref().map(|w| w.len() <= tokens.len() && w.iter().zip(&tokens).all(|(a, &b)| a == &json!(b))).unwrap_or(false);
                all_equal &= tokens_equal;
                println!("{}", json!({"prompt": i, "rep": rep, "mode": "overlapped", "prompt_tokens": ids.len(), "ttft_s": (ttft / 1e3 * 1e4).round() / 1e4,
                                      "prefill_tok_per_s": (ids.len() as f64 / (ttft / 1e3)).round(), "decode_itl_ms": (itl * 1e3).round() / 1e3,
                                      "tokens_equal": tokens_equal}));
            }
        }
    }
    if all_equal { Ok(()) } else { Err("native outputs differ from the reference".into()) }
}

/// Integer tensor of a gold (int32 or int64) as i64.
#[cfg(feature = "cuda")]
fn gold_ints(g: &dsv4_flash_native::gold::Gold, name: &str) -> Result<Vec<i64>, String> {
    let (meta, raw) = (g.meta(name)?, g.bytes(name)?);
    Ok(match meta.dtype.trim_start_matches("torch.") {
        "int64" => raw.chunks_exact(8).map(|v| i64::from_le_bytes(v.try_into().unwrap())).collect(),
        "int32" => raw.chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap()) as i64).collect(),
        d => return Err(format!("gold {name}: dtype {d}")),
    })
}

/// SGLang's schedule from a concurrent gold run: every captured forward in order, a prefill (its
/// requests' chunks, each request named by its first chunk's input ids) or a decode step (its
/// rows by request-pool row, in batch order). A request whose prefill starts over was retracted
/// (a Retract step before it).
/// Returns the steps (requests index `prompts`).
#[cfg(feature = "cuda")]
fn schedule_from_golds(gold: &Path, prompts: &[Vec<i64>]) -> Result<Vec<dsv4_flash_native::model::Step>, String> {
    use dsv4_flash_native::{gold::Gold, model::{Chunk, Step}};
    let mut steps = vec![];
    let mut pool: std::collections::HashMap<i64, usize> = Default::default();
    let mut admitted = vec![false; prompts.len()];
    for k in 1.. {
        let dir = gold.join(format!("fwd-{k:03}"));
        if !dir.join("meta.json").exists() { break; }
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        // SGLang's ForwardMode (an IntEnum, saved as its value): EXTEND = 1, DECODE = 2.
        let mode = match meta["forward_mode"].as_str().unwrap_or("") { "1" => "EXTEND".to_string(), "2" => "DECODE".to_string(), m => m.to_string() };
        let g = Gold::open(&dir);
        let rp = gold_ints(&g, "batch.req_pool_indices")?;
        if mode.ends_with("DECODE") {
            let rows = rp.iter().map(|p| pool.get(p).copied().ok_or(format!("fwd {k}: decode row of request-pool row {p} before its prefill"))).collect::<Result<Vec<_>, _>>()?;
            steps.push(Step::Decode { rows });
        } else if mode.ends_with("EXTEND") {
            let ids = gold_ints(&g, "batch.input_ids")?;
            let (lens, starts) = (gold_ints(&g, "batch.extend_seq_lens")?, gold_ints(&g, "batch.extend_prefix_lens")?);
            let mut chunks = vec![];
            let mut off = 0usize;
            for (j, p) in rp.iter().enumerate() {
                let (len, start) = (lens[j] as usize, starts[j] as usize);
                let part = &ids[off..off + len];
                off += len;
                let req = if start == 0 {
                    let i = (0..prompts.len()).find(|&i| prompts[i].len() >= len && prompts[i][..len] == *part)
                        .ok_or(format!("fwd {k}: prefill matches no prompt"))?;
                    // Prefilled before: SGLang retracted it (what it keeps is known once its new
                    // prefill ends, below).
                    if admitted[i] { steps.push(Step::Retract { req: i, keep: usize::MAX }); }
                    admitted[i] = true;
                    pool.insert(*p, i);
                    i
                } else { *pool.get(p).ok_or(format!("fwd {k}: continued prefill of an unknown request"))? };
                chunks.push(Chunk { req, start, len });
            }
            steps.push(Step::Prefill { chunks });
        } else { return Err(format!("fwd {k}: forward mode {mode}")); }
    }
    // A retracted request keeps the tokens its new prefill covers past the prompt: the end of its
    // last chunk before it decodes again.
    for r in 0..steps.len() {
        let Step::Retract { req, .. } = steps[r] else { continue };
        let mut end = 0;
        for st in &steps[r + 1..] {
            match st {
                Step::Prefill { chunks } => for c in chunks { if c.req == req { end = c.start + c.len; } },
                Step::Decode { rows } if rows.contains(&req) => break,
                _ => {}
            }
        }
        steps[r] = Step::Retract { req, keep: end.checked_sub(prompts[req].len()).ok_or(format!("request {req}: retracted, prefilled again short of its prompt"))? };
    }
    Ok(steps)
}

/// Tokens each request holds after `schedule` (requests `prompts`).
#[cfg(feature = "cuda")]
fn held_after(schedule: &[dsv4_flash_native::model::Step], prompts: &[Vec<i64>]) -> Vec<usize> {
    use dsv4_flash_native::model::Step;
    let mut held = vec![0usize; prompts.len()];
    for st in schedule {
        match st {
            Step::Prefill { chunks } => for c in chunks { if c.start + c.len == prompts[c.req].len() + held[c.req] { held[c.req] += 1 } },
            Step::Decode { rows } => for &i in rows { held[i] += 1 },
            Step::Retract { req, keep } => held[*req] = *keep,
        }
    }
    held
}

/// `schedule` up to its last prefill (SGLang's admission of the requests), then decode steps to
/// `max_new` tokens per request as SGLang's running batch goes on: each decode step's rows in
/// order, a request whose prefill finishes appended (a retracted one leaves until then), a request
/// leaving once it holds `max_new` tokens.
#[cfg(feature = "cuda")]
fn continue_schedule(schedule: &[dsv4_flash_native::model::Step], prompts: &[Vec<i64>], max_new: usize) -> Vec<dsv4_flash_native::model::Step> {
    use dsv4_flash_native::model::Step;
    let cut = schedule.iter().rposition(|s| matches!(s, Step::Prefill { .. })).map(|i| i + 1).unwrap_or(0);
    let mut out: Vec<Step> = schedule[..cut].to_vec();
    let mut held = vec![0usize; prompts.len()];
    let mut order: Vec<usize> = vec![];
    for st in &out {
        match st {
            Step::Decode { rows } => { order = rows.clone(); for &i in rows { held[i] += 1; } }
            Step::Prefill { chunks } => for c in chunks {
                if c.start + c.len == prompts[c.req].len() + held[c.req] { held[c.req] += 1; order.retain(|&i| i != c.req); order.push(c.req); }
            },
            Step::Retract { req, keep } => { held[*req] = *keep; order.retain(|&i| i != *req); }
        }
    }
    loop {
        order.retain(|&i| held[i] < max_new);
        if order.is_empty() { break; }
        for &i in &order { held[i] += 1; }
        out.push(Step::Decode { rows: order.clone() });
    }
    out
}

/// `dsv4 bench-concurrent <prepared> <bundle> <concurrent gold run> <decode> <reps>`: serve the
/// run's prompts concurrently on SGLang's schedule (from the run's golds: its prefills, waves
/// interleaved with decode steps past the SWA pool, and its batch row order; without golds, all
/// prefilled then one decode batch), timing the wall time as SGLang's bench_concurrent does, and
/// check each request's tokens against the reference server's (outputs-a.json: a prefix).
#[cfg(feature = "cuda")]
fn bench_concurrent(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::{Model, Step};
    let [prepared, bundle, run_dir, decode, reps] = [&args[0], &args[1], &args[2], &args[3], &args[4]];
    let decode: usize = decode.parse().map_err(|e| format!("decode: {e}"))?;
    let reps: usize = reps.parse().map_err(|e| format!("reps: {e}"))?;
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    let run = Path::new(run_dir);
    let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let reference: Value = serde_json::from_slice(&std::fs::read(run.join("outputs-a.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let gold = run.join("gold");
    let schedule = match if gold.exists() { Some(schedule_from_golds(&gold, &prompts)) } else { None } {
        Some(Ok(golds)) => {
            let full = continue_schedule(&golds, &prompts, decode + 1);
            // The continuation must reproduce the captured decode steps it overlaps.
            let overlap = golds.len().min(full.len());
            let agrees = golds[..overlap] == full[..overlap];
            let summary: Vec<String> = golds.iter().map(|s| match s {
                Step::Prefill { chunks } if chunks.len() == 1 && chunks[0].len == 16384 && chunks[0].start == 0 => "P".to_string(),
                Step::Prefill { chunks } => format!("P[{}]", chunks.iter().map(|c| format!("{}@{}+{}", c.req, c.start, c.len)).collect::<Vec<_>>().join(",")),
                Step::Decode { rows } => format!("D{}", rows.len()),
                Step::Retract { req, keep } => format!("R{req}k{keep}") }).collect();
            let mut runs: Vec<(String, usize)> = vec![];
            for t in summary { if runs.last().map(|r| r.0 == t).unwrap_or(false) { runs.last_mut().unwrap().1 += 1 } else { runs.push((t, 1)) } }
            // Golds that take every request to decode + 1 tokens are the schedule as is.
            let complete = held_after(&golds, &prompts).iter().all(|&h| h == decode + 1);
            println!("{}", json!({"schedule": if complete { "golds" } else { "golds, continued" }, "gold_forwards": golds.len(), "steps": full.len(),
                                  "continuation_agrees": agrees, "gold_runs": runs.iter().map(|(t, c)| format!("{t}x{c}")).collect::<Vec<_>>()}));
            if complete { golds } else { full }
        }
        Some(Err(e)) => { println!("{}", json!({"schedule": "prefill then decode", "golds": e})); Step::prefill_then_decode(&lens, decode + 1) }
        None => Step::prefill_then_decode(&lens, decode + 1),
    };
    let mut all_equal = true;
    for rep in 0..reps {
        let r = m.serve(&prompts, decode + 1, &schedule)?;
        let equal: Vec<bool> = r.tokens.iter().enumerate().map(|(i, toks)| {
            reference[i]["output_ids"].as_array().map(|w| w.len() <= toks.len() && w.iter().zip(toks).all(|(a, &b)| a == &json!(b))).unwrap_or(false)
        }).collect();
        let n_equal = equal.iter().filter(|&&e| e).count();
        all_equal &= n_equal == prompts.len();
        let c = prompts.len();
        let first_diff = equal.iter().position(|&e| !e).map(|i| {
            let want: Vec<i64> = reference[i]["output_ids"].as_array().map(|w| w.iter().filter_map(|v| v.as_i64()).collect()).unwrap_or_default();
            let at = want.iter().zip(&r.tokens[i]).position(|(a, &b)| *a != b as i64);
            json!({"request": i, "token": at})
        });
        println!("{}", json!({"rep": rep, "concurrency": c, "prompt_tokens": prompts[0].len(), "decode": decode, "wall_s": (r.wall_s * 1e3).round() / 1e3,
                              "prefill_s": (r.prefill_s * 1e3).round() / 1e3, "decode_s": (r.decode_s * 1e3).round() / 1e3,
                              "decode_steps": r.decode_steps,
                              "decode_step_ms": (r.decode_s * 1e3 / r.decode_steps.max(1) as f64 * 1e3).round() / 1e3,
                              "output_tok_per_s": (c as f64 * (decode + 1) as f64 / r.wall_s * 10.0).round() / 10.0,
                              "requests_equal": n_equal, "first_mismatch": first_diff}));
    }
    if all_equal { Ok(()) } else { Err("native outputs differ from the reference".into()) }
}

/// `dsv4 mtp-target <prepared> <bundle> <MTP gold run>`: the target model's half of SGLang's MTP
/// run (prefill, then each verify forward with the golds' tokens), forward by forward.
#[cfg(feature = "cuda")]
fn mtp_target(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::Model;
    let [prepared, bundle, run_dir] = [&args[0], &args[1], &args[2]];
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    dsv4_flash_native::compress::set_speculative(true);
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    m.mtp_target_check(Path::new(run_dir))
}

/// `dsv4 mtp-draft <prepared> <bundle> <MTP gold run> <checkpoint>`: SGLang's MTP run forward by
/// forward, the draft's forwards included (draft weights rebuilt from the checkpoint).
#[cfg(feature = "cuda")]
fn mtp_draft(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::Model;
    let [prepared, bundle, run_dir, ckpt] = [&args[0], &args[1], &args[2], &args[3]];
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    dsv4_flash_native::compress::set_speculative(true);
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    m.mtp_draft_check(Path::new(run_dir), Path::new(ckpt))
}

/// `dsv4 mtp-generate <prepared> <bundle> <MTP run dir> <checkpoint> [max new tokens]`: greedy MTP
/// generation of the run's first prompt (Model::generate_mtp), compared with the reference
/// server's tokens (outputs-a.json) when the run has them.
#[cfg(feature = "cuda")]
fn mtp_generate(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::Model;
    let [prepared, bundle, run_dir, ckpt] = [&args[0], &args[1], &args[2], &args[3]];
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    let run = Path::new(run_dir);
    let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let reference: Option<Vec<usize>> = std::fs::read(run.join("outputs-a.json")).ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| serde_json::from_value(v[0]["output_ids"].clone()).ok());
    let max_new: usize = match args.get(4) { Some(v) => v.parse().map_err(|_| "max new tokens")?, None => reference.as_ref().map(|r| r.len()).ok_or("max new tokens")? };
    dsv4_flash_native::compress::set_speculative(true);
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    m.load_draft_weights(Path::new(ckpt))?;
    let r = m.generate_mtp(&prompts[0], max_new)?;
    let rounds = r.accepted.len();
    let mean_accept = r.accepted.iter().map(|&a| a as f64 + 1.0).sum::<f64>() / rounds.max(1) as f64;
    // Compared over the tokens both have (a reference may be shorter than the generation).
    let first_diff = reference.as_ref().and_then(|g| g.iter().zip(&r.tokens).position(|(a, b)| a != b));
    println!("{}", json!({"tokens": r.tokens.len(), "rounds": rounds, "accept_length": mean_accept, "accepted": r.accepted,
                          "prefill_s": r.prefill_s, "decode_s": r.decode_s, "decode_tok_s": (r.tokens.len() - 1) as f64 / r.decode_s,
                          "reference_tokens": reference.as_ref().map(|g| g.len()), "first_mismatch": first_diff,
                          "compared": reference.as_ref().map(|g| g.len().min(r.tokens.len())), "equal": reference.as_ref().map(|_| first_diff.is_none())}));
    if let (Some(g), Some(i)) = (&reference, first_diff) {
        let lo = i.saturating_sub(3);
        return Err(format!("native differs from the reference at token {i}: native {:?} vs reference {:?}", &r.tokens[lo..(i + 4).min(r.tokens.len())], &g[lo..(i + 4).min(g.len())]));
    }
    Ok(())
}

/// SGLang's concurrent MTP schedule from a gold run's forwards: each target EXTEND is a prefill of
/// its chunks (a chunk from position 0 matched to its prompt by tokens; an admitted request
/// starting over was retracted), each TARGET_VERIFY a round over its request-pool rows in batch
/// order; the draft's forwards follow from those.
#[cfg(feature = "cuda")]
fn mtp_schedule_from_golds(gold: &Path, prompts: &[Vec<i64>]) -> Result<Vec<dsv4_flash_native::model::MtpStep>, String> {
    use dsv4_flash_native::{gold::Gold, model::{Chunk, MtpStep}};
    let mut steps = vec![];
    let mut pool: std::collections::HashMap<i64, usize> = Default::default();
    let mut admitted = vec![false; prompts.len()];
    for k in 1.. {
        let dir = gold.join(format!("fwd-{k:03}"));
        if !dir.join("meta.json").exists() { break; }
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if meta["runner"] != "target" { continue; }
        let g = Gold::open(&dir);
        let rp = gold_ints(&g, "batch.req_pool_indices")?;
        match meta["forward_mode"].as_str().unwrap_or("") {
            "1" => {
                let ids = gold_ints(&g, "batch.input_ids")?;
                let (lens, starts) = (gold_ints(&g, "batch.extend_seq_lens")?, gold_ints(&g, "batch.extend_prefix_lens")?);
                let mut chunks = vec![];
                let mut off = 0usize;
                for (j, p) in rp.iter().enumerate() {
                    let (len, start) = (lens[j] as usize, starts[j] as usize);
                    let part = &ids[off..off + len];
                    off += len;
                    let req = if start == 0 {
                        let i = (0..prompts.len()).find(|&i| { let m = len.min(prompts[i].len()); prompts[i][..m] == part[..m] })
                            .ok_or(format!("fwd {k}: prefill matches no prompt"))?;
                        if admitted[i] { steps.push(MtpStep::Retract { req: i, keep: usize::MAX }); }
                        admitted[i] = true;
                        pool.insert(*p, i);
                        i
                    } else { *pool.get(p).ok_or(format!("fwd {k}: continued prefill of an unknown request"))? };
                    chunks.push(Chunk { req, start, len });
                }
                steps.push(MtpStep::Prefill(chunks));
            }
            "5" => steps.push(MtpStep::Round(rp.iter().map(|p| pool.get(p).copied().ok_or(format!("fwd {k}: verify of an unknown request"))).collect::<Result<Vec<_>, _>>()?)),
            m => return Err(format!("fwd {k}: target forward mode {m}")),
        }
    }
    // A retracted request keeps the tokens its new prefill covers past the prompt: the end of its
    // last chunk before its next round.
    for r in 0..steps.len() {
        let MtpStep::Retract { req, .. } = steps[r] else { continue };
        let mut end = 0;
        for st in &steps[r + 1..] {
            match st {
                MtpStep::Prefill(chunks) => for c in chunks { if c.req == req { end = c.start + c.len; } },
                MtpStep::Round(rows) if rows.contains(&req) => break,
                _ => {}
            }
        }
        steps[r] = MtpStep::Retract { req, keep: end.checked_sub(prompts[req].len()).ok_or(format!("request {req}: retracted, prefilled again short of its prompt"))? };
    }
    Ok(steps)
}

/// `dsv4 mtp-serve <prepared> <bundle> <MTP gold run> <checkpoint> [reps]`: concurrent MTP serving
/// of the run's prompts on SGLang's schedule (from its golds), each request's tokens checked
/// against the reference servers' (eager B, whose golds these are, and graph A).
#[cfg(feature = "cuda")]
fn mtp_serve(args: &[String]) -> Result<(), String> {
    use dsv4_flash_native::model::Model;
    let [prepared, bundle, run_dir, ckpt] = [&args[0], &args[1], &args[2], &args[3]];
    let reps: usize = args.get(4).map(|v| v.parse().unwrap_or(1)).unwrap_or(1);
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "DSV4_MANIFEST_SHA256 pins the prepared package")?;
    let run = Path::new(run_dir);
    let prompts: Vec<Vec<i64>> = serde_json::from_slice(&std::fs::read(run.join("prompts.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let outputs = |name: &str| -> Option<Vec<Vec<usize>>> {
        let v: Value = serde_json::from_slice(&std::fs::read(run.join(name)).ok()?).ok()?;
        v.as_array()?.iter().map(|o| serde_json::from_value(o["output_ids"].clone()).ok()).collect()
    };
    let (a, b) = (outputs("outputs-a.json"), outputs("outputs-b.json"));
    let reference = b.clone().or(a.clone()).ok_or("the run has no reference outputs")?;
    let max_new = reference[0].len();
    let schedule = mtp_schedule_from_golds(&run.join("gold"), &prompts)?;
    let rounds = schedule.iter().filter(|s| matches!(s, dsv4_flash_native::model::MtpStep::Round(_))).count();
    let retracts = schedule.iter().filter(|s| matches!(s, dsv4_flash_native::model::MtpStep::Retract { .. })).count();
    let packed = schedule.iter().filter(|s| matches!(s, dsv4_flash_native::model::MtpStep::Prefill(c) if c.len() > 1)).count();
    println!("{}", json!({"schedule_rounds": rounds, "retracts": retracts, "packed_prefills": packed, "steps": schedule.len()}));
    dsv4_flash_native::compress::set_speculative(true);
    let t0 = std::time::Instant::now();
    let mut timeline = Vec::new();
    let mut m = Model::start_prepared(Path::new(prepared), &sha, Path::new(bundle))?.finish(t0, &mut timeline)?;
    m.load_draft_weights(Path::new(ckpt))?;
    let mut all_equal = true;
    for rep in 0..reps {
        let r = m.serve_mtp(&prompts, max_new, &schedule)?;
        let eq = |g: &Option<Vec<Vec<usize>>>| g.as_ref().map(|g| g.iter().zip(&r.tokens).filter(|(x, y)| x == y).count());
        let first_diff: Vec<Option<usize>> = reference.iter().zip(&r.tokens).map(|(g, t)| g.iter().zip(t).position(|(x, y)| x != y)).collect();
        let (eq_b, eq_a) = (eq(&b), eq(&a));
        all_equal &= eq_b.or(eq_a) == Some(prompts.len());
        let out_tokens: usize = r.tokens.iter().map(|t| t.len()).sum();
        println!("{}", json!({"rep": rep, "requests": prompts.len(), "rounds": r.decode_steps, "schedule_rounds": rounds, "wall_s": r.wall_s, "prefill_s": r.prefill_s,
                              "decode_s": r.decode_s, "decode_tokens": r.decode_tokens, "decode_tok_s": r.decode_tokens as f64 / r.decode_s, "output_tokens": out_tokens,
                              "equal_eager_b": eq_b, "equal_graph_a": eq_a, "first_mismatch": first_diff}));
    }
    if all_equal { Ok(()) } else { Err("native MTP outputs differ from the reference".into()) }
}

/// Diagnostics: time cuBLASLt's first calls for the model's GEMM shapes in a fresh process.
#[cfg(feature = "cuda")]
fn lt_warm() -> Result<(), String> {
    use dsv4_flash_native::{cublaslt::Lt, cuda::Device};
    let t0 = std::time::Instant::now();
    let dev = Device::open(0)?;
    let ctx_s = t0.elapsed().as_secs_f64();
    let lt = Lt::open(&std::env::var("DSV4_CUBLASLT").unwrap_or_else(|_| "/opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13".into()), dev.sm_count as i32)?;
    let open_s = t0.elapsed().as_secs_f64();
    for (n, f) in [(2048usize, true), (1024, true), (129280, false)] {
        let ms: Vec<usize> = (1..=32).collect();
        let calls = lt.warm(n, 4096, f, &ms)?;
        let calls: Vec<_> = calls.into_iter().filter(|c| c.1 > 5.0 || c.0 == 1).collect();
        println!("{}", json!({"n": n, "calls_m_ms_algo": calls.iter().map(|c| json!([c.0, (c.1 * 10.0).round() / 10.0, c.2])).collect::<Vec<_>>()}));
    }
    println!("{}", json!({"context_s": ctx_s, "lt_open_s": open_s, "total_s": t0.elapsed().as_secs_f64()}));
    Ok(())
}

fn prepare(args: &[String]) -> Result<(), String> {
    let [ckpt, params, tables, out] = [&args[0], &args[1], &args[2], &args[3]];
    let workers: usize = args.get(4).map(|v| v.parse().unwrap_or(8)).unwrap_or(8);
    let summary = dsv4_flash_native::prepare::prepare(Path::new(ckpt), Path::new(params), Path::new(tables), Path::new(out), workers)?;
    println!("{summary}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn rung(args: &[String]) -> Result<(), String> {
    use std::path::Path;
    let [which, ckpt, bundle, gold] = [&args[0], &args[1], &args[2], &args[3]];
    let results = match which.as_str() {
        "prefill-l0-pre" => dsv4_flash_native::parity::prefill_layer0_pre_attention(Path::new(ckpt), Path::new(bundle), Path::new(gold))?,
        "prefill-layers" => {
            let tables = args.get(4).ok_or("prefill-layers needs <tables dir> [layers]")?;
            let n: usize = args.get(5).map(|v| v.parse().unwrap()).unwrap_or(2);
            dsv4_flash_native::parity::prefill_layers(Path::new(ckpt), Path::new(bundle), Path::new(gold), Path::new(tables), n)?
        }
        "determinism" => {
            let tables = args.get(4).ok_or("determinism needs <tables dir> <prompt> <repeats>")?;
            let prompt: usize = args.get(5).map(|v| v.parse().unwrap()).unwrap_or(0);
            let repeats: usize = args.get(6).map(|v| v.parse().unwrap()).unwrap_or(4);
            dsv4_flash_native::parity::determinism(Path::new(ckpt), Path::new(bundle), Path::new(gold), Path::new(tables), prompt, repeats)?
        }
        "generate" => {
            let tables = args.get(4).ok_or("generate needs <tables dir> <prompt> <steps>")?;
            let prompts: Vec<usize> = args.get(5).map(|v| v.split(',').map(|p| p.parse().unwrap()).collect()).unwrap_or(vec![0]);
            let steps: usize = args.get(6).map(|v| v.parse().unwrap()).unwrap_or(5);
            dsv4_flash_native::parity::generate(Path::new(ckpt), Path::new(bundle), Path::new(gold), Path::new(tables), &prompts, steps)?
        }
        "prefill-l0-ffn" => dsv4_flash_native::parity::prefill_layer0_ffn(Path::new(ckpt), Path::new(bundle), Path::new(gold))?,
        "long-prefill" => {
            // <prepared dir> <bundle> <long capture run dir> <prompt> [layers]
            let prompt: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(0);
            let layers: usize = args.get(5).map(|v| v.parse().unwrap()).unwrap_or(43);
            dsv4_flash_native::parity::long_prefill(Path::new(ckpt), Path::new(bundle), Path::new(gold), prompt, layers)?
        }
        "pro-prefill" => {
            // <rank package dir> <bundle> <gold fwd dir> [layers]
            let layers: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(dsv4_flash_native::model::LAYERS);
            dsv4_flash_native::parity::pro_prefill(Path::new(ckpt), Path::new(bundle), Path::new(gold), layers)?
        }
        "prefill-l0-attn" => {
            let tables = args.get(4).ok_or("prefill-l0-attn needs <tables dir>")?;
            dsv4_flash_native::parity::prefill_layer0_attention(Path::new(ckpt), Path::new(bundle), Path::new(gold), Path::new(tables))?
        }
        other => return Err(format!("unknown rung {other}")),
    };
    let equal = results.iter().filter(|c| c.equal).count();
    println!("{}", json!({"rung": which, "equal": equal, "total": results.len()}));
    if equal == results.len() { Ok(()) } else { Err("rung differs".into()) }
}

/// `load-bench <prepared dir>`: the cold start's weight load alone, on device DSV4_GPU (default 0):
/// readers start before the CUDA context, every byte lands on the device. With DSV4_START_EPOCH
/// (seconds), the readers wait for that wall-clock time, so several processes (one per GPU) start
/// their loads together. Prints one JSON line.
#[cfg(feature = "cuda")]
fn load_bench(args: &[String], t0: std::time::Instant) -> Result<(), String> {
    let dir = Path::new(&args[0]);
    let now = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    for (key, value) in [("NATIVE_LOAD_FILE_IO", "direct"), ("NATIVE_LOAD_WORKERS", "64"), ("NATIVE_LOAD_CHUNK_MIB", "16")] {
        if std::env::var_os(key).is_none() { std::env::set_var(key, value); }
    }
    let ordinal: i32 = std::env::var("DSV4_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let sha = std::env::var("DSV4_MANIFEST_SHA256").map_err(|_| "a prepared package needs DSV4_MANIFEST_SHA256 (its manifest's pin)")?;
    if let Some(at) = std::env::var("DSV4_START_EPOCH").ok().and_then(|v| v.parse::<f64>().ok()) {
        let wait = at - now();
        if wait > 0.0 { std::thread::sleep(std::time::Duration::from_secs_f64(wait)); }
    }
    let begin = t0.elapsed().as_secs_f64();
    let epoch_begin = now();
    // DSV4_CTX_LOCK=<file>: rank processes create their CUDA contexts one after another (an
    // exclusive file lock). Created together, eight contexts all become ready late and every
    // rank's readers wait; in turn, the first ranks start uploading while the others are created.
    let device = std::thread::spawn(move || -> Result<(dsv4_flash_native::cuda::Device, f64, f64), String> {
        let lock = match std::env::var("DSV4_CTX_LOCK") {
            Ok(path) if !path.is_empty() => {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{path}: {e}"))?;
                f.lock().map_err(|e| format!("{path}: {e}"))?;
                Some(f)
            }
            _ => None,
        };
        let waited = t0.elapsed().as_secs_f64() - begin;
        let dev = dsv4_flash_native::cuda::Device::open(ordinal)?;
        drop(lock);
        Ok((dev, waited, t0.elapsed().as_secs_f64() - begin - waited))
    });
    let load = dsv4_flash_native::prepared::start(dir, &sha)?;
    let readers = t0.elapsed().as_secs_f64() - begin;
    let (dev, ctx_wait, ctx_create) = device.join().map_err(|_| "CUDA context thread panicked")??;
    dsv4_flash_native::cuda::set_current(dev.ctx)?;
    let context = t0.elapsed().as_secs_f64() - begin;
    let weights = load.finish(dev.stream, dev.ctx)?;
    dev.synchronize()?;
    let done = t0.elapsed().as_secs_f64() - begin;
    let gb = weights.stats.read_bytes as f64 / 1e9;
    println!("{}", json!({"load_bench": {"gpu": ordinal, "epoch_begin": epoch_begin, "epoch_done": now(), "readers_started_s": readers,
        "cuda_context_s": context, "ctx_lock_wait_s": ctx_wait, "ctx_create_s": ctx_create, "weights_on_device_s": done, "gb": gb, "gb_per_s": gb / done,
        "loader_gb_per_s": gb / weights.stats.seconds, "tensors": weights.stats.tensors}}));
    // DSV4_VERIFY=1: read every tensor back from the device and compare its SHA-256 with the
    // package's params.json (the reference rank's device bytes).
    if std::env::var("DSV4_VERIFY").as_deref() == Ok("1") {
        use sha2::{Digest, Sha256};
        let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("params.json")).map_err(|e| format!("params.json: {e}"))?).map_err(|e| e.to_string())?;
        let (mut equal, mut total, mut bytes, mut differ) = (0usize, 0usize, 0usize, Vec::new());
        for row in meta["params"].as_array().ok_or("params.json has no params")? {
            let (Some(name), Some(want), Some(n)) = (row["name"].as_str(), row["device_sha256"].as_str(), row["device_bytes"].as_u64()) else { continue };
            let ptr = weights.get(name)?;
            let mut h = Sha256::new();
            let mut at = 0usize;
            while at < n as usize {
                let len = (n as usize - at).min(256 << 20);
                h.update(dsv4_flash_native::cuda::download(ptr + at as u64, len)?);
                at += len;
            }
            total += 1;
            bytes += n as usize;
            if format!("{:x}", h.finalize()) == want { equal += 1; } else if differ.len() < 10 { differ.push(name.to_string()); }
        }
        println!("{}", json!({"verify": {"gpu": ordinal, "rank": meta["rank"], "equal": equal, "total": total, "gb": bytes as f64 / 1e9, "differ": differ}}));
        if equal != total { return Err("device bytes differ from the reference's".into()); }
    }
    Ok(())
}

fn main() {
    let t0 = std::time::Instant::now();
    // Wall clock at process start, for end-to-end cold-start timing across machines.
    let epoch0 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    std::env::set_var("DSV4_EPOCH0", format!("{epoch0:.6}"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("verify-weights") if args.len() >= 3 => verify_weights(&args[1..]),
        Some("tokenize") if args.len() >= 3 => tokenize(&args[1..]),
        Some("prepare") if args.len() >= 5 => prepare(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("run") if args.len() >= 7 => run(&args[1..], t0),
        #[cfg(feature = "cuda")]
        Some("lt-warm") => lt_warm(),
        #[cfg(feature = "cuda")]
        Some("load-bench") if args.len() >= 2 => load_bench(&args[1..], t0),
        // pro-rank <rank package dir> <bundle> <gold fwd dir (prompt ids, checks)> <shm dir>
        #[cfg(feature = "cuda")]
        Some("pro-rank") if args.len() >= 5 => {
            if let Some(at) = std::env::var("DSV4_START_EPOCH").ok().and_then(|v| v.parse::<f64>().ok()) {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                if at > now { std::thread::sleep(std::time::Duration::from_secs_f64(at - now)); }
            }
            let t0 = std::time::Instant::now();
            let gold = dsv4_flash_native::gold::Gold::open(&args[3]);
            gold.bytes("batch.input_ids").map(|b| b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<i64>>())
                .and_then(|ids| dsv4_flash_native::parity::pro_rank(Path::new(&args[1]), Path::new(&args[2]), &ids, Path::new(&args[4]), Some(Path::new(&args[3])), t0))
        }
        #[cfg(feature = "cuda")]
        Some("rung") if args.len() >= 5 => rung(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("bench-long") if args.len() >= 7 => bench_long(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("bench-concurrent") if args.len() >= 6 => bench_concurrent(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("mtp-target") if args.len() >= 4 => mtp_target(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("mtp-draft") if args.len() >= 5 => mtp_draft(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("mtp-generate") if args.len() >= 5 => mtp_generate(&args[1..]),
        #[cfg(feature = "cuda")]
        Some("mtp-serve") if args.len() >= 5 => mtp_serve(&args[1..]),
        _ => Err("usage: dsv4 verify-weights <checkpoint> <params.json> [prefix] | tokenize <checkpoint> <outputs.json> | \
                  prepare <checkpoint> <params.json> <tables> <out> [workers] | run <checkpoint|prepared> <bundle> <tables> <prompts.json> <max new tokens> <out.json> | rung ...".into()),
    };
    let _ = t0;
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
