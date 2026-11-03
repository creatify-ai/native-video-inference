//! Optional request-owned handoff before decoder GPU allocations begin.
use crate::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{io::{BufRead, BufReader, Read, Write}, os::unix::net::UnixStream, path::{Component, PathBuf}, time::{Duration, Instant}};

const ENVIRONMENT_KEY: &str = "NATIVE_DECODE_PHASE_GATE";
const SCHEMA: &str = "native-decode-phase-gate-v1";
const RETIRE_SCHEMA: &str = "native-decode-phase-gate-v2";
const MAX_FRAME_BYTES: u64 = 1024;

/// A native owner may publish this phase only after its final possible encoder,
/// DiT and upsampler use, synchronized retirement on every rank, and preservation
/// of decoder resources. It authorizes foreign weight preparation, never a
/// concurrent request forward. The framework still admits the actual byte budget.
pub const DECODE_WEIGHTS_RETIRED: &str = "decode_weights_retired";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin { schema: String, path: PathBuf, token: String, stage: String, geometry: Vec<usize> }

/// The same request pin is accepted by the standard native actor and by
/// subprocess pipelines. No model-specific transport or scheduler is required.
pub fn acknowledge(pin: &Pin, stage: &str, geometry: &[usize]) -> Result<Value> {
    connect(pin.clone(), stage, geometry)
}

/// A busy native actor retires its own terminal weights on its CUDA owner.
/// The framework decides from the live successor, then submits lookahead only
/// after this callback has synchronized and preserved the decoder's resources.
pub fn acknowledge_with_retirement(pin: &Pin, stage: &str, geometry: &[usize],
    retire: impl FnOnce() -> Result<Value>) -> Result<Value> {
    exchange(pin.clone(), stage, geometry, Some(retire))
}

fn connect(pin: Pin, stage: &str, geometry: &[usize]) -> Result<Value> {
    exchange(pin, stage, geometry, None::<fn() -> Result<Value>>)
}

fn exchange(pin: Pin, stage: &str, geometry: &[usize], retire: Option<impl FnOnce() -> Result<Value>>) -> Result<Value> {
    if !matches!(pin.schema.as_str(), SCHEMA | RETIRE_SCHEMA) ||
        (pin.schema == RETIRE_SCHEMA) != retire.is_some() || pin.stage != stage || pin.geometry != geometry || geometry.is_empty() ||
        geometry.iter().any(|n| *n == 0) || pin.token.len() != 32 ||
        !pin.token.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) ||
        !pin.path.is_absolute() || pin.path.components().any(|v| matches!(v, Component::ParentDir)) {
        return Err("decode phase differs from the request-owned gate".into());
    }
    let begun = Instant::now();
    let mut stream = UnixStream::connect(&pin.path)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut request = serde_json::to_vec(&json!({"schema": pin.schema, "token": pin.token,
        "stage": stage, "geometry": geometry}))?;
    request.push(b'\n');
    if request.len() > MAX_FRAME_BYTES as usize { return Err("decode phase request exceeds its bound".into()); }
    stream.write_all(&request)?;
    let mut retirement = Value::Null;
    if let Some(retire) = retire {
        let mut reader = BufReader::new(&mut stream);
        let mut decision = Vec::new();
        reader.by_ref().take(MAX_FRAME_BYTES + 1).read_until(b'\n', &mut decision)?;
        if decision.len() > MAX_FRAME_BYTES as usize { return Err("retirement decision exceeds its bound".into()); }
        let decision: Value = serde_json::from_slice(&decision)?;
        let release = decision["retire_primary"].as_bool().ok_or("explicit retirement decision required")?;
        if decision != json!({"schema": pin.schema, "token": pin.token, "retire_primary": release}) {
            return Err("weight retirement belongs to another request".into());
        }
        retirement = if release { retire()? } else { json!({"retired": false}) };
        let mut reply = serde_json::to_vec(&json!({"schema": pin.schema, "token": pin.token,
            "retirement_complete": true, "retire_primary": release, "receipt": retirement}))?;
        reply.push(b'\n');
        if reply.len() > MAX_FRAME_BYTES as usize { return Err("retirement receipt exceeds its bound".into()); }
        stream.write_all(&reply)?;
    }
    let mut raw = Vec::new();
    stream.take(MAX_FRAME_BYTES + 1).read_to_end(&mut raw)?;
    if raw.len() > MAX_FRAME_BYTES as usize { return Err("decode phase acknowledgement exceeds its bound".into()); }
    let reply: Value = serde_json::from_slice(&raw)?;
    if reply != json!({"schema": pin.schema, "token": pin.token, "continued": true}) {
        return Err("decode phase acknowledgement belongs to another request".into());
    }
    Ok(json!({"schema": pin.schema, "stage": stage, "geometry": geometry, "retirement": retirement,
        "submission_barrier_seconds": begun.elapsed().as_secs_f64(), "waited_for_dma_completion": false}))
}

/// The caller supplies a phase after the final possible DiT use and before
/// decoder allocations. No gate means the existing execution continues directly.
pub fn before(stage: &str, geometry: &[usize]) -> Result<Option<Value>> {
    match std::env::var(ENVIRONMENT_KEY) {
        Ok(raw) => {
            if raw.len() > MAX_FRAME_BYTES as usize { return Err("decode phase pin exceeds its bound".into()); }
            Ok(Some(connect(serde_json::from_str(&raw)?, stage, geometry)?))
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::{BufRead, BufReader}, os::unix::net::UnixListener, sync::atomic::{AtomicUsize, Ordering}, thread};
    fn exchange(reply: Value) -> Result<Value> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("phase-{}-{}.sock", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let listener = UnixListener::bind(&path).unwrap();
        let peer = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new(); BufReader::new(&mut stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request, json!({"schema": SCHEMA, "token": "a".repeat(32), "stage": "decode", "geometry": [768, 768, 124]}));
            stream.write_all(&serde_json::to_vec(&reply).unwrap()).unwrap();
        });
        let result = connect(Pin { schema: SCHEMA.into(), path: path.clone(), token: "a".repeat(32),
            stage: "decode".into(), geometry: vec![768, 768, 124] }, "decode", &[768, 768, 124]);
        peer.join().unwrap(); std::fs::remove_file(path).unwrap(); result
    }
    #[test]
    fn native_decode_requires_its_own_submission_acknowledgement() {
        let receipt = exchange(json!({"schema": SCHEMA, "token": "a".repeat(32), "continued": true})).unwrap();
        assert_eq!(receipt["waited_for_dma_completion"], false);
        assert!(exchange(json!({"schema": SCHEMA, "token": "b".repeat(32), "continued": true})).is_err());
        assert!(exchange(json!({"schema": SCHEMA, "token": "a".repeat(32), "continued": false})).is_err());
    }
    #[test]
    fn phase_and_geometry_must_match_before_connecting() {
        let make = || Pin { schema: SCHEMA.into(), path: "/missing-phase.sock".into(), token: "a".repeat(32),
            stage: "decode".into(), geometry: vec![768, 768, 124] };
        assert!(connect(make(), "another", &[768, 768, 124]).unwrap_err().to_string().contains("request-owned"));
        assert!(connect(make(), "decode", &[768, 1376, 124]).unwrap_err().to_string().contains("request-owned"));
        let mut pin = make(); pin.path = "/tmp/../missing-phase.sock".into();
        assert!(connect(pin, "decode", &[768, 768, 124]).unwrap_err().to_string().contains("request-owned"));
    }
    #[test]
    fn busy_owner_retires_before_submission_only_for_a_foreign_successor() {
        for release in [false,true]{
            let path=std::env::temp_dir().join(format!("retire-{}-{release}.sock",std::process::id()));
            let listener=UnixListener::bind(&path).unwrap();
            let peer=thread::spawn(move||{
                let (mut socket,_)=listener.accept().unwrap();
                let mut line=String::new();BufReader::new(&mut socket).read_line(&mut line).unwrap();
                let request:Value=serde_json::from_str(&line).unwrap();
                assert_eq!(request["schema"],RETIRE_SCHEMA);
                writeln!(socket,"{}",json!({"schema":RETIRE_SCHEMA,"token":"a".repeat(32),"retire_primary":release})).unwrap();
                line.clear();BufReader::new(&mut socket).read_line(&mut line).unwrap();
                let retired:Value=serde_json::from_str(&line).unwrap();
                assert_eq!(retired["receipt"]["retired"],release);
                // Framework can start DMA now; it never waits for that DMA.
                writeln!(socket,"{}",json!({"schema":RETIRE_SCHEMA,"token":"a".repeat(32),"continued":true})).unwrap();
            });
            let called=std::cell::Cell::new(false);
            let receipt=acknowledge_with_retirement(&Pin{schema:RETIRE_SCHEMA.into(),path:path.clone(),
                token:"a".repeat(32),stage:"decode".into(),geometry:vec![832,1472,49]},"decode",&[832,1472,49],||{
                    called.set(true);Ok(json!({"retired":true}))
                }).unwrap();
            assert_eq!(called.get(),release);
            assert_eq!(receipt["waited_for_dma_completion"],false);
            peer.join().unwrap();std::fs::remove_file(path).unwrap();
        }
    }
}
