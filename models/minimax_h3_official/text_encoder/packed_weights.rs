//! Verified FP8 storage decoding before the unchanged BF16 device upload.
use crate::{prepare::hash, Result};
use serde_json::Value;
use std::{fs, path::{Path, PathBuf}};

fn path(root: &Path, spec: &Value) -> Result<PathBuf> {
    let name=spec["file"].as_str().ok_or("tensor file")?;
    if Path::new(name).is_absolute() || name.split('/').any(|p|p=="..") {
        return Err("invalid tensor path".into());
    }
    Ok(root.join(name))
}
fn verify(data:&[u8],spec:&Value)->Result<()> {
    if Some(data.len() as u64)!=spec["bytes"].as_u64() ||
        Some(hash(data).as_str())!=spec["sha256"].as_str() {
        return Err("tensor hash/size mismatch".into());
    }
    Ok(())
}
fn restore(data:&[u8])->Result<Vec<u8>> {
    let table:[u16;256]=std::array::from_fn(|code| {
        let exponent=(code>>3)&15;let fraction=code&7;
        let value=if exponent==0 {fraction as f32/512.0}
            else {(8+fraction) as f32*2f32.powi(exponent as i32-10)};
        ((value.to_bits()>>16) as u16)|(((code&128) as u16)<<8)
    });
    let size=data.len().checked_mul(2).ok_or("FP8 restored size overflow")?;
    let mut result=Vec::with_capacity(size);
    for &code in data {
        if code&127==127{return Err("nonfinite FP8 model tensor".into());}
        result.extend_from_slice(&table[code as usize].to_le_bytes());
    }
    Ok(result)
}
pub fn read(root:&Path,spec:&Value)->Result<Vec<u8>> {
    let expanded=path(root,spec)?;
    // Existing BF16 packages and verified local staging keep their original path.
    let data=if let Some(storage)=spec.get("storage").filter(|_|!expanded.exists()) {
        if storage["dtype"]!="float8_e4m3fn" || spec["dtype"]!="torch.bfloat16" ||
            storage["bytes"].as_u64().and_then(|n|n.checked_mul(2))!=spec["bytes"].as_u64() {
            return Err("unsupported packed model tensor".into());
        }
        let stored=crate::startup_trace::read(path(root,storage)?)?;verify(&stored,storage)?;
        restore(&stored)?
    }else{crate::startup_trace::read(expanded)?};
    verify(&data,spec)?;Ok(data)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn matches_accepted_restoration_for_every_finite_encoding() {
        let codes:Vec<_>=(0..=255u8).filter(|n|n&127!=127).collect();
        // SHA of the accepted Python converter, including both signed zeros.
        assert_eq!(hash(&restore(&codes).unwrap()),"216e2e0390539de6d4441627856e58b228815f45906442235e5b80b58be178c2");
        assert!(restore(&[127]).is_err());assert!(restore(&[255]).is_err());
    }
    #[test] fn rejects_changed_tensor_and_escaping_path() {
        let spec=serde_json::json!({"file":"a.raw","bytes":2,"sha256":hash(&[1,2])});
        assert!(verify(&[1,2],&spec).is_ok());assert!(verify(&[1,3],&spec).is_err());
        assert!(verify(&[1],&spec).is_err());
        for name in ["../weights","/weights"] {
            assert!(path(Path::new("/model"),&serde_json::json!({"file":name})).is_err());
        }
    }
}
