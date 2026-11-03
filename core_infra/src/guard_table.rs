//! SHA-pinned, content-addressed light guards for raw-file checkpoint adapters.
//! Packed checkpoints supply the same guards directly in checkpoint::Manifest.
use crate::integrity::Expected;
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path,sync::OnceLock};
type Result<T> = std::result::Result<T,String>;
pub struct Table {entries:BTreeMap<String,Expected>}
impl Table {
    pub fn parse(raw:&[u8],wanted:&str)->Result<Self>{
        if format!("{:x}",Sha256::digest(raw))!=wanted{return Err("guard manifest SHA256 mismatch".into());}
        let value:serde_json::Value=serde_json::from_slice(raw).map_err(|e|e.to_string())?;
        // The H3 schema is the legacy spelling of this exact trusted metadata.
        if !matches!(value["schema"].as_str(),Some("native-light-sha-v1"|"h3-light-sha-v1"))
            || value["segment_bytes"]!=1024 || value["full_sha256_verified"]!=true {
            return Err("untrusted guard manifest".into());
        }
        let digest=|v:&serde_json::Value|->Result<String>{let s=v.as_str().ok_or("guard digest missing")?;
            if s.len()!=64 || !s.bytes().all(|b|b.is_ascii_hexdigit()&&!b.is_ascii_uppercase()){
                return Err("invalid guard digest".into());}Ok(s.into())};
        let mut entries=BTreeMap::new();
        for (full,item) in value["entries"].as_object().ok_or("guard entries missing")?{
            let full=digest(&serde_json::Value::String(full.clone()))?;
            let bytes=item["bytes"].as_u64().and_then(|n|usize::try_from(n).ok()).ok_or("guard bytes missing")?;
            entries.insert(full,Expected{bytes,head_sha256:digest(&item["head_sha256"])?,tail_sha256:digest(&item["tail_sha256"])?});
        }
        Ok(Self{entries})
    }
    pub fn load(path:&Path,sha:&str)->Result<Self>{Self::parse(&std::fs::read(path).map_err(|e|e.to_string())?,sha)}
    pub fn expected(&self,sha:&str,bytes:usize,required:bool)->Result<Option<Expected>>{
        let Some(entry)=self.entries.get(sha) else{return if required{Err("payload absent from trusted guard manifest".into())}else{Ok(None)}};
        if entry.bytes!=bytes{return Err("guard payload size mismatch".into());}Ok(Some(entry.clone()))
    }
}
pub fn expected(sha:&str,bytes:usize,required:bool)->Result<Option<Expected>>{
    static TABLE:OnceLock<Result<Table>>=OnceLock::new();
    let table=TABLE.get_or_init(||{
        let path=std::env::var("NATIVE_LOAD_GUARD_MANIFEST").map_err(|_|"guard manifest missing")?;
        let sha=std::env::var("NATIVE_LOAD_GUARD_SHA256").map_err(|_|"guard manifest identity missing")?;
        Table::load(Path::new(&path),&sha)
    });
    match table{Ok(t)=>t.expected(sha,bytes,required),Err(e)=>Err(e.clone())}
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn requires_trusted_metadata_identity(){
        let raw=br#"{"schema":"native-light-sha-v1","segment_bytes":1024,"full_sha256_verified":true,"entries":{}}"#;
        assert!(Table::parse(raw,&format!("{:x}",Sha256::digest(raw))).is_ok());
        assert!(Table::parse(raw,&"0".repeat(64)).is_err());
        let bad=String::from_utf8(raw.to_vec()).unwrap().replace("true","false");
        assert!(Table::parse(bad.as_bytes(),&format!("{:x}",Sha256::digest(bad.as_bytes()))).is_err());
    }
}
