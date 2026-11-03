//! Official Wan component inventory: manifest prefixes, runtime dtypes and preload priority.
#[cfg(feature="cuda")]
use native_core_infra::{checkpoint::{Dtype,Manifest},model_load::Request};
pub const COMPONENTS:[&str;4]=["t5","clip","vae","dit"];
pub fn priority(name:&str)->u8{u8::from(name=="dit")}
pub fn component(name:&str)->Option<&'static str>{
    if name.starts_with("t5."){Some("t5")}else if name.starts_with("clip.visual."){Some("clip")}
    else if name.starts_with("vae."){Some("vae")}else if name.starts_with("dit."){Some("dit")}else{None}
}
#[cfg(feature="cuda")]
pub fn requests(manifest:&Manifest,name:&str)->Result<Vec<Request>,String>{
    if !COMPONENTS.contains(&name){return Err("unknown Wan preload component".into());}
    let values=manifest.tensors.iter().filter(|(n,_)|component(n)==Some(name)).map(|(n,t)|{
        let valid=match name{"t5"=>t.dtype==Dtype::BF16,"clip"=>t.dtype==Dtype::F16,
            "vae"=>t.dtype==Dtype::F32,"dit"=>matches!(t.dtype,Dtype::BF16|Dtype::F64),_=>false};
        if !valid{return Err("Wan runtime export dtype differs".into());}
        Ok(Request{name:n.clone(),dtype:t.dtype})
    }).collect::<Result<Vec<_>,String>>()?;
    if values.is_empty(){return Err("Wan component inventory missing".into());}Ok(values)
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn dependencies_and_visual_tower_are_explicit(){
        assert_eq!(component("clip.visual.transformer.0.norm1.weight"),Some("clip"));
        assert_eq!(component("clip.textual.token_embedding.weight"),None);
        assert_eq!(priority("vae"),0);assert_eq!(priority("dit"),1);
        assert_eq!(component("unknown.weight"),None);
    }
}
