//! CPU admission for byte-preserving, two-, four- or eight-rank layer weight packages.
//! Sharding never changes a tensor's dtype, shape, or GEMM reduction dimension.
use crate::{integrity, Result, TensorSpec};
use serde::Deserialize;
use std::{collections::{BTreeMap, BTreeSet}, path::{Component, Path}};

pub const SUPPORTED_WORLDS: [usize; 3] = [2, 4, 8];
pub const ALIGNMENT: usize = 256;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum GatherMode {P2p,AllGather,Cache,CacheOverlap}
impl GatherMode {
    pub fn parse(value:&str)->Result<Self>{match value{
        "p2p"=>Ok(Self::P2p),"allgather"=>Ok(Self::AllGather),"request_cache"=>Ok(Self::Cache),
        "request_cache_overlap"=>Ok(Self::CacheOverlap),_=>Err("invalid H3_WEIGHT_GATHER_MODE".into())}}
    pub fn name(self)->&'static str{match self{Self::P2p=>"p2p",Self::AllGather=>"allgather",Self::Cache=>"request_cache",Self::CacheOverlap=>"request_cache_overlap"}}
    pub fn overlap(self)->bool{self==Self::CacheOverlap}
    pub fn cache(self)->bool{matches!(self,Self::Cache|Self::CacheOverlap)}
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub file: String,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub names: Vec<String>,
    pub offset: usize,
    pub tensor: TensorSpec,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
    pub entries: Vec<Entry>,
    pub ranks: Vec<Shard>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub world_size: usize,
    pub workflow: String,
    pub base_weights_manifest_sha256: String,
    pub groups: Vec<Group>,
}

pub fn aligned(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 { return Err("zero weight alignment".into()); }
    value.checked_add(alignment-1).and_then(|n| (n/alignment).checked_mul(alignment))
        .ok_or_else(|| "weight package size overflow".into())
}

/// Bind each base tensor to its execution layer.
pub fn group_for_key(key: &str) -> Result<String> {
    let name=key.strip_prefix("weights/").ok_or("unknown weight namespace")?;
    for (prefix,count) in [("transformer_blocks.",50),("token_refiner.refiner_blocks.",2)] {
        if let Some(suffix)=name.strip_prefix(prefix) {
            let (index,tail)=suffix.split_once('.').ok_or("weight layer suffix missing")?;
            let layer:usize=index.parse()?;
            if layer>=count || index!=layer.to_string() || tail.is_empty() {
                return Err("weight layer index outside H3 architecture".into());
            }
            return Ok(format!("{prefix}{layer}"));
        }
    }
    Ok("global".into())
}

pub fn expected_tensors(model: &serde_json::Value)
    -> Result<BTreeMap<String,TensorSpec>> {
    let mut values=BTreeMap::new();
    for (name,value) in model["weights"].as_object().ok_or("missing model weights")? {
        values.insert(format!("weights/{name}"),TensorSpec::parse(value)?);
    }
    Ok(values)
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {Ok(serde_json::from_slice(bytes)?)}

    pub fn validate(&self, expected: &BTreeMap<String,TensorSpec>, workflow: &str,
                    base_sha: &str) -> Result<()> {
        if self.schema!="h3-layer-weight-shards-v1" || !SUPPORTED_WORLDS.contains(&self.world_size) ||
            crate::recipe::Recipe::partition(workflow).is_none() || self.workflow!=workflow ||
            self.base_weights_manifest_sha256!=base_sha || !integrity::valid_sha(base_sha) ||
            expected.is_empty() {
            return Err("weight shard recipe/base identity mismatch".into());
        }
        let mut names=BTreeSet::new();let mut groups=BTreeSet::new();let mut paths=BTreeSet::new();
        for group in &self.groups {
            if !groups.insert(group.name.clone()) || group.entries.is_empty() ||
                group.ranks.len()!=self.world_size || !integrity::valid_sha(&group.sha256) {
                return Err("invalid or duplicate weight group".into());
            }
            let mut end=0;
            for entry in &group.entries {
                entry.tensor.validate()?;
                if entry.names.is_empty() || entry.offset!=aligned(end,ALIGNMENT)? ||
                    !entry.tensor.sha256.as_deref().is_some_and(integrity::valid_sha) {
                    return Err("weight group has a gap, overlap, or unpinned tensor".into());
                }
                end=entry.offset.checked_add(entry.tensor.bytes).ok_or("weight offset overflow")?;
                for name in &entry.names {
                    let want=expected.get(name).ok_or("unexpected sharded tensor")?;
                    if !names.insert(name.clone()) || group_for_key(name)?!=group.name ||
                        want.dtype!=entry.tensor.dtype || want.shape!=entry.tensor.shape ||
                        want.stride!=entry.tensor.stride || want.bytes!=entry.tensor.bytes ||
                        want.sha256!=entry.tensor.sha256 {
                        return Err("weight group changes tensor identity or layer ownership".into());
                    }
                }
            }
            if group.bytes!=aligned(end,ALIGNMENT*self.world_size)? {
                return Err("weight group must use minimal equal-rank padding".into());
            }
            for shard in &group.ranks {
                let path=Path::new(&shard.file);
                if !paths.insert(shard.file.clone()) || path.components().next().is_none() ||
                    path.components().any(|p|!matches!(p,Component::Normal(_))) ||
                    shard.bytes!=group.bytes/self.world_size || !integrity::valid_sha(&shard.sha256) {
                    return Err("invalid equal-size rank shard".into());
                }
            }
        }
        if names.len()!=expected.len() {return Err("weight shard package omits model tensors".into());}
        Ok(())
    }

    pub fn rank_bytes(&self) -> Result<usize> {
        if !SUPPORTED_WORLDS.contains(&self.world_size){return Err("unsupported shard world size".into());}
        self.groups.iter().try_fold(0usize,|total,g|total.checked_add(g.bytes/self.world_size)
            .ok_or_else(||"resident shard size overflow".into()))
    }
    pub fn gather_bytes(&self) -> usize {self.groups.iter().map(|g|g.bytes).max().unwrap_or(0)}
    pub fn cache_layout(&self)->Result<(usize,BTreeMap<String,(usize,usize)>)>{
        let mut total=0usize;let mut layout=BTreeMap::new();
        for group in &self.groups {
            if group.bytes==0 || group.bytes%ALIGNMENT!=0 || layout.insert(group.name.clone(),(total,group.bytes)).is_some(){
                return Err("invalid request cache extent or duplicate group".into());
            }
            total=total.checked_add(group.bytes).ok_or("request weight cache size overflow")?;
        }
        if total==0{return Err("empty request weight cache".into());}Ok((total,layout))
    }
}

/// One range of a rank's shard, read from the model's own weight file: bytes
/// `offset..offset+bytes` of `file` fill `dest..dest+bytes` of the rank's shard of `group`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slice {
    pub group: usize,
    pub dest: usize,
    pub file: String,
    pub offset: u64,
    pub bytes: usize,
    pub file_bytes: u64,
    pub sha256: String,
}
/// Every rank's shard as slices of the unsharded weight files (`prepare/cp.py`). Preparation
/// rebuilt each shard from its slices, with zero padding between tensors, and checked its SHA-256.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slices {
    pub schema: String,
    pub shard_manifest_sha256: String,
    pub full_sha256_verified: bool,
    pub ranks: Vec<Vec<Slice>>,
}
impl Slices {
    /// This rank's slices, in shard order, each inside its group's shard and none overlapping.
    pub fn rank(&self, manifest: &Manifest, manifest_sha256: &str, rank: usize) -> Result<&[Slice]> {
        if self.schema!="h3-dit-shard-slices-v1" || !self.full_sha256_verified ||
            self.shard_manifest_sha256!=manifest_sha256 || self.ranks.len()!=manifest.world_size {
            return Err("DiT shard slices belong to another package".into());
        }
        let pieces=self.ranks.get(rank).ok_or("DiT shard slices miss this rank")?;
        let mut covered=(0usize,0usize);
        for piece in pieces {
            let shard=manifest.groups.get(piece.group).ok_or("DiT slice group outside its manifest")?.ranks[rank].bytes;
            let end=piece.dest.checked_add(piece.bytes).ok_or("DiT slice size overflow")?;
            if piece.bytes==0 || end>shard || (piece.group,piece.dest)<covered || !integrity::valid_sha(&piece.sha256) {
                return Err("DiT slices overlap or exceed their shard".into());
            }
            covered=(piece.group,end);
        }
        Ok(pieces)
    }
}

/// The CP package a rank process was given: H3_DIT_SHARDS (its directory) with the SHA-256 of its
/// manifest.json and, if the shards are read as slices, of its slices.json.
pub fn configured() -> Result<Option<serde_json::Value>> {
    let Some(root)=std::env::var_os("H3_DIT_SHARDS") else {return Ok(None)};
    let digest=|name:&str|->Result<Option<String>> {match std::env::var(name) {
        Ok(value) if integrity::valid_sha(&value)=>Ok(Some(value)),
        Err(std::env::VarError::NotPresent)=>Ok(None),_=>Err(format!("invalid {name}").into())}};
    let manifest=digest("H3_DIT_SHARDS_SHA256")?.ok_or("H3_DIT_SHARDS requires H3_DIT_SHARDS_SHA256")?;
    let mut value=serde_json::json!({"root":root.to_str().ok_or("invalid H3_DIT_SHARDS")?,"manifest_sha256":manifest});
    if let Some(slices)=digest("H3_DIT_SHARD_SLICES_SHA256")? {value["slices_sha256"]=slices.into();}
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn slices_stay_inside_their_rank_shard_in_order() {
        let (manifest,_)=fixture();
        let piece=|group,dest,bytes|json!({"group":group,"dest":dest,"file":"tensor.raw","offset":0,"bytes":bytes,
            "file_bytes":30,"sha256":"e".repeat(64)});
        let slices=|rank0:Vec<serde_json::Value>|serde_json::from_value::<Slices>(json!({"schema":"h3-dit-shard-slices-v1",
            "shard_manifest_sha256":"f".repeat(64),"full_sha256_verified":true,
            "ranks":(0..8).map(|r|if r==0{rank0.clone()}else{vec![]}).collect::<Vec<_>>()})).unwrap();
        let good=slices(vec![piece(0,0,16),piece(0,16,14)]);
        assert_eq!(good.rank(&manifest,&"f".repeat(64),0).unwrap().len(),2);
        assert!(good.rank(&manifest,&"0".repeat(64),0).is_err());
        assert!(good.rank(&manifest,&"f".repeat(64),8).is_err());
        for bad in [vec![piece(0,0,257)],vec![piece(0,16,4),piece(0,0,4)],vec![piece(0,0,8),piece(0,4,8)],
                    vec![piece(1,0,8)],vec![piece(0,0,0)]] {
            assert!(slices(bad).rank(&manifest,&"f".repeat(64),0).is_err());
        }
    }
    #[test]
    fn request_cache_has_disjoint_complete_extents_and_rejects_overflow() {
        let(mut m,_)=fixture();let mut next=m.groups[0].clone();next.name="transformer_blocks.1".into();
        m.groups.push(next);let(total,layout)=m.cache_layout().unwrap();
        assert_eq!(total,4096);assert_eq!(layout["transformer_blocks.0"],(0,2048));
        assert_eq!(layout["transformer_blocks.1"],(2048,2048));
        m.groups[1].name=m.groups[0].name.clone();assert!(m.cache_layout().is_err());
        m.groups[1].name="transformer_blocks.1".into();m.groups[0].bytes=usize::MAX-255;
        assert!(m.cache_layout().is_err());
        assert!(GatherMode::parse("auto").is_err());
    }
    fn fixture() -> (Manifest,BTreeMap<String,TensorSpec>) {
        let spec=TensorSpec::parse(&json!({"file":"tensor.raw","dtype":"torch.bfloat16","shape":[5,3],
            "stride":[3,1],"bytes":30,"contiguous":true,"storage_offset":0,"sha256":"a".repeat(64)})).unwrap();
        let key="weights/transformer_blocks.0.attn.to_q.weight".to_owned();
        let manifest=Manifest::parse(&serde_json::to_vec(&json!({"schema":"h3-layer-weight-shards-v1",
            "world_size":8,"workflow":"fl2va","base_weights_manifest_sha256":"b".repeat(64),
            "groups":[{"name":"transformer_blocks.0","bytes":2048,
                "sha256":"c".repeat(64),"entries":[{"names":[key],"offset":0,"tensor":{
                    "file":"tensor.raw","dtype":"torch.bfloat16","shape":[5,3],"stride":[3,1],
                    "bytes":30,"contiguous":true,"storage_offset":0,"sha256":"a".repeat(64)}}],
                "ranks":(0..8).map(|r|json!({"file":format!("rank-{r}.raw"),"bytes":256,"sha256":"d".repeat(64)})).collect::<Vec<_>>()}]})).unwrap()).unwrap();
        (manifest,BTreeMap::from([(key,spec)]))
    }
    #[test]
    fn equal_byte_shards_preserve_nondivisible_tensor() {
        let (manifest,expected)=fixture();manifest.validate(&expected,"fl2va",&"b".repeat(64)).unwrap();
        let mut original=vec![0u8;2048];for (i,b) in original[..30].iter_mut().enumerate(){*b=(i*7) as u8;}
        let shards:Vec<_>=original.chunks_exact(256).map(|s|s.to_vec()).collect();
        assert_eq!(shards.concat(),original);assert_eq!(manifest.rank_bytes().unwrap(),256);
        assert_eq!(manifest.gather_bytes(),2048);
    }
    #[test]
    fn reject_wrong_recipe_coverage_offsets_and_rank_paths() {
        let (mut m,e)=fixture();assert!(m.validate(&e,"ref2va",&"b".repeat(64)).is_err());
        m.groups[0].entries[0].offset=256;assert!(m.validate(&e,"fl2va",&"b".repeat(64)).is_err());
        m.groups[0].entries[0].offset=0;m.groups[0].ranks[0].file="../rank.raw".into();
        assert!(m.validate(&e,"fl2va",&"b".repeat(64)).is_err());
        let (mut m,e)=fixture();m.groups.clear();assert!(m.validate(&e,"fl2va",&"b".repeat(64)).is_err());
        assert!(aligned(usize::MAX,256).is_err());
    }
    #[test]
    fn four_rank_padding_and_rank_coverage_are_admitted_independently() {
        let (mut m,e)=fixture();m.world_size=4;m.groups[0].bytes=1024;m.groups[0].ranks.truncate(4);
        m.validate(&e,"fl2va",&"b".repeat(64)).unwrap();
        assert_eq!(m.rank_bytes().unwrap(),256);
        m.world_size=8;assert!(m.validate(&e,"fl2va",&"b".repeat(64)).is_err());
        m.world_size=4;m.groups[0].bytes=2048;
        assert!(m.validate(&e,"fl2va",&"b".repeat(64)).is_err());
        m.world_size=0;assert!(m.rank_bytes().is_err());
    }
    #[test]
    fn base_tensors_follow_their_execution_layer() {
        assert_eq!(group_for_key("weights/time_embedder.linear_1.weight").unwrap(),"global");
        for name in ["weights/transformer_blocks.50.attn.weight","weights/transformer_blocks.01.attn.weight",
                     "unknown/time_embedder", "unknown/transformer_blocks.0.attn.to_q"] {
            assert!(group_for_key(name).is_err(),"{name}");
        }
    }
}
