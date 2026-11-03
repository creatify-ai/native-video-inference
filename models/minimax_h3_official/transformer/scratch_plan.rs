//! Base projection scratch follows context-parallel row ownership.
use crate::Result;

/// Keep headroom for CUDA plans after reserving the complete request cache
/// and the video/audio embeddings that are allocated on first execution.
pub fn cache_execution_budget(cache:usize,staging:usize)->Result<usize> {
    cache.checked_add(staging).and_then(|v|v.checked_add(2usize<<30))
        .ok_or_else(||"request cache memory budget overflow".into())
}

pub struct ScratchPlan { pub rows: usize, pub ff: usize, pub ff2: usize, pub fp_hidden:usize,
    pub small_input:usize,pub small_output:usize }
/// Packed rows owned by each CP rank: the balanced split, remainder to the lowest ranks.
/// Every rank and the scratch plan must derive the split from this one function.
pub fn row_counts(seq:usize,world:usize)->Result<Vec<usize>> {
    if ![1,2,4,8].contains(&world) || seq<world {return Err("invalid CP row partition".into());}
    Ok((0..world).map(|rank|seq/world+usize::from(rank<seq%world)).collect())
}

/// Views into the FFN projection allocation, used outside its live interval.
/// Q/K normalization views are disjoint; FP32 modality projection runs before
/// the refiner or after all transformer blocks. Reference diagnostics retain Q/K past attention.
pub struct ProjectionAliases { pub qk_bytes:usize, pub kn_offset:usize, pub fp_bytes:usize }
/// While packing input, the text refiner uses only the first `text` rows of
/// full-sequence normalization storage. Modality embeddings occupy its disjoint
/// tail until scattered into hidden, before main-block normalization begins.
pub struct EmbeddingAliases {pub video_offset:usize,pub video_bytes:usize,
    pub audio_offset:usize,pub audio_bytes:usize}
pub fn embedding_aliases(seq:usize,text:usize,video:usize,audio:usize)->Result<EmbeddingAliases>{
    if text==0 || text.checked_add(video).and_then(|v|v.checked_add(audio))!=Some(seq){
        return Err("embedding views require the complete packed row partition".into());
    }
    let bytes=|rows:usize|->Result<usize>{rows.checked_mul(5376*2).ok_or("embedding view size overflow".into())};
    let video_offset=bytes(text)?;let video_bytes=bytes(video)?;
    let audio_offset=bytes(text.checked_add(video).ok_or("embedding view offset overflow")?)?;
    let audio_bytes=bytes(audio)?;bytes(seq)?;
    Ok(EmbeddingAliases{video_offset,video_bytes,audio_offset,audio_bytes})
}
impl ScratchPlan {
    pub fn projection_aliases(&self,world:usize,reference:bool)->Result<Option<ProjectionAliases>> {
        if world!=1 || reference {return Ok(None);}
        let qk=self.rows.checked_mul(7168*2).ok_or("Q/K scratch size overflow")?;
        if qk.checked_mul(2).ok_or("Q/K scratch size overflow")?>self.ff2 || self.fp_hidden>self.ff2 {
            return Err("FFN projection scratch cannot contain projection views".into());
        }
        Ok(Some(ProjectionAliases{qk_bytes:qk,kn_offset:qk,fp_bytes:self.fp_hidden}))
    }
    pub fn new(seq:usize,text:usize,world:usize,rank:usize)->Result<Self> {
        if ![1,2,4,8].contains(&world) || rank>=world || seq<world || text==0 || text>seq {
            return Err("invalid scratch row partition".into());
        }
        let rows=row_counts(seq,world)?[rank].max(text);
        let bytes=|r:usize,w:usize|->Result<usize>{r.checked_mul(w).ok_or_else(||"scratch size overflow".into())};
        let small=world>1 && seq<4096;
        // FP32 input/output projections are dead throughout the transformer
        // blocks. Keep FFN scratch local to the owning CP rank.
        Ok(Self{rows,ff:bytes(rows,14336*2)?,ff2:bytes(rows,14336*4)?,
            fp_hidden:bytes(seq,5376*4)?,
            small_input:if small{bytes(seq,14336*2)?}else{0},
            small_output:if small{bytes(seq,14336*4)?}else{0}})
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn local_projection_extents_follow_owned_rows() {
        for rank in 0..8 {
            let plan=ScratchPlan::new(65539,513,8,rank).unwrap();
            let local=65539/8+usize::from(rank<65539%8);
            assert_eq!(plan.rows,local.max(513));
            assert_eq!(plan.ff,plan.rows*14336*2);
            assert_eq!(plan.ff2,plan.rows*14336*4);
            assert_eq!(plan.fp_hidden,65539*5376*4);
        }
    }
    #[test] fn small_context_parallel_projections_preserve_full_matrix_geometry() {
        for rank in 0..8 {
            let plan=ScratchPlan::new(595,33,8,rank).unwrap();
            assert_eq!(plan.small_input,595*14336*2);
            assert_eq!(plan.small_output,595*14336*4);
        }
    }
    #[test] fn aliases_never_overlap_live_embeddings() {
        let views=embedding_aliases(595,33,548,14).unwrap();
        assert_eq!(views.video_offset,33*5376*2);
        assert_eq!(views.audio_offset,581*5376*2);
        assert_eq!(views.audio_offset+views.audio_bytes,595*5376*2);
        let plan=ScratchPlan::new(595,33,1,0).unwrap();
        let aliases=plan.projection_aliases(1,false).unwrap().unwrap();
        assert!(aliases.kn_offset+aliases.qk_bytes<=plan.ff2);
        assert!(aliases.fp_bytes<=plan.ff2);
    }
    #[test] fn rejects_invalid_and_overflowing_layouts() {
        assert!(ScratchPlan::new(0,1,8,0).is_err());
        assert!(ScratchPlan::new(8192,8193,8,0).is_err());
        assert!(ScratchPlan::new(usize::MAX,1,1,0).is_err());
        assert!(embedding_aliases(595,33,548,15).is_err());
        assert!(cache_execution_budget(usize::MAX,1).is_err());
    }
}
