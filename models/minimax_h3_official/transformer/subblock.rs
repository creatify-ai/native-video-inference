//! Only video queries may be sparse; conditioning images stay dense.
use crate::Result;
use h3_native_pipeline::ref2va::{Kind,ReferenceGeometry};

pub fn storage_rows(rows:usize)->Result<usize> {
    if rows==0 {return Err("empty sparse input".into());}
    rows.checked_add(63).map(|n|n/64*64).ok_or_else(||"sparse storage row overflow".into())
}

/// Packed visual rows also contain images. FL2VA keyframes and Ref2VA image
/// references are dense; Ref2VA video references retain video eligibility.
pub fn image_condition_positions(video:&[i64],condition_rows:usize,references:Option<&[ReferenceGeometry]>)->Result<Vec<i64>> {
    if condition_rows>video.len(){return Err("image condition rows exceed visual rows".into());}
    let Some(references)=references else{return Ok(video[..condition_rows].to_vec());};
    let mut cursor=0usize;let mut dense=Vec::new();
    for reference in references {
        if reference.kind==Kind::Audio {continue;}
        let rows=reference.latent_frames.checked_mul(reference.latent_height/2)
            .and_then(|n|n.checked_mul(reference.latent_width/2)).ok_or("image condition size overflow")?;
        let end=cursor.checked_add(rows).ok_or("image condition size overflow")?;
        if end>condition_rows{return Err("reference visual rows exceed condition prefix".into());}
        if reference.kind==Kind::Image {dense.extend_from_slice(&video[cursor..end]);}
        cursor=end;
    }
    if cursor!=condition_rows{return Err("reference visual rows differ from condition prefix".into());}
    Ok(dense)
}

/// Visual modality tags do not distinguish Qwen image and video tokens.
/// Missing video provenance follows SGLang's fallback: keep Qwen rows dense.
pub fn text_image_positions(tags:&[i64],text:&[i64],video_tokens:&[usize])->Result<Vec<i64>> {
    let mut video=std::collections::BTreeSet::new();
    for &index in video_tokens {
        if index>=text.len() || !video.insert(index){return Err("invalid Qwen video token index".into());}
        let row=usize::try_from(text[index]).map_err(|_|"negative text position")?;
        if tags.get(row)!=Some(&0){return Err("Qwen video token is not a visual row".into());}
    }
    let mut dense=Vec::new();
    for (index,&position) in text.iter().enumerate() {
        let row=usize::try_from(position).map_err(|_|"negative text position")?;
        let tag=tags.get(row).ok_or("text query position out of range")?;
        if *tag==0 && !video.contains(&index){dense.push(position);}
    }
    Ok(dense)
}

pub fn sparse_query_blocks(tags:&[i64],dense_image_positions:&[i64])->Result<Vec<i32>> {
    if tags.is_empty() || tags.iter().any(|tag|!(-1..=2).contains(tag)) {
        return Err("invalid packed modality tags for SubBlock".into());
    }
    let mut eligible=tags.iter().map(|&tag|tag==0).collect::<Vec<_>>();
    for &position in dense_image_positions {
        let row=usize::try_from(position).map_err(|_|"negative image query position")?;
        if row>=tags.len() || tags[row]!=0{return Err("image query position is not a visual row".into());}
        eligible[row]=false;
    }
    Ok(eligible.chunks(64).map(|block|i32::from(block.iter().all(|&video|video))).collect())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn partial_blocks_have_physical_guard_rows_without_changing_logical_lengths() {
        for n in [1,63,64,65,4097,21743,30559] {
            let padded=storage_rows(n).unwrap();
            assert!(padded>=n && padded-n<64);assert_eq!(padded%64,0);
        }
        assert!(storage_rows(0).is_err());assert!(storage_rows(usize::MAX).is_err());
    }
    #[test] fn mixed_blocks_and_non_video_rows_are_dense() {
        let mut tags=vec![0;129];
        assert_eq!(sparse_query_blocks(&tags,&[]).unwrap(),vec![1,1,1]);
        tags[63]=1;tags[64]=2;tags[128]=-1;
        assert_eq!(sparse_query_blocks(&tags,&[]).unwrap(),vec![0,0,0]);
        tags[63]=0;tags[64]=0;
        assert_eq!(sparse_query_blocks(&tags,&[]).unwrap(),vec![1,1,0]);
        assert!(sparse_query_blocks(&[],&[]).is_err());
        assert!(sparse_query_blocks(&[3],&[]).is_err());
    }
    #[test] fn image_anchors_stay_dense_without_disabling_qwen_video_tokens() {
        let tags=vec![0;192];let video=(64..192).collect::<Vec<i64>>();
        let dense=image_condition_positions(&video,64,None).unwrap();
        assert_eq!(sparse_query_blocks(&tags,&dense).unwrap(),vec![1,0,1]);
        assert!(sparse_query_blocks(&tags,&[-1]).is_err());
        assert!(sparse_query_blocks(&tags,&[192]).is_err());
        assert!(sparse_query_blocks(&[1],&[0]).is_err());
        assert!(image_condition_positions(&video,129,None).is_err());
    }
    #[test] fn reference_videos_remain_sparse_and_reference_images_remain_dense() {
        let references=[ReferenceGeometry{kind:Kind::Image,latent_frames:1,latent_height:16,latent_width:16,audio_rows:0},
            ReferenceGeometry{kind:Kind::Video,latent_frames:2,latent_height:16,latent_width:8,audio_rows:0}];
        let video=(0..192).collect::<Vec<i64>>();
        let dense=image_condition_positions(&video,128,Some(&references)).unwrap();
        assert_eq!(sparse_query_blocks(&vec![0;192],&dense).unwrap(),vec![0,1,1]);
        assert!(image_condition_positions(&video,127,Some(&references)).is_err());
        assert!(image_condition_positions(&video,129,Some(&references)).is_err());
    }
    #[test] fn qwen_images_are_dense_and_only_explicit_video_tokens_are_sparse() {
        let tags=vec![0;192];let text=(0..128).collect::<Vec<i64>>();
        let images=text_image_positions(&tags,&text,&[]).unwrap();
        assert_eq!(sparse_query_blocks(&tags,&images).unwrap(),vec![0,0,1]);
        let images=text_image_positions(&tags,&text,&(64..128).collect::<Vec<_>>()).unwrap();
        assert_eq!(sparse_query_blocks(&tags,&images).unwrap(),vec![0,1,1]);
        assert!(text_image_positions(&tags,&text,&[128]).is_err());
        assert!(text_image_positions(&tags,&text,&[1,1]).is_err());
    }
}
