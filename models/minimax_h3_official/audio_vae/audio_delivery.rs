//! Channel-major export spans use the decoder's declared output geometry.
//! The final graph view can alias a tensor whose internal rank is still three.
pub fn prefix_ranges(shape:&[i64],input_bytes:usize,samples:usize)->Result<[(usize,usize);2],String> {
    if shape.len()!=2||shape[0]!=2||shape[1]<=0 {
        return Err("audio delivery requires decoded stereo [2,N] geometry".into());
    }
    let decoded=usize::try_from(shape[1]).map_err(|e|e.to_string())?;
    let stride=decoded.checked_mul(4).ok_or("decoded audio byte count overflow")?;
    if stride.checked_mul(2)!=Some(input_bytes)||samples==0||samples>decoded {
        return Err("audio delivery exceeds decoded stereo waveform".into());
    }
    Ok([(0,samples*4),(stride,samples*4)])
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn channel_prefixes_preserve_order_and_do_not_take_the_left_channel_twice() {
        let input=(0..12u32).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        let ranges=prefix_ranges(&[2,6],input.len(),2).unwrap();
        let output=ranges.into_iter().flat_map(|(first,len)|input[first..first+len].iter().copied()).collect::<Vec<_>>();
        assert_eq!(output,[0u32,1,6,7].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
        assert_eq!(prefix_ranges(&[2,6],48,6).unwrap(),[(0,24),(24,24)]);
    }
    #[test] fn rejects_empty_oversized_or_inconsistent_export_geometry() {
        for (shape,bytes,samples) in [(vec![2,6],48,0),(vec![2,6],48,7),(vec![1,2,6],48,2),
            (vec![2,6],40,2),(vec![2,0],0,1),(vec![2,-1],8,1)] {
            assert!(prefix_ranges(&shape,bytes,samples).is_err());
        }
    }
}
