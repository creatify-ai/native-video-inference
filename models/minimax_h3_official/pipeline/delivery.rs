//! Exact 24fps delivery on the H3 17n+5 frame grid.
use crate::Result;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortDelivery { pub frames: usize, pub samples: usize }
impl ShortDelivery {
    pub fn from_request(request:&Value,rendered_frames:usize)->Result<Option<Self>> {
        let Some(value)=request.get("short_video_delivery").filter(|v|!v.is_null()) else {return Ok(None);};
        let number=|key:&str|->Result<usize>{Ok(usize::try_from(value[key].as_u64().ok_or("invalid short delivery geometry")?)?)};
        let result=Self{frames:number("num_frames")?,samples:number("samples")?};
        if result.frames==0 || result.frames>120 || result.frames>rendered_frames ||
           result.samples!=(result.frames*32000+12)/24 {
            return Err("short delivery duration must match rendered video at 24fps/32000Hz".into());
        }
        Ok(Some(result))
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use serde_json::json;
    #[test] fn matches_video_and_audio_without_extending_a_render() {
        assert_eq!(ShortDelivery::from_request(&json!({}),124).unwrap(),None);
        for frames in 1..=120 {
            let rendered=frames+(5+17-frames%17)%17;
            let request=json!({"short_video_delivery":{"num_frames":frames,"samples":(frames*32000+12)/24}});
            assert_eq!(ShortDelivery::from_request(&request,rendered).unwrap().unwrap().frames,frames);
            assert!(ShortDelivery::from_request(&request,frames-1).is_err());
        }
        for (frames,samples) in [(0,0),(24,32001),(121,161333)] {
            assert!(ShortDelivery::from_request(&json!({"short_video_delivery":{"num_frames":frames,"samples":samples}}),124).is_err());
        }
    }
}
