//! Checked native operator budgets; larger envelopes require trusted runtime opt-in.
pub const MAX_PACKED_ROWS: usize = 524_288;
pub const MAX_TEXT_ROWS: usize = 262_144;
pub const MAX_VIDEO_ELEMENTS: usize = u32::MAX as usize;

fn budget(value: Option<&str>, default: usize, ceiling: usize) -> Result<usize, &'static str> {
    let n = value.map_or(Ok(default), |s| s.parse::<usize>().map_err(|_| "invalid native resource budget"))?;
    if n == 0 || n > ceiling {return Err("native resource budget exceeds operator envelope");}
    Ok(n)
}
fn configured(name: &str, default: usize, ceiling: usize) -> Result<usize, &'static str> {
    let value=match std::env::var(name) {Ok(s)=>Some(s),Err(std::env::VarError::NotPresent)=>None,
        Err(_)=>return Err("invalid native resource budget")};
    budget(value.as_deref(),default,ceiling)
}
fn checked_rows(parts: &[usize], limit: usize) -> Result<usize, &'static str> {
    let total=parts.iter().try_fold(0usize, |n,&v| n.checked_add(v)).ok_or("packed sequence overflow")?;
    if total==0 || total>limit {return Err("packed sequence exceeds configured native row budget");}
    Ok(total)
}
pub fn packed_rows(parts: &[usize]) -> Result<usize, &'static str> {
    checked_rows(parts,configured("H3_NATIVE_MAX_PACKED_ROWS",131_072,MAX_PACKED_ROWS)?)
}
pub fn video_elements(width: usize,height: usize,frames: usize) -> Result<usize,&'static str> {
    let n=width.checked_mul(height).and_then(|x|x.checked_mul(frames)).and_then(|x|x.checked_mul(3))
        .ok_or("video output element overflow")?;
    checked_rows(&[n],configured("H3_NATIVE_MAX_VIDEO_ELEMENTS",i32::MAX as usize,MAX_VIDEO_ELEMENTS)?)
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn production_nine_image_and_operator_boundaries() {
        for n in [131458,255713,262145,376538,MAX_PACKED_ROWS] {
            assert_eq!(checked_rows(&[n-1094,1094],MAX_PACKED_ROWS).unwrap(),n);
        }
        for parts in [vec![0],vec![MAX_PACKED_ROWS,1],vec![usize::MAX,1]] {
            assert!(checked_rows(&parts,MAX_PACKED_ROWS).is_err());
        }
        assert!(checked_rows(&[131073],131072).is_err());
    }
    #[test] fn conservative_defaults_and_invalid_overrides() {
        assert_eq!(budget(None,131072,MAX_PACKED_ROWS).unwrap(),131072);
        assert_eq!(budget(Some("524288"),131072,MAX_PACKED_ROWS).unwrap(),MAX_PACKED_ROWS);
        for value in ["0","-1","524289","x"] {assert!(budget(Some(value),131072,MAX_PACKED_ROWS).is_err());}
        for (w,h,f) in [(1088usize,1920usize,345usize),(1440,2592,192)] {
            let n=3*w*h*f;assert!(n>i32::MAX as usize);assert!(n<MAX_VIDEO_ELEMENTS);
            assert!(3*28*w*h<i32::MAX as usize);
        }
        assert!(video_elements(usize::MAX,2,345).is_err());
    }
}
