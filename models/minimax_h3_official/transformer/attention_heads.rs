//! Keep the original router and BSA's signed 32-bit offsets within range.
//! Each group owns independent heads over the complete packed sequence.
// Group near 2K/243-frame geometry before the full-head C++ BSA workspace
// consumes about 14 GiB. The existing kernels and per-head arithmetic are unchanged.
pub const MAX_UNGROUPED_ROWS: usize = 256000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadPlan { pub heads: usize, pub groups: usize }
impl HeadPlan {
    pub fn new(sequence: usize) -> crate::Result<Self> {
        if sequence == 0 || sequence > crate::limits::MAX_PACKED_ROWS {
            return Err("attention head plan requires 1<=S<=524288".into());
        }
        Ok(if sequence > MAX_UNGROUPED_ROWS { Self { heads: 7, groups: 8 } }
           else { Self { heads: 56, groups: 1 } })
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn every_supported_geometry_keeps_original_kernel_offsets_signed() {
        for sequence in 1..=crate::limits::MAX_PACKED_ROWS {
            let p = HeadPlan::new(sequence).unwrap();
            let blocks = sequence.div_ceil(64);
            let kept = (blocks.div_ceil(4).div_ceil(8) * 8).min(blocks);
            assert_eq!(p.heads * p.groups, 56);
            for elements in [sequence * p.heads * 128, p.heads * blocks * blocks,
                             p.heads * blocks * kept, p.heads * blocks * 4 * 128] {
                assert!(elements <= i32::MAX as usize);
            }
        }
        assert_eq!(HeadPlan::new(MAX_UNGROUPED_ROWS).unwrap().groups, 1);
        assert_eq!(HeadPlan::new(MAX_UNGROUPED_ROWS+1).unwrap().groups, 8);
        assert!(HeadPlan::new(0).is_err());
        assert!(HeadPlan::new(524289).is_err());
    }
}

#[cfg(test)] mod memory_tests {
    #[test] fn two_k_243_frames_uses_bounded_cpp_workspace() {
        // 1440x2560, T=(243-5)/17*5+2=72; 2x2 spatial token packing.
        let video_rows = 72 * (1440 / 16 / 2) * (2560 / 16 / 2);
        let plan = super::HeadPlan::new(video_rows + 810 + 512).unwrap();
        assert_eq!(plan.heads, 7);
        assert_eq!(plan.groups, 8);
        assert!(plan.heads * (video_rows + 810 + 512) * 128 * 2 * 4 < 2 * 1024usize.pow(3));
    }
}
