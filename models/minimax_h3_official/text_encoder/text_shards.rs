//! Placement of the Qwen text decoder's projection weights across a conditioner
//! rank group (every group of 2, 4 or 8 ranks). The seven projections of one decoder
//! layer form a block of 256-byte-aligned tensors in PROJECTIONS order; the block
//! splits evenly into one contiguous shard per rank. A rank keeps only its shards
//! and copies a layer's whole block together before the layer runs.
use crate::Result;
use std::ops::Range;

pub const LAYERS: usize = 50;
/// (name below `language_model.layers.N.`, rows, columns) of each BF16 weight.
pub const PROJECTIONS: [(&str, usize, usize); 7] = [
    ("self_attn.q_proj", 8192, 5120), ("self_attn.k_proj", 1024, 5120), ("self_attn.v_proj", 1024, 5120),
    ("self_attn.o_proj", 5120, 8192), ("mlp.gate_proj", 25600, 5120), ("mlp.up_proj", 25600, 5120),
    ("mlp.down_proj", 5120, 25600)];

pub fn weight_name(layer: usize, projection: usize) -> String {
    format!("language_model.layers.{layer}.{}.weight", PROJECTIONS[projection].0)
}
/// The projection index of a sharded weight name, if it is one.
pub fn sharded_weight(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("language_model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|l| *l < LAYERS)?;
    let projection = PROJECTIONS.iter().position(|(p, _, _)| rest.strip_suffix(".weight") == Some(*p))?;
    Some((layer, projection))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout { pub world: usize, pub rank: usize, pub offsets: [usize; 7], pub bytes: [usize; 7], pub block: usize, pub shard: usize }
impl Layout {
    pub fn new(world: usize, rank: usize) -> Result<Self> {
        if ![2, 4, 8].contains(&world) || rank >= world { return Err("invalid text shard group".into()); }
        let (mut offsets, mut bytes, mut at) = ([0; 7], [0; 7], 0usize);
        for (i, &(_, n, k)) in PROJECTIONS.iter().enumerate() {
            offsets[i] = at; bytes[i] = n * k * 2; at += (bytes[i] + 255) & !255;
        }
        if at % (world * 256) != 0 { return Err("text layer block does not split into aligned shards".into()); }
        Ok(Self { world, rank, offsets, bytes, block: at, shard: at / world })
    }
    /// The bytes of projection `i` this rank stores, and where they start in its shard.
    pub fn owned(&self, i: usize) -> Option<(Range<usize>, usize)> {
        let (low, high) = (self.rank * self.shard, (self.rank + 1) * self.shard);
        let (start, end) = (self.offsets[i].max(low), (self.offsets[i] + self.bytes[i]).min(high));
        (start < end).then(|| ((start - self.offsets[i])..(end - self.offsets[i]), start - low))
    }
    /// Device bytes a rank keeps: its shard of every layer plus two gathered blocks.
    pub fn resident_bytes(&self) -> usize { LAYERS * self.shard + 2 * self.block }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn every_byte_has_one_owner_at_every_group_size() {
        for world in [2, 4, 8] {
            let layouts: Vec<_> = (0..world).map(|r| Layout::new(world, r).unwrap()).collect();
            let block = layouts[0].block;
            assert_eq!(block, 975_175_680);
            let mut seen = vec![0u8; block / 256];
            for layout in &layouts {
                assert_eq!(layout.shard * world, block);
                for i in 0..7 {
                    assert_eq!(layout.offsets[i] % 256, 0);
                    if let Some((range, at)) = layout.owned(i) {
                        // Owned ranges map one to one onto this rank's slice of the block.
                        assert_eq!(layout.offsets[i] + range.start, layout.rank * layout.shard + at);
                        assert!(at + range.len() <= layout.shard);
                        for unit in (layout.offsets[i] + range.start) / 256..(layout.offsets[i] + range.end) / 256 { seen[unit] += 1; }
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "world {world}");
        }
    }
    #[test] fn eight_ranks_keep_about_an_eighth() {
        let layout = Layout::new(8, 3).unwrap();
        assert_eq!(layout.shard, 121_896_960);
        assert_eq!(layout.resident_bytes(), 50 * 121_896_960 + 2 * 975_175_680);
    }
    #[test] fn names_round_trip_and_reject_others() {
        for layer in [0, 17, 49] { for p in 0..7 { assert_eq!(sharded_weight(&weight_name(layer, p)), Some((layer, p))); } }
        for name in ["language_model.layers.50.mlp.up_proj.weight", "language_model.layers.3.input_layernorm.weight",
                     "language_model.layers.3.self_attn.q_proj.bias", "visual.blocks.0.attn.qkv.weight",
                     "language_model.embed_tokens.weight"] {
            assert_eq!(sharded_weight(name), None, "{name}");
        }
    }
    #[test] fn invalid_groups_are_rejected() {
        assert!(Layout::new(1, 0).is_err()); assert!(Layout::new(3, 0).is_err()); assert!(Layout::new(8, 8).is_err());
    }
}
