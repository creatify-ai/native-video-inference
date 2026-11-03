//! Preserve the two tensor namespaces while sharing one bounded IO pool.
use crate::{Result,TensorSpec,weight_loader::Job};
use serde_json::Value;
use std::{path::Path};

pub struct Groups {
    pub weights:Vec<(String,TensorSpec)>,
    pub buffers:Vec<(String,TensorSpec)>,
}

impl Groups {
    pub fn parse(weights:&Value,buffers:&Value)->Result<Self> {
        fn entries(value:&Value)->Result<Vec<(String,TensorSpec)>> {
            value.as_object().ok_or("model tensor group missing")?.iter()
                .map(|(name,spec)|Ok((name.clone(),TensorSpec::parse(spec)?))).collect()
        }
        Ok(Self{weights:entries(weights)?,buffers:entries(buffers)?})
    }
    /// Packet indices are weight entries, then buffer entries. Roots can differ.
    pub fn jobs(&self,weights_root:&Path,buffers_root:&Path)->Result<Vec<Job>> {
        self.weights.iter().map(|(_,s)|(weights_root,s)).chain(
            self.buffers.iter().map(|(_,s)|(buffers_root,s)))
            .map(|(root,s)|Job::new(root,&s.file,s.bytes,s.sha256.clone()).map_err(Into::into)).collect()
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use crate::weight_loader::{Config,ReaderPool};
    use sha2::{Digest,Sha256};
    use serde_json::json;
    #[test] fn duplicate_names_and_paths_keep_their_roots_and_packet_indices() {
        let root=std::env::temp_dir().join(format!("h3-payload-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let weights=root.join("weights");let buffers=root.join("buffers");
        std::fs::create_dir(&weights).unwrap();std::fs::create_dir(&buffers).unwrap();
        let data=[vec![1u8;8],vec![9u8;8]];
        for (path,bytes) in [(&weights,&data[0]),(&buffers,&data[1])] {
            std::fs::write(path.join("same.raw"),bytes).unwrap();
        }
        let spec=|bytes:&[u8]|json!({"same":{"file":"same.raw","dtype":"torch.bfloat16",
            "shape":[4],"stride":[1],"bytes":8,"contiguous":true,"storage_offset":0,
            "sha256":format!("{:x}",Sha256::digest(bytes))}});
        let groups=Groups::parse(&spec(&data[0]),&spec(&data[1])).unwrap();
        let jobs=groups.jobs(&weights,&buffers).unwrap();
        assert_eq!(jobs[0].path,weights.join("same.raw"));
        assert_eq!(jobs[1].path,buffers.join("same.raw"));
        let mut pool=ReaderPool::start_options(jobs,Config{workers:2,chunk_bytes:3,pinned_slots:2,budget_bytes:18},false,true).unwrap();
        let mut actual=vec![Vec::new(),Vec::new()];
        pool.consume_all(|i,offset,chunk|{assert_eq!(offset,actual[i].len());actual[i].extend_from_slice(chunk);Ok(())}).unwrap();
        assert_eq!(actual,data);
        drop(pool);std::fs::remove_dir_all(root).unwrap();
    }
    #[test] fn invalid_auxiliary_specs_fail_before_starting_readers() {
        let bad=json!({"x":{"file":"../escape","dtype":"torch.bfloat16","shape":[4],
            "stride":[1],"bytes":8,"contiguous":true,"storage_offset":0}});
        assert!(Groups::parse(&json!({}),&bad).is_err());
        let bad_size=json!({"x":{"file":"a","dtype":"torch.bfloat16","shape":[4],
            "stride":[1],"bytes":7,"contiguous":true,"storage_offset":0}});
        assert!(Groups::parse(&json!({}),&bad_size).is_err());
    }
}
