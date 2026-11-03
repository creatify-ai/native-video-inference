//! Streaming first/last guards supplied by a trusted caller. Full SHA is a separate policy.
use sha2::{Digest,Sha256};
#[derive(Clone,Debug)]
pub struct Expected {pub bytes:usize,pub head_sha256:String,pub tail_sha256:String}
fn compare(data:&[u8],wanted:&str,part:&str)->Result<(),String>{
    if format!("{:x}",Sha256::digest(data))!=wanted{return Err(format!("light SHA mismatch: {part} 1 KiB"));}Ok(())
}
pub struct ReaderGuard {expected:Expected,head:Vec<u8>,tail:Vec<u8>}
impl ReaderGuard {
    pub fn new(expected:Expected)->Self{Self{expected,head:Vec::with_capacity(1024),tail:Vec::with_capacity(1024)}}
    pub fn observe(&mut self,offset:usize,data:&[u8]){
        let head_end=self.expected.bytes.min(1024);
        if offset<head_end{self.head.extend_from_slice(&data[..data.len().min(head_end-offset)]);}
        let tail_start=self.expected.bytes.saturating_sub(1024);
        if offset+data.len()>tail_start{self.tail.extend_from_slice(&data[tail_start.saturating_sub(offset)..]);}
    }
    pub fn finish(self)->Result<(),String>{
        if self.head.len()!=self.expected.bytes.min(1024)||self.tail.len()!=self.expected.bytes.min(1024){return Err("incomplete light SHA segments".into());}
        compare(&self.head,&self.expected.head_sha256,"first")?;
        compare(&self.tail,&self.expected.tail_sha256,"last")
    }
}
