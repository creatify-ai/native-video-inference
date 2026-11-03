//! Qwen3-VL vision patch rows: resized RGB frames in 2x2 merge-group order, each
//! row 3 channels x 2 frames x 16 x 16 FP32 values normalized as (value-mean)/std.
//! Any row range can be produced on its own, so each rank of a parallel
//! conditioner group prepares only the rows it embeds; the full range is the
//! single-process tensor.
use crate::Result;
use std::{fs, io::{Read, Seek, SeekFrom}, ops::Range, path::PathBuf};

/// One image (one sampled frame) or video: sampled frames, two per temporal block.
pub(crate) struct Visual {pub path:PathBuf,pub w:usize,pub h:usize,pub rw:usize,pub rh:usize,pub sampled:Vec<usize>}
impl Visual {
    pub fn blocks(&self)->usize {self.sampled.len().div_ceil(2)}
    pub fn grid(&self)->(usize,usize) {(self.rh/16,self.rw/16)}
    fn rows(&self)->usize {self.blocks()*(self.rh/16)*(self.rw/16)}
}

/// One pixel tensor (images, or the videos of a reference set).
pub(crate) struct Pixels {pub visuals:Vec<Visual>,pub mean:[f32;3],pub std:[f32;3]}
impl Pixels {
    pub fn total_rows(&self)->usize {self.visuals.iter().map(Visual::rows).sum()}
    /// Rows `range`, reading and resizing only the frames they use. Each row's
    /// values depend on its frames alone, so contiguous row ranges fill on up to
    /// eight threads (about 1024 rows each) with unchanged order and arithmetic.
    pub fn rows(&self,range:Range<usize>)->Result<Vec<f32>> {
        let mut loaded=vec![];let (mut first,mut at)=(0,0);
        for visual in &self.visuals {
            let count=visual.rows();
            let (a,b)=(range.start.max(first),range.end.min(first+count));
            if a<b {loaded.push(Loaded{visual,frames:frames(visual,a-first..b-first)?,first:a-first,at});at+=b-a;}
            first+=count;
        }
        if at!=range.len() {return Err("pixel row range outside the tensor".into());}
        let mut out=vec![0f32;range.len()*1536];
        let share=range.len().div_ceil((range.len()/1024).clamp(1,8)).max(1);
        let (mean,std)=(self.mean,self.std);
        let results:Vec<Result<()>>=std::thread::scope(|scope| {
            let workers:Vec<_>=out.chunks_mut(share*1536).enumerate().map(|(i,part)|{
                let loaded=&loaded;scope.spawn(move||fill(loaded,i*share,part,mean,std))}).collect();
            workers.into_iter().map(|w|w.join().unwrap_or_else(|_|Err("pixel worker panicked".into()))).collect()
        });
        for result in results {result?;}
        if out.iter().any(|x|!x.is_finite()) {return Err("nonfinite normalized pixels".into());}
        Ok(out)
    }
}

/// One visual's resized frames for the requested rows, which start at its local
/// row `first` and at output row `at`.
struct Loaded<'a> {visual:&'a Visual,frames:Vec<Option<Vec<u8>>>,first:usize,at:usize}

fn frames(v:&Visual,rows:Range<usize>)->Result<Vec<Option<Vec<u8>>>> {
    let (gh,gw)=v.grid();let per=gh*gw;let last=v.sampled.len().checked_sub(1).ok_or("visual without frames")?;
    // Temporal block t uses sampled frames 2t and 2t+1; an odd last frame repeats.
    let used=2*(rows.start/per)..=(2*((rows.end-1)/per)+1).min(last);
    let bytes=v.w*v.h*3;let mut file=fs::File::open(&v.path)?;
    let mut frames=vec![None;v.sampled.len()];let mut raw=vec![0u8;bytes];
    for s in used {
        file.seek(SeekFrom::Start((v.sampled[s]*bytes) as u64))?;file.read_exact(&mut raw)?;
        frames[s]=Some(crate::resize::rgb(&raw,v.w,v.h,v.rw,v.rh)?);
    }
    Ok(frames)
}

/// Output rows first.. of the requested range, 1536 values each.
fn fill(loaded:&[Loaded],first:usize,out:&mut [f32],mean:[f32;3],std:[f32;3])->Result<()> {
    for (k,row) in out.chunks_exact_mut(1536).enumerate() {
        let r=first+k;
        let item=&loaded[loaded.partition_point(|l|l.at<=r).checked_sub(1).ok_or("pixel row owner")?];
        let v=item.visual;let (gh,gw)=v.grid();let last=v.sampled.len()-1;
        let local=item.first+(r-item.at);let (t,rem)=(local/(gh*gw),local%(gh*gw));
        let (br,bc,ir,ic)=(rem/(2*gw),(rem/4)%(gw/2),(rem/2)%2,rem%2);
        let mut next=0;
        for c in 0..3 {for dt in 0..2 {
            let frame=item.frames[(2*t+dt).min(last)].as_deref().ok_or("pixel frame not loaded")?;
            for y in 0..16 {for x in 0..16 {
                let value=frame[(((br*2+ir)*16+y)*v.rw+(bc*2+ic)*16+x)*3+c] as f32;
                row[next]=(value-mean[c])/std[c];next+=1;
            }}
        }}
    }
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    // The accepted single-process loop: every frame resized, rows in order.
    fn reference(v:&Visual,mean:[f32;3],std:[f32;3])->Vec<f32> {
        let raw=fs::read(&v.path).unwrap();let bytes=v.w*v.h*3;
        let frames:Vec<_>=v.sampled.iter().map(|&i|crate::resize::rgb(&raw[i*bytes..(i+1)*bytes],v.w,v.h,v.rw,v.rh).unwrap()).collect();
        let (gh,gw)=v.grid();let mut out=vec![];
        for t in 0..v.blocks() {for br in 0..gh/2 {for bc in 0..gw/2 {for ir in 0..2 {for ic in 0..2 {for c in 0..3 {for dt in 0..2 {for y in 0..16 {for x in 0..16 {
            let frame=&frames[(2*t+dt).min(frames.len()-1)];let value=frame[(((br*2+ir)*16+y)*v.rw+(bc*2+ic)*16+x)*3+c] as f32;
            out.push((value-mean[c])/std[c]);
        }}}}}}}}}
        out
    }
    #[test] fn any_range_matches_the_full_tensor() {
        let dir=std::env::temp_dir().join(format!("h3-pixels-{}",std::process::id()));fs::create_dir_all(&dir).unwrap();
        let mut seed=7u32;let mut next=||{seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);(seed>>24) as u8};
        let mut visuals=vec![];
        // The 900x700 image (2464 rows) makes the full range fill on two threads.
        for (i,&(w,h,frames,ref sampled)) in [(70usize,45usize,1usize,vec![0usize]),(64,64,1,vec![0]),(50,33,7,vec![0,3,6]),
                                                  (900,700,1,vec![0]),(40,72,5,vec![0,2,4,1])].iter().enumerate() {
            let path=dir.join(format!("{i}.rgb"));fs::write(&path,(0..w*h*3*frames).map(|_|next()).collect::<Vec<_>>()).unwrap();
            let (rw,rh)=crate::resize::smart(w,h,32,1024,1<<20).unwrap();
            visuals.push(Visual{path,w,h,rw,rh,sampled:sampled.clone()});
        }
        let (mean,std)=([122.77f32,116.75,104.09],[68.5f32,66.63,70.32]);
        let full:Vec<f32>=visuals.iter().flat_map(|v|reference(v,mean,std)).collect();
        let pixels=Pixels{visuals,mean,std};let total=pixels.total_rows();
        assert_eq!(full.len(),total*1536);
        let bits=|v:&[f32]|v.iter().map(|x|x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&pixels.rows(0..total).unwrap()),bits(&full));
        for world in [2,3,4,8] {for rank in 0..world {
            let (a,b)=(total*rank/world,total*(rank+1)/world);
            assert_eq!(bits(&pixels.rows(a..b).unwrap()),bits(&full[a*1536..b*1536]));
        }}
        assert!(pixels.rows(total..total+1).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
