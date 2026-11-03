//! One bounded CPU reader cohort for immutable tensor archives. Components
//! publish independently; CUDA and the final light-guard checks stay with the
//! owning thread. Workers write disjoint ranges of unpublished host storage.
use crate::{host_buffer::HostBuffer,integrity::{Expected,ReaderGuard},weight_loader::Config,Result};
use std::{collections::HashMap,fs::File,os::unix::fs::FileExt,path::PathBuf,sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering},mpsc},time::Instant};

pub struct Archive {
    pub path:PathBuf,pub bytes:usize,pub storage:Arc<HostBuffer>,
    pub guards:Vec<(usize,Expected)>,
}
#[derive(Clone,Debug,serde::Serialize)]
pub struct Stats {pub bytes:usize,pub read_complete_seconds:f64,pub guard_seconds:f64,pub direct_read_bytes:usize}
#[derive(Clone,Copy)]struct Job {archive:usize,offset:usize,bytes:usize}

/// All archives share `config.workers`; adding a model component does not add
/// another worker pool or another CPU payload copy. Callback runs only after
/// all ranges and trusted tensor guards of that component have passed.
pub fn read(archives:&[Archive],config:Config,ready:impl FnMut(usize,Stats)->Result<()>)->Result<()>{
    read_with_prepare(archives,config,||Ok(()),ready)
}
/// Run weight-independent preparation on the owner after its CPU readers
/// start. A preparation error cancels and joins the cohort before returning.
pub fn read_with_prepare(archives:&[Archive],config:Config,prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Stats)->Result<()>)->Result<()>{
    read_prioritized_with_prepare(archives,&vec![0;archives.len()],config,prepare,ready)
}
/// Lower ranks are first-use prerequisites. One work-conserving reader pool
/// keeps the cold-start worker/buffer budget; lower-priority publication cannot
/// overtake an unfinished higher-priority component.
pub fn read_prioritized_with_prepare(archives:&[Archive],priorities:&[u8],config:Config,
    prepare:impl FnOnce()->Result<()>,mut ready:impl FnMut(usize,Stats)->Result<()>)->Result<()>{
    config.validate()?;
    if priorities.len()!=archives.len(){return Err("archive priority declaration differs from its components".into());}
    if archives.is_empty(){return prepare();}
    let mut ranges=archives.iter().map(|a|{
        if a.storage.writable_bytes().is_none_or(|n|a.bytes>n){return Err("archive destination is read-only or too small");}
        let begin=a.storage.pointer() as usize;
        Ok((begin,begin.checked_add(a.bytes).ok_or("archive destination overflow")?))
    }).collect::<std::result::Result<Vec<_>,_>>()?;
    ranges.sort_unstable();
    if ranges.windows(2).any(|p|p[0].1>p[1].0){return Err("archive destinations overlap".into());}
    let stripe=config.chunk_bytes.checked_mul(4).ok_or("archive stripe overflow")?;
    let direct=match std::env::var("NATIVE_LOAD_FILE_IO").as_deref(){
        Err(_)|Ok("auto")=>usize::from(cfg!(target_os="linux")),Ok("buffered")=>0,
        Ok("direct")if cfg!(target_os="linux")=>2,
        _=>return Err("archive NATIVE_LOAD_FILE_IO must be auto, buffered or direct".into()),
    };
    let direct_bytes=archives.iter().map(|_|AtomicUsize::new(0)).collect::<Vec<_>>();
    let mut pending=Vec::new();
    for a in archives{
        if a.bytes==0 || std::fs::metadata(&a.path)?.len()!=a.bytes as u64{return Err("archive extent changed".into());}
        let mut end=0;
        for (offset,guard) in &a.guards{
            if *offset<end || guard.bytes==0 || offset.checked_add(guard.bytes).is_none_or(|n|n>a.bytes){
                return Err("archive tensor guards overlap or escape payload".into());
            }
            end=offset+guard.bytes;
        }
        if a.guards.is_empty(){return Err("archive needs trusted tensor guards".into());}
        pending.push(AtomicUsize::new(a.bytes.div_ceil(stripe)));
    }
    // Round-robin ranges start small encoders beside DiT, then all free
    // readers take remaining ranges. A huge embedding is never one job.
    let mut jobs=Vec::new();let mut offset=0;
    loop{
        let before=jobs.len();
        for(index,a)in archives.iter().enumerate(){if offset<a.bytes{
            jobs.push(Job{archive:index,offset,bytes:stripe.min(a.bytes-offset)});
        }}
        if before==jobs.len(){break;}
        offset=offset.checked_add(stripe).ok_or("archive range overflow")?;
    }
    jobs.sort_by_key(|job|priorities[job.archive]);
    let begun=Instant::now();let next=AtomicUsize::new(0);let stop=AtomicBool::new(false);
    let (sender,receiver)=mpsc::channel::<std::result::Result<(usize,f64),String>>();
    std::thread::scope(|scope|->Result<()>{
        for _ in 0..config.workers.min(jobs.len()){
            let sender=sender.clone();let jobs=&jobs;let next=&next;let stop=&stop;let pending=&pending;let direct_bytes=&direct_bytes;
            scope.spawn(move||{
                let mut files=HashMap::<usize,File>::new();let mut direct_files=HashMap::<usize,Option<File>>::new();
                let work=(||->std::result::Result<(),String>{
                    while !stop.load(Ordering::Acquire){
                        let Some(job)=jobs.get(next.fetch_add(1,Ordering::Relaxed))else{break;};
                        let archive=&archives[job.archive];
                        if !files.contains_key(&job.archive){
                            let file=File::open(&archive.path).map_err(|e|e.to_string())?;
                            if file.metadata().map_err(|e|e.to_string())?.len()!=archive.bytes as u64{return Err("archive extent changed".into());}
                            files.insert(job.archive,file);
                        }
                        let file=&files[&job.archive];let mut copied=0;
                        while copied<job.bytes{
                            if stop.load(Ordering::Acquire){return Ok(());}
                            let count=config.chunk_bytes.min(job.bytes-copied);let position=job.offset+copied;
                            // Job construction partitions each allocation; no
                            // range is read/published until its archive is done.
                            let target=unsafe{std::slice::from_raw_parts_mut(archive.storage.pointer().add(position),count)};
                            let mut used=0;
                            #[cfg(target_os="linux")]
                            if direct>0&&position%4096==0&&(target.as_ptr() as usize)%4096==0{
                                use std::os::unix::fs::OpenOptionsExt;
                                let handle=direct_files.entry(job.archive).or_insert_with(||std::fs::OpenOptions::new().read(true).custom_flags(0o40000).open(&archive.path).ok());
                                if direct==2&&handle.is_none(){return Err("direct archive file open failed".into());}
                                let aligned=count/4096*4096;
                                if let Some(input)=handle.as_ref().filter(|_|aligned>0){
                                    match input.read_exact_at(&mut target[..aligned],position as u64){
                                        Ok(())=>{used=aligned;direct_bytes[job.archive].fetch_add(aligned,Ordering::Relaxed);},
                                        Err(error)if direct==1&&matches!(error.raw_os_error(),Some(22|38|95))=>{*handle=None;},
                                        Err(error)=>return Err(format!("direct archive {} at {position}: {error}",archive.path.display())),
                                    }
                                }
                            }
                            if used<count{file.read_exact_at(&mut target[used..],(position+used) as u64)
                                .map_err(|e|format!("archive {} at {position}: {e}",archive.path.display()))?;}
                            copied+=count;
                        }
                        if pending[job.archive].fetch_sub(1,Ordering::AcqRel)==1{
                            sender.send(Ok((job.archive,begun.elapsed().as_secs_f64()))).map_err(|_|"archive consumer closed")?;
                        }
                    }Ok(())
                })();
                if let Err(error)=work{stop.store(true,Ordering::Release);let _=sender.send(Err(error));}
            });
        }
        drop(sender);
        let result=(||->Result<()>{
            prepare()?;
            let mut completed=vec![None::<Stats>;archives.len()];
            let mut published=vec![false;archives.len()];
            for _ in archives{
                let(index,read_complete_seconds)=receiver.recv().map_err(|_|"archive readers exited before completion")??;
                let archive=&archives[index];
                if std::fs::metadata(&archive.path)?.len()!=archive.bytes as u64{return Err("archive extent changed during read".into());}
                let start=Instant::now();
                for(offset,expected)in &archive.guards{
                    let bytes=unsafe{std::slice::from_raw_parts(archive.storage.pointer().add(*offset),expected.bytes)};
                    verify(bytes,expected)?;
                }
                completed[index]=Some(Stats{bytes:archive.bytes,read_complete_seconds,guard_seconds:start.elapsed().as_secs_f64(),direct_read_bytes:direct_bytes[index].load(Ordering::Relaxed)});
                loop{
                    let next=(0..archives.len()).find(|&i|completed[i].is_some()&&!published[i]&&
                        (0..archives.len()).all(|j|priorities[j]>=priorities[i]||published[j]));
                    let Some(next)=next else{break;};
                    ready(next,completed[next].take().unwrap())?;published[next]=true;
                }
            }Ok(())
        })();
        stop.store(true,Ordering::Release);
        // Scoped joins precede every allocation release, including callback,
        // reader and integrity errors. No worker escapes its host owner.
        result
    })
}
fn verify(bytes:&[u8],expected:&Expected)->Result<()>{
    let mut guard=ReaderGuard::new(expected.clone());
    // One observation preserves overlapping first/last segments for small
    // tensors without scanning/copying the complete tensor.
    if bytes.len()<=2048{guard.observe(0,bytes);}else{
        guard.observe(0,&bytes[..1024]);guard.observe(bytes.len()-1024,&bytes[bytes.len()-1024..]);
    }
    guard.finish().map_err(Into::into)
}

#[cfg(test)]mod tests{
    use super::*;use sha2::{Digest,Sha256};
    fn expected(bytes:&[u8])->Expected{Expected{bytes:bytes.len(),head_sha256:format!("{:x}",Sha256::digest(&bytes[..bytes.len().min(1024)])),tail_sha256:format!("{:x}",Sha256::digest(&bytes[bytes.len().saturating_sub(1024)..]))}}
    #[test]fn cohort_publishes_exact_archives_and_checks_short_overlapping_guards(){
        let root=std::env::temp_dir().join(format!("archive-read-{}-{:?}",std::process::id(),std::thread::current().id()));std::fs::create_dir_all(&root).unwrap();
        let inputs=[vec![19u8;4194321],vec![37u8;1500],vec![51u8;31]];
        let archives=inputs.iter().enumerate().map(|(i,bytes)|{
            let path=root.join(i.to_string());std::fs::write(&path,bytes).unwrap();
            Archive{path,bytes:bytes.len(),storage:Arc::new(HostBuffer::new(bytes.len()).unwrap()),guards:vec![(0,expected(bytes))]}
        }).collect::<Vec<_>>();
        let config=Config{workers:3,chunk_bytes:1<<20,pinned_slots:1,budget_bytes:7<<20};let mut seen=Vec::new();
        read(&archives,config,|i,stats|{assert_eq!(stats.bytes,inputs[i].len());assert_eq!(unsafe{std::slice::from_raw_parts(archives[i].storage.pointer(),inputs[i].len())},inputs[i]);seen.push(i);Ok(())}).unwrap();
        seen.sort();assert_eq!(seen,vec![0,1,2]);std::fs::remove_dir_all(root).unwrap();
    }
    #[test]fn aliases_and_read_only_mappings_are_rejected_before_writing(){
        let path=std::env::temp_dir().join(format!("archive-alias-{}-{:?}",std::process::id(),std::thread::current().id()));let bytes=vec![7u8;4096];std::fs::write(&path,&bytes).unwrap();
        let buffer=Arc::new(HostBuffer::new(4096).unwrap());
        let make=|storage|Archive{path:path.clone(),bytes:4096,storage,guards:vec![(0,expected(&bytes))]};
        let config=Config{workers:1,chunk_bytes:1<<20,pinned_slots:1,budget_bytes:3<<20};
        assert!(read(&[make(buffer.clone()),make(buffer)],config,|_,_|Ok(())).unwrap_err().to_string().contains("overlap"));
        let mapped=Arc::new(HostBuffer::map_file(&path,0,4096).unwrap());
        assert!(read(&[make(mapped)],config,|_,_|Ok(())).unwrap_err().to_string().contains("read-only"));
        std::fs::remove_file(path).unwrap();
    }
    #[test]fn corrupted_guard_is_never_published(){
        let path=std::env::temp_dir().join(format!("archive-bad-{}-{:?}",std::process::id(),std::thread::current().id()));let bytes=vec![1u8;4096];std::fs::write(&path,&bytes).unwrap();
        let mut guard=expected(&bytes);guard.head_sha256="0".repeat(64);
        let archives=[Archive{path:path.clone(),bytes:bytes.len(),storage:Arc::new(HostBuffer::new(bytes.len()).unwrap()),guards:vec![(0,guard)]}];
        let config=Config{workers:1,chunk_bytes:1<<20,pinned_slots:1,budget_bytes:3<<20};
        let mut published=false;assert!(read(&archives,config,|_,_|{published=true;Ok(())}).is_err());assert!(!published);std::fs::remove_file(path).unwrap();
    }
    #[test]fn first_use_priority_reuses_the_complete_reader_pool_and_orders_publication(){
        let root=std::env::temp_dir().join(format!("archive-priority-{}-{:?}",std::process::id(),std::thread::current().id()));std::fs::create_dir_all(&root).unwrap();
        let archives=[31usize,9<<20,1500].into_iter().enumerate().map(|(i,n)|{
            let bytes=vec![i as u8+1;n];let path=root.join(i.to_string());std::fs::write(&path,&bytes).unwrap();
            Archive{path,bytes:n,storage:Arc::new(HostBuffer::new(n).unwrap()),guards:vec![(0,expected(&bytes))]}
        }).collect::<Vec<_>>();
        let config=Config{workers:3,chunk_bytes:1<<20,pinned_slots:1,budget_bytes:7<<20};let mut seen=Vec::new();
        // The tiny decoder would otherwise finish before the large encoder.
        read_prioritized_with_prepare(&archives,&[2,0,1],config,||Ok(()),|i,_|{seen.push(i);Ok(())}).unwrap();
        assert_eq!(seen,vec![1,2,0]);
        assert!(read_prioritized_with_prepare(&archives,&[0],config,||Ok(()),|_,_|Ok(())).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
