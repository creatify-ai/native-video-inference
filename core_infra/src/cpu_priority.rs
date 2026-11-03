//! Optional CPU work yields scheduling priority to request preparation and GPU
//! submission. Lower only the calling worker, never its process or siblings.
#[cfg(target_os="linux")]
unsafe extern "C" {
    fn gettid()->i32;
    fn getpriority(which:i32,who:u32)->i32;
    fn setpriority(which:i32,who:u32,priority:i32)->i32;
}
pub fn background()->std::io::Result<()> {
    #[cfg(target_os="linux")]
    unsafe {
        let thread=gettid() as u32;
        if getpriority(0,thread)<10 && setpriority(0,thread,10)!=0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(all(test,target_os="linux"))]
mod tests {
    use super::*;
    #[test]
    fn background_priority_is_local_to_the_worker() {
        let before=unsafe{getpriority(0,gettid() as u32)};
        let child=std::thread::spawn(||{
            background().unwrap();
            unsafe{getpriority(0,gettid() as u32)}
        }).join().unwrap();
        assert!(child>=10);
        assert_eq!(unsafe{getpriority(0,gettid() as u32)},before);
    }
}
