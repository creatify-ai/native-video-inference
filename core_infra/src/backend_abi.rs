//! Versioned native backend ABI. All allocation ownership stays with its DSO.
#[repr(C)]
pub struct Bytes {pub data:*mut u8,pub len:usize}
pub type Call=unsafe extern "C" fn(*const u8,usize,*mut Bytes)->i32;
pub type Free=unsafe extern "C" fn(*mut u8,usize);
pub type Version=unsafe extern "C" fn()->u32;
pub const VERSION:u32=1;
pub const MAX_REQUEST_BYTES:usize=1<<20;
pub const MAX_REPLY_BYTES:usize=4<<20;
