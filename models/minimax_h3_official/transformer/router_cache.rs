// Cache compiled modules and scalar layouts; buffers remain per request.
struct RouterCache {
    plans:std::cell::RefCell<Vec<((usize,usize),std::rc::Rc<Program>)>>,
    builds:std::cell::Cell<usize>,hits:std::cell::Cell<usize>,evictions:std::cell::Cell<usize>,
}
impl RouterCache {
    const CAPACITY:usize=16;
    fn new()->Self{Self{plans:Default::default(),builds:Default::default(),hits:Default::default(),evictions:Default::default()}}
    fn get(&self,root:&Path,sequence:usize,heads:usize)->Result<std::rc::Rc<Program>>{
        let key=(sequence,heads);let mut plans=self.plans.borrow_mut();
        if let Some(index)=plans.iter().position(|(k,_)|*k==key){
            let entry=plans.remove(index);let result=entry.1.clone();plans.push(entry);
            self.hits.set(self.hits.get()+1);return Ok(result);
        }
        // Program Drop synchronizes before unloading driver modules.
        if plans.len()>=Self::CAPACITY{plans.remove(0);self.evictions.set(self.evictions.get()+1);}
        let program=std::rc::Rc::new(Program::load_for_shape(root,sequence,heads)?);
        plans.push((key,program.clone()));self.builds.set(self.builds.get()+1);Ok(program)
    }
    fn report(&self)->Value{json!({"policy":"LRU","capacity":Self::CAPACITY,"entries":self.plans.borrow().len(),
        "builds":self.builds.get(),"hits":self.hits.get(),"evictions":self.evictions.get(),
        "modules_per_plan":4,"request_buffers_cached":false})}
}
