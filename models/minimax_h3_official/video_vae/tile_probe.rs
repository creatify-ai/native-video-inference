// Explicit diagnostic commands; never used as an ordinary request input source.
fn spatial_probe(args:&[String])->Result<()>{
    if args.len()!=8{return Err("usage: tile-spatial PROGRAM_ROOT STAGE HEIGHT WIDTH INPUT_RAW OUTPUT_DIR".into());}
    let root=Path::new(&args[2]);let stage=&args[3];let h=args[4].parse()?;let w=args[5].parse()?;
    let mut program=Program::load_execution(root,false)?;program.select_spatial(stage,h,w)?;
    let input=arr(&program.manifest["buffers"])?.iter().find(|x|x["role"]=="input").ok_or("tile runtime input")?;
    let width=if input["dtype"]=="torch.float32"{4}else if input["dtype"]=="torch.float16"{2}else{return Err("tile input dtype".into());};
    let bytes=fs::read(&args[6])?;let count=ints(&input["shape"])?.iter().product::<i64>() as usize*width;
    if bytes.len()!=count{return Err("tile input byte count".into());}
    program.arena.upload(program.buffers[num(&input["id"])?],&bytes)?;program.run(false)?;
    let output=program.output()?;let out=Path::new(&args[7]);fs::create_dir_all(out)?;fs::write(out.join("output.raw"),&output)?;
    let report=json!({"passed":true,"stage":stage,"input_sha256":hash(&bytes),"output_sha256":hash(&output),"output_bytes":output.len(),"spatial_plans":program.spatial_report()});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;Ok(())
}
fn spatial_plan_probe(args:&[String])->Result<()>{
    if args.len()!=7{return Err("usage: shape-plan PROGRAM_ROOT STAGE HEIGHT WIDTH OUTPUT_JSON".into());}
    let base:Value=serde_json::from_slice(&fs::read(Path::new(&args[2]).join("program.json"))?)?;
    let mut plans=SpatialPlans::new(&std::sync::Arc::new(base),None)?;let result=plans.select(&args[3],args[4].parse()?,args[5].parse()?)?;
    fs::write(&args[6],serde_json::to_vec(&*result)?)?;Ok(())
}
