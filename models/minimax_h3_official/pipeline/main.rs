fn main()->h3_official_model_pipeline::Result<()> {
    h3_official_model_pipeline::audit::begin_request()?;
    let result=h3_official_model_pipeline::cli::run(std::env::args().collect(),None)?;
    println!("{}",serde_json::to_string_pretty(&result)?);
    Ok(())
}
