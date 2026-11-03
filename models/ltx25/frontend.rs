//! LTXGemmaTokenizer's raw-prompt path using the same tokenizers 0.22.2 engine.
use crate::{weights::Checkpoint,Result};
use serde_json::Value;
use tokenizers::{Tokenizer,TokenizerImpl,TruncationParams,TruncationDirection,TruncationStrategy,
    models::{bpe::BPE,unigram::Unigram,wordlevel::WordLevel,wordpiece::WordPiece},
    normalizers::NormalizerWrapper,pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,decoders::DecoderWrapper};

pub const MAX_TOKENS:usize=1024;
pub struct TextInput {pub ids:Vec<i64>,pub attention_mask:Vec<i64>}
pub struct GemmaTokenizer {tokenizer:Tokenizer,bos:u32,pad:u32}

fn content(value:&Value)->Option<&str>{value.as_str().or_else(||value["content"].as_str())}
fn python_whitespace(c:char)->bool{c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)}

// ModelWrapper first buffers/clones the full model through serde's untagged
// representation. Select its declared model type without allocating vocabulary
// entries, then use the same upstream model and tokenizer deserializers directly.
// Converting into Tokenizer moves all tokenizer state, including added tokens,
// normalizers, post-processing, padding and truncation, without rebuilding it.
fn parse_tokenizer(bytes:&[u8])->tokenizers::Result<Tokenizer>{
    #[derive(serde::Deserialize)]
    struct ModelKind {#[serde(rename="type")] kind:Option<String>}
    #[derive(serde::Deserialize)]
    struct Kind {model:ModelKind}
    type Typed<M> = TokenizerImpl<M,NormalizerWrapper,PreTokenizerWrapper,PostProcessorWrapper,DecoderWrapper>;
    let kind:Kind=serde_json::from_slice(bytes)?;
    match kind.model.kind.as_deref(){
        Some("BPE")=>Typed::<BPE>::from_bytes(bytes).map(Into::into),
        Some("Unigram")=>Typed::<Unigram>::from_bytes(bytes).map(Into::into),
        Some("WordLevel")=>Typed::<WordLevel>::from_bytes(bytes).map(Into::into),
        Some("WordPiece")=>Typed::<WordPiece>::from_bytes(bytes).map(Into::into),
        // Preserve the upstream legacy type inference and unknown-type errors.
        _=>Tokenizer::from_bytes(bytes),
    }
}

impl GemmaTokenizer {
    pub fn from_checkpoint(checkpoint:&Checkpoint)->Result<Self>{
        let vocabulary=native_core_infra::startup_trace::measure("tokenizer_prepare","read vocabulary",0,||
            checkpoint.asset("tokenizer_json"))?;
        let vocabulary_bytes=vocabulary.len();
        let mut tokenizer=native_core_infra::startup_trace::measure("tokenizer_prepare","parse vocabulary",vocabulary_bytes,||{
            let value=parse_tokenizer(&vocabulary);drop(vocabulary);value})
            .map_err(|e|e.to_string())?;
        let config_bytes=native_core_infra::startup_trace::measure("tokenizer_prepare","read configuration",0,||
            checkpoint.asset("hf_asset__tokenizer_config.json"))?;
        let _configure=native_core_infra::startup_trace::Span::new("tokenizer_prepare","configure tokenizer",config_bytes.len());
        let config:Value=serde_json::from_slice(&config_bytes)?;
        drop(config_bytes);
        let bos_name=content(&config["bos_token"]).ok_or("missing BOS token in tokenizer config")?;
        let bos=tokenizer.token_to_id(bos_name).ok_or("BOS absent from vocabulary")?;
        let pad_name=content(&config["pad_token"]).or_else(||content(&config["eos_token"]))
            .ok_or("no pad/EOS token in tokenizer config")?;
        let pad=tokenizer.token_to_id(pad_name).ok_or("pad token absent from vocabulary")?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams {max_length:MAX_TOKENS,stride:0,
            direction:TruncationDirection::Right,strategy:TruncationStrategy::LongestFirst}))
            .map_err(|e|e.to_string())?;
        Ok(Self{tokenizer,bos,pad})
    }
    pub fn encode(&self,prompt:&str)->Result<TextInput>{
        let prompt=prompt.trim_matches(python_whitespace);
        let encoding=self.tokenizer.encode(prompt,true).map_err(|e|e.to_string())?;
        let mut ids=encoding.get_ids().to_vec();
        if ids.first()!=Some(&self.bos){ids.insert(0,self.bos);ids.truncate(MAX_TOKENS);}
        let count=ids.len();
        if count>MAX_TOKENS{return Err("tokenizer exceeded fixed context".into());}
        let mut padded=vec![self.pad as i64;MAX_TOKENS-count];padded.extend(ids.into_iter().map(i64::from));
        let mut mask=vec![0;MAX_TOKENS-count];mask.extend(std::iter::repeat_n(1,count));
        Ok(TextInput{ids:padded,attention_mask:mask})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn typed_model_parser_preserves_tokenizer_state_and_legacy_models(){
        let models=[
            serde_json::json!({"type":"BPE","vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]],"unk_token":"<unk>"}),
            serde_json::json!({"type":"Unigram","vocab":[["<unk>",0.],["a",-1.],["b",-2.],["ab",-3.]],"unk_id":0,"byte_fallback":false}),
            serde_json::json!({"type":"WordPiece","vocab":{"<unk>":0,"a":1,"b":2,"##b":3},"unk_token":"<unk>","continuing_subword_prefix":"##","max_input_chars_per_word":100}),
            serde_json::json!({"type":"WordLevel","vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"unk_token":"<unk>"}),
            serde_json::json!({"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"unk_token":"<unk>"}),
        ];
        for model in models{
            let bytes=serde_json::to_vec(&serde_json::json!({"version":"1.0","truncation":null,"padding":null,
                "added_tokens":[{"id":4,"content":"<special>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
                "normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null,"decoder":null,"model":model})).unwrap();
            let original=Tokenizer::from_bytes(&bytes).unwrap();let typed=parse_tokenizer(&bytes).unwrap();
            assert_eq!(original.to_string(false).unwrap(),typed.to_string(false).unwrap());
            for prompt in ["AB", " a b ", "<special> ab", "世界 🌊", ""]{
                assert_eq!(original.encode(prompt,true).unwrap(),typed.encode(prompt,true).unwrap());
            }
        }
        for bytes in [br#"{"model":{"type":"BPE","vocab":{}}}"#.as_slice(),
            br#"{"model":{"type":"unsupported"}}"#.as_slice(),br#"{"model":"#.as_slice()]{
            assert!(Tokenizer::from_bytes(bytes).is_err());assert!(parse_tokenizer(bytes).is_err());
        }
    }
    #[test]
    fn python_prompt_strip_includes_control_separators(){
        assert_eq!("\u{1c}\u{85}\u{2003}hello\u{1f}".trim_matches(python_whitespace),"hello");
        assert_eq!("\u{200b}hello\u{200b}".trim_matches(python_whitespace),"\u{200b}hello\u{200b}");
    }
    #[test]
    fn immutable_tokenizer_survives_source_close_and_encodes_each_request(){
        let model=serde_json::json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"<bos>":0,"<pad>":1,"hello":2},"unk_token":"<pad>"}});
        let model=serde_json::to_vec(&model).unwrap();
        let config=br#"{"bos_token":"<bos>","pad_token":"<pad>"}"#;
        let header=serde_json::to_vec(&serde_json::json!({
            "tokenizer_json":{"dtype":"U8","shape":[model.len()],"data_offsets":[0,model.len()]},
            "hf_asset__tokenizer_config.json":{"dtype":"U8","shape":[config.len()],
                "data_offsets":[model.len(),model.len()+config.len()]}})).unwrap();
        let path=std::env::temp_dir().join(format!("ltx25-tokenizer-cache-{}",std::process::id()));
        let mut bytes=(header.len() as u64).to_le_bytes().to_vec();bytes.extend(header);bytes.extend(model);bytes.extend(config);
        std::fs::write(&path,bytes).unwrap();let checkpoint=Checkpoint::open(&path).unwrap();
        let first=checkpoint.tokenizer().unwrap();let before=first.encode("hello").unwrap();
        std::fs::remove_file(path).unwrap();let second=checkpoint.tokenizer().unwrap();
        assert!(std::rc::Rc::ptr_eq(&first,&second));assert!(checkpoint.tokenizer_loaded());
        assert_eq!(before.ids,second.encode("hello").unwrap().ids);
        assert_eq!(&before.ids[MAX_TOKENS-2..],&[0,2]);
        let other=second.encode("unknown").unwrap();assert_eq!(&other.ids[MAX_TOKENS-2..],&[0,1]);
        assert_eq!(before.attention_mask,other.attention_mask);
    }
    #[test]
    #[ignore="requires LTX_TOKENIZER_CHECKPOINT: the pinned Gemma text-encoder file"]
    fn typed_tokenizer_deserialization_benchmark(){
        use tokenizers::models::ModelWrapper;
        let path=std::env::var("LTX_TOKENIZER_CHECKPOINT").expect("LTX_TOKENIZER_CHECKPOINT");
        let checkpoint=Checkpoint::open(path).unwrap();let bytes=checkpoint.asset("tokenizer_json").unwrap();
        let typed_first=std::env::var("LTX_TYPED_BENCH_ORDER").as_deref()==Ok("typed-first");
        let mut values=Vec::new();
        for typed in if typed_first {[true,false]}else{[false,true]}{
            let started=std::time::Instant::now();
            let value=if typed {parse_tokenizer(&bytes).unwrap()}else{Tokenizer::from_bytes(&bytes).unwrap()};
            values.push((typed,value,started.elapsed().as_secs_f64()));
        }
        let (baseline,baseline_seconds,typed,typed_seconds)=if typed_first {
            let (second_type,baseline,baseline_seconds)=values.pop().unwrap();assert!(!second_type);
            let (_,typed,typed_seconds)=values.pop().unwrap();(baseline,baseline_seconds,typed,typed_seconds)
        }else{
            let (second_type,typed,typed_seconds)=values.pop().unwrap();assert!(second_type);
            let (_,baseline,baseline_seconds)=values.pop().unwrap();(baseline,baseline_seconds,typed,typed_seconds)
        };
        assert_eq!(baseline.get_vocab(true),typed.get_vocab(true));
        assert_eq!(baseline.to_string(false).unwrap(),typed.to_string(false).unwrap());
        let long="A fox, 世界 🌊. ".repeat(1200);
        let prompts=["Ocean waves breaking against dark rocks at sunset.","A kingfisher on a branch.",
            ""," \n\t", "你好，世界。 🌊 café e\u{301}","<bos><pad><eos>",long.as_str()];
        for prompt in prompts{
            for special in [false,true]{
                let a=baseline.encode(prompt,special).unwrap();let b=typed.encode(prompt,special).unwrap();
                assert_eq!(a,b,"prompt {prompt:?};special={special}");
            }
        }
        println!("LTX25_TYPED_TOKENIZER_BENCH {}",serde_json::json!({"vocabulary_bytes":bytes.len(),
            "model":match baseline.get_model(){ModelWrapper::BPE(_)=>"BPE",ModelWrapper::Unigram(_)=>"Unigram",
                ModelWrapper::WordLevel(_)=>"WordLevel",ModelWrapper::WordPiece(_)=>"WordPiece"},
            "generic_seconds":baseline_seconds,"typed_seconds":typed_seconds,
            "order":if typed_first {"typed-first"}else{"generic-first"},
            "serialized_tokenizer_exact":true,"vocabulary_exact":true,"encoding_cases":14,"encodings_exact":true}));
    }
}
