//! Text <-> token ids the way the reference serves `/generate`: SGLang encodes `text` with the
//! checkpoint's fast tokenizer (tokenizer.json: ByteLevel post-processor, add_bos_token false, so
//! nothing is added around the prompt) and returns the generated ids decoded with
//! skip_special_tokens (clean_up_tokenization_spaces is false).
use crate::Result;
use std::path::Path;

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    pub fn open(checkpoint: &Path) -> Result<Self> {
        let path = checkpoint.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&path).map(Self).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.0.encode(text, true).map_err(|e| e.to_string())?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.0.decode(ids, true).map_err(|e| e.to_string())
    }
}
