//! The request's CPU front end as WanI2V.generate computes it before the encoders run:
//! - image: PNG decode (`Image.open`, `ImageOps.exif_transpose`, `convert('RGB')`; PNGs carrying
//!   EXIF are refused), `TF.to_tensor` and `.sub_(0.5).div_(0.5)`;
//! - size: lat_h / lat_w from max_area and the aspect ratio (numpy float64 math);
//! - the VAE conditioning frame's bicubic taps, as torch's CPU kernel computes them (the
//!   interpolation itself runs on the device, kernels/vae.cu `w21_bicubic`);
//! - text: HuggingfaceTokenizer's whitespace cleaning and the umt5 fast tokenizer.
use crate::Result;
use std::path::Path;

// ---- image ---------------------------------------------------------------------------------------

/// 8-bit RGB, row-major HWC.
pub struct Rgb { pub width: usize, pub height: usize, pub data: Vec<u8> }

/// Image.open(path).convert("RGB") for PNG (palette expanded, 16-bit reduced to the high byte,
/// alpha dropped, gray replicated).
pub fn decode_png(bytes: &[u8]) -> Result<Rgb> {
    if has_exif(bytes) {
        return Err("PNG carries EXIF (ImageOps.exif_transpose is not ported); store the image transposed".into());
    }
    let mut decoder = png::Decoder::new(bytes);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3, png::ColorType::Rgba => 4, png::ColorType::Grayscale => 1, png::ColorType::GrayscaleAlpha => 2,
        other => return Err(format!("unsupported PNG color type {other:?}").into()),
    };
    let mut data = Vec::with_capacity(w * h * 3);
    for px in buf[..info.buffer_size()].chunks_exact(channels) {
        match channels { 1 | 2 => data.extend([px[0], px[0], px[0]]), _ => data.extend(&px[..3]) }
    }
    Ok(Rgb { width: w, height: h, data })
}

/// Whether Pillow would find EXIF in the PNG (an eXIf chunk, or a "Raw profile type exif" text chunk).
fn has_exif(bytes: &[u8]) -> bool {
    let mut at = 8;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
        let kind = &bytes[at + 4..at + 8];
        let data = &bytes[(at + 8).min(bytes.len())..(at + 8 + len).min(bytes.len())];
        if kind == b"eXIf" || (matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") && data.starts_with(b"Raw profile type exif")) {
            return true;
        }
        if kind == b"IEND" { break; }
        at += 12 + len;
    }
    false
}

/// TF.to_tensor(img): fp32 [3, H, W], each byte divided by 255 (true division).
pub fn to_tensor(img: &Rgb) -> Vec<f32> {
    let plane = img.width * img.height;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in img.data.chunks_exact(3).enumerate() {
        for c in 0..3 { out[c * plane + i] = px[c] as f32 / 255.0; }
    }
    out
}

/// `.sub_(0.5).div_(0.5)` on the to_tensor output (each op rounded; the division is exact).
pub fn normalize(x: &[f32]) -> Vec<f32> { x.iter().map(|&v| (v - 0.5) / 0.5).collect() }

/// The latent grid for a source image of `height` x `width`: lat = round(sqrt(max_area * aspect) //
/// vae_stride // patch * patch) in float64 (aspect = h / w for rows, its inverse for columns).
pub fn latent_size(height: usize, width: usize, max_area: f64) -> (usize, usize) {
    const VAE_STRIDE: f64 = 8.0;
    const PATCH: f64 = 2.0;
    let aspect = height as f64 / width as f64;
    let lat = |v: f64| ((v.sqrt() / VAE_STRIDE).floor() / PATCH).floor() * PATCH;
    (lat(max_area * aspect).round() as usize, lat(max_area / aspect).round() as usize)
}

/// HelperInterpCubic::compute_indices_weights (align_corners false, no scale factor; A = -0.75) as
/// torch's AVX2 and AVX-512 CPU builds compute it: GCC contracts the source index and the first
/// multiply-add(s) of each cubic polynomial into FMAs, not their last step (fitted on every tap of
/// 400 random sizes against torch's own weights). Returns the 4 clamped source indices and weights
/// of every output position.
pub fn cubic_taps(input: usize, output: usize) -> (Vec<i32>, Vec<f32>) {
    const A: f32 = -0.75;
    let cc1 = |x: f32| (A + 2.0).mul_add(x, -(A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f32| A.mul_add(x, -(5.0 * A)).mul_add(x, 8.0 * A) * x - 4.0 * A;
    let scale = input as f32 / output as f32;
    let (mut indices, mut weights) = (Vec::with_capacity(4 * output), Vec::with_capacity(4 * output));
    for i in 0..output {
        let real = scale.mul_add(i as f32 + 0.5, -0.5);
        let index = (real.floor() as i64).min(input as i64 - 1);
        let lambda = (real - index as f32).max(0.0).min(1.0);
        let x2 = 1.0 - lambda;
        weights.extend([cc2(lambda + 1.0), cc1(lambda), cc1(x2), cc2(x2 + 1.0)]);
        indices.extend((0..4).map(|j| (index + j - 1).clamp(0, input as i64 - 1) as i32));
    }
    (indices, weights)
}

// ---- text ----------------------------------------------------------------------------------------

/// i2v-14B sample_neg_prompt (wan/configs/wan_i2v_14B.py), used when the request's n_prompt is "".
pub const NEGATIVE_PROMPT: &str = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

fn lookup(table: &'static [(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.binary_search_by(|(k, _)| k.cmp(&key)).ok().map(|i| table[i].1)
}

/// ftfy.fixes.unescape_html: `&#?[0-9A-Za-z]{1,24};` matches are replaced from ftfy's entity
/// table, or (numeric ones) by html.unescape when that consumes the semicolon.
fn ftfy_unescape_html(text: &str) -> String {
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '&' {
            let mut j = i + 1;
            if c.get(j) == Some(&'#') { j += 1; }
            let start = j;
            while j < c.len() && j - start < 24 && c[j].is_ascii_alphanumeric() { j += 1; }
            if j > start && c.get(j) == Some(&';') {
                let entity: String = c[i..=j].iter().collect();
                let replaced = if let Some(v) = lookup(crate::html5_entities::FTFY_ENTITIES, &entity) { v.to_string() }
                    else if entity.starts_with("&#") {
                        let u = html_unescape(&entity);
                        if u.contains(';') { entity.clone() } else { u }
                    } else { entity.clone() };
                out.push_str(&replaced);
                i = j + 1;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

/// Python html.unescape: `&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)` through
/// html._replace_charref (numeric references, html5 names, longest-prefix legacy names).
fn html_unescape(text: &str) -> String {
    use crate::html5_entities::{HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS};
    if !text.contains('&') { return text.to_string(); }
    let c: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] != '&' { out.push(c[i]); i += 1; continue; }
        let r = &c[i + 1..];
        let numeric = |digits_from: usize, radix: u32| -> Option<(usize, String)> {
            let mut j = digits_from;
            let mut num: u64 = 0;
            while j < r.len() && r[j].is_digit(radix) {
                num = (num * radix as u64 + r[j].to_digit(radix).unwrap() as u64).min(0x110000);
                j += 1;
            }
            if j == digits_from { return None; }
            if r.get(j) == Some(&';') { j += 1; }
            let n = num as u32;
            let text = if let Some((_, v)) = INVALID_CHARREFS.iter().find(|(k, _)| *k == n) { v.to_string() }
                else if (0xD800..=0xDFFF).contains(&n) || n > 0x10FFFF { "\u{FFFD}".to_string() }
                else if INVALID_CODEPOINTS.contains(&n) { String::new() }
                else { char::from_u32(n).map(String::from).unwrap_or_default() };
            Some((j, text))
        };
        let matched = if r.first() == Some(&'#') {
            if r.get(1).is_some_and(|x| x.is_ascii_digit()) { numeric(1, 10) }
            else if r.get(1).is_some_and(|x| *x == 'x' || *x == 'X') { numeric(2, 16) }
            else { None }
        } else {
            let mut j = 0;
            while j < r.len() && j < 32 && !matches!(r[j], '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';') { j += 1; }
            if j == 0 { None } else {
                if r.get(j) == Some(&';') { j += 1; }
                let s: Vec<char> = r[..j].to_vec();
                let name: String = s.iter().collect();
                let text = if let Some(v) = lookup(HTML5, &name) { v.to_string() } else {
                    let mut found = None;
                    for x in (2..s.len()).rev() {
                        let prefix: String = s[..x].iter().collect();
                        if let Some(v) = lookup(HTML5, &prefix) {
                            found = Some(format!("{v}{}", s[x..].iter().collect::<String>()));
                            break;
                        }
                    }
                    found.unwrap_or_else(|| format!("&{name}"))
                };
                Some((j, text))
            }
        };
        match matched {
            Some((consumed, text)) => { out.push_str(&text); i += 1 + consumed; }
            None => { out.push('&'); i += 1; }
        }
    }
    out
}

/// whitespace_clean(basic_clean(text)) for the text ftfy and html.unescape leave alone except
/// for width/quote/line-break/control normalization: ASCII, CJK ideographs and punctuation,
/// fullwidth ASCII forms and curly quotes. Other text (possible mojibake, HTML entities, other
/// scripts) is refused rather than approximated.
pub fn clean_prompt(text: &str) -> Result<String> {
    // ftfy.fix_text with unescape_html="auto": text with '<' may be HTML and is refused below;
    // otherwise its entities are decoded first.
    let unescaped = if text.contains('<') { text.to_string() } else { ftfy_unescape_html(text) };
    let text = unescaped.as_str();
    let mut fixed = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    // Accented Latin letters pass through ftfy unchanged unless its fix_encoding sees mojibake:
    // UTF-8 read as Latin-1/cp1252 leaves a lead letter (Ã, Â, or â/Ð/Ñ) before a continuation
    // byte shown as U+0080..U+00BF or a cp1252 symbol, or C1 controls. Such text is refused, as are
    // combining marks (NFC would compose them).
    let continuation = |c: char| matches!(c as u32, 0x80..=0xBF)
        || "€‚ƒ„…†‡ˆ‰Š‹ŒŽ\u{2018}\u{2019}\u{201C}\u{201D}•–—˜™š›œžŸ".contains(c);
    for (k, &c) in chars.iter().enumerate() {
        let next = chars.get(k + 1).copied();
        if matches!(c, 'Ã' | 'Â') || matches!(c as u32, 0x80..=0x9F | 0x0300..=0x036F)
            || (matches!(c, 'â' | 'Ð' | 'Ñ') && next.is_some_and(continuation)) {
            return Err(format!("prompt character U+{:04X} may be mojibake for ftfy (fix_encoding not ported)", c as u32).into());
        }
    }
    use crate::html5_entities::{COMBINING, FTFY_CONTROL_REMOVED, FTFY_LIGATURES, FTFY_WIDTH, NFC_UNSTABLE};
    let table = |t: &'static [(u32, &'static str)], u: u32| t.binary_search_by_key(&u, |(k, _)| *k).ok().map(|i| t[i].1);
    // uncurl_quotes: [\u02bc\u2018-\u201b] -> ' and [\u201c-\u201f] -> "
    let uncurl = |c: char| match c as u32 { 0x02BC | 0x2018..=0x201B => '\'', 0x201C..=0x201F => '"', _ => c };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let u = c as u32;
        match u {
            // remove_terminal_escapes (ESC [ ... letter) is not ported; replace_lossy_sequences works on U+FFFD.
            0x1B => return Err("prompt has a terminal escape (ftfy remove_terminal_escapes not ported)".into()),
            0xFFFD => return Err("prompt has U+FFFD (ftfy replace_lossy_sequences not ported)".into()),
            0x3C => return Err("prompt has '<' (ftfy html policy not ported)".into()),
            // fix_line_breaks
            0x0D => { fixed.push('\n'); if chars.get(i + 1) == Some(&'\n') { i += 1; } }
            0x2028 | 0x2029 => fixed.push('\n'),
            _ if FTFY_CONTROL_REMOVED.binary_search(&u).is_ok() => {}
            _ if COMBINING.binary_search(&u).is_ok() || NFC_UNSTABLE.binary_search(&u).is_ok() => {
                return Err(format!("prompt character U+{u:04X} changes under NFC (normalization not ported)").into());
            }
            _ => {
                // fix_latin_ligatures, then fix_character_width, then uncurl_quotes.
                if let Some(expanded) = table(FTFY_LIGATURES, u) {
                    fixed.extend(expanded.chars().map(uncurl));
                } else if let Some(narrow) = table(FTFY_WIDTH, u) {
                    fixed.extend(narrow.chars().map(uncurl));
                } else {
                    fixed.push(uncurl(c));
                }
            }
        }
        i += 1;
    }
    // html.unescape twice and strip, then re.sub(r'\s+', ' ') and strip.
    let fixed = html_unescape(&html_unescape(&fixed));
    Ok(fixed.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// HuggingfaceTokenizer(umt5-xxl, seq_len 512, clean='whitespace')([text]) after `clean_prompt`:
/// ids and mask as int64 [512] (padded and truncated to 512, with EOS) and the token count.
pub fn tokenize(tokenizer_json: &Path, text: &str) -> Result<(Vec<i64>, Vec<i64>, usize)> {
    encode_tokens(&load_tokenizer(tokenizer_json)?,text)
}
fn load_tokenizer(tokenizer_json:&Path)->Result<tokenizers::Tokenizer> {
    let tokenizer=native_core_infra::startup_trace::measure("wan_frontend","tokenizer read and parse",0,||->Result<_>{
        let bytes=native_core_infra::startup_trace::measure("wan_tokenizer","read JSON",0,||
            native_core_infra::startup_trace::read(tokenizer_json))?;
        parse_tokenizer(&bytes).map_err(|e|format!("tokenizer: {e}").into())
    })?;
    configure_tokenizer(tokenizer)
}

// UMT5 explicitly declares its Unigram model. The generic ModelWrapper
// deserializer buffers the vocabulary as an untagged serde value and then as
// serde_json::Value before deserializing the model. Use the same upstream
// Unigram/TokenizerImpl deserializers directly, then move every tokenizer field
// into Tokenizer. Legacy/other models keep upstream inference and errors.
fn parse_tokenizer(bytes:&[u8])->tokenizers::Result<tokenizers::Tokenizer>{
    use tokenizers::{Tokenizer,TokenizerImpl,models::unigram::Unigram,
        normalizers::NormalizerWrapper,pre_tokenizers::PreTokenizerWrapper,
        processors::PostProcessorWrapper,decoders::DecoderWrapper};
    type Typed=TokenizerImpl<Unigram,NormalizerWrapper,PreTokenizerWrapper,PostProcessorWrapper,DecoderWrapper>;
    // The upstream Unigram visitor validates its own type. Other/legacy model
    // formats retain the generic parser, without scanning UMT5's full JSON twice.
    match native_core_infra::startup_trace::measure("wan_tokenizer","deserialize explicit Unigram",bytes.len(),||
        Typed::from_bytes(bytes).map(Into::into)) {
        Ok(value)=>Ok(value),
        Err(_)=>{
        native_core_infra::startup_trace::measure("wan_tokenizer","deserialize generic model",bytes.len(),||
            Tokenizer::from_bytes(bytes))
        }
    }
}

fn configure_tokenizer(mut tokenizer:tokenizers::Tokenizer)->Result<tokenizers::Tokenizer>{
    use tokenizers::{PaddingDirection,PaddingParams,PaddingStrategy,TruncationDirection,TruncationParams,TruncationStrategy};
    let _configure=native_core_infra::startup_trace::Span::new("wan_tokenizer","configure tokenizer",0);
    tokenizer.with_truncation(Some(TruncationParams { max_length: 512, strategy: TruncationStrategy::LongestFirst,
                                                      stride: 0, direction: TruncationDirection::Right }))
        .map_err(|e| format!("truncation: {e}"))?;
    tokenizer.with_padding(Some(PaddingParams { strategy: PaddingStrategy::Fixed(512), direction: PaddingDirection::Right,
                                                pad_to_multiple_of: None, pad_id: 0, pad_type_id: 0, pad_token: "<pad>".into() }));
    Ok(tokenizer)
}
fn encode_tokens(tokenizer:&tokenizers::Tokenizer,text:&str)->Result<(Vec<i64>,Vec<i64>,usize)> {
    let e = tokenizer.encode(clean_prompt(text)?, true).map_err(|e| format!("encode: {e}"))?;
    let ids = e.get_ids().iter().map(|&v| v as i64).collect();
    let mask: Vec<i64> = e.get_attention_mask().iter().map(|&v| v as i64).collect();
    let len = mask.iter().filter(|&&m| m > 0).count();
    Ok((ids, mask, len))
}

// ---- whole front end -----------------------------------------------------------------------------

/// generate()'s arguments as the official generate.py passes them for i2v-14B (720P defaults).
#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub negative_prompt: String,
    pub max_area: f64,
    pub frame_num: usize,
    pub sample_steps: usize,
    pub shift: f64,
    pub guide_scale: f64,
    pub seed: u64,
}

impl Request {
    /// A request's JSON: prompt plus optional max_area, frame_num, sample_steps,
    /// shift, guide_scale, seed, negative_prompt, sample_solver. Unported branches are refused.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let num = |key: &str, default: f64| v.get(key).and_then(|x| x.as_f64()).unwrap_or(default);
        if v.get("sample_solver").and_then(|x| x.as_str()).is_some_and(|s| s != "unipc") {
            return Err("only sample_solver 'unipc' is ported".into());
        }
        let seed = num("seed", -1.0);
        if seed < 0.0 || seed.fract() != 0.0 { return Err("seed must be a non-negative integer (random seeds are not ported)".into()); }
        let frame_num = num("frame_num", 81.0) as usize;
        if frame_num % 4 != 1 { return Err(format!("frame_num {frame_num} is not 4n+1").into()); }
        let negative = v.get("negative_prompt").and_then(|x| x.as_str()).unwrap_or("");
        Ok(Self {
            prompt: v["prompt"].as_str().ok_or("request.prompt")?.to_string(),
            negative_prompt: if negative.is_empty() { NEGATIVE_PROMPT.to_string() } else { negative.to_string() },
            max_area: num("max_area", 720.0 * 1280.0),
            frame_num,
            sample_steps: num("sample_steps", 40.0) as usize,
            shift: num("shift", 5.0),
            guide_scale: num("guide_scale", 5.0),
            seed: seed as u64,
        })
    }
}

/// A request after the CPU front end.
#[derive(Clone)]
pub struct Prepared {
    pub request: Request,
    /// TF.to_tensor output, fp32 [3, src_h, src_w] (recorded as image/to_tensor in a gold).
    pub to_tensor: Vec<f32>,
    /// Normalized image, fp32 [3, src_h, src_w]: the CLIP input and the resize source.
    pub image: Vec<f32>,
    pub src_h: usize,
    pub src_w: usize,
    pub lat_h: usize,
    pub lat_w: usize,
    /// Bicubic taps (indices, weights) for rows (src_h -> 8 lat_h) and columns (src_w -> 8 lat_w).
    pub taps_y: (Vec<i32>, Vec<f32>),
    pub taps_x: (Vec<i32>, Vec<f32>),
    /// (ids, mask, token count) of the prompt and the negative prompt.
    pub tokens: [(Vec<i64>, Vec<i64>, usize); 2],
}

impl Prepared {
    pub fn height(&self) -> usize { 8 * self.lat_h }
    pub fn width(&self) -> usize { 8 * self.lat_w }
    pub fn latent_frames(&self) -> usize { (self.request.frame_num - 1) / 4 + 1 }
}

/// `inputs/`: request.json and its image_file; `tokenizer_json`: the checkpoint's umt5 tokenizer.
pub fn prepare(inputs: &Path, tokenizer_json: &Path) -> Result<Prepared> {
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(inputs.join("request.json"))?)?;
    let png = std::fs::read(inputs.join(json["image_file"].as_str().unwrap_or("image.png")))?;
    prepare_request(Request::from_json(&json)?, &png, tokenizer_json)
}

/// The front end for a request and its PNG bytes.
pub fn prepare_request(request: Request, png: &[u8], tokenizer_json: &Path) -> Result<Prepared> {
    let _frontend=native_core_infra::startup_trace::Span::new("wan_frontend","complete CPU frontend",png.len());
    // Issue the immutable metadata read before image work and the shared
    // checkpoint readers can fill the serving filesystem's IO queue.
    let tokenizer=load_tokenizer(tokenizer_json)?;
    let png = native_core_infra::startup_trace::measure("wan_frontend","decode PNG",png.len(),||decode_png(png))?;
    let to_tensor = native_core_infra::startup_trace::measure("wan_frontend","image to tensor",png.data.len(),||to_tensor(&png));
    let image = native_core_infra::startup_trace::measure("wan_frontend","normalize image",to_tensor.len()*4,||normalize(&to_tensor));
    let (lat_h, lat_w) = latent_size(png.height, png.width, request.max_area);
    if lat_h == 0 || lat_w == 0 { return Err("image aspect ratio too extreme for max_area".into()); }
    let (taps_y,taps_x) = native_core_infra::startup_trace::measure("wan_frontend","bicubic taps",0,||
        (cubic_taps(png.height,8*lat_h),cubic_taps(png.width,8*lat_w)));
    // Encoding borrows the configured tokenizer. Positive and negative text
    // have identical settings; parsing its large vocabulary twice adds no
    // preparation and previously delayed cold request readiness.
    let tokens = [
        native_core_infra::startup_trace::measure("wan_frontend","encode positive prompt",request.prompt.len(),||encode_tokens(&tokenizer,&request.prompt))?,
        native_core_infra::startup_trace::measure("wan_frontend","encode negative prompt",request.negative_prompt.len(),||encode_tokens(&tokenizer,&request.negative_prompt))?];
    Ok(Prepared { request, to_tensor, image, src_h: png.height, src_w: png.width, lat_h, lat_w, taps_y, taps_x, tokens })
}

/// cache_video's unchanged FP32 normalization, from C/F/H/W to F/H/W/C.
/// Each worker owns complete output frames; no reduction or floating-point
/// operation crosses a worker boundary.
pub fn video_to_u8(video:&[f32],frames:usize,h:usize,w:usize)->Vec<u8> {
    let plane=h*w;
    let mut out=vec![0u8;frames*plane*3];
    if out.is_empty(){return out;}
    let workers=if frames*plane<1<<20 {1}else{
        std::thread::available_parallelism().map(usize::from).unwrap_or(1).min(8).min(frames)
    };
    let frames_per_worker=frames.div_ceil(workers);
    std::thread::scope(|scope|{
        for (part,output) in out.chunks_mut(frames_per_worker*plane*3).enumerate(){
            scope.spawn(move || {
                let first=part*frames_per_worker;
                for (local,frame) in output.chunks_mut(plane*3).enumerate(){
                    let f=first+local;
                    for i in 0..plane {for c in 0..3 {
                        let x=video[(c*frames+f)*plane+i].clamp(-1.0,1.0);
                        frame[i*3+c]=(((x+1.0)/2.0)*255.0) as u8;
                    }}
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    #[derive(Clone,serde::Serialize)]
    #[serde(transparent)]
    struct ProfiledUnigram(tokenizers::models::unigram::Unigram);
    impl<'de> serde::Deserialize<'de> for ProfiledUnigram {
        fn deserialize<D:serde::Deserializer<'de>>(deserializer:D)->std::result::Result<Self,D::Error>{
            #[derive(serde::Deserialize)]
            struct Spec {
                #[serde(rename="type")] kind:String,
                vocab:Vec<(String,f64)>,unk_id:Option<usize>,
                #[serde(default)] byte_fallback:bool,
            }
            let spec=native_core_infra::startup_trace::measure("wan_tokenizer_detail","decode Unigram vocabulary",0,||
                Spec::deserialize(deserializer))?;
            if spec.kind!="Unigram" {return Err(serde::de::Error::custom("expected Unigram"));}
            native_core_infra::startup_trace::measure("wan_tokenizer_detail","build Unigram model and trie",spec.vocab.len(),||
                tokenizers::models::unigram::Unigram::from(spec.vocab,spec.unk_id,spec.byte_fallback))
                .map(Self).map_err(serde::de::Error::custom)
        }
    }
    impl tokenizers::Model for ProfiledUnigram {
        type Trainer=tokenizers::models::unigram::UnigramTrainer;
        fn tokenize(&self,s:&str)->tokenizers::Result<Vec<tokenizers::Token>>{tokenizers::Model::tokenize(&self.0,s)}
        fn token_to_id(&self,s:&str)->Option<u32>{tokenizers::Model::token_to_id(&self.0,s)}
        fn id_to_token(&self,id:u32)->Option<String>{tokenizers::Model::id_to_token(&self.0,id)}
        fn get_vocab(&self)->std::collections::HashMap<String,u32>{tokenizers::Model::get_vocab(&self.0)}
        fn get_vocab_size(&self)->usize{tokenizers::Model::get_vocab_size(&self.0)}
        fn save(&self,path:&Path,prefix:Option<&str>)->tokenizers::Result<Vec<std::path::PathBuf>>{
            tokenizers::Model::save(&self.0,path,prefix)
        }
        fn get_trainer(&self)->Self::Trainer{tokenizers::Model::get_trainer(&self.0)}
    }
    impl From<ProfiledUnigram> for tokenizers::models::ModelWrapper {
        fn from(value:ProfiledUnigram)->Self{Self::Unigram(value.0)}
    }
    fn profile_tokenizer_details(bytes:&[u8])->tokenizers::Result<tokenizers::Tokenizer>{
        use tokenizers::{TokenizerImpl,normalizers::NormalizerWrapper,pre_tokenizers::PreTokenizerWrapper,
            processors::PostProcessorWrapper,decoders::DecoderWrapper};
        type Typed=TokenizerImpl<ProfiledUnigram,NormalizerWrapper,PreTokenizerWrapper,PostProcessorWrapper,DecoderWrapper>;
        native_core_infra::startup_trace::measure("wan_tokenizer_detail","complete profiled tokenizer",bytes.len(),||
            Typed::from_bytes(bytes).map(Into::into))
    }
    #[test]
    #[ignore="requires the three frozen tokenizer JSON assets on one CPU host"]
    fn tokenizer_cross_model_profile(){
        use tokenizers::{Tokenizer,TokenizerImpl,models::{bpe::BPE,unigram::Unigram},
            normalizers::NormalizerWrapper,pre_tokenizers::PreTokenizerWrapper,
            processors::PostProcessorWrapper,decoders::DecoderWrapper};
        use sha2::{Digest,Sha256};
        native_core_infra::startup_trace::configure_from_env();
        let spec=std::env::var("TOKENIZER_COMPARE_SPEC").expect("TOKENIZER_COMPARE_SPEC");
        let spec:Vec<(String,String)>=serde_json::from_slice(&std::fs::read(spec).unwrap()).unwrap();
        type Typed<M>=TokenizerImpl<M,NormalizerWrapper,PreTokenizerWrapper,PostProcessorWrapper,DecoderWrapper>;
        #[derive(serde::Deserialize)]struct ModelKind {#[serde(rename="type")] kind:String}
        #[derive(serde::Deserialize)]struct Kind {model:ModelKind}
        let order=std::env::var("WAN_TOKENIZER_BENCH_ORDER").unwrap_or_default();
        let long="A fox, 世界 🌊. ".repeat(1200);
        let prompts=["Ocean waves breaking against dark rocks at sunset.","A kingfisher on a branch.",
            ""," \n\t","你好，世界。 🌊 café e\u{301}","<pad><bos><eos></s><unk><extra_id_0>",long.as_str()];
        for (name,path) in spec {
            let bytes=native_core_infra::startup_trace::read(&path).unwrap();
            let kind:Kind=serde_json::from_slice(&bytes).unwrap();
            let mut values=Vec::new();
            for typed in if order=="typed-first" {[true,false]}else{[false,true]} {
                let started=std::time::Instant::now();
                let label=format!("{name}:{}",if typed {"typed"}else{"generic"});
                let value=native_core_infra::startup_trace::measure("tokenizer_cross_model",&label,bytes.len(),||{
                    if !typed {return Tokenizer::from_bytes(&bytes);}
                    match kind.model.kind.as_str() {
                        "BPE"=>Typed::<BPE>::from_bytes(&bytes).map(Into::into),
                        "Unigram"=>Typed::<Unigram>::from_bytes(&bytes).map(Into::into),
                        other=>panic!("unexpected tokenizer model {other}"),
                    }
                }).unwrap();
                values.push((typed,value,started.elapsed().as_secs_f64()));
            }
            let (reference,generic_seconds,typed,typed_seconds)=if order=="typed-first" {
                let (_,reference,generic_seconds)=values.pop().unwrap();
                let (_,typed,typed_seconds)=values.pop().unwrap();(reference,generic_seconds,typed,typed_seconds)
            }else{
                let (_,typed,typed_seconds)=values.pop().unwrap();
                let (_,reference,generic_seconds)=values.pop().unwrap();(reference,generic_seconds,typed,typed_seconds)
            };
            assert_eq!(reference.to_string(false).unwrap(),typed.to_string(false).unwrap());
            assert_eq!(reference.get_vocab(true),typed.get_vocab(true));
            let serialized_sha256=format!("{:x}",Sha256::digest(reference.to_string(false).unwrap().as_bytes()));
            let vocabulary:std::collections::BTreeMap<_,_>=reference.get_vocab(true).into_iter().collect();
            let vocabulary_sha256=format!("{:x}",Sha256::digest(serde_json::to_vec(&vocabulary).unwrap()));
            let mut cases:Vec<String>=prompts.iter().map(|s|s.to_string()).collect();
            for i in 0..128 {
                let id=(i*1009)%reference.get_vocab_size(false);
                cases.push(format!("{} / 世界 🌊 {i} café",reference.id_to_token(id as u32).unwrap()));
            }
            let mut encodings=Sha256::new();
            for prompt in &cases {for special in [false,true] {
                let a=reference.encode(prompt.as_str(),special).unwrap();let b=typed.encode(prompt.as_str(),special).unwrap();
                assert_eq!(a,b);encodings.update(serde_json::to_vec(&a).unwrap());
            }}
            println!("TOKENIZER_CROSS_MODEL_PROFILE {}",serde_json::json!({"name":name,"model":kind.model.kind,
                "tokenizer_bytes":bytes.len(),"tokenizer_sha256":format!("{:x}",Sha256::digest(&bytes)),
                "vocabulary_size":reference.get_vocab_size(false),"added_vocabulary_size":reference.get_vocab_size(true)-reference.get_vocab_size(false),
                "generic_seconds":generic_seconds,"typed_seconds":typed_seconds,"order":order,
                "serialized_tokenizer_exact":true,"vocabulary_exact":true,"encodings_exact":true,"encoding_cases":2*cases.len(),
                "serialized_tokenizer_sha256":serialized_sha256,"vocabulary_sha256":vocabulary_sha256,
                "encodings_sha256":format!("{:x}",encodings.finalize())}));
        }
    }
    #[test]
    #[ignore="requires the frozen UMT5 vocabulary; CPU-only construction microprofile"]
    fn unigram_trie_microprofile(){
        use std::{collections::HashMap,hash::BuildHasher,time::Instant};
        native_core_infra::startup_trace::configure_from_env();
        #[derive(serde::Deserialize)]struct Model {vocab:Vec<(String,f64)>}
        #[derive(serde::Deserialize)]struct Document {model:Model}
        let bytes=std::fs::read(std::env::var("WAN_TOKENIZER_JSON").unwrap()).unwrap();
        let document:Document=serde_json::from_slice(&bytes).unwrap();
        let words:Vec<&[u8]>=document.model.vocab.iter().map(|(s,_)|s.as_bytes()).collect();
        struct Node<S>{children:HashMap<u8,Node<S>,S>,leaf:bool}
        fn map_trie<S:BuildHasher+Clone+Default>(words:&[&[u8]],seed:S,shared:bool)->serde_json::Value{
            let started=Instant::now();let mut count=1;
            let mut root=Node{children:HashMap::with_hasher(seed.clone()),leaf:false};
            for word in words {
                let mut node=&mut root;
                for label in *word {
                    node=node.children.entry(*label).or_insert_with(||{count+=1;Node{
                        children:HashMap::with_hasher(if shared {seed.clone()}else{S::default()}),leaf:false}});
                }
                node.leaf=true;
            }
            std::hint::black_box(&root);let construct=started.elapsed().as_secs_f64();
            for word in words {
                let mut node=&root;
                for label in *word {node=node.children.get(label).unwrap();}
                assert!(node.leaf);
            }
            let started=Instant::now();drop(root);let retire=started.elapsed().as_secs_f64();
            serde_json::json!({"construct_seconds":construct,"drop_seconds":retire,"nodes":count,
                "node_bytes":std::mem::size_of::<Node<S>>(),"all_vocabulary_memberships_exact":true})
        }
        let seed=tokenizers::models::bpe::Vocab::default().hasher().clone();
        let original=map_trie(&words,seed.clone(),false);let shared_seed=map_trie(&words,seed,true);
        struct FlatNode{head:usize,leaf:bool}
        struct Edge{label:u8,child:usize,next:usize}
        let capacity=words.iter().map(|w|w.len()).sum::<usize>()+1;
        let started=Instant::now();let mut nodes=Vec::with_capacity(capacity);let mut edges:Vec<Edge>=Vec::with_capacity(capacity);
        nodes.push(FlatNode{head:usize::MAX,leaf:false});let mut comparisons=0;
        for word in &words {
            let mut node=0;
            for label in *word {
                let mut edge=nodes[node].head;
                while edge!=usize::MAX {comparisons+=1;if edges[edge].label==*label {break;}edge=edges[edge].next;}
                if edge==usize::MAX {
                    let child=nodes.len();nodes.push(FlatNode{head:usize::MAX,leaf:false});
                    let next=nodes[node].head;nodes[node].head=edges.len();edges.push(Edge{label:*label,child,next});node=child;
                }else{node=edges[edge].child;}
            }
            nodes[node].leaf=true;
        }
        std::hint::black_box((&nodes,&edges));let construct=started.elapsed().as_secs_f64();
        for word in &words {
            let mut node=0;
            for label in *word {
                let mut edge=nodes[node].head;
                while edge!=usize::MAX && edges[edge].label!=*label {edge=edges[edge].next;}
                assert!(edge!=usize::MAX);node=edges[edge].child;
            }
            assert!(nodes[node].leaf);
        }
        println!("WAN21_TRIE_MICROPROFILE {}",serde_json::json!({"words":words.len(),"word_bytes":capacity-1,
            "per_node_random_seed":original,"shared_random_seed":shared_seed,
            "flat_trie":{"construct_seconds":construct,"nodes":nodes.len(),"edges":edges.len(),"comparisons":comparisons,
                "allocated_capacity_bytes":nodes.capacity()*std::mem::size_of::<FlatNode>()+edges.capacity()*std::mem::size_of::<Edge>(),
                "all_vocabulary_memberships_exact":true}}));
    }
    #[test]
    fn frame_partitioning_preserves_channel_order_and_fp32_clamp_rounding() {
        let cases=[(f32::NEG_INFINITY,0),(-1.,0),(-0.5,63),(-0.,127),(0.,127),(0.5,191),(1.,255),(f32::INFINITY,255),(f32::NAN,0)];
        let (frames,h,w)=(9,257,1031);let plane=h*w;
        let mut video=vec![0f32;3*frames*plane];
        for c in 0..3 {for f in 0..frames {video[(c*frames+f)*plane..(c*frames+f+1)*plane].fill(cases[(c*3+f)%cases.len()].0);}}
        let output=video_to_u8(&video,frames,h,w);
        for f in 0..frames {
            let expected=[cases[f%9].1,cases[(3+f)%9].1,cases[(6+f)%9].1];
            assert!(output[f*plane*3..(f+1)*plane*3].chunks_exact(3).all(|pixel|pixel==expected));
        }
        assert!(video_to_u8(&[],0,h,w).is_empty());
    }
    #[test]
    fn shared_tokenizer_keeps_prompts_independent_and_preserves_eos_truncation_and_padding() {
        use tokenizers::{models::wordlevel::WordLevel,pre_tokenizers::whitespace::Whitespace,
            processors::template::TemplateProcessing,Tokenizer};
        let model=WordLevel::builder().vocab([("<pad>",0),("</s>",1),("<unk>",2),("sun",3),("rain",4)]
            .into_iter().map(|(key,value)|(key.to_string(),value)).collect()).unk_token("<unk>".into()).build().unwrap();
        let mut tokenizer=Tokenizer::new(model);tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(TemplateProcessing::builder().try_single("$A </s>").unwrap()
            .special_tokens(vec![("</s>",1)]).build().unwrap()));
        let path=std::env::temp_dir().join(format!("wan-shared-tokenizer-{}-{}.json",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        tokenizer.save(&path,false).unwrap();let shared=load_tokenizer(&path).unwrap();
        let first=encode_tokens(&shared,"sun sun").unwrap();
        let negative=encode_tokens(&shared,"rain").unwrap();
        let long=encode_tokens(&shared,&vec!["sun";600].join(" ")).unwrap();
        assert_eq!(&first.0[..4],&[3,3,1,0]);assert_eq!(&first.1[..4],&[1,1,1,0]);assert_eq!(first.2,3);
        assert_eq!(&negative.0[..3],&[4,1,0]);assert_eq!(negative.2,2);
        assert_eq!(long.0.len(),512);assert_eq!(long.2,512);assert_eq!(long.0[511],1);
        assert_eq!(encode_tokens(&shared,"sun sun").unwrap(),first);
        assert_eq!(tokenize(&path,"rain").unwrap(),negative);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn typed_unigram_preserves_complete_state_encodings_and_fallback_errors(){
        use tokenizers::Tokenizer;
        let models=[
            serde_json::json!({"type":"Unigram","vocab":[["<unk>",0.],["a",-1.],["b",-2.],["ab",-3.]],"unk_id":0,"byte_fallback":false}),
            serde_json::json!({"type":"WordLevel","vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"unk_token":"<unk>"}),
            serde_json::json!({"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"unk_token":"<unk>"}),
        ];
        for model in models {
            let bytes=serde_json::to_vec(&serde_json::json!({"version":"1.0","truncation":null,"padding":null,
                "added_tokens":[{"id":4,"content":"<special>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
                "normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null,"decoder":null,"model":model})).unwrap();
            let reference=Tokenizer::from_bytes(&bytes).unwrap();let typed=parse_tokenizer(&bytes).unwrap();
            assert_eq!(reference.to_string(false).unwrap(),typed.to_string(false).unwrap());
            assert_eq!(reference.get_vocab(true),typed.get_vocab(true));
            for prompt in ["AB"," a b ","<special> ab","世界 🌊",""] {
                for special in [false,true] {
                    assert_eq!(reference.encode(prompt,special).unwrap(),typed.encode(prompt,special).unwrap());
                }
            }
        }
        for bytes in [br#"{"model":{"type":"Unigram","vocab":[],"unk_id":0}}"#.as_slice(),
            br#"{"model":{"type":"unsupported"}}"#.as_slice(),br#"{"model":"#.as_slice()] {
            assert!(Tokenizer::from_bytes(bytes).is_err());assert!(parse_tokenizer(bytes).is_err());
        }
    }
    #[test]
    fn unigram_branching_prefixes_preserve_longest_scored_pieces_and_clone(){
        use tokenizers::{models::unigram::Unigram,Model};
        let mut vocabulary=vec![("<unk>".to_string(),0.)];
        for value in b'a'..=b'z' {vocabulary.push(((value as char).to_string(),-3.));}
        for value in b'a'..=b'z' {vocabulary.push((format!("a{}",value as char),-0.1));}
        let model=Unigram::from(vocabulary,Some(0),false).unwrap();
        for current in [&model,&model.clone()] {
            for value in b'a'..=b'z' {
                let input=format!("a{}ab",value as char);
                let encoded=current.tokenize(&input).unwrap();
                assert_eq!(encoded.iter().map(|t|t.id).collect::<Vec<_>>(),vec![27+(value-b'a') as u32,28]);
                assert_eq!(encoded.iter().map(|t|t.offsets).collect::<Vec<_>>(),vec![(0,2),(2,4)]);
            }
            let encoded=current.tokenize("az🌊").unwrap();
            assert_eq!(encoded.iter().map(|t|t.id).collect::<Vec<_>>(),vec![52,0]);
            assert_eq!(encoded.iter().map(|t|t.offsets).collect::<Vec<_>>(),vec![(0,2),(2,6)]);
        }
    }
    #[test]
    fn unigram_position_index_preserves_all_ids_duplicates_and_byte_fallback(){
        use tokenizers::{models::unigram::Unigram,Model};
        let mut vocabulary=vec![("<unk>".to_string(),0.)];
        for i in 0..1024 {vocabulary.push((format!("item{i:04}"),-1.));}
        vocabulary.push(("item0007".into(),-0.5));
        let model=Unigram::from(vocabulary,Some(0),false).unwrap();
        for current in [&model,&model.clone()] {
            for i in 0..1024 {assert_eq!(current.token_to_id(&format!("item{i:04}")),Some(if i==7 {1025}else{i+1}));}
            assert_eq!(current.token_to_id("missing"),None);
            assert_eq!(current.get_vocab().len(),1025);assert_eq!(current.get_vocab_size(),1026);
        }
        let fallback=Unigram::from(vec![("<unk>".into(),0.),("<0xF0>".into(),-1.),("<0x9F>".into(),-1.),
            ("<0x8C>".into(),-1.),("<0x8A>".into(),-1.)],Some(0),true).unwrap();
        let encoded=fallback.tokenize("🌊").unwrap();
        assert_eq!(encoded.iter().map(|t|t.id).collect::<Vec<_>>(),vec![1,2,3,4]);
        assert!(encoded.iter().all(|t|t.offsets==(0,4)));
    }
    #[test]
    #[ignore="requires the frozen UMT5 tokenizer and Wan request mounted read-only"]
    fn tokenizer_deserialization_profile_and_parity(){
        use sha2::{Digest,Sha256};
        use tokenizers::Tokenizer;
        native_core_infra::startup_trace::configure_from_env();
        let path=std::env::var("WAN_TOKENIZER_JSON").expect("WAN_TOKENIZER_JSON");
        let read_started=std::time::Instant::now();
        let bytes=native_core_infra::startup_trace::measure("wan_tokenizer_bench","read JSON",0,||
            native_core_infra::startup_trace::read(&path)).unwrap();
        let read_seconds=read_started.elapsed().as_secs_f64();
        let typed_first=std::env::var("WAN_TOKENIZER_BENCH_ORDER").as_deref()==Ok("typed-first");
        let mut values=Vec::new();
        for typed in if typed_first {[true,false]}else{[false,true]} {
            let started=std::time::Instant::now();
            let value=if typed {parse_tokenizer(&bytes).unwrap()}else {
                native_core_infra::startup_trace::measure("wan_tokenizer_bench","generic deserialize",bytes.len(),||
                    Tokenizer::from_bytes(&bytes)).unwrap()
            };
            values.push((typed,value,started.elapsed().as_secs_f64()));
        }
        let (reference,generic_seconds,typed,typed_seconds)=if typed_first {
            let (second_type,reference,generic_seconds)=values.pop().unwrap();assert!(!second_type);
            let (_,typed,typed_seconds)=values.pop().unwrap();(reference,generic_seconds,typed,typed_seconds)
        }else {
            let (second_type,typed,typed_seconds)=values.pop().unwrap();assert!(second_type);
            let (_,reference,generic_seconds)=values.pop().unwrap();(reference,generic_seconds,typed,typed_seconds)
        };
        assert_eq!(reference.to_string(false).unwrap(),typed.to_string(false).unwrap());
        assert_eq!(reference.get_vocab(true),typed.get_vocab(true));
        if std::env::var("WAN_TOKENIZER_DEEP_PROFILE").as_deref()==Ok("1") {
            let profiled=profile_tokenizer_details(&bytes).unwrap();
            assert_eq!(reference.to_string(false).unwrap(),profiled.to_string(false).unwrap());
            assert_eq!(reference.get_vocab(true),profiled.get_vocab(true));
        }
        let request=std::env::var("WAN_TOKENIZER_REQUEST").expect("WAN_TOKENIZER_REQUEST");
        let request:serde_json::Value=serde_json::from_slice(&std::fs::read(request).unwrap()).unwrap();
        let request=Request::from_json(&request).unwrap();
        let long="A fox, 世界 🌊. ".repeat(1200);
        let prompts=[request.prompt.as_str(),request.negative_prompt.as_str(),""," \n\t", "你好，世界。 🌊 café e\u{301}",
            "<pad></s><unk><extra_id_0>","&amp;amp; &#x1f30a; ﬁ ＡＢＣ", "a\u{1c}b\u{2003}c",long.as_str()];
        let mut cases=0;
        for prompt in prompts {
            for special in [false,true] {
                assert_eq!(reference.encode(prompt,special).unwrap(),typed.encode(prompt,special).unwrap());cases+=1;
            }
        }
        let reference=configure_tokenizer(reference).unwrap();let typed=configure_tokenizer(typed).unwrap();
        for prompt in prompts {
            match (encode_tokens(&reference,prompt),encode_tokens(&typed,prompt)) {
                (Ok(reference),Ok(typed))=>assert_eq!(reference,typed),
                (Err(reference),Err(typed))=>assert_eq!(reference.to_string(),typed.to_string()),
                _=>panic!("frontend acceptance changed for {prompt:?}"),
            }
            cases+=1;
            assert_eq!(reference.encode(prompt,true).unwrap(),typed.encode(prompt,true).unwrap());cases+=1;
        }
        println!("WAN21_TOKENIZER_PROFILE {}",serde_json::json!({"tokenizer_bytes":bytes.len(),
            "tokenizer_sha256":format!("{:x}",Sha256::digest(&bytes)),"model":"Unigram",
            "read_seconds":read_seconds,"generic_seconds":generic_seconds,"typed_seconds":typed_seconds,
            "order":if typed_first {"typed-first"}else{"generic-first"},"encoding_cases":cases,
            "serialized_tokenizer_exact":true,"vocabulary_exact":true,"encodings_exact":true}));
    }
}
