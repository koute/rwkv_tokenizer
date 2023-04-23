use pyo3::prelude::*;
use ahash::AHashMap as HashMap;
use ahash::AHashSet as HashSet;

struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u16>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

lazy_static::lazy_static! {
    static ref TOKENIZER: Tokenizer = {
        let raw_data = include_bytes!("../rwkv_vocab.bin");
        let list: Vec<(Vec<u8>, u16)> = speedy::Readable::read_from_buffer(&raw_data[..]).unwrap();
        let mut first_bytes_to_len = Vec::new();
        first_bytes_to_len.resize(u16::MAX as usize, 2);

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = HashSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        token_index_to_bytes.resize_with(u16::MAX as usize, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                let max_length = &mut first_bytes_to_len[key];
                if token_bytes.len() > *max_length {
                    *max_length = token_bytes.len();
                }

                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths.into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            }).collect();

        Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        }
    };
}

#[pyfunction]
pub fn encode(input: &str) -> Vec<u16> {
    let mut input = input.as_bytes();
    let tokenizer = &*TOKENIZER;
    let mut output = Vec::new();
    'next_token: while !input.is_empty() {
        let lengths;
        if input.len() >= 2 {
            let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
            lengths = &tokenizer.first_bytes_to_lengths[key][..];
        } else {
            lengths = &[1][..];
        }

        for &length in lengths {
            let length = length as usize;
            if length > input.len() {
                continue;
            }

            if let Some(&token_index) = tokenizer.bytes_to_token_index.get(&input[..length]) {
                output.push(token_index);
                input = &input[length..];
                continue 'next_token;
            }
        }

        unreachable!("no matching token found");
    }

    output
}

#[pyfunction]
pub fn decode(tokens: Vec<u16>) -> String {
    let tokenizer = &*TOKENIZER;
    let mut output = Vec::new();
    output.reserve(tokens.len());
    for token in tokens {
        output.extend_from_slice(&tokenizer.token_index_to_bytes[token as usize]);
    }

    String::from_utf8(output).unwrap()
}

#[pymodule]
fn librwkv_tokenizer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    Ok(())
}
