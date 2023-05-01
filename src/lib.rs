use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyRuntimeError};

mod tokenizer;

#[pyclass]
struct Tokenizer {
    tokenizer: crate::tokenizer::Tokenizer
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(vocabulary_path: &str) -> PyResult<Self> {
        let vocabulary_json = std::fs::read_to_string(vocabulary_path).map_err(PyIOError::new_err)?;
        Ok(Self {
            tokenizer: crate::tokenizer::Tokenizer::new(&vocabulary_json)
                .map_err(|error| ToString::to_string(&error))
                .map_err(PyRuntimeError::new_err)?
        })
    }

    fn encode(&self, input: &str) -> PyResult<Vec<u16>> {
        self.tokenizer.encode(input.as_bytes())
            .map_err(|error| ToString::to_string(&error))
            .map_err(PyRuntimeError::new_err)
    }

    fn decode(&self, tokens: Vec<u16>) -> PyResult<String> {
        let bytes = self.tokenizer.decode(&tokens)
            .map_err(|error| ToString::to_string(&error))
            .map_err(PyRuntimeError::new_err)?;

        String::from_utf8(bytes)
            .map_err(|_| PyRuntimeError::new_err("tokens do not represent a valid UTF-8 string"))
    }
}

#[pymodule]
fn librwkv_tokenizer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
