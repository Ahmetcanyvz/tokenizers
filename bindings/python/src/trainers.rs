use std::sync::{Arc, RwLock};
use std::fs::File;

use ahash::AHashMap;
use compact_str::CompactString;

use crate::models::PyModel;
use crate::tokenizer::{PyAddedToken, PyTokenizer};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};
use tk::models::TrainerWrapper;
use tk::Trainer;
use tokenizers as tk;

/// Base class for all trainers
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of a
/// Trainer will return an instance of this class when instantiated.
#[pyclass(
    module = "tokenizers.trainers",
    name = "Trainer",
    subclass,
    from_py_object
)]
#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct PyTrainer {
    pub trainer: Arc<RwLock<TrainerWrapper>>,
}

impl PyTrainer {
    #[cfg(test)]
    pub(crate) fn new(trainer: Arc<RwLock<TrainerWrapper>>) -> Self {
        PyTrainer { trainer }
    }
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let base = self.clone();
        Ok(match *self.trainer.as_ref().read().unwrap() {
            TrainerWrapper::BpeTrainer(_) => Py::new(py, (PyBpeTrainer {}, base))?.into_any(),
            TrainerWrapper::WordPieceTrainer(_) => {
                Py::new(py, (PyWordPieceTrainer {}, base))?.into_any()
            }
            TrainerWrapper::WordLevelTrainer(_) => {
                Py::new(py, (PyWordLevelTrainer {}, base))?.into_any()
            }
            TrainerWrapper::UnigramTrainer(_) => {
                Py::new(py, (PyUnigramTrainer {}, base))?.into_any()
            }
        })
    }
}
#[pymethods]
impl PyTrainer {
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let data = serde_json::to_string(&self.trainer).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle PyTrainer: {e}"
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                let unpickled = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle PyTrainer: {e}"
                    ))
                })?;
                self.trainer = unpickled;
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }
}

impl Trainer for PyTrainer {
    type Model = PyModel;

    fn should_show_progress(&self) -> bool {
        self.trainer.read().unwrap().should_show_progress()
    }

    fn train(&self, model: &mut PyModel) -> tk::Result<Vec<tk::AddedToken>> {
        self.trainer
            .read()
            .unwrap()
            .train(&mut model.model.write().unwrap())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> tk::Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> tk::Result<Vec<String>> + Sync,
    {
        self.trainer.write().unwrap().feed(iterator, process)
    }
}

impl<I> From<I> for PyTrainer
where
    I: Into<TrainerWrapper>,
{
    fn from(trainer: I) -> Self {
        PyTrainer {
            trainer: Arc::new(RwLock::new(trainer.into())),
        }
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref trainer) = *super_.trainer.read().unwrap() {
            trainer.$($name)+
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref mut trainer) = *super_.trainer.write().unwrap() {
            trainer.$name = $value;
        }
    }};
    ($self: ident, $variant: ident, @$name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref mut trainer) = *super_.trainer.write().unwrap() {
            trainer.$name($value);
        }
    }};
}

/// Trainer capable of training a BPE model
///
/// Args:
///     vocab_size (:obj:`int`, `optional`):
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency (:obj:`int`, `optional`):
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress (:obj:`bool`, `optional`):
///         Whether to show progress bars while training.
///
///     special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
///         A list of special tokens the model should know of.
///
///     limit_alphabet (:obj:`int`, `optional`):
///         The maximum different characters to keep in the alphabet.
///
///     initial_alphabet (:obj:`List[str]`, `optional`):
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     continuing_subword_prefix (:obj:`str`, `optional`):
///         A prefix to be used for every subword that is not a beginning-of-word.
///
///     end_of_word_suffix (:obj:`str`, `optional`):
///         A suffix to be used for every subword that is a end-of-word.
///
///     max_token_length (:obj:`int`, `optional`):
///         Prevents creating tokens longer than the specified size.
///         This can help with reducing polluting your vocabulary with
///         highly repetitive tokens like `======` for wikipedia
///
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "BpeTrainer")]
pub struct PyBpeTrainer {}
#[pymethods]
impl PyBpeTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, BpeTrainer, vocab_size)
    }

    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, BpeTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, BpeTrainer, min_frequency)
    }

    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, BpeTrainer, min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, BpeTrainer, show_progress)
    }

    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, BpeTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            BpeTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }

    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            BpeTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_limit_alphabet(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, BpeTrainer, limit_alphabet)
    }

    #[setter]
    fn set_limit_alphabet(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, BpeTrainer, limit_alphabet, limit);
    }

    #[getter]
    fn get_max_token_length(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, BpeTrainer, max_token_length)
    }

    #[setter]
    fn set_max_token_length(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, BpeTrainer, max_token_length, limit);
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            BpeTrainer,
            initial_alphabet.iter().map(|c| c.to_string()).collect()
        )
    }

    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            BpeTrainer,
            initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BpeTrainer, continuing_subword_prefix.clone())
    }

    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, prefix: Option<String>) {
        setter!(self_, BpeTrainer, continuing_subword_prefix, prefix);
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BpeTrainer, end_of_word_suffix.clone())
    }

    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, suffix: Option<String>) {
        setter!(self_, BpeTrainer, end_of_word_suffix, suffix);
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix=None, end_of_word_suffix=None, max_token_length=None, words={})"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::bpe::BpeTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "max_token_length" => builder = builder.max_token_length(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {key}"),
                };
            }
        }
        Ok((PyBpeTrainer {}, builder.build().into()))
    }
}

/// Trainer capable of training a WordPiece model
///
/// Args:
///     vocab_size (:obj:`int`, `optional`):
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency (:obj:`int`, `optional`):
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress (:obj:`bool`, `optional`):
///         Whether to show progress bars while training.
///
///     special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
///         A list of special tokens the model should know of.
///
///     limit_alphabet (:obj:`int`, `optional`):
///         The maximum different characters to keep in the alphabet.
///
///     initial_alphabet (:obj:`List[str]`, `optional`):
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     continuing_subword_prefix (:obj:`str`, `optional`):
///         A prefix to be used for every subword that is not a beginning-of-word.
///
///     end_of_word_suffix (:obj:`str`, `optional`):
///         A suffix to be used for every subword that is a end-of-word.
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "WordPieceTrainer")]
pub struct PyWordPieceTrainer {}
#[pymethods]
impl PyWordPieceTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, WordPieceTrainer, vocab_size())
    }

    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, WordPieceTrainer, @set_vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, WordPieceTrainer, min_frequency())
    }

    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, WordPieceTrainer, @set_min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, WordPieceTrainer, show_progress())
    }

    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, WordPieceTrainer, @set_show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            WordPieceTrainer,
            special_tokens()
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }

    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            WordPieceTrainer,
            @set_special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_limit_alphabet(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, WordPieceTrainer, limit_alphabet())
    }

    #[setter]
    fn set_limit_alphabet(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, WordPieceTrainer, @set_limit_alphabet, limit);
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            WordPieceTrainer,
            initial_alphabet().iter().map(|c| c.to_string()).collect()
        )
    }

    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            WordPieceTrainer,
            @set_initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, WordPieceTrainer, continuing_subword_prefix().clone())
    }

    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, prefix: Option<String>) {
        setter!(self_, WordPieceTrainer, @set_continuing_subword_prefix, prefix);
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, WordPieceTrainer, end_of_word_suffix().clone())
    }

    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, suffix: Option<String>) {
        setter!(self_, WordPieceTrainer, @set_end_of_word_suffix, suffix);
    }

    #[new]
    #[pyo3(
        signature = (** kwargs),
        text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix=\"##\", end_of_word_suffix=None)"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {key}"),
                };
            }
        }

        Ok((PyWordPieceTrainer {}, builder.build().into()))
    }
}

/// Trainer capable of training a WorldLevel model
///
/// Args:
///     vocab_size (:obj:`int`, `optional`):
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency (:obj:`int`, `optional`):
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress (:obj:`bool`, `optional`):
///         Whether to show progress bars while training.
///
///     special_tokens (:obj:`List[Union[str, AddedToken]]`):
///         A list of special tokens the model should know of.
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "WordLevelTrainer")]
pub struct PyWordLevelTrainer {}
#[pymethods]
impl PyWordLevelTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, WordLevelTrainer, vocab_size)
    }

    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, WordLevelTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, WordLevelTrainer, min_frequency)
    }

    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, WordLevelTrainer, min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, WordLevelTrainer, show_progress)
    }

    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, WordLevelTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            WordLevelTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }

    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            WordLevelTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[])"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::wordlevel::WordLevelTrainer::builder();

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => {
                        builder.vocab_size(val.extract()?);
                    }
                    "min_frequency" => {
                        builder.min_frequency(val.extract()?);
                    }
                    "show_progress" => {
                        builder.show_progress(val.extract()?);
                    }
                    "special_tokens" => {
                        builder.special_tokens(
                            val.cast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    _ => println!("Ignored unknown kwargs option {key}"),
                }
            }
        }

        Ok((
            PyWordLevelTrainer {},
            builder
                .build()
                .expect("WordLevelTrainerBuilder cannot fail")
                .into(),
        ))
    }
}

/// Trainer capable of training a Unigram model
///
/// Args:
///     vocab_size (:obj:`int`):
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     show_progress (:obj:`bool`):
///         Whether to show progress bars while training.
///
///     special_tokens (:obj:`List[Union[str, AddedToken]]`):
///         A list of special tokens the model should know of.
///
///     initial_alphabet (:obj:`List[str]`):
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     shrinking_factor (:obj:`float`):
///         The shrinking factor used at each step of the training to prune the
///         vocabulary.
///
///     unk_token (:obj:`str`):
///         The token used for out-of-vocabulary tokens.
///
///     max_piece_length (:obj:`int`):
///         The maximum length of a given token.
///
///     n_sub_iterations (:obj:`int`):
///         The number of iterations of the EM algorithm to perform before
///         pruning the vocabulary.
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "UnigramTrainer")]
pub struct PyUnigramTrainer {}
#[pymethods]
impl PyUnigramTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> u32 {
        getter!(self_, UnigramTrainer, vocab_size)
    }

    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: u32) {
        setter!(self_, UnigramTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, UnigramTrainer, show_progress)
    }

    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, UnigramTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            UnigramTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }

    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            UnigramTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            UnigramTrainer,
            initial_alphabet.iter().map(|c| c.to_string()).collect()
        )
    }

    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            UnigramTrainer,
            initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=8000, show_progress=True, special_tokens=[], initial_alphabet=[], shrinking_factor=0.75, unk_token=None, max_piece_length=16, n_sub_iterations=2)"
    )]
    pub fn new(kwargs: Option<Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::unigram::UnigramTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder.vocab_size(val.extract()?),
                    "show_progress" => builder.show_progress(val.extract()?),
                    "n_sub_iterations" => builder.n_sub_iterations(val.extract()?),
                    "shrinking_factor" => builder.shrinking_factor(val.extract()?),
                    "unk_token" => builder.unk_token(val.extract()?),
                    "max_piece_length" => builder.max_piece_length(val.extract()?),
                    "seed_size" => builder.seed_size(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        )
                    }
                    "special_tokens" => builder.special_tokens(
                        val.cast::<PyList>()?
                            .into_iter()
                            .map(|token| {
                                if let Ok(content) = token.extract::<String>() {
                                    Ok(PyAddedToken::from(content, Some(true)).get_token())
                                } else if let Ok(mut token) =
                                    token.extract::<PyRefMut<PyAddedToken>>()
                                {
                                    token.special = true;
                                    Ok(token.get_token())
                                } else {
                                    Err(exceptions::PyTypeError::new_err(
                                        "special_tokens must be a List[Union[str, AddedToken]]",
                                    ))
                                }
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    ),
                    _ => {
                        println!("Ignored unknown kwargs option {key}");
                        &mut builder
                    }
                };
            }
        }

        let trainer: tokenizers::models::unigram::UnigramTrainer =
            builder.build().map_err(|e| {
                exceptions::PyException::new_err(format!("Cannot build UnigramTrainer: {e}"))
            })?;
        Ok((PyUnigramTrainer {}, trainer.into()))
    }
}

use tk::models::bpe::parity_utils;

fn map_tk_err(
    result: tk::tokenizer::Result<AHashMap<CompactString, u64>>,
) -> PyResult<AHashMap<CompactString, u64>> {
    result.map_err(|e| exceptions::PyRuntimeError::new_err(format!("{}", e)))
}

/// Trainer for parity-aware BPE that ensures cross-lingual fairness in tokenization.
///
/// Unlike standard BPE, this trainer takes per-language training files and balances
/// merge operations across languages using a development set or target compression ratios.
///
/// Args:
///     num_merges (:obj:`int`, `optional`):
///         Number of BPE merge operations to perform. Defaults to ``32000``.
///
///     variant (:obj:`str`, `optional`):
///         Algorithm variant: ``"base"`` (default) or ``"window"`` (moving-window balancing).
///
///     min_frequency (:obj:`int`, `optional`):
///         Minimum pair frequency to merge. Defaults to ``0``.
///
///     global_merges (:obj:`int`, `optional`):
///         Number of initial standard BPE merges before switching to parity mode. Defaults to ``0``.
///
///     window_size (:obj:`int`, `optional`):
///         Window size for the ``"window"`` variant. Defaults to ``100``.
///
///     alpha (:obj:`float`, `optional`):
///         Alpha parameter for the ``"window"`` variant. Defaults to ``2.0``.
///
///     total_symbols (:obj:`bool`, `optional`):
///         If True, subtract unique character count from ``num_merges``. Defaults to ``False``.
///
/// Example::
///
///     from tokenizers import Tokenizer
///     from tokenizers.models import BPE
///     from tokenizers import pre_tokenizers
///     from tokenizers.trainers import ParityBpeTrainer
///
///     tokenizer = Tokenizer(BPE())
///     tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
///
///     trainer = ParityBpeTrainer(num_merges=32000, variant="base")
///     trainer.train(tokenizer, train_files=["train_en.txt", "train_de.txt"],
///                   dev_files=["dev_en.txt", "dev_de.txt"])
///     output = tokenizer.encode("Hello world")
///
#[pyclass(module = "tokenizers.trainers", name = "ParityBpeTrainer")]
pub struct PyParityBpeTrainer {
    num_merges: usize,
    variant: String,
    min_frequency: u64,
    ratio: Option<Vec<f64>>,
    global_merges: usize,
    window_size: usize,
    alpha: f64,
    total_symbols: bool,
    special_tokens: Vec<tk::AddedToken>,
    show_progress: bool,
    limit_alphabet: Option<usize>,
    initial_alphabet: Vec<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
}

impl Default for PyParityBpeTrainer {
    fn default() -> Self {
        Self {
            num_merges: 32000,
            variant: "base".to_string(),
            min_frequency: 0,
            ratio: None,
            global_merges: 0,
            window_size: 100,
            alpha: 2.0,
            total_symbols: false,
            special_tokens: Vec::new(),
            show_progress: true,
            limit_alphabet: None,
            initial_alphabet: Vec::new(),
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            max_token_length: None,
        }
    }
}

impl PyParityBpeTrainer {
    /// Build a Rust `ParityBpeTrainerBuilder` from the current Python-side settings.
    fn make_builder(
        &self,
        parity_variant: tk::models::bpe::ParityVariant,
    ) -> tk::models::bpe::ParityBpeTrainerBuilder {
        use tk::models::bpe::ParityBpeTrainer as RustTrainer;

        let mut builder = RustTrainer::builder()
            .min_frequency(self.min_frequency)
            .num_merges(self.num_merges)
            .show_progress(self.show_progress)
            .variant(parity_variant)
            .global_merges(self.global_merges)
            .window_size(self.window_size)
            .alpha(self.alpha)
            .total_symbols(self.total_symbols)
            .special_tokens(self.special_tokens.clone());

        if let Some(limit) = self.limit_alphabet {
            builder = builder.limit_alphabet(limit);
        }
        if !self.initial_alphabet.is_empty() {
            builder = builder.initial_alphabet(
                self.initial_alphabet.iter().copied().collect(),
            );
        }
        if let Some(ref prefix) = self.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix.clone());
        }
        if let Some(ref suffix) = self.end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix.clone());
        }
        builder = builder.max_token_length(self.max_token_length);
        builder
    }
}

#[pymethods]
impl PyParityBpeTrainer {
    #[new]
    #[pyo3(signature = (
        num_merges = 32000,
        variant = "base",
        min_frequency = 0,
        ratio = None,
        global_merges = 0,
        window_size = 100,
        alpha = 2.0,
        total_symbols = false,
        special_tokens = None,
        show_progress = true,
        limit_alphabet = None,
        initial_alphabet = None,
        continuing_subword_prefix = None,
        end_of_word_suffix = None,
        max_token_length = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_merges: usize,
        variant: &str,
        min_frequency: u64,
        ratio: Option<Vec<f64>>,
        global_merges: usize,
        window_size: usize,
        alpha: f64,
        total_symbols: bool,
        special_tokens: Option<&Bound<'_, PyList>>,
        show_progress: bool,
        limit_alphabet: Option<usize>,
        initial_alphabet: Option<Vec<char>>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        max_token_length: Option<usize>,
    ) -> PyResult<Self> {
        match variant {
            "base" | "window" => {}
            _ => return Err(exceptions::PyValueError::new_err(
                format!("Unknown variant '{}'. Use 'base' or 'window'.", variant)
            )),
        }

        let parsed_special_tokens = if let Some(tokens) = special_tokens {
            tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "special_tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            Vec::new()
        };

        Ok(PyParityBpeTrainer {
            num_merges,
            variant: variant.to_string(),
            min_frequency,
            ratio,
            global_merges,
            window_size,
            alpha,
            total_symbols,
            special_tokens: parsed_special_tokens,
            show_progress,
            limit_alphabet,
            initial_alphabet: initial_alphabet.unwrap_or_default(),
            continuing_subword_prefix,
            end_of_word_suffix,
            max_token_length,
        })
    }

    /// Train a user-configured tokenizer with parity-aware BPE in-place.
    ///
    /// The tokenizer's normalizer and pre-tokenizer (if set) are used during
    /// training, matching the standard ``tokenizer.train(files, trainer)``
    /// workflow.  After training the model is set on the tokenizer directly.
    ///
    /// Args:
    ///     tokenizer (:class:`~tokenizers.Tokenizer`):
    ///         A tokenizer instance to train.  Its pre-tokenizer (and optionally
    ///         normalizer) should already be configured.
    ///
    ///     train_files (:obj:`List[str]`, `optional`):
    ///         List of training file paths, one per language.
    ///         Either ``train_files`` or ``config`` must be provided.
    ///
    ///     dev_files (:obj:`List[str]`, `optional`):
    ///         List of development file paths for parity computation.
    ///         If provided, dev-set token lengths drive language selection
    ///         (overrides ``ratio``).
    ///
    ///     ratio (:obj:`List[float]`, `optional`):
    ///         Target compression ratios per language (alternative to ``dev_files``).
    ///
    ///     output (:obj:`str`, `optional`):
    ///         Path to write merge rules to a file.
    ///
    ///     config (:obj:`str`, `optional`):
    ///         Path to a JSON config file defining language groups with input files,
    ///         compression ratios, and optional dev files.  When provided, overrides
    ///         ``train_files``, ``dev_files``, and ``ratio``.
    #[pyo3(signature = (tokenizer, train_files = None, dev_files = None, ratio = None, output = None, config = None))]
    fn train(
        &self,
        tokenizer: &mut PyTokenizer,
        train_files: Option<Vec<String>>,
        dev_files: Option<Vec<String>>,
        ratio: Option<Vec<f64>>,
        output: Option<String>,
        config: Option<String>,
    ) -> PyResult<()> {
        use tk::models::bpe::{ParityVariant, TrainingConfig, BPE};

        let parity_variant = match self.variant.as_str() {
            "base" => ParityVariant::Base,
            "window" => ParityVariant::Window,
            _ => unreachable!(),
        };

        // Extract normalizer and pre-tokenizer from the user-configured tokenizer
        let normalizer = tokenizer.tokenizer.get_normalizer().cloned();
        let pre_tokenizer = tokenizer.tokenizer.get_pre_tokenizer().cloned();
        let norm_ref: Option<&dyn tk::Normalizer> = normalizer.as_ref().map(|n| n as &dyn tk::Normalizer);
        let pretok_ref: Option<&dyn tk::PreTokenizer> = pre_tokenizer.as_ref().map(|p| p as &dyn tk::PreTokenizer);

        if let Some(ref cfg_path) = config {
            // Config-driven training
            let training_config = TrainingConfig::from_file(cfg_path)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Cannot load config {}: {}", cfg_path, e)))?;

            let has_dev = training_config.has_dev();

            let mut builder = self.make_builder(parity_variant);

            // Only use ratios if no dev files are present in the config
            if !has_dev {
                builder = builder.ratio(training_config.ratios());
            }

            let mut trainer = builder.build();

            // Feed training data
            for (lang_idx, lang_cfg) in training_config.languages.iter().enumerate() {
                let mut merged_counts: AHashMap<CompactString, u64> = AHashMap::new();
                for file_path in &lang_cfg.input {
                    let file_counts = map_tk_err(parity_utils::pre_tokenize_auto(
                        file_path,
                        &lang_cfg.text_column,
                        norm_ref,
                        pretok_ref,
                    ))?;
                    for (word, count) in file_counts {
                        *merged_counts.entry(word).or_default() += count;
                    }
                }
                trainer.feed_language(lang_idx, merged_counts);
            }

            // Feed dev data from config
            if has_dev {
                for (lang_idx, lang_cfg) in training_config.languages.iter().enumerate() {
                    if let Some(ref dev_paths) = lang_cfg.dev {
                        let mut dev_counts: AHashMap<CompactString, u64> = AHashMap::new();
                        for file_path in dev_paths {
                            let file_counts = map_tk_err(parity_utils::pre_tokenize_auto(
                                file_path,
                                &lang_cfg.text_column,
                                norm_ref,
                                pretok_ref,
                            ))?;
                            for (word, count) in file_counts {
                                *dev_counts.entry(word).or_default() += count;
                            }
                        }
                        trainer.feed_dev_language(lang_idx, dev_counts);
                    }
                }
            }

            // Train
            let mut model = BPE::default();
            let (special_tokens, merge_strings) = trainer
                .do_train(&mut model)
                .map_err(|e| exceptions::PyRuntimeError::new_err(format!("Training error: {}", e)))?;

            // Write output file if requested
            if let Some(ref out_path) = output {
                use std::io::Write;
                let mut out = File::create(out_path)
                    .map_err(|e| exceptions::PyIOError::new_err(format!("Cannot create {}: {}", out_path, e)))?;
                writeln!(out, "#version: 0.2").map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
                for merge_line in &merge_strings {
                    writeln!(out, "{}", merge_line).map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
                }
            }

            // Set the trained model on the tokenizer in-place
            let py_model: PyModel = model.into();
            tokenizer.tokenizer.with_model(py_model);
            tokenizer.tokenizer.add_special_tokens(&special_tokens);
        } else {
            // File-per-language mode
            let train_files = train_files.ok_or_else(|| {
                exceptions::PyValueError::new_err("Either 'train_files' or 'config' must be provided")
            })?;

            if train_files.is_empty() {
                return Err(exceptions::PyValueError::new_err("train_files must not be empty"));
            }

            // Dev files take precedence over ratios
            let has_dev = dev_files.as_ref().is_some_and(|d| !d.is_empty());
            let effective_ratio = if has_dev {
                None
            } else {
                ratio.or_else(|| self.ratio.clone())
            };

            let mut builder = self.make_builder(parity_variant);

            if let Some(r) = effective_ratio {
                builder = builder.ratio(r);
            }

            let mut trainer = builder.build();

            // Feed training data
            for (lang, path) in train_files.iter().enumerate() {
                let word_counts = map_tk_err(parity_utils::pre_tokenize_file(path, norm_ref, pretok_ref))?;
                trainer.feed_language(lang, word_counts);
            }

            // Feed dev data
            if let Some(ref dev) = dev_files {
                if dev.len() != train_files.len() {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "dev_files length ({}) must match train_files length ({})",
                        dev.len(),
                        train_files.len()
                    )));
                }
                for (lang, path) in dev.iter().enumerate() {
                    let word_counts = map_tk_err(parity_utils::pre_tokenize_file(path, norm_ref, pretok_ref))?;
                    trainer.feed_dev_language(lang, word_counts);
                }
            }

            // Train
            let mut model = BPE::default();
            let (special_tokens, merge_strings) = trainer
                .do_train(&mut model)
                .map_err(|e| exceptions::PyRuntimeError::new_err(format!("Training error: {}", e)))?;

            // Write output file if requested
            if let Some(ref out_path) = output {
                use std::io::Write;
                let mut out = File::create(out_path)
                    .map_err(|e| exceptions::PyIOError::new_err(format!("Cannot create {}: {}", out_path, e)))?;
                writeln!(out, "#version: 0.2").map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
                for merge_line in &merge_strings {
                    writeln!(out, "{}", merge_line).map_err(|e| exceptions::PyIOError::new_err(e.to_string()))?;
                }
            }

            // Set the trained model on the tokenizer in-place
            let py_model: PyModel = model.into();
            tokenizer.tokenizer.with_model(py_model);
            tokenizer.tokenizer.add_special_tokens(&special_tokens);
        }

        Ok(())
    }

    #[getter]
    fn get_num_merges(&self) -> usize { self.num_merges }

    #[setter]
    fn set_num_merges(&mut self, v: usize) { self.num_merges = v; }

    #[getter]
    fn get_variant(&self) -> &str { &self.variant }

    #[getter]
    fn get_min_frequency(&self) -> u64 { self.min_frequency }

    #[setter]
    fn set_min_frequency(&mut self, v: u64) { self.min_frequency = v; }

    #[getter]
    fn get_global_merges(&self) -> usize { self.global_merges }

    #[setter]
    fn set_global_merges(&mut self, v: usize) { self.global_merges = v; }

    #[getter]
    fn get_window_size(&self) -> usize { self.window_size }

    #[setter]
    fn set_window_size(&mut self, v: usize) { self.window_size = v; }

    #[getter]
    fn get_alpha(&self) -> f64 { self.alpha }

    #[setter]
    fn set_alpha(&mut self, v: f64) { self.alpha = v; }

    #[getter]
    fn get_total_symbols(&self) -> bool { self.total_symbols }

    #[setter]
    fn set_total_symbols(&mut self, v: bool) { self.total_symbols = v; }

    #[getter]
    fn get_show_progress(&self) -> bool { self.show_progress }

    #[setter]
    fn set_show_progress(&mut self, v: bool) { self.show_progress = v; }

    #[getter]
    fn get_special_tokens(&self) -> Vec<PyAddedToken> {
        self.special_tokens.iter().map(|tok| tok.clone().into()).collect()
    }

    #[setter]
    fn set_special_tokens(&mut self, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        self.special_tokens = special_tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(tk::tokenizer::AddedToken::from(content, true))
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.special = true;
                    Ok(token.get_token())
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "special_tokens must be a List[Union[str, AddedToken]]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(())
    }

    #[getter]
    fn get_limit_alphabet(&self) -> Option<usize> { self.limit_alphabet }

    #[setter]
    fn set_limit_alphabet(&mut self, v: Option<usize>) { self.limit_alphabet = v; }

    #[getter]
    fn get_initial_alphabet(&self) -> Vec<String> {
        self.initial_alphabet.iter().map(|c| c.to_string()).collect()
    }

    #[setter]
    fn set_initial_alphabet(&mut self, alphabet: Vec<char>) {
        self.initial_alphabet = alphabet;
    }

    #[getter]
    fn get_continuing_subword_prefix(&self) -> Option<&str> {
        self.continuing_subword_prefix.as_deref()
    }

    #[setter]
    fn set_continuing_subword_prefix(&mut self, v: Option<String>) {
        self.continuing_subword_prefix = v;
    }

    #[getter]
    fn get_end_of_word_suffix(&self) -> Option<&str> {
        self.end_of_word_suffix.as_deref()
    }

    #[setter]
    fn set_end_of_word_suffix(&mut self, v: Option<String>) {
        self.end_of_word_suffix = v;
    }

    #[getter]
    fn get_max_token_length(&self) -> Option<usize> { self.max_token_length }

    #[setter]
    fn set_max_token_length(&mut self, v: Option<usize>) { self.max_token_length = v; }

    fn __repr__(&self) -> String {
        format!(
            "ParityBpeTrainer(num_merges={}, variant=\"{}\", min_frequency={}, \
             global_merges={}, window_size={}, alpha={}, total_symbols={})",
            self.num_merges, self.variant, self.min_frequency,
            self.global_merges, self.window_size, self.alpha, self.total_symbols,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("num_merges", self.num_merges)?;
        dict.set_item("variant", &self.variant)?;
        dict.set_item("min_frequency", self.min_frequency)?;
        dict.set_item("global_merges", self.global_merges)?;
        dict.set_item("window_size", self.window_size)?;
        dict.set_item("alpha", self.alpha)?;
        dict.set_item("total_symbols", self.total_symbols)?;
        dict.set_item("show_progress", self.show_progress)?;
        dict.set_item(
            "ratio",
            self.ratio.as_ref().map(|r| PyList::new(py, r).unwrap()),
        )?;
        let special: Vec<String> = self
            .special_tokens
            .iter()
            .map(|t| t.content.clone())
            .collect();
        dict.set_item("special_tokens", PyList::new(py, &special)?)?;
        dict.set_item("limit_alphabet", self.limit_alphabet)?;
        let alphabet_strs: Vec<String> = self.initial_alphabet.iter().map(|c| c.to_string()).collect();
        dict.set_item("initial_alphabet", PyList::new(py, &alphabet_strs)?)?;
        dict.set_item("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        dict.set_item("end_of_word_suffix", &self.end_of_word_suffix)?;
        dict.set_item("max_token_length", self.max_token_length)?;
        Ok(dict.into_any().unbind())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let dict = state.cast_bound::<PyDict>(py)?;
        self.num_merges = dict
            .get_item("num_merges")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("num_merges"))?
            .extract()?;
        self.variant = dict
            .get_item("variant")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("variant"))?
            .extract()?;
        self.min_frequency = dict
            .get_item("min_frequency")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("min_frequency"))?
            .extract()?;
        self.global_merges = dict
            .get_item("global_merges")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("global_merges"))?
            .extract()?;
        self.window_size = dict
            .get_item("window_size")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("window_size"))?
            .extract()?;
        self.alpha = dict
            .get_item("alpha")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("alpha"))?
            .extract()?;
        self.total_symbols = dict
            .get_item("total_symbols")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("total_symbols"))?
            .extract()?;
        self.show_progress = dict
            .get_item("show_progress")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("show_progress"))?
            .extract()?;
        self.ratio = dict
            .get_item("ratio")?
            .and_then(|v| if v.is_none() { None } else { Some(v.extract().ok()?) });
        let special_strs: Vec<String> = dict
            .get_item("special_tokens")?
            .ok_or_else(|| exceptions::PyKeyError::new_err("special_tokens"))?
            .extract()?;
        self.special_tokens = special_strs
            .into_iter()
            .map(|s| tk::tokenizer::AddedToken::from(s, true))
            .collect();
        self.limit_alphabet = dict
            .get_item("limit_alphabet")?
            .and_then(|v| if v.is_none() { None } else { Some(v.extract().ok()?) });
        self.initial_alphabet = dict
            .get_item("initial_alphabet")?
            .and_then(|v| v.extract::<Vec<String>>().ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|s| s.chars().next())
            .collect();
        self.continuing_subword_prefix = dict
            .get_item("continuing_subword_prefix")?
            .and_then(|v| if v.is_none() { None } else { Some(v.extract().ok()?) });
        self.end_of_word_suffix = dict
            .get_item("end_of_word_suffix")?
            .and_then(|v| if v.is_none() { None } else { Some(v.extract().ok()?) });
        self.max_token_length = dict
            .get_item("max_token_length")?
            .and_then(|v| if v.is_none() { None } else { Some(v.extract().ok()?) });
        Ok(())
    }
}

/// Trainers Module
#[pymodule]
pub mod trainers {
    #[pymodule_export]
    pub use super::PyBpeTrainer;
    #[pymodule_export]
    pub use super::PyTrainer;
    #[pymodule_export]
    pub use super::PyUnigramTrainer;
    #[pymodule_export]
    pub use super::PyWordLevelTrainer;
    #[pymodule_export]
    pub use super::PyWordPieceTrainer;
    #[pymodule_export]
    pub use super::PyParityBpeTrainer;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk::models::bpe::trainer::BpeTrainer;

    #[test]
    fn get_subtype() {
        Python::attach(|py| {
            let py_trainer = PyTrainer::new(Arc::new(RwLock::new(BpeTrainer::default().into())));
            let py_bpe = py_trainer.get_as_subtype(py).unwrap();
            assert_eq!("BpeTrainer", py_bpe.bind(py).get_type().qualname().unwrap());
        })
    }
}
