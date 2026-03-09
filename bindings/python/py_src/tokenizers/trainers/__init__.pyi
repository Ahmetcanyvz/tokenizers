# Generated content DO NOT EDIT
class Trainer:
    """
    Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.
    """

class BpeTrainer(Trainer):
    """
    Trainer capable of training a BPE model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.

        max_token_length (:obj:`int`, `optional`):
            Prevents creating tokens longer than the specified size.
            This can help with reducing polluting your vocabulary with
            highly repetitive tokens like `======` for wikipedia

    """
    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        max_token_length=None,
        words={},
    ):
        pass

class UnigramTrainer(Trainer):
    """
    Trainer capable of training a Unigram model

    Args:
        vocab_size (:obj:`int`):
            The size of the final vocabulary, including all tokens and alphabet.

        show_progress (:obj:`bool`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.

        initial_alphabet (:obj:`List[str]`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        shrinking_factor (:obj:`float`):
            The shrinking factor used at each step of the training to prune the
            vocabulary.

        unk_token (:obj:`str`):
            The token used for out-of-vocabulary tokens.

        max_piece_length (:obj:`int`):
            The maximum length of a given token.

        n_sub_iterations (:obj:`int`):
            The number of iterations of the EM algorithm to perform before
            pruning the vocabulary.
    """
    def __init__(
        self,
        vocab_size=8000,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=[],
        shrinking_factor=0.75,
        unk_token=None,
        max_piece_length=16,
        n_sub_iterations=2,
    ):
        pass

class WordLevelTrainer(Trainer):
    """
    Trainer capable of training a WorldLevel model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.
    """
    def __init__(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[]):
        pass

class ParityBpeTrainer:
    """
    Trainer for parity-aware BPE that ensures cross-lingual fairness in tokenization.

    Unlike standard BPE, this trainer takes per-language training files and balances
    merge operations across languages using a development set or target compression ratios.

    Args:
        num_merges (:obj:`int`, `optional`):
            Number of BPE merge operations to perform. Defaults to ``32000``.

        variant (:obj:`str`, `optional`):
            Algorithm variant: ``"base"`` (default) or ``"window"`` (moving-window balancing).

        min_frequency (:obj:`int`, `optional`):
            Minimum pair frequency to merge. Defaults to ``2``.

        global_merges (:obj:`int`, `optional`):
            Number of initial standard BPE merges before switching to parity mode. Defaults to ``0``.

        window_size (:obj:`int`, `optional`):
            Window size for the ``"window"`` variant. Defaults to ``100``.

        alpha (:obj:`float`, `optional`):
            Alpha parameter for the ``"window"`` variant. Defaults to ``2.0``.

        total_symbols (:obj:`bool`, `optional`):
            If True, subtract unique character count from ``num_merges``. Defaults to ``False``.
    """
    def __init__(
        self,
        num_merges=32000,
        variant="base",
        min_frequency=2,
        ratio=None,
        global_merges=0,
        window_size=100,
        alpha=2.0,
        total_symbols=False,
    ):
        pass

    def train(
        self,
        train_files,
        dev_files=None,
        ratio=None,
        output=None,
    ):
        """
        Train parity-aware BPE and return a ready-to-use Tokenizer.

        Args:
            train_files (:obj:`List[str]`):
                List of training file paths, one per language.

            dev_files (:obj:`List[str]`, `optional`):
                List of development file paths for parity computation.

            ratio (:obj:`List[float]`, `optional`):
                Target compression ratios per language (alternative to dev_files).

            output (:obj:`str`, `optional`):
                Path to write merge rules to a file.

        Returns:
            :class:`~tokenizers.Tokenizer`
        """
        pass

class WordPieceTrainer(Trainer):
    """
    Trainer capable of training a WordPiece model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.
    """
    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix="##",
        end_of_word_suffix=None,
    ):
        pass
