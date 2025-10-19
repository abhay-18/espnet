"""HuggingFace tokenizer-based text I/O implementation."""

from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer, AutoConfig

from espnet2.speechlm.multimodal_io.abs_io import AbsIO


class HuggingFaceTextIO(AbsIO):
    """Text I/O using HuggingFace tokenizers.

    This class implements text encoding/decoding using HuggingFace's
    pretrained tokenizers. Text is discrete with a single stream.
    """

    def __init__(self, tokenizer_name: str):
        """Initialize HuggingFace text tokenizer.

        Args:
            tokenizer_name: HuggingFace model name or path for the tokenizer
                           (e.g., "bert-base-uncased", "gpt2",
                           "facebook/opt-125m")
        """
        super().__init__(modality="text", is_discrete=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

        # NOTE(Jinchuan): As the model would reserve some unused slots for future
        # expansion, this vocabulary length is smaller than model embeddings size.
        # Here records the real model embedding size.
        self.real_vocab_size = AutoConfig.from_pretrained(tokenizer_name).vocab_size

    def encode_batch(self, batch_data: List[str]) -> Dict[str, np.ndarray]:
        """Encode a batch of text strings into token IDs.

        Args:
            batch_data: List of text strings

        Returns:
            Dictionary containing:
                - 'data': Token IDs array [batch_size, max_seq_len]
                - 'lengths': Sequence lengths array [batch_size]
        """
        if len(batch_data) == 0:
            raise ValueError("encode_batch requires at least one text string")

        # Tokenize batch with padding to the longest sequence
        encoded = self.tokenizer(
            batch_data,
            padding=True,  # Pad to longest sequence in batch
            truncation=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"]  # Shape: [batch_size, max_seq_len]
        attention_mask = encoded["attention_mask"]  # Shape: [batch_size, max_seq_len]

        # Calculate actual lengths from attention mask
        lengths = np.sum(attention_mask, axis=1, dtype=np.int32)  # Shape: [batch_size]

        return {
            "data": input_ids,
            "lengths": lengths,
        }

    def decode_batch(self, batch_encoded: Dict[str, np.ndarray]) -> List[str]:
        """Decode a batch of token ID sequences back to text strings.

        Args:
            batch_encoded: Dictionary containing 'data' with token IDs
                          [batch_size, seq_len]

        Returns:
            List of decoded text strings
        """
        token_ids = batch_encoded["data"]

        if len(token_ids.shape) != 2:
            raise ValueError(
                f"Expected 2D array [batch_size, seq_len], "
                f"got shape {token_ids.shape}"
            )

        # Decode each sequence in the batch
        texts = []
        for i in range(token_ids.shape[0]):
            text = self.tokenizer.decode(
                token_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            texts.append(text)

        return texts

    def find_length(self, data: str) -> int:
        """Calculate token sequence length without full encoding.

        Args:
            data: Single text string

        Returns:
            Token count after tokenization
        """
        if not isinstance(data, str):
            raise ValueError(f"find_length expects a string, got {type(data)}")

        # Use the tokenizer's encode method to get accurate length
        # including special tokens (e.g., [CLS], [SEP] for BERT)
        token_ids = self.tokenizer.encode(
            data, truncation=True, add_special_tokens=True
        )

        return len(token_ids)

    def feature_dim(self) -> Optional[int]:
        """Get feature dimension (None for discrete text modality).

        Returns:
            None (text is discrete, not continuous)
        """
        return None

    def num_stream(self) -> Optional[int]:
        """Get number of streams (1 for text).

        Returns:
            1 (text uses single stream)
        """
        return 1

    def get_vocabulary(self) -> Optional[List[str]]:
        """Get the complete vocabulary list of the tokenizer.

        Returns:
            List of all tokens in the vocabulary
        """
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
        vocab = [token for token, _ in sorted_tokens]

        while len(vocab) < self.real_vocab_size:
            vocab.append(f"<|unused_text_{len(vocab)}|>")

        return vocab

    def get_stream_interval(self) -> Optional[List[tuple]]:
        """Get vocabulary intervals for all streams.

        Returns:
            List containing single tuple [(0, vocab_size)] for text's single stream
        """
        return [(0, self.real_vocab_size)]

    def get_stream_weight(self) -> Optional[List[float]]:
        """Get loss weights for all streams.

        Returns:
            List containing [1.0] for single text stream
        """
        return [1.0]
