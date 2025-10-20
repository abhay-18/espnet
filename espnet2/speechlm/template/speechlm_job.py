#!/usr/bin/env python3
"""SpeechLM job template for training configuration."""

from typing import Any, Callable, Dict
import yaml
import re
import torch.nn as nn

from espnet2.speechlm.template import AbsJobTemplate
from espnet2.speechlm.configuration import SPEECHLM_TASK_CONFIGS
from espnet2.speechlm.multimodal_io import AbsIO, MULTIMODAL_IOS
from espnet2.speechlm.model.speechlm import SPEECHLM_MODELS

class SpeechLMJobTemplate(AbsJobTemplate):
    """Job template for SpeechLM training tasks.

    This class implements the specific model and data processing
    configurations for speech language modeling tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the SpeechLM job template.

        Args:
            config: Dictionary containing job configuration parameters.
        """
        super().__init__(config)

        # (1) keep other configs
        self.config = config
        
        # (2) build tokenizers and vocabulary
        io_config = config['multimodal_io']
        self.multimodal_io = dict()
        for io_name, io_config in io_config.items():
            multimodal_io_class = MULTIMODAL_IOS[io_name]
            assert issubclass(multimodal_io_class, AbsIO)
            self.multimodal_io[io_name] = multimodal_io_class(**io_config)
        
        self.vocab, self.vocab_interval = self._build_vocabulary()

    def _build_vocabulary(self, num_special_tokens=256):
        # (1) Initial special token. We keep a fixed number of slots
        vocab_interval = {
            "special_token": (0, num_special_tokens)
        }
        vocab = [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|eot|>",
            "<|text|>",
            "<|audio|>",
            "<|image|>",
            "<|toolcall|>",
        ]
        while len(vocab) < num_special_tokens:
            vocab.append(f"<|unused_{len(vocab)}|>")
        
        # (2) add vocabulary from each discrete multimodal IO.
        start = num_special_tokens
        for io_name, io in self.multimodal_io.items():
            if io.is_discrete:
                vocab.extend(io.get_vocabulary())
                vocab_interval[io_name] = (start, len(vocab))
                start = len(vocab)

        assert len(vocab) == len(set(vocab)), "There are duplicated tokens in the vocab"

        return vocab, vocab_interval
        

    def build_preprocessor(self) -> Callable:
        """Build the data collation function for SpeechLM.

        Returns:
            A callable function for collating SpeechLM batch data.
        """

        processor_config = self.config['preprocessor']
        multimodal_io = {
            io_name: io.copy_for_worker()
            for io_name, io in self.multimodal_io.items()
        }
        return SpeechLMPreprocessor(
            multimodal_io=multimodal_io,
            vocab=self.vocab,
            vocab_interval=self.vocab_interval,
            audio_input=processor_config['audio_input'],
            audio_output=processor_config['audio_output'],
        )

    def build_model(self) -> nn.Module:
        """Build the SpeechLM model.

        Returns:
            A SpeechLM model instance.
        """

        model_config = self.config['model']
        model_class = SPEECHLM_MODELS[model_config['model_choice']]

        model = model_class(
            model_hf_tag=model_config['model_hf_tag'],
            multimodal_io=self.multimodal_io,
            vocab_interval=self.vocab_interval,
            **model_config['model_conf']
        )

        if model_config.get('activation_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        return model


class SpeechLMPreprocessor:
    def __init__(
        self,
        multimodal_io,
        vocab,
        vocab_interval,
        audio_input: str = "continuous_audio",
        audio_output: str = "discrete_audio",
    ):
        
        # (1) keep all multimodal_io
        self.multimodal_io = multimodal_io
        self.audio_input = audio_input
        self.audio_output = audio_output

        # (2) vocabulary
        self.vocab = vocab
        self.vocab_interval = vocab_interval

        # (3) Additional add-on operations:
    
    def find_length(self, key, data_dict):
        task, _, _ = key
        messages = self._apply_chat_template(task, data_dict)

        # (1) <bos>
        length = 2

        # (2) each message, consider role, modality and end of <eot>/<eos>
        for _, this_io, this_data in messages:
            length += 3
            length += self.multimodal_io[this_io].find_length(this_data)
        
        return length

    def __call__(self, data_lst):
        raise NotImplementedError
    
    def process_single(self, data):
        (task, data_name, example_id), data_dict = data

        if "dialogue" not in data_dict:
            messages = self._apply_chat_template(data_dict, task)
        else:
            messages = data_dict['dialogue']

        raise NotImplementedError
    
    def _apply_chat_template(self, task, data_dict):
        if "dialogue" in data_dict:
            if len(data_dict) != 1:
                raise ValueError("If dialogue exist, there should be no more other entries")
            return data_dict['dialogue']
        else:
            task_config = SPEECHLM_TASK_CONFIGS[task]
            messages = list()
            for role, entry in task_config:
                if bool(re.match(r"^audio", entry)):
                    if role == "user" or role == "system":
                        this_io = self.audio_input
                    elif role == "assistant":
                        this_io = self.audio_output
                elif bool(re.match(r"^text", entry)):
                    this_io = "text"
                else:
                    raise ValueError(f"Not supported data entry in template: {entry}")
                
                this_data = data_dict[entry]
                message = (role, this_io, this_data)
                messages.append(message)
            return messages

    def _special_token(self, name):
        token = f"<|{name}|>"
        special_range = self.vocab_interval["special_token"][1]
        return self.vocab[:special_range].index(token)


if __name__ == "__main__":
    config = "test.yaml"
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
        template = SpeechLMJobTemplate(config)
        template.build_model()
