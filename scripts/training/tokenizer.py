from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
)

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
@dataclass
class TokenizerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper

@singleton
class TokenizerSingleton:
    def __init__(self, token_args):
        if token_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(token_args.tokenizer_name, **token_args)
        elif token_args.tokenizer_name_or_path:
            self.tokenizer = LlamaTokenizer.from_pretrained(token_args.tokenizer_name_or_path, **token_args)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        self.tokenizer.add_eos_token = True

def create_tokenizer(token_args):
    return TokenizerSingleton(token_args).tokenizer




