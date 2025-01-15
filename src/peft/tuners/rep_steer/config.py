from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class RepSteerConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RepSteerModel`].
    Args:
        r (`int`):
            Number of representations to learn.
        # TODO: fill this out
    """
    r: int = field(default=1, metadata={"help": "Number of representations to learn."})
    shared_reps: bool = field(default=False, metadata={"help": "Whether to share the steering vector across layers."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.REP_STEER